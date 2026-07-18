#!/usr/bin/env python
"""Backfill Union V0 rows that repeatedly fail full JSON parsing.

This utility is intentionally narrow. It is for rare cases where the main Union V0
forward prompt produces useful but malformed/too-long JSON and the JSON-repair
prompt also fails. It reruns only selected missing rows with a stricter compact
forward prompt, appends validated forward records, then runs the standard Union V0
backward step for the new rows.

The compact prompt keeps the same Union V0 data dictionary and valid ID set, but
asks for only the essential meaning-preserving annotations and interpretation
units. This is preferable to leaving rows missing from paired evaluation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


COMPACT_MAX_ANNOTATIONS = 14


def load_union_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "03_run_union_v0_roundtrip.py"
    spec = importlib.util.spec_from_file_location("union_v0_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["union_v0_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def compact_dictionary_text(inv, max_chars: int | None = None) -> str:
    """Use the same inventory, but with a more compact line format."""
    lines = []
    for _, row in inv.iterrows():
        definition = str(row.get("source_element_definition", "") or "").strip()
        label = str(row.get("source_element_label", "") or "").strip()
        desc = f"{label}: {definition}" if definition else label
        line = f"- {row['union_element_id']} [{row['source_model']}; {row.get('element_scope', 'span')}] {desc}"
        lines.append(" ".join(line.split()))
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        # Keep all IDs visible, but definitions may be shortened globally by line.
        shortened = []
        per_line = max(160, max_chars // max(1, len(lines)))
        for line in lines:
            shortened.append(line[:per_line])
        text = "\n".join(shortened)
    return text


def build_compact_forward_messages(sentence: str, dictionary_text: str) -> list[dict[str, str]]:
    system = (
        "You are an NLP annotator for informed-consent documents. "
        "Return one syntactically valid JSON object only. No markdown. No comments."
    )
    user = f"""
Annotate the sentence using ONLY union_element_id values copied exactly from the data dictionary.

This is a compact recovery run for a row whose previous full Union V0 JSON was malformed.
Prioritize meaning preservation over exhaustive labeling.

Rules:
- Return VALID JSON only.
- Use at most {COMPACT_MAX_ANNOTATIONS} annotation objects.
- Select the most important spans needed to reconstruct the consent meaning: permission/denial, action, data/specimen/resource, actor/recipient, purpose, condition, restriction, withdrawal, and time period.
- Use exact contiguous span_text from the sentence.
- Copy union_element_id exactly, including double colons.
- Do not invent IDs.
- If no ID fits a meaningful phrase, put it in unmatched_language.
- sentence_decision must be one of: permit, deny, mixed, unclear.
- sentence-level elements may appear only in sentence_level_elements.
- interpretation_units should summarize the meaning needed for reconstruction, not list ontology labels.

Data dictionary:
{dictionary_text}

Sentence:
{sentence}

Return exactly this JSON object shape:
{{
  "sentence_decision": "permit",
  "sentence_level_elements": [],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "union_element_id": "SOURCE::ELEMENT_ID",
      "overlap_group_id": null,
      "span_relation": "single",
      "rationale": "brief rationale"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "evidence_span_text": "span or phrase represented by this unit",
      "annotation_ids": ["a1"],
      "relationship": "single",
      "combined_meaning": "meaning to preserve for reconstruction",
      "backward_mapping_decision": "use_as_core_meaning",
      "rationale": "brief explanation"
    }}
  ],
  "unmatched_language": []
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_failed_forward_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if str(obj.get("stage")) in {"forward", "forward_repair"} and obj.get("source_id"):
                ids.add(str(obj["source_id"]))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--source_ids", default=None, help="Comma-separated source_ids to compact-backfill. Defaults to missing failed rows.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    ap.add_argument("--dictionary_max_chars", type=int, default=None, help="Optional total character cap for compact dictionary text.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mod = load_union_runner(repo_root)

    model_out = Path(args.output_dir) / args.model_key
    model_out.mkdir(parents=True, exist_ok=True)
    forward_path = model_out / "union_v0_forward_mappings.jsonl"
    backward_path = model_out / "union_v0_backward_reconstructions.jsonl"
    failures_path = model_out / "failed_requests.jsonl"
    audit_path = model_out / "compact_backfill_audit.jsonl"

    rows = mod.load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    inv = mod.load_inventory(Path(args.inventory_csv))
    dictionary_text_full = mod.build_dictionary_text(inv)
    dictionary_text_compact = compact_dictionary_text(inv, args.dictionary_max_chars)
    maps = mod.build_inventory_maps(inv)
    model_cfg = mod.load_model_config(Path(args.model_config_yaml), args.model_key)
    client = mod.make_client(model_cfg)

    done_forward = mod.read_done_keys(forward_path)
    failed_forward = read_failed_forward_ids(failures_path)
    all_rows_by_id = {str(r["_source_id"]): r for _, r in rows.iterrows()}

    if args.source_ids:
        requested = {x.strip() for x in args.source_ids.split(",") if x.strip()}
    else:
        missing = set(all_rows_by_id) - set(done_forward)
        requested = missing | (failed_forward & missing)

    target_ids = [sid for sid in rows["_source_id"].astype(str).tolist() if sid in requested and sid not in done_forward]

    if not target_ids:
        print("No missing forward rows to compact-backfill.")
        print(json.dumps({"compact_backfilled": 0, "compact_failures": 0}, indent=2))
        return

    print(f"Compact-backfilling {len(target_ids)} forward rows: {target_ids}")

    ok = 0
    failed = 0
    for sid in target_ids:
        row = all_rows_by_id[sid]
        raw = ""
        try:
            raw = mod.call_chat(client, model_cfg, build_compact_forward_messages(row["_source_text"], dictionary_text_compact))
            parsed_raw = mod.extract_json(raw)
            parsed, validation = mod.validate_forward_obj(parsed_raw, maps)
            mod.append_jsonl(forward_path, {
                "source_id": sid,
                "source_text": row["_source_text"],
                "model_key": model_cfg["model_key"],
                "model": model_cfg["model"],
                "stage": "forward",
                "parse_ok": True,
                "parse_repaired": False,
                "compact_backfill": True,
                "validation_summary": validation,
                "parsed_forward": parsed,
                "raw_response": raw,
            })
            mod.append_jsonl(audit_path, {
                "source_id": sid,
                "compact_backfill_success": True,
                "validation_summary": validation,
                "raw_response": raw,
            })
            ok += 1
            print(f"[compact forward] ok {sid} valid={validation['n_annotations_valid']} invalid={validation['n_annotations_invalid']}")
        except Exception as exc:
            failed += 1
            mod.append_jsonl(failures_path, {"source_id": sid, "stage": "forward_compact_backfill", "error": repr(exc)})
            mod.append_jsonl(audit_path, {
                "source_id": sid,
                "compact_backfill_success": False,
                "error": repr(exc),
                "raw_response": raw,
            })
            print(f"[compact forward] FAILED {sid}: {exc}", file=sys.stderr)

    if ok:
        print("Running standard backward for newly compact-backfilled rows...")
        mod.run_backward(client, model_cfg, dictionary_text_full, model_out)
        mod.write_roundtrip_csv(forward_path, backward_path, model_out / "union_v0_roundtrip_outputs.csv")

    print(json.dumps({"compact_backfilled": ok, "compact_failures": failed}, indent=2))


if __name__ == "__main__":
    main()
