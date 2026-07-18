#!/usr/bin/env python
"""Repair Union V0 forward rows that failed because the model returned malformed JSON.

This is a narrow backfill utility for cases where the model produced a useful but
syntactically invalid JSON forward mapping. It reloads the existing Union V0 runner,
reruns only missing/failed forward rows, asks the same model to repair malformed JSON
when needed, appends successful forward records, runs backward for the newly available
forward records, and rewrites the round-trip CSV.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_union_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "03_run_union_v0_roundtrip.py"
    spec = importlib.util.spec_from_file_location("union_v0_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["union_v0_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


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
            if str(obj.get("stage")) == "forward" and obj.get("source_id"):
                ids.add(str(obj["source_id"]))
    return ids


def build_json_repair_messages(raw_response: str, parse_error: str) -> list[dict[str, str]]:
    system = "You repair malformed JSON. Return valid JSON only. Do not add explanations."
    user = f"""
The following model output was intended to be a Union V0 informed-consent forward mapping JSON object, but it failed JSON parsing.

Parse error:
{parse_error}

Repair rules:
- Return one valid JSON object only.
- Preserve the same meaning, labels, spans, annotation IDs, interpretation units, and unmatched language already present.
- Do not add new annotations or remove substantive annotations unless required only to fix syntax.
- Use null, not the string "null", when a value is null.
- Ensure all arrays/objects have commas between items.
- Ensure strings are quoted and internal quotes are escaped.
- The repaired object must contain these top-level keys if possible:
  sentence_decision, sentence_level_elements, annotations, interpretation_units, unmatched_language.

Malformed output:
{raw_response}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_or_repair(mod, client, model_cfg: dict[str, Any], raw: str) -> tuple[dict[str, Any], str | None, str | None]:
    try:
        return mod.extract_json(raw), None, None
    except Exception as exc:
        parse_error = repr(exc)
    repair_raw = mod.call_chat(client, model_cfg, build_json_repair_messages(raw, parse_error))
    repaired = mod.extract_json(repair_raw)
    return repaired, repair_raw, parse_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--source_ids", default=None, help="Optional comma-separated source_ids to repair. Defaults to failed/missing forward rows.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mod = load_union_runner(repo_root)

    model_out = Path(args.output_dir) / args.model_key
    model_out.mkdir(parents=True, exist_ok=True)
    forward_path = model_out / "union_v0_forward_mappings.jsonl"
    backward_path = model_out / "union_v0_backward_reconstructions.jsonl"
    failures_path = model_out / "failed_requests.jsonl"
    repair_audit_path = model_out / "repaired_forward_parse_failures.jsonl"

    rows = mod.load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    inv = mod.load_inventory(Path(args.inventory_csv))
    dictionary_text = mod.build_dictionary_text(inv)
    maps = mod.build_inventory_maps(inv)
    model_cfg = mod.load_model_config(Path(args.model_config_yaml), args.model_key)
    client = mod.make_client(model_cfg)

    done_forward = mod.read_done_keys(forward_path)
    failed_forward = read_failed_forward_ids(failures_path)
    all_rows_by_id = {str(r["_source_id"]): r for _, r in rows.iterrows()}

    if args.source_ids:
        target_ids = {x.strip() for x in args.source_ids.split(",") if x.strip()}
    else:
        missing = set(all_rows_by_id) - set(done_forward)
        target_ids = missing | (failed_forward & missing)

    target_ids = [sid for sid in rows["_source_id"].astype(str).tolist() if sid in target_ids and sid not in done_forward]

    if not target_ids:
        print("No missing/failed forward rows to repair.")
    else:
        print(f"Repairing {len(target_ids)} forward rows: {target_ids}")

    repaired_count = 0
    failed_count = 0
    for sid in target_ids:
        row = all_rows_by_id[sid]
        try:
            raw = mod.call_chat(client, model_cfg, mod.build_forward_messages(row["_source_text"], dictionary_text))
            parsed_raw, repair_raw, parse_error = parse_or_repair(mod, client, model_cfg, raw)
            parsed, validation = mod.validate_forward_obj(parsed_raw, maps)
            mod.append_jsonl(forward_path, {
                "source_id": sid,
                "source_text": row["_source_text"],
                "model_key": model_cfg["model_key"],
                "model": model_cfg["model"],
                "stage": "forward",
                "parse_ok": True,
                "parse_repaired": repair_raw is not None,
                "initial_parse_error": parse_error,
                "validation_summary": validation,
                "parsed_forward": parsed,
                "raw_response": raw,
                "repair_response": repair_raw,
            })
            mod.append_jsonl(repair_audit_path, {
                "source_id": sid,
                "repair_success": True,
                "parse_repaired": repair_raw is not None,
                "initial_parse_error": parse_error,
                "validation_summary": validation,
            })
            repaired_count += 1
            print(f"[repair forward] ok {sid} repaired={repair_raw is not None} valid={validation['n_annotations_valid']} invalid={validation['n_annotations_invalid']}")
        except Exception as exc:
            failed_count += 1
            mod.append_jsonl(failures_path, {"source_id": sid, "stage": "forward_repair", "error": repr(exc)})
            mod.append_jsonl(repair_audit_path, {"source_id": sid, "repair_success": False, "error": repr(exc)})
            print(f"[repair forward] FAILED {sid}: {exc}", file=sys.stderr)

    if repaired_count:
        print("Running backward for any newly repaired forward rows...")
        mod.run_backward(client, model_cfg, dictionary_text, model_out)
        mod.write_roundtrip_csv(forward_path, backward_path, model_out / "union_v0_roundtrip_outputs.csv")

    print(json.dumps({"forward_repaired_or_backfilled": repaired_count, "repair_failures": failed_count}, indent=2))


if __name__ == "__main__":
    main()
