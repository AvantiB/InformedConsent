#!/usr/bin/env python
"""Run Union V0 full-dictionary forward/backward round-trip experiments.

Designed for one model at a time. Open-source models can be served with vLLM's
OpenAI-compatible API, while closed-source models can use the same OpenAI client
interface. Outputs are append-only JSONL files so interrupted runs can resume.

Forward mapping intentionally has two layers:
1. raw annotations, including same-span, overlapping, and nested labels; and
2. interpretation_units, where the LLM decides how related annotations should be
   considered together for backward reconstruction.

The runner validates Union V0 IDs against the inventory. Common unambiguous ID
formatting errors, such as ICO:0000108 instead of ICO::ICO:0000108, are repaired.
Remaining invalid IDs are moved to invalid_annotations and are not treated as
primary dictionary-grounded evidence for backward reconstruction.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. Install with: pip install openai") from exc

TEXT_COL_CANDIDATES = [
    "canonical_full_text",
    "full_text_original",
    "original_sentence",
    "full_text",
    "sentence",
    "text",
]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "id"]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")
    return None


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_rows(roundtrips_csv: Path, limit: int | None, no_dedupe_sentences: bool) -> pd.DataFrame:
    df = pd.read_csv(roundtrips_csv)
    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    id_col = pick_col(df, ID_COL_CANDIDATES, required=False)
    out = df.copy()
    out["_source_text"] = out[text_col].map(norm_text)
    out["_source_id"] = out[id_col].astype(str) if id_col else out["_source_text"].map(stable_id)
    out = out[out["_source_text"].astype(bool)].copy()
    if not no_dedupe_sentences:
        out = out.drop_duplicates(subset=["_source_text"]).copy()
        out["_source_id"] = out["_source_text"].map(stable_id)
    out = out.reset_index(drop=True)
    if limit is not None:
        out = out.head(limit).copy()
    return out[["_source_id", "_source_text"]]


def load_inventory(inventory_csv: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv).fillna("")
    required = [
        "union_element_id",
        "source_model",
        "source_element_id",
        "source_element_label",
        "source_element_definition",
    ]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    if "element_scope" not in inv.columns:
        inv["element_scope"] = "span"
    return inv


def build_dictionary_text(inv: pd.DataFrame) -> str:
    lines = []
    for _, row in inv.iterrows():
        scope = row.get("element_scope", "span") or "span"
        label = norm_text(row["source_element_label"])
        definition = norm_text(row["source_element_definition"])
        desc = f"{label}: {definition}" if definition else label
        lines.append(f"- {row['union_element_id']} [{row['source_model']}; {scope}] {desc}")
    return "\n".join(lines)


def build_inventory_maps(inv: pd.DataFrame) -> dict[str, Any]:
    valid_ids = set(inv["union_element_id"].astype(str))
    by_pair: dict[tuple[str, str], str] = {}
    for _, row in inv.iterrows():
        source_model = str(row["source_model"])
        source_element_id = str(row["source_element_id"])
        union_element_id = str(row["union_element_id"])
        for alias in {source_model, source_model.replace("_Consent", "")}:
            by_pair[(alias, source_element_id)] = union_element_id
            if ":" in source_element_id:
                prefix, suffix = source_element_id.split(":", 1)
                if alias == prefix:
                    by_pair[(alias, suffix)] = union_element_id
    return {"valid_ids": valid_ids, "by_pair": by_pair}


def repair_union_id(uid: Any, maps: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(uid, str):
        return str(uid), "invalid"
    uid = uid.strip()
    if uid in maps["valid_ids"]:
        return uid, "valid"
    if "::" not in uid and ":" in uid:
        source_model, rest = uid.split(":", 1)
        for key in [(source_model, rest), (source_model, f"{source_model}:{rest}")]:
            if key in maps["by_pair"]:
                return maps["by_pair"][key], "repaired"
    return uid, "invalid"


def validate_forward_obj(forward_obj: dict[str, Any], maps: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = copy.deepcopy(forward_obj)
    annotations = obj.get("annotations") or []
    if not isinstance(annotations, list):
        annotations = []
    valid_annotations = []
    invalid_annotations = []
    n_valid = n_repaired = n_invalid = 0

    for ann in annotations:
        if not isinstance(ann, dict):
            n_invalid += 1
            invalid_annotations.append({"raw_annotation": ann, "id_validation_status": "invalid_non_object"})
            continue
        original_uid = ann.get("union_element_id", "")
        repaired_uid, status = repair_union_id(original_uid, maps)
        ann["id_validation_status"] = status
        if status == "valid":
            n_valid += 1
            valid_annotations.append(ann)
        elif status == "repaired":
            n_repaired += 1
            ann["original_union_element_id"] = original_uid
            ann["union_element_id"] = repaired_uid
            valid_annotations.append(ann)
        else:
            n_invalid += 1
            bad = copy.deepcopy(ann)
            bad["invalid_union_element_id"] = original_uid
            invalid_annotations.append(bad)

    obj["annotations"] = valid_annotations
    obj["invalid_annotations"] = invalid_annotations

    units = obj.get("interpretation_units") or []
    if not isinstance(units, list):
        units = []
    valid_ids = {str(a.get("annotation_id")) for a in valid_annotations if a.get("annotation_id") is not None}
    invalid_ids = {str(a.get("annotation_id")) for a in invalid_annotations if isinstance(a, dict) and a.get("annotation_id") is not None}
    for unit in units:
        if isinstance(unit, dict):
            ids = [str(x) for x in (unit.get("annotation_ids") or [])]
            unit["valid_annotation_ids"] = [x for x in ids if x in valid_ids]
            unit["invalid_annotation_ids"] = [x for x in ids if x in invalid_ids]
    obj["interpretation_units"] = units

    validation = {
        "n_annotations_raw": len(annotations),
        "n_annotations_valid": n_valid,
        "n_annotations_repaired": n_repaired,
        "n_annotations_invalid": n_invalid,
        "n_annotations_backward_eligible": len(valid_annotations),
        "n_interpretation_units": len(units),
        "has_invalid_ids": n_invalid > 0,
    }
    obj["validation_summary"] = validation
    return obj, validation


def load_model_config(path: Path, model_key: str) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    model_cfg = {**(cfg.get("defaults", {}) or {}), **((cfg.get("models", {}) or {}).get(model_key, {}))}
    if not model_cfg:
        raise KeyError(f"model_key={model_key!r} not found in {path}")
    model_cfg["model_key"] = model_key
    return model_cfg


def make_client(model_cfg: dict[str, Any]) -> OpenAI:
    api_key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if not api_key:
        api_key = "EMPTY"
    base_url = model_cfg.get("base_url")
    if base_url in {"", "null", None}:
        return OpenAI(api_key=api_key)
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_chat(client: OpenAI, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    kwargs = {
        "model": model_cfg["model"],
        "messages": messages,
        "max_tokens": int(model_cfg.get("max_tokens", 2200)),
        "timeout": float(model_cfg.get("timeout_seconds", 120)),
    }
    if model_cfg.get("temperature") is not None:
        kwargs["temperature"] = model_cfg.get("temperature", 0)
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_exc}")


def build_forward_messages(sentence: str, dictionary_text: str) -> list[dict[str, str]]:
    system = (
        "You are an NLP annotator for informed-consent documents. Your task is to consistently apply "
        "the combined source-model data dictionary to the input sentence. Return valid JSON only."
    )
    user = f"""
Task: annotate the informed-consent sentence using ONLY element IDs from the data dictionary below.

Important context:
- This dictionary is a naive union of multiple information models, not a reduced meta-model.
- Several elements may overlap, duplicate, specialize, or complement each other.
- The same or similar text span MAY receive more than one label.
- A larger phrase may receive a broader role, while a nested shorter phrase may receive a narrower or more specific role.
- Preserve overlaps/nesting relationships rather than forcing a single label too early.

Annotation rules:
- Find the smallest meaningful contiguous text span for each concept when possible.
- Assign one best union_element_id per annotation object.
- Copy union_element_id EXACTLY from the data dictionary, including punctuation such as "::".
- Do not create new IDs, abbreviate IDs, or convert double-colon IDs to single-colon IDs.
- If no exact dictionary ID fits, put the phrase in unmatched_language instead of creating an annotation.
- If the same span maps clearly to multiple source-model elements, output multiple annotation objects with the same span_text and a shared overlap_group_id.
- If a broad phrase and a nested narrower phrase both carry meaning, output both annotations and link them with a shared overlap_group_id.
- Sentence-level elements may be used only in sentence_level_elements, not as span annotations.
- sentence_decision must be one of: permit, deny, mixed, unclear.

Interpretation rules for backward mapping:
- After producing raw annotations, create interpretation_units.
- Each interpretation_unit should explain how related annotations should be considered together for reconstruction.
- Use interpretation_units to decide whether overlapping labels are equivalent, complementary, broad/narrow, conflicting, or uncertain.
- Do not merely collapse overlapping labels as redundant. Preserve specificity when a nested or narrower annotation adds meaning.
- If two labels mean essentially the same thing for the sentence, select the interpretation that best preserves meaning and record the redundancy in rationale.
- If one label is broad and another is narrower, preserve both in a combined meaning when both are needed.

Data dictionary:
{dictionary_text}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "sentence_level_elements": [{{"union_element_id": "...", "value": "..."}}],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "union_element_id": "...",
      "overlap_group_id": "g1 or null",
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span",
      "rationale": "brief rationale"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "evidence_span_text": "span or phrase represented by this unit",
      "annotation_ids": ["a1", "a2"],
      "relationship": "single|same_span_multiple_labels|nested_broad_narrow|complementary_roles|conflicting_or_uncertain",
      "combined_meaning": "final meaning to preserve for backward reconstruction",
      "backward_mapping_decision": "use_as_core_meaning|use_as_modifier|preserve_broad_and_specific|choose_more_specific|choose_broader|flag_uncertain",
      "rationale": "brief explanation of how the annotations should be considered together"
    }}
  ],
  "unmatched_language": [{{"span_text": "exact text span", "reason": "brief reason"}}]
}}

Sentence:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_backward_messages(forward_obj: dict[str, Any], dictionary_text: str) -> list[dict[str, str]]:
    system = "You reconstruct informed-consent sentence meaning from structured annotations. Do not see the original sentence. Return valid JSON only."
    mapping_text = json.dumps(forward_obj, ensure_ascii=False, indent=2)
    user = f"""
Task: reconstruct a concise natural-language consent sentence that preserves the meaning of the structured mapping.

Use the mapping as follows:
- interpretation_units are the primary source for reconstruction.
- annotations are valid dictionary-grounded supporting evidence and should be used to understand exact source-model elements and span relationships.
- invalid_annotations are not dictionary-grounded. Do not treat their invalid IDs as authoritative. Use their span text only if needed as unmatched or cautionary evidence.
- Do not reconstruct by simply listing every annotation label.
- If multiple labels refer to the same span and interpretation_units mark them as equivalent, express the shared meaning once.
- If a broader span and a nested narrower span both add meaning, preserve the combined broad+narrow meaning.
- If annotations are complementary, include all complementary meaning needed for preservation.
- Do not add details that are not in the mapping.
- Preserve permission/denial, action, object, actor/recipient, purpose, condition, restriction, and temporal meaning when present.

Data dictionary:
{dictionary_text}

Structured mapping:
{mapping_text}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_done_keys(path: Path, key_field: str = "source_id") -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if key_field in obj:
                    done.add(str(obj[key_field]))
            except Exception:
                continue
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[str(obj["source_id"])] = obj
    return out


def write_roundtrip_csv(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    forward = load_jsonl_by_id(forward_path)
    backward = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, fwd in forward.items():
        parsed = fwd.get("parsed_forward") or {}
        validation = fwd.get("validation_summary") or parsed.get("validation_summary") or {}
        annotations = parsed.get("annotations") or []
        invalid_annotations = parsed.get("invalid_annotations") or []
        units = parsed.get("interpretation_units") or []
        bwd = backward.get(source_id, {})
        rows.append({
            "source_id": source_id,
            "source_text": fwd.get("source_text", ""),
            "sentence_decision": parsed.get("sentence_decision", ""),
            "n_annotations_raw": validation.get("n_annotations_raw", ""),
            "n_annotations_valid": validation.get("n_annotations_valid", ""),
            "n_annotations_repaired": validation.get("n_annotations_repaired", ""),
            "n_annotations_invalid": validation.get("n_annotations_invalid", ""),
            "n_annotations_backward_eligible": validation.get("n_annotations_backward_eligible", ""),
            "n_interpretation_units": len(units) if isinstance(units, list) else "",
            "forward_parse_ok": fwd.get("parse_ok", False),
            "backward_parse_ok": bwd.get("parse_ok", False),
            "reconstructed_sentence": (bwd.get("parsed_backward") or {}).get("reconstructed_sentence", ""),
            "annotations_json": json.dumps(annotations, ensure_ascii=False),
            "invalid_annotations_json": json.dumps(invalid_annotations, ensure_ascii=False),
            "interpretation_units_json": json.dumps(units, ensure_ascii=False),
            "forward_raw": fwd.get("raw_response", ""),
            "backward_raw": bwd.get("raw_response", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_forward(rows: pd.DataFrame, client: OpenAI, model_cfg: dict[str, Any], dictionary_text: str, maps: dict[str, Any], out_dir: Path) -> None:
    forward_path = out_dir / "union_v0_forward_mappings.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    done = read_done_keys(forward_path)
    for i, row in rows.iterrows():
        source_id = str(row["_source_id"])
        if source_id in done:
            continue
        try:
            raw = call_chat(client, model_cfg, build_forward_messages(row["_source_text"], dictionary_text))
            parsed_raw = extract_json(raw)
            parsed, validation = validate_forward_obj(parsed_raw, maps)
            append_jsonl(forward_path, {
                "source_id": source_id,
                "source_text": row["_source_text"],
                "model_key": model_cfg["model_key"],
                "model": model_cfg["model"],
                "stage": "forward",
                "parse_ok": True,
                "validation_summary": validation,
                "parsed_forward": parsed,
                "raw_response": raw,
            })
            done.add(source_id)
            print(f"[forward] {i + 1}/{len(rows)} ok {source_id} valid={validation['n_annotations_valid']} repaired={validation['n_annotations_repaired']} invalid={validation['n_annotations_invalid']}")
        except Exception as exc:
            append_jsonl(failures_path, {"source_id": source_id, "stage": "forward", "error": repr(exc)})
            print(f"[forward] {i + 1}/{len(rows)} FAILED {source_id}: {exc}", file=sys.stderr)


def run_backward(client: OpenAI, model_cfg: dict[str, Any], dictionary_text: str, out_dir: Path) -> None:
    forward_path = out_dir / "union_v0_forward_mappings.jsonl"
    backward_path = out_dir / "union_v0_backward_reconstructions.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    forward = load_jsonl_by_id(forward_path)
    done = read_done_keys(backward_path)
    for i, (source_id, fwd) in enumerate(forward.items()):
        if source_id in done:
            continue
        try:
            raw = call_chat(client, model_cfg, build_backward_messages(fwd.get("parsed_forward") or {}, dictionary_text))
            parsed = extract_json(raw)
            append_jsonl(backward_path, {
                "source_id": source_id,
                "source_text": fwd.get("source_text", ""),
                "model_key": model_cfg["model_key"],
                "model": model_cfg["model"],
                "stage": "backward",
                "parse_ok": True,
                "parsed_backward": parsed,
                "raw_response": raw,
            })
            done.add(source_id)
            print(f"[backward] {i + 1}/{len(forward)} ok {source_id}")
        except Exception as exc:
            append_jsonl(failures_path, {"source_id": source_id, "stage": "backward", "error": repr(exc)})
            print(f"[backward] {i + 1}/{len(forward)} FAILED {source_id}: {exc}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    inv = load_inventory(Path(args.inventory_csv))
    dictionary_text = build_dictionary_text(inv)
    maps = build_inventory_maps(inv)
    model_cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(model_cfg)

    run_meta = {
        "model_key": args.model_key,
        "model": model_cfg.get("model"),
        "stage": args.stage,
        "n_input_rows": int(len(rows)),
        "n_union_elements": int(len(inv)),
        "inventory_csv": args.inventory_csv,
        "roundtrips_csv": args.roundtrips_csv,
        "prompt_design": "overlap_aware_raw_annotations_plus_interpretation_units",
        "id_validation": "repair_unambiguous_ids_and_flag_remaining_invalid",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    if args.stage in {"forward", "both"}:
        run_forward(rows, client, model_cfg, dictionary_text, maps, output_dir)
    if args.stage in {"backward", "both"}:
        run_backward(client, model_cfg, dictionary_text, output_dir)

    write_roundtrip_csv(
        output_dir / "union_v0_forward_mappings.jsonl",
        output_dir / "union_v0_backward_reconstructions.jsonl",
        output_dir / "union_v0_roundtrip_outputs.csv",
    )
    print(f"Wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
