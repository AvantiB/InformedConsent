#!/usr/bin/env python
"""Filter Phase 1 round-trip outputs before meta-model development.

This script removes NA-only/deferred-tagging rows from schema-induction evidence.
It is intended for both fresh strict Phase 1 outputs and legacy researcher/output
CSVs that may contain entries such as "NA [NA] (permit)". Such rows can still be
kept for coverage-inclusive evaluation, but they must not contribute to
frequency counts, co-occurrence, relationship evidence, functionally validated
annotation sets, or LLM schema-induction cards.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

NA_LABELS = {"", "na", "n/a", "none", "null", "unknown", "unmatched", "unmatched_language", "no_match", "invalid"}
DECISION_VALUES = {"permit", "permission", "deny", "denial", "prohibit", "prohibition", "mixed", "unclear", "obligation", "duty", "yes", "no"}
BANNED_BACKWARD_KEYS = [
    "unmatched_language",
    "evidence_span_text",
    "combined_meaning",
    "backward_mapping_decision",
    "rationale",
    "raw_response",
    "original_sentence",
]


def norm_text(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = norm_text(x).casefold()
    return s in {"true", "1", "yes", "y"}


def to_num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_json_loads(x: Any) -> Any:
    s = norm_text(x)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_items_from_packet(row: pd.Series) -> list[dict[str, Any]]:
    for col in ["backward_packet_json", "sanitized_forward_material_json"]:
        if col in row.index:
            obj = safe_json_loads(row.get(col))
            if isinstance(obj, dict):
                items = obj.get("ordered_reconstruction_items") or []
                if isinstance(items, list):
                    return [x for x in items if isinstance(x, dict)]
    return []


def is_naish_label(label: Any) -> bool:
    s = norm_text(label).casefold().strip("[](){} .,:;-")
    return s in NA_LABELS


def annotations_serialized_is_na_only(text: Any) -> bool:
    s = norm_text(text)
    if not s:
        return False
    low = s.casefold()
    # Explicit old format: NA [NA] (permit), NA[NA](deny), [NA].
    if re.fullmatch(r"(?:na|n/a|none|null|unknown)?\s*\[\s*(?:na|n/a|none|null|unknown)\s*\]\s*(?:\((?:permit|deny|mixed|unclear|permission|prohibition|yes|no)\))?", low):
        return True
    # If every bracketed label is NA-like and there is no non-NA annotation span, treat as NA-only.
    labels = re.findall(r"\[([^\]]+)\]", low)
    if labels and all(is_naish_label(x) for x in labels):
        non_na_text = re.sub(r"\[[^\]]+\]", "", low)
        non_na_text = re.sub(r"\([^)]*\)", "", non_na_text)
        non_na_text = non_na_text.strip(" .,:;-[](){}")
        if not non_na_text or non_na_text in NA_LABELS:
            return True
    return False


def row_has_valid_annotation_evidence(row: pd.Series) -> bool:
    # Fresh strict outputs should expose the count directly.
    for col in ["n_annotations_backward_eligible", "annotation_count", "n_annotations_valid"]:
        if col in row.index and to_num(row.get(col)) > 0:
            return True
    items = extract_items_from_packet(row)
    if items:
        for item in items:
            span = norm_text(item.get("span_text"))
            label = norm_text(item.get("label_id") or item.get("label") or item.get("source_element_id") or item.get("union_label_id"))
            if span and not is_naish_label(label):
                return True
    # Legacy source columns.
    for col in ["annotations_serialized", "annotations_combined", "annotations_json"]:
        if col in row.index and norm_text(row.get(col)):
            if annotations_serialized_is_na_only(row.get(col)):
                return False
            if "[" in norm_text(row.get(col)) or safe_json_loads(row.get(col)) is not None:
                return True
    return False


def row_is_na_only_or_no_evidence(row: pd.Series) -> tuple[bool, str]:
    if "exclude_from_schema_induction" in row.index and as_bool(row.get("exclude_from_schema_induction")):
        return True, norm_text(row.get("exclude_reason")) or "preflagged_exclude_from_schema_induction"
    if "na_only_annotation_row" in row.index and as_bool(row.get("na_only_annotation_row")):
        return True, "preflagged_na_only_annotation_row"
    for col in ["annotations_serialized", "annotations_combined", "annotations_json"]:
        if col in row.index and annotations_serialized_is_na_only(row.get(col)):
            return True, f"legacy_na_only_annotation:{col}"
    if not row_has_valid_annotation_evidence(row):
        return True, "zero_valid_span_annotation_evidence"
    return False, ""


def backward_packet_contains_banned_keys(row: pd.Series) -> list[str]:
    packets = []
    for col in ["backward_packet_json", "sanitized_forward_material_json"]:
        if col in row.index:
            packets.append(norm_text(row.get(col)))
    text = "\n".join(packets)
    if not text:
        return []
    return [key for key in BANNED_BACKWARD_KEYS if key in text]


def read_input_tables(paths: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            path = Path(p)
            if not path.exists() or path.is_dir():
                continue
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path).fillna("")
            elif path.suffix.lower() in {".jsonl", ".ndjson"}:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                df = pd.DataFrame(rows).fillna("")
            else:
                continue
            df["_source_file"] = str(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No input CSV/JSONL files matched: {paths}")
    return pd.concat(frames, ignore_index=True, sort=False).fillna("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV/JSONL paths or globs, e.g. 'meta_model/phase1/**/*.csv'")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", default="phase1_schema_induction")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_input_tables(args.inputs)

    exclude_flags = []
    exclude_reasons = []
    banned_hits = []
    for _, row in df.iterrows():
        exclude, reason = row_is_na_only_or_no_evidence(row)
        exclude_flags.append(bool(exclude))
        exclude_reasons.append(reason)
        banned_hits.append(";".join(backward_packet_contains_banned_keys(row)))

    df["exclude_from_schema_induction_final"] = exclude_flags
    df["schema_induction_exclude_reason"] = exclude_reasons
    df["backward_packet_banned_key_hits"] = banned_hits
    df["safe_for_schema_induction"] = (~df["exclude_from_schema_induction_final"]) & (df["backward_packet_banned_key_hits"].astype(str) == "")

    all_path = out_dir / f"{args.prefix}_all_rows_with_flags.csv"
    filtered_path = out_dir / f"{args.prefix}_filtered_rows.csv"
    excluded_path = out_dir / f"{args.prefix}_excluded_rows.csv"
    summary_path = out_dir / f"{args.prefix}_summary.json"

    df.to_csv(all_path, index=False)
    df[df["safe_for_schema_induction"]].to_csv(filtered_path, index=False)
    df[~df["safe_for_schema_induction"]].to_csv(excluded_path, index=False)

    summary = {
        "n_rows_total": int(len(df)),
        "n_rows_safe_for_schema_induction": int(df["safe_for_schema_induction"].sum()),
        "n_rows_excluded": int((~df["safe_for_schema_induction"]).sum()),
        "exclude_reason_counts": df.loc[~df["safe_for_schema_induction"], "schema_induction_exclude_reason"].value_counts(dropna=False).to_dict(),
        "banned_key_hit_counts": df.loc[df["backward_packet_banned_key_hits"].astype(str) != "", "backward_packet_banned_key_hits"].value_counts(dropna=False).to_dict(),
        "outputs": {
            "all_rows_with_flags": str(all_path),
            "filtered_rows": str(filtered_path),
            "excluded_rows": str(excluded_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
