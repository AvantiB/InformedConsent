#!/usr/bin/env python
"""Validate Union V0 round-trip smoke/full outputs.

Checks parse completion, Union V0 ID validity, repairable ID formatting errors,
overlap/nesting annotations, and interpretation-unit coverage. Works for older
outputs and newer runner outputs that include validation_summary and
invalid_annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                rows.append({"_jsonl_parse_error": str(exc), "_line_no": line_no, "_raw": line[:500]})
    return rows


def load_inventory(inventory_csv: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv).fillna("")
    required = ["union_element_id", "source_model", "source_element_id"]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    return inv


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
                return maps["by_pair"][key], "repairable"
    return uid, "invalid"


def validate_outputs(model_output_dir: Path, inventory_csv: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    inv = load_inventory(inventory_csv)
    maps = build_inventory_maps(inv)
    valid_ids = maps["valid_ids"]

    forward = load_jsonl(model_output_dir / "union_v0_forward_mappings.jsonl")
    backward = load_jsonl(model_output_dir / "union_v0_backward_reconstructions.jsonl")
    failed = load_jsonl(model_output_dir / "failed_requests.jsonl")

    invalid_rows = []
    n_annotations_total = 0
    n_valid_ids_exact = 0
    n_repaired_by_runner = 0
    n_repairable_not_yet_repaired = 0
    n_invalid_remaining = 0
    n_overlap_annotations = 0
    n_interpretation_units = 0
    n_records_with_interpretation = 0
    n_forward_parse_errors = sum(1 for x in forward if "_jsonl_parse_error" in x)
    span_relation_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}

    for rec in forward:
        source_id = rec.get("source_id", "")
        parsed = rec.get("parsed_forward") or {}
        annotations = parsed.get("annotations") or []
        invalid_annotations = parsed.get("invalid_annotations") or []
        units = parsed.get("interpretation_units") or []

        if isinstance(units, list):
            n_interpretation_units += len(units)
            if units:
                n_records_with_interpretation += 1
            for unit in units:
                if isinstance(unit, dict):
                    rel = unit.get("relationship", "")
                    relationship_counts[rel] = relationship_counts.get(rel, 0) + 1

        if isinstance(annotations, list):
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                n_annotations_total += 1
                uid = str(ann.get("union_element_id", ""))
                repaired_uid, status = repair_union_id(uid, maps)
                if uid in valid_ids:
                    n_valid_ids_exact += 1
                elif status == "repairable":
                    n_repairable_not_yet_repaired += 1
                    invalid_rows.append({
                        "source_id": source_id,
                        "span_text": ann.get("span_text", ""),
                        "union_element_id": uid,
                        "status": "repairable_not_yet_repaired",
                        "suggested_union_element_id": repaired_uid,
                        "annotation_id": ann.get("annotation_id", ""),
                        "rationale": ann.get("rationale", ""),
                    })
                else:
                    n_invalid_remaining += 1
                    invalid_rows.append({
                        "source_id": source_id,
                        "span_text": ann.get("span_text", ""),
                        "union_element_id": uid,
                        "status": "invalid",
                        "suggested_union_element_id": "",
                        "annotation_id": ann.get("annotation_id", ""),
                        "rationale": ann.get("rationale", ""),
                    })

                if ann.get("id_validation_status") == "repaired":
                    n_repaired_by_runner += 1
                if ann.get("overlap_group_id"):
                    n_overlap_annotations += 1
                span_rel = ann.get("span_relation", "")
                span_relation_counts[span_rel] = span_relation_counts.get(span_rel, 0) + 1

        if isinstance(invalid_annotations, list):
            for ann in invalid_annotations:
                if not isinstance(ann, dict):
                    continue
                n_invalid_remaining += 1
                invalid_rows.append({
                    "source_id": source_id,
                    "span_text": ann.get("span_text", ""),
                    "union_element_id": ann.get("invalid_union_element_id") or ann.get("union_element_id", ""),
                    "status": "invalid_annotations_from_runner",
                    "suggested_union_element_id": "",
                    "annotation_id": ann.get("annotation_id", ""),
                    "rationale": ann.get("rationale", ""),
                })

    n_problem = n_repairable_not_yet_repaired + n_invalid_remaining
    summary = {
        "model_output_dir": str(model_output_dir),
        "inventory_csv": str(inventory_csv),
        "n_forward_records": len(forward),
        "n_backward_records": len(backward),
        "n_failed_requests": len(failed),
        "n_forward_jsonl_parse_errors": n_forward_parse_errors,
        "n_annotations_total": n_annotations_total,
        "n_valid_ids_exact": n_valid_ids_exact,
        "n_repaired_ids_by_runner": n_repaired_by_runner,
        "n_repairable_ids_not_yet_repaired": n_repairable_not_yet_repaired,
        "n_invalid_ids_remaining": n_invalid_remaining,
        "n_problem_ids_total": n_problem,
        "problem_id_rate": n_problem / max(1, n_annotations_total),
        "n_overlap_annotations": n_overlap_annotations,
        "n_interpretation_units": n_interpretation_units,
        "n_records_with_interpretation_units": n_records_with_interpretation,
        "span_relation_counts": span_relation_counts,
        "interpretation_relationship_counts": relationship_counts,
        "ready_for_full_run": (
            len(forward) > 0
            and len(backward) == len(forward)
            and len(failed) == 0
            and n_forward_parse_errors == 0
            and n_repairable_not_yet_repaired == 0
            and n_invalid_remaining == 0
            and n_records_with_interpretation == len(forward)
        ),
    }
    return summary, pd.DataFrame(invalid_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_output_dir", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--output_prefix", default=None)
    args = ap.parse_args()

    model_output_dir = Path(args.model_output_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else model_output_dir / "validation"
    summary, invalid_df = validate_outputs(model_output_dir, Path(args.inventory_csv))

    summary_path = output_prefix.with_suffix(".summary.json")
    invalid_path = output_prefix.with_suffix(".invalid_annotations.csv")
    summary_path.write_text(json.dumps(summary, indent=2))
    invalid_df.to_csv(invalid_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {invalid_path}")


if __name__ == "__main__":
    main()
