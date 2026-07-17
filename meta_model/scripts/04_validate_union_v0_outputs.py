#!/usr/bin/env python
"""Validate Union V0 round-trip smoke/full outputs.

This script checks JSONL readability, request completion, Union V0 ID validity,
overlap/nesting annotations, and interpretation-unit coverage.

Readiness gates:
- strict: no invalid IDs anywhere.
- pragmatic: primary annotations are clean and quarantined invalid IDs are below
  a configurable rate threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

INVALID_COLUMNS = [
    "source_id",
    "span_text",
    "union_element_id",
    "status",
    "suggested_union_element_id",
    "annotation_id",
    "rationale",
]


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
                rows.append({
                    "_jsonl_parse_error": str(exc),
                    "_line_no": line_no,
                    "_raw": line[:500],
                })
    return rows


def load_inventory(path: Path) -> pd.DataFrame:
    inv = pd.read_csv(path).fillna("")
    required = ["union_element_id", "source_model", "source_element_id"]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    return inv


def build_inventory_maps(inv: pd.DataFrame) -> dict[str, Any]:
    valid_ids = set(inv["union_element_id"].astype(str))
    by_pair: dict[tuple[str, str], str] = {}
    by_source_id: dict[str, list[str]] = {}

    for _, row in inv.iterrows():
        union_id = str(row["union_element_id"])
        source_model = str(row["source_model"])
        source_id = str(row["source_element_id"])
        aliases = {
            source_model,
            source_model.replace("_Consent", ""),
            source_model.replace("_", ""),
        }
        for alias in aliases:
            by_pair[(alias, source_id)] = union_id
            if ":" in source_id:
                prefix, suffix = source_id.split(":", 1)
                if alias == prefix:
                    by_pair[(alias, suffix)] = union_id
        by_source_id.setdefault(source_id, []).append(union_id)

    return {"valid_ids": valid_ids, "by_pair": by_pair, "by_source_id": by_source_id}


def repair_union_id(uid: Any, maps: dict[str, Any]) -> tuple[str, str]:
    """Return (possibly repaired ID, status).

    status is one of: valid, repairable, invalid.
    """
    if not isinstance(uid, str):
        return str(uid), "invalid"
    uid = uid.strip()
    if not uid:
        return uid, "invalid"
    if uid in maps["valid_ids"]:
        return uid, "valid"

    # A bare source element ID can be repaired only if it maps uniquely.
    matches = maps["by_source_id"].get(uid, [])
    if len(matches) == 1:
        return matches[0], "repairable"

    # Common LLM error: ICO:0000108 instead of ICO::ICO:0000108,
    # or FHIR:Consent.provision.actor instead of FHIR_Consent::Consent.provision.actor.
    if "::" not in uid and ":" in uid:
        source_model, rest = uid.split(":", 1)
        candidates = [
            (source_model, rest),
            (source_model, f"{source_model}:{rest}"),
            (source_model.replace("FHIR", "FHIR_Consent"), rest),
        ]
        for key in candidates:
            if key in maps["by_pair"]:
                return maps["by_pair"][key], "repairable"

    return uid, "invalid"


def nonempty_value(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return bool(s) and s not in {"null", "none", "nan", ""}


def add_count(d: dict[str, int], key: Any) -> None:
    k = str(key or "")
    d[k] = d.get(k, 0) + 1


def validate_outputs(
    model_output_dir: Path,
    inventory_csv: Path,
    quarantine_rate_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inv = load_inventory(inventory_csv)
    maps = build_inventory_maps(inv)
    valid_ids = maps["valid_ids"]

    forward = load_jsonl(model_output_dir / "union_v0_forward_mappings.jsonl")
    backward = load_jsonl(model_output_dir / "union_v0_backward_reconstructions.jsonl")
    failed = load_jsonl(model_output_dir / "failed_requests.jsonl")

    invalid_rows: list[dict[str, Any]] = []
    n_annotations_total = 0
    n_valid_ids_exact = 0
    n_repaired_by_runner = 0
    n_repairable_not_repaired = 0
    n_invalid_primary = 0
    n_invalid_quarantined = 0
    n_overlap_annotations = 0
    n_interpretation_units = 0
    n_records_with_interpretation = 0
    span_relation_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}

    n_forward_parse_errors = sum(1 for x in forward if "_jsonl_parse_error" in x)
    n_backward_parse_errors = sum(1 for x in backward if "_jsonl_parse_error" in x)

    for rec in forward:
        if "_jsonl_parse_error" in rec:
            continue
        source_id = str(rec.get("source_id", ""))
        parsed = rec.get("parsed_forward") or {}
        annotations = parsed.get("annotations") or []
        invalid_annotations = parsed.get("invalid_annotations") or []
        units = parsed.get("interpretation_units") or []

        if isinstance(units, list):
            n_interpretation_units += len(units)
            if len(units) > 0:
                n_records_with_interpretation += 1
            for unit in units:
                if isinstance(unit, dict):
                    add_count(relationship_counts, unit.get("relationship", ""))

        if isinstance(annotations, list):
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                n_annotations_total += 1
                uid = str(ann.get("union_element_id", ""))
                suggested_uid, status = repair_union_id(uid, maps)

                if uid in valid_ids:
                    n_valid_ids_exact += 1
                elif status == "repairable":
                    n_repairable_not_repaired += 1
                    invalid_rows.append({
                        "source_id": source_id,
                        "span_text": ann.get("span_text", ""),
                        "union_element_id": uid,
                        "status": "repairable_not_yet_repaired",
                        "suggested_union_element_id": suggested_uid,
                        "annotation_id": ann.get("annotation_id", ""),
                        "rationale": ann.get("rationale", ""),
                    })
                else:
                    n_invalid_primary += 1
                    invalid_rows.append({
                        "source_id": source_id,
                        "span_text": ann.get("span_text", ""),
                        "union_element_id": uid,
                        "status": "invalid_primary_annotation",
                        "suggested_union_element_id": "",
                        "annotation_id": ann.get("annotation_id", ""),
                        "rationale": ann.get("rationale", ""),
                    })

                if ann.get("id_validation_status") == "repaired":
                    n_repaired_by_runner += 1
                if nonempty_value(ann.get("overlap_group_id")):
                    n_overlap_annotations += 1
                add_count(span_relation_counts, ann.get("span_relation", ""))

        if isinstance(invalid_annotations, list):
            for ann in invalid_annotations:
                if not isinstance(ann, dict):
                    continue
                n_invalid_quarantined += 1
                invalid_rows.append({
                    "source_id": source_id,
                    "span_text": ann.get("span_text", ""),
                    "union_element_id": ann.get("invalid_union_element_id") or ann.get("union_element_id", ""),
                    "status": "invalid_annotations_from_runner",
                    "suggested_union_element_id": "",
                    "annotation_id": ann.get("annotation_id", ""),
                    "rationale": ann.get("rationale", ""),
                })

    n_problem_primary = n_repairable_not_repaired + n_invalid_primary
    n_problem_total = n_problem_primary + n_invalid_quarantined
    quarantine_rate = n_invalid_quarantined / max(1, n_annotations_total)

    base_ok = (
        len(forward) > 0
        and len(backward) == len(forward)
        and len(failed) == 0
        and n_forward_parse_errors == 0
        and n_backward_parse_errors == 0
        and n_records_with_interpretation == len(forward)
    )
    ready_strict = base_ok and n_problem_total == 0
    ready_pragmatic = (
        base_ok
        and n_problem_primary == 0
        and n_repairable_not_repaired == 0
        and quarantine_rate <= quarantine_rate_threshold
    )

    summary = {
        "model_output_dir": str(model_output_dir),
        "inventory_csv": str(inventory_csv),
        "n_forward_records": len(forward),
        "n_backward_records": len(backward),
        "n_failed_requests": len(failed),
        "n_forward_jsonl_parse_errors": n_forward_parse_errors,
        "n_backward_jsonl_parse_errors": n_backward_parse_errors,
        "n_annotations_total": n_annotations_total,
        "n_valid_ids_exact": n_valid_ids_exact,
        "n_repaired_ids_by_runner": n_repaired_by_runner,
        "n_repairable_ids_not_yet_repaired": n_repairable_not_repaired,
        "n_invalid_ids_in_primary_annotations": n_invalid_primary,
        "n_invalid_ids_quarantined": n_invalid_quarantined,
        "n_invalid_ids_remaining": n_problem_total,
        "n_problem_ids_total": n_problem_total,
        "problem_id_rate": n_problem_total / max(1, n_annotations_total),
        "quarantined_invalid_id_rate": quarantine_rate,
        "quarantine_rate_threshold": quarantine_rate_threshold,
        "n_overlap_annotations": n_overlap_annotations,
        "n_interpretation_units": n_interpretation_units,
        "n_records_with_interpretation_units": n_records_with_interpretation,
        "span_relation_counts": span_relation_counts,
        "interpretation_relationship_counts": relationship_counts,
        "ready_for_full_run_strict": ready_strict,
        "ready_for_full_run_pragmatic": ready_pragmatic,
        "ready_for_full_run": ready_pragmatic,
    }

    invalid_df = pd.DataFrame(invalid_rows, columns=INVALID_COLUMNS)
    return summary, invalid_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_output_dir", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--output_prefix", default=None)
    ap.add_argument("--quarantine_rate_threshold", type=float, default=0.10)
    args = ap.parse_args()

    model_output_dir = Path(args.model_output_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else model_output_dir / "validation"

    summary, invalid_df = validate_outputs(
        model_output_dir=model_output_dir,
        inventory_csv=Path(args.inventory_csv),
        quarantine_rate_threshold=args.quarantine_rate_threshold,
    )

    summary_path = output_prefix.with_suffix(".summary.json")
    invalid_path = output_prefix.with_suffix(".invalid_annotations.csv")
    summary_path.write_text(json.dumps(summary, indent=2))
    invalid_df.to_csv(invalid_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {invalid_path}")


if __name__ == "__main__":
    main()
