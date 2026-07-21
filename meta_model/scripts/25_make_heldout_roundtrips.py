#!/usr/bin/env python
"""Create held-out roundtrip input files for refined meta-model CV evaluation.

This script filters the main roundtrips.csv by fold-level consent-form assignments.
It is intentionally separate from schema induction so held-out evaluation inputs are
created only after fold assignments and selected schemas are frozen.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

FORM_COLS = ["form_key", "form_id", "source_file", "source_file_original", "source_id", "input_workbook"]
TEXT_COLS = ["canonical_full_text", "full_text_original", "full_text", "sentence_text", "sentence", "text"]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates}; available={list(df.columns)}")
    return None


def strip_workflow_suffixes(v: str) -> str:
    v = re.sub(r"\.(txt|csv|xlsx?)$", "", v, flags=re.I)
    v = re.sub(r"_annotated$", "", v, flags=re.I)
    v = re.sub(r"_output$", "", v, flags=re.I)
    v = re.sub(r"\s+annotated$", "", v, flags=re.I)
    v = re.sub(r"\s+output$", "", v, flags=re.I)
    v = re.sub(r"\s+copy(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    v = re.sub(r"_copy(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    return re.sub(r"\s+", " ", v).strip(" _-")


def form_value_from_row(row: pd.Series) -> str:
    for c in FORM_COLS:
        if c in row.index and norm(row.get(c)):
            v = strip_workflow_suffixes(norm(row.get(c)))
            if v and v.lower() not in {"nan", "none", "null"}:
                if v.startswith("FORM_") and "e3b0c442" in v:
                    continue
                return v
    return ""


def form_match_key(raw: Any) -> str:
    v = strip_workflow_suffixes(norm(raw)).lower()
    v = v.replace("’", "'").replace("‘", "'").replace("`", "'")
    v = re.sub(r"[^a-z0-9]+", " ", v)
    return " ".join(v.split())


def build_assignment_maps(folds: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    fold_col = "canonical_form_id" if "canonical_form_id" in folds.columns else "form_id"
    exact: dict[str, int] = {}
    fuzzy: dict[str, int] = {}
    for _, r in folds.iterrows():
        fid = norm(r.get(fold_col))
        if not fid:
            continue
        fold_id = int(r["fold_id"])
        exact[fid] = fold_id
        fuzzy[form_match_key(fid)] = fold_id
        if norm(r.get("alias_for_canonical_form_id")):
            fuzzy[form_match_key(r.get("alias_for_canonical_form_id"))] = fold_id
    return exact, fuzzy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrips_csv", required=True, help="Main roundtrips.csv used for individual and Union V0 evaluations.")
    ap.add_argument("--fold_assignments_csv", required=True, help="Prefer fold_assignments.repaired.csv.")
    ap.add_argument("--output_dir", required=True, help="Usually meta_model/refined_cv.")
    ap.add_argument("--write_train", action="store_true", help="Also write train_roundtrips.csv per fold for auditing only.")
    args = ap.parse_args()

    df = pd.read_csv(args.roundtrips_csv).fillna("")
    folds = pd.read_csv(args.fold_assignments_csv).fillna("")
    pick_col(df, TEXT_COLS, required=True)
    exact, fuzzy = build_assignment_maps(folds)

    rows = []
    for i, r in df.iterrows():
        form_id = form_value_from_row(r)
        fold = exact.get(form_id)
        match_type = "exact"
        if fold is None:
            fold = fuzzy.get(form_match_key(form_id))
            match_type = "fuzzy_form_key" if fold is not None else "unassigned"
        x = r.to_dict()
        x["cv_form_id"] = form_id
        x["cv_fold_id"] = fold if fold is not None else ""
        x["cv_form_match_type"] = match_type
        x["cv_source_row"] = int(i) + 2
        rows.append(x)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    annotated = pd.DataFrame(rows)
    annotated.to_csv(out_root / "roundtrips_with_cv_folds.csv", index=False)

    unassigned = annotated[annotated["cv_form_match_type"] == "unassigned"].copy()
    if not unassigned.empty:
        unassigned[["cv_source_row", "cv_form_id", "cv_form_match_type"]].drop_duplicates().to_csv(out_root / "roundtrips_unassigned_form_review.csv", index=False)

    counts: dict[str, Any] = {
        "roundtrips_csv": args.roundtrips_csv,
        "fold_assignments_csv": args.fold_assignments_csv,
        "n_rows": int(len(annotated)),
        "n_unassigned_rows": int(len(unassigned)),
        "folds": {},
    }

    for fold_id in sorted(int(x) for x in folds["fold_id"].dropna().unique()):
        fold_dir = out_root / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        heldout = annotated[pd.to_numeric(annotated["cv_fold_id"], errors="coerce") == fold_id].copy()
        heldout.to_csv(fold_dir / "heldout_roundtrips.csv", index=False)
        if args.write_train:
            train = annotated[(annotated["cv_form_match_type"] != "unassigned") & (pd.to_numeric(annotated["cv_fold_id"], errors="coerce") != fold_id)].copy()
            train.to_csv(fold_dir / "train_roundtrips_for_audit.csv", index=False)
        form_summary = heldout.groupby("cv_form_id", dropna=False).size().reset_index(name="n_rows")
        form_summary.to_csv(fold_dir / "heldout_form_summary.csv", index=False)
        counts["folds"][f"fold_{fold_id:02d}"] = {
            "n_heldout_rows": int(len(heldout)),
            "n_heldout_forms": int(heldout["cv_form_id"].nunique()),
        }

    (out_root / "heldout_roundtrips_metadata.json").write_text(json.dumps(counts, indent=2))
    print(f"Wrote held-out roundtrip files under {out_root}/fold_XX")
    if not unassigned.empty:
        print(f"WARNING: {len(unassigned)} rows were unassigned. Review {out_root / 'roundtrips_unassigned_form_review.csv'}")


if __name__ == "__main__":
    main()
