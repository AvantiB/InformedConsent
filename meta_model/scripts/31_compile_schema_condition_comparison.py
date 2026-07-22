#!/usr/bin/env python
"""Compile scored/diagnostic round-trip outputs into paper-facing schema summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

SCORE_CANDIDATES = [
    "classifier_preservation_score",
    "meaning_preserved_score",
    "meaning_preservation_score",
    "classifier_score",
    "predicted_probability",
    "probability",
    "score",
    "meaning_preserved_pred_proba",
    "meaning_preserved_pred",
]

MEAN_METRICS = [
    "annotation_count",
    "unique_element_count",
    "forward_parse_ok",
    "backward_parse_ok",
    "content_word_recall",
    "content_word_precision",
    "content_word_f1",
    "content_word_jaccard",
    "dropped_content_word_rate",
    "added_content_word_rate",
    "important_category_presence_recall",
    "important_cue_exact_recall",
    "important_cue_jaccard",
    "modal_category_changed",
    "modal_word_change_ratio",
    "modal_word_recall",
    "unmatched_language_available",
    "unmatched_language_rate",
    "suspected_error_count",
]

RENAME = {
    "roundtrip_id": "n",
    "annotation_count": "mean_annotation_count",
    "unique_element_count": "mean_unique_fields",
    "forward_parse_ok": "forward_parse_rate",
    "backward_parse_ok": "backward_parse_rate",
    "content_word_recall": "mean_content_word_recall",
    "content_word_precision": "mean_content_word_precision",
    "content_word_f1": "mean_content_word_f1",
    "content_word_jaccard": "mean_content_word_jaccard",
    "dropped_content_word_rate": "mean_dropped_content_word_rate",
    "added_content_word_rate": "mean_added_content_word_rate",
    "important_category_presence_recall": "mean_important_category_presence_recall",
    "important_cue_exact_recall": "mean_important_cue_exact_recall",
    "important_cue_jaccard": "mean_important_cue_jaccard",
    "modal_category_changed": "modal_category_change_rate",
    "modal_word_change_ratio": "mean_modal_word_change_ratio",
    "modal_word_recall": "mean_modal_word_recall",
    "unmatched_language_available": "unmatched_language_availability_rate",
    "unmatched_language_rate": "mean_unmatched_language_rate_when_available",
    "suspected_error_count": "mean_suspected_error_flags",
}


def numeric_mean(x: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").mean()


def score_column(df: pd.DataFrame) -> str | None:
    return next((c for c in SCORE_CANDIDATES if c in df.columns), None)


def summarize(df: pd.DataFrame, group_cols: list[str], score_col: str | None) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        df = df.copy()
        df["__all__"] = "all"
        cols = ["__all__"]
    count_col = "roundtrip_id" if "roundtrip_id" in df.columns else df.columns[0]
    agg: dict[str, Any] = {count_col: "count"}
    for col in MEAN_METRICS:
        if col in df.columns:
            agg[col] = numeric_mean
    if score_col:
        agg[score_col] = numeric_mean
    out = df.groupby(cols, dropna=False).agg(agg).reset_index()
    rename = dict(RENAME)
    rename[count_col] = "n"
    if score_col:
        rename[score_col] = "mean_classifier_score"
    return out.rename(columns=rename)


def paired_tables(df: pd.DataFrame, score_col: str | None, out: Path) -> None:
    if not score_col or not {"condition", "llm"}.issubset(df.columns):
        return
    index_cols = [c for c in ["source_id", "roundtrip_id", "llm", "cv_fold", "fold_id"] if c in df.columns]
    if "source_id" not in index_cols and "roundtrip_id" not in index_cols:
        return
    if "source_id" in index_cols and "roundtrip_id" in index_cols:
        index_cols.remove("roundtrip_id")
    wide = df.pivot_table(index=index_cols, columns="condition", values=score_col, aggfunc="mean").reset_index()
    wide.to_csv(out / "paired_condition_scores_wide.csv", index=False)

    cond_cols = [c for c in wide.columns if c not in index_cols]
    diffs = []
    for a in cond_cols:
        for b in cond_cols:
            if a >= b:
                continue
            tmp = wide[index_cols].copy()
            tmp["condition_a"] = a
            tmp["condition_b"] = b
            tmp["score_a"] = wide[a]
            tmp["score_b"] = wide[b]
            tmp["score_diff_a_minus_b"] = wide[a] - wide[b]
            diffs.append(tmp)
    if diffs:
        pd.concat(diffs, ignore_index=True).to_csv(out / "paired_condition_score_differences.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored_csv", required=True, help="Scored CSV or diagnostic row-level CSV from script 32.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.scored_csv, low_memory=False).fillna("")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = score_column(df)

    summarize(df, ["condition", "information_model", "llm"], sc).to_csv(out / "schema_condition_summary.csv", index=False)
    summarize(df, ["condition", "llm"], sc).to_csv(out / "schema_condition_by_llm.csv", index=False)
    summarize(df, ["condition", "information_model"], sc).to_csv(out / "schema_condition_by_information_model.csv", index=False)
    summarize(df, ["condition"], sc).to_csv(out / "schema_condition_overall.csv", index=False)
    paired_tables(df, sc, out)
    print(f"Wrote schema-condition comparison summaries to {out}")


if __name__ == "__main__":
    main()
