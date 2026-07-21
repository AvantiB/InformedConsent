#!/usr/bin/env python
"""Compile scored round-trip outputs into paper-facing schema-condition summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.scored_csv).fillna("")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_cols = [c for c in ["meaning_preserved_score", "score", "predicted_probability", "probability", "meaning_preserved_pred"] if c in df.columns]
    score_col = score_cols[0] if score_cols else None
    group_cols = [c for c in ["condition", "information_model", "llm"] if c in df.columns]
    agg = {
        "roundtrip_id": "count",
        "annotation_count": lambda x: pd.to_numeric(x, errors="coerce").mean(),
        "unique_element_count": lambda x: pd.to_numeric(x, errors="coerce").mean(),
        "forward_parse_ok": lambda x: pd.Series(x).astype(str).str.lower().isin(["true", "1"]).mean(),
        "backward_parse_ok": lambda x: pd.Series(x).astype(str).str.lower().isin(["true", "1"]).mean(),
    }
    if score_col:
        agg[score_col] = lambda x: pd.to_numeric(x, errors="coerce").mean()
    summary = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary = summary.rename(columns={"roundtrip_id": "n", "annotation_count": "mean_annotation_count", "unique_element_count": "mean_unique_fields", "forward_parse_ok": "forward_parse_rate", "backward_parse_ok": "backward_parse_rate"})
    if score_col:
        summary = summary.rename(columns={score_col: "mean_classifier_score"})
    summary.to_csv(out / "schema_condition_summary.csv", index=False)

    # Pairwise wide table by source_id + llm when possible.
    if {"source_id", "llm", "condition"}.issubset(df.columns) and score_col:
        wide = df.pivot_table(index=["source_id", "llm"], columns="condition", values=score_col, aggfunc="mean").reset_index()
        wide.to_csv(out / "paired_condition_scores_wide.csv", index=False)
    print(f"Wrote schema-condition comparison summaries to {out}")


if __name__ == "__main__":
    main()
