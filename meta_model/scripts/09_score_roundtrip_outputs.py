#!/usr/bin/env python
"""Score standardized round-trip outputs with the final classifier.

Inputs:
  - standardized_roundtrips.csv from 07_standardize_roundtrip_outputs.py
  - final_meaning_preservation_classifier.joblib from 08_train_final_meaning_classifier.py

Outputs:
  - scored_roundtrips.csv
  - score_summary_by_condition.csv
  - paired_union_vs_individual.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MEANING_DIR = REPO_ROOT / "meaning_preservation"
sys.path.insert(0, str(MEANING_DIR))

import run_classifier_experiments as base  # noqa: E402


def apply_cue_dictionary(data: dict[str, Any] | None) -> None:
    if not data or "cue_groups" not in data:
        return
    cue_groups = {str(k): [str(vv).lower() for vv in v] for k, v in data["cue_groups"].items()}
    base.CUE_GROUPS = cue_groups
    base.PERMISSION = cue_groups.get("permission", base.PERMISSION)
    base.OBLIGATION = cue_groups.get("obligation", base.OBLIGATION)
    base.PROHIBITION = cue_groups.get("prohibition", base.PROHIBITION)


def ensure_classifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["roundtrip_id"] = out.get("roundtrip_id", pd.Series([f"RT_{i}" for i in range(len(out))])).astype(str)
    out["sentence_id"] = out.get("source_id", out["roundtrip_id"]).astype(str)
    out["form_id"] = out.get("condition", "FORM_UNKNOWN").astype(str)
    out["original_text"] = out.get("original_text", out.get("source_text", "")).fillna("").astype(str)
    out["reconstructed_text"] = out.get("reconstructed_text", out.get("reconstructed_sentence", "")).fillna("").astype(str)
    out["forward_mapping"] = out.get("forward_mapping", "").fillna("").astype(str)
    out["llm"] = out.get("llm", out.get("model", "unknown")).fillna("unknown").astype(str)
    out["information_model"] = out.get("information_model", out.get("info_model", "unknown")).fillna("unknown").astype(str)
    if "annotation_count" not in out:
        out["annotation_count"] = np.nan
    if "unique_element_count" not in out:
        out["unique_element_count"] = np.nan
    return out


def add_optional_semantics(df: pd.DataFrame, feats: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    feats = base.add_embeddings(
        df,
        feats,
        bundle.get("embedding_model"),
        int(bundle.get("embedding_batch_size") or 64),
        bundle.get("embedding_backend") or "auto",
        bundle.get("embedding_device"),
    )
    feats = base.add_nli(
        df,
        feats,
        bundle.get("nli_model"),
        int(bundle.get("nli_batch_size") or 16),
        bundle.get("nli_device"),
    )
    return feats


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X).astype(float)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["llm", "condition", "information_model"]
    for keys, g in scored.groupby(group_cols, dropna=False):
        rows.append({
            "llm": keys[0],
            "condition": keys[1],
            "information_model": keys[2],
            "n": len(g),
            "mean_score": g["classifier_preservation_score"].mean(),
            "median_score": g["classifier_preservation_score"].median(),
            "pct_ge_0_5": (g["classifier_preservation_score"] >= 0.5).mean(),
            "pct_ge_0_7": (g["classifier_preservation_score"] >= 0.7).mean(),
            "pct_ge_0_8": (g["classifier_preservation_score"] >= 0.8).mean(),
            "pct_ge_0_9": (g["classifier_preservation_score"] >= 0.9).mean(),
        })
    return pd.DataFrame(rows).sort_values(group_cols)


def paired_union_vs_individual(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    union = scored[scored["condition"].eq("union_v0_full_dictionary")].copy()
    indiv = scored[scored["condition"].eq("individual_source_model_json")].copy()
    if union.empty or indiv.empty:
        return pd.DataFrame(rows)
    for llm in sorted(set(union["llm"]) & set(indiv["llm"])):
        u = union[union["llm"].eq(llm)][["source_id", "classifier_preservation_score"]].rename(columns={"classifier_preservation_score": "union_score"})
        for info in sorted(indiv[indiv["llm"].eq(llm)]["information_model"].unique()):
            x = indiv[(indiv["llm"].eq(llm)) & (indiv["information_model"].eq(info))][["source_id", "classifier_preservation_score"]].rename(columns={"classifier_preservation_score": "individual_score"})
            m = u.merge(x, on="source_id", how="inner")
            if m.empty:
                continue
            delta = m["union_score"] - m["individual_score"]
            rows.append({
                "llm": llm,
                "comparison": f"Union_V0_minus_{info}",
                "individual_information_model": info,
                "n_pairs": len(m),
                "mean_union_score": m["union_score"].mean(),
                "mean_individual_score": m["individual_score"].mean(),
                "mean_delta_union_minus_individual": delta.mean(),
                "median_delta_union_minus_individual": delta.median(),
                "union_win_rate": (delta > 0).mean(),
                "individual_win_rate": (delta < 0).mean(),
                "tie_rate": (delta == 0).mean(),
            })
        best = indiv[indiv["llm"].eq(llm)].pivot_table(index="source_id", values="classifier_preservation_score", aggfunc="max").reset_index().rename(columns={"classifier_preservation_score": "best_individual_score"})
        m = u.merge(best, on="source_id", how="inner")
        if not m.empty:
            delta = m["union_score"] - m["best_individual_score"]
            rows.append({
                "llm": llm,
                "comparison": "Union_V0_minus_best_individual",
                "individual_information_model": "best_individual_per_sentence",
                "n_pairs": len(m),
                "mean_union_score": m["union_score"].mean(),
                "mean_individual_score": m["best_individual_score"].mean(),
                "mean_delta_union_minus_individual": delta.mean(),
                "median_delta_union_minus_individual": delta.median(),
                "union_win_rate": (delta > 0).mean(),
                "individual_win_rate": (delta < 0).mean(),
                "tie_rate": (delta == 0).mean(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standardized_csv", required=True)
    ap.add_argument("--classifier_bundle", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.classifier_bundle)
    apply_cue_dictionary(bundle.get("cue_dictionary"))

    raw = pd.read_csv(args.standardized_csv)
    df = ensure_classifier_columns(raw)
    feats = base.extract_features(df)
    feats = add_optional_semantics(df, feats, bundle)
    feature_columns = list(bundle["feature_columns"])
    missing = [c for c in feature_columns if c not in feats.columns]
    if missing:
        raise ValueError(f"Scoring features are missing columns required by classifier: {missing}")
    X = feats[feature_columns].copy()
    scores = predict_scores(bundle["model"], X)

    scored = raw.copy()
    scored["classifier_preservation_score"] = scores
    scored["classifier_pred_0_5"] = (scores >= 0.5).astype(int)
    scored["classifier_pred_0_7"] = (scores >= 0.7).astype(int)
    scored["classifier_pred_0_8"] = (scores >= 0.8).astype(int)
    scored["classifier_pred_0_9"] = (scores >= 0.9).astype(int)
    scored.to_csv(out / "scored_roundtrips.csv", index=False)
    summarize(scored).to_csv(out / "score_summary_by_condition.csv", index=False)
    paired_union_vs_individual(scored).to_csv(out / "paired_union_vs_individual.csv", index=False)
    (out / "scoring_metadata.json").write_text(json.dumps({"n_scored": int(len(scored)), "classifier_bundle": args.classifier_bundle}, indent=2))
    print(f"Wrote scored outputs to {out}")


if __name__ == "__main__":
    main()
