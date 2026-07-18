#!/usr/bin/env python
"""Score standardized round-trip outputs with the final classifier.

Inputs:
  - standardized_roundtrips.csv from 07_standardize_roundtrip_outputs.py
  - final_meaning_preservation_classifier.joblib from 08_train_final_meaning_classifier.py

Outputs:
  - scored_roundtrips.csv
  - score_summary_by_condition.csv
  - paired_union_vs_individual.csv
  - high_classifier_low_overlap_audit.csv

The classifier score is a proxy meaning-preservation estimate. This script also
adds independent lexical/content-coverage metrics so high classifier scores that
omit large parts of the original sentence can be audited.
"""
from __future__ import annotations

import argparse
import json
import re
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

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "having", "he", "her", "hers", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "that", "the",
    "their", "theirs", "them", "they", "this", "to", "was", "we", "were", "will", "with",
    "you", "your", "yours", "can", "may", "could", "would", "should", "do", "does", "did",
    "about", "above", "below", "also", "than", "then", "there", "these", "those", "such",
}


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


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", "" if pd.isna(text) else str(text).lower())


def content_tokens(text: Any) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def jaccard(a: list[str], b: list[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def recall(a: list[str], b: list[str]) -> float:
    """Recall of original tokens a in reconstruction tokens b."""
    aa, bb = set(a), set(b)
    if not aa:
        return 1.0
    return len(aa & bb) / len(aa)


def precision(a: list[str], b: list[str]) -> float:
    """Precision of reconstruction tokens b with respect to original tokens a."""
    aa, bb = set(a), set(b)
    if not bb:
        return 1.0 if not aa else 0.0
    return len(aa & bb) / len(bb)


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def bigrams(tokens: list[str]) -> list[str]:
    return [tokens[i] + " " + tokens[i + 1] for i in range(len(tokens) - 1)]


def add_lexical_coverage_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in scored.iterrows():
        orig = r.get("original_text", r.get("source_text", ""))
        rec = r.get("reconstructed_text", r.get("reconstructed_sentence", ""))
        otok, rtok = tokenize(orig), tokenize(rec)
        octok, rctok = content_tokens(orig), content_tokens(rec)
        obg, rbg = bigrams(otok), bigrams(rtok)
        p = precision(octok, rctok)
        rc = recall(octok, rctok)
        len_ratio = len(rtok) / max(1, len(otok))
        missing = sorted(set(octok) - set(rctok))
        rows.append({
            "lexical_token_jaccard": jaccard(otok, rtok),
            "content_token_jaccard": jaccard(octok, rctok),
            "content_token_precision": p,
            "content_token_recall": rc,
            "content_token_f1": f1(p, rc),
            "bigram_recall": recall(obg, rbg),
            "orig_token_count": len(otok),
            "recon_token_count": len(rtok),
            "recon_to_orig_length_ratio": len_ratio,
            "missing_content_tokens_json": json.dumps(missing, ensure_ascii=False),
            "n_missing_content_tokens": len(missing),
        })
    metrics = pd.DataFrame(rows)
    out = pd.concat([scored.reset_index(drop=True), metrics], axis=1)
    # Audit flags: intentionally conservative. These are not automatic failures;
    # they identify rows where a high classifier score may be driven by salient cue words
    # while substantial source content was omitted.
    high_score = out["classifier_preservation_score"] >= 0.80
    low_content = out["content_token_recall"] < 0.65
    low_bigram = out["bigram_recall"] < 0.35
    compressed = (out["orig_token_count"] >= 12) & (out["recon_to_orig_length_ratio"] < 0.55)
    out["audit_high_score_low_overlap"] = high_score & (low_content | low_bigram | compressed)
    out["audit_low_content_recall"] = low_content
    out["audit_low_bigram_recall"] = low_bigram
    out["audit_heavy_compression"] = compressed
    return out


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
            "mean_content_token_recall": g["content_token_recall"].mean(),
            "median_content_token_recall": g["content_token_recall"].median(),
            "mean_content_token_f1": g["content_token_f1"].mean(),
            "mean_bigram_recall": g["bigram_recall"].mean(),
            "mean_length_ratio": g["recon_to_orig_length_ratio"].mean(),
            "pct_high_score_low_overlap_audit": g["audit_high_score_low_overlap"].mean(),
            "pct_low_content_recall": g["audit_low_content_recall"].mean(),
            "pct_heavy_compression": g["audit_heavy_compression"].mean(),
        })
    return pd.DataFrame(rows).sort_values(group_cols)


def paired_union_vs_individual(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    union = scored[scored["condition"].eq("union_v0_full_dictionary")].copy()
    indiv = scored[scored["condition"].eq("individual_source_model_json")].copy()
    if union.empty or indiv.empty:
        return pd.DataFrame(rows)
    metrics = [
        "classifier_preservation_score",
        "content_token_recall",
        "content_token_f1",
        "bigram_recall",
        "recon_to_orig_length_ratio",
    ]
    for llm in sorted(set(union["llm"]) & set(indiv["llm"])):
        ucols = ["source_id"] + metrics
        u = union[union["llm"].eq(llm)][ucols].rename(columns={m: f"union_{m}" for m in metrics})
        for info in sorted(indiv[indiv["llm"].eq(llm)]["information_model"].unique()):
            xcols = ["source_id"] + metrics
            x = indiv[(indiv["llm"].eq(llm)) & (indiv["information_model"].eq(info))][xcols].rename(columns={m: f"individual_{m}" for m in metrics})
            m = u.merge(x, on="source_id", how="inner")
            if m.empty:
                continue
            score_delta = m["union_classifier_preservation_score"] - m["individual_classifier_preservation_score"]
            coverage_delta = m["union_content_token_recall"] - m["individual_content_token_recall"]
            rows.append({
                "llm": llm,
                "comparison": f"Union_V0_minus_{info}",
                "individual_information_model": info,
                "n_pairs": len(m),
                "mean_union_score": m["union_classifier_preservation_score"].mean(),
                "mean_individual_score": m["individual_classifier_preservation_score"].mean(),
                "mean_delta_union_minus_individual": score_delta.mean(),
                "median_delta_union_minus_individual": score_delta.median(),
                "union_win_rate": (score_delta > 0).mean(),
                "individual_win_rate": (score_delta < 0).mean(),
                "tie_rate": (score_delta == 0).mean(),
                "mean_union_content_recall": m["union_content_token_recall"].mean(),
                "mean_individual_content_recall": m["individual_content_token_recall"].mean(),
                "mean_delta_content_recall_union_minus_individual": coverage_delta.mean(),
                "mean_union_bigram_recall": m["union_bigram_recall"].mean(),
                "mean_individual_bigram_recall": m["individual_bigram_recall"].mean(),
            })
        best = indiv[indiv["llm"].eq(llm)].sort_values("classifier_preservation_score").groupby("source_id", as_index=False).tail(1)
        best = best[["source_id"] + metrics].rename(columns={m: f"best_individual_{m}" for m in metrics})
        m = u.merge(best, on="source_id", how="inner")
        if not m.empty:
            score_delta = m["union_classifier_preservation_score"] - m["best_individual_classifier_preservation_score"]
            coverage_delta = m["union_content_token_recall"] - m["best_individual_content_token_recall"]
            rows.append({
                "llm": llm,
                "comparison": "Union_V0_minus_best_individual",
                "individual_information_model": "best_individual_per_sentence",
                "n_pairs": len(m),
                "mean_union_score": m["union_classifier_preservation_score"].mean(),
                "mean_individual_score": m["best_individual_classifier_preservation_score"].mean(),
                "mean_delta_union_minus_individual": score_delta.mean(),
                "median_delta_union_minus_individual": score_delta.median(),
                "union_win_rate": (score_delta > 0).mean(),
                "individual_win_rate": (score_delta < 0).mean(),
                "tie_rate": (score_delta == 0).mean(),
                "mean_union_content_recall": m["union_content_token_recall"].mean(),
                "mean_individual_content_recall": m["best_individual_content_token_recall"].mean(),
                "mean_delta_content_recall_union_minus_individual": coverage_delta.mean(),
                "mean_union_bigram_recall": m["union_bigram_recall"].mean(),
                "mean_individual_bigram_recall": m["best_individual_bigram_recall"].mean(),
            })
    return pd.DataFrame(rows)


def write_audit_tables(scored: pd.DataFrame, out: Path) -> None:
    audit_cols = [
        "source_id", "llm", "condition", "information_model", "source_text", "reconstructed_text",
        "classifier_preservation_score", "content_token_recall", "content_token_f1", "bigram_recall",
        "recon_to_orig_length_ratio", "n_missing_content_tokens", "missing_content_tokens_json",
        "audit_low_content_recall", "audit_low_bigram_recall", "audit_heavy_compression",
    ]
    cols = [c for c in audit_cols if c in scored.columns]
    high_low = scored[scored["audit_high_score_low_overlap"]].sort_values(
        ["classifier_preservation_score", "content_token_recall"], ascending=[False, True]
    )
    high_low[cols].to_csv(out / "high_classifier_low_overlap_audit.csv", index=False)
    scored.sort_values(["content_token_recall", "classifier_preservation_score"], ascending=[True, False]).head(200)[cols].to_csv(
        out / "lowest_content_coverage_top200.csv", index=False
    )


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
    # Keep canonical text columns in scored output even when standardized input used source_text/reconstructed_sentence names.
    scored["original_text"] = df["original_text"].values
    scored["reconstructed_text"] = df["reconstructed_text"].values
    scored["classifier_preservation_score"] = scores
    scored["classifier_pred_0_5"] = (scores >= 0.5).astype(int)
    scored["classifier_pred_0_7"] = (scores >= 0.7).astype(int)
    scored["classifier_pred_0_8"] = (scores >= 0.8).astype(int)
    scored["classifier_pred_0_9"] = (scores >= 0.9).astype(int)
    scored = add_lexical_coverage_metrics(scored)

    scored.to_csv(out / "scored_roundtrips.csv", index=False)
    summarize(scored).to_csv(out / "score_summary_by_condition.csv", index=False)
    paired_union_vs_individual(scored).to_csv(out / "paired_union_vs_individual.csv", index=False)
    write_audit_tables(scored, out)
    (out / "scoring_metadata.json").write_text(json.dumps({
        "n_scored": int(len(scored)),
        "classifier_bundle": args.classifier_bundle,
        "secondary_metrics": [
            "lexical_token_jaccard",
            "content_token_recall",
            "content_token_f1",
            "bigram_recall",
            "recon_to_orig_length_ratio",
            "audit_high_score_low_overlap",
        ],
        "audit_rule": "flag if classifier_score>=0.80 and content_recall<0.65 or bigram_recall<0.35 or length_ratio<0.55 for source sentences with >=12 tokens",
    }, indent=2))
    print(f"Wrote scored outputs to {out}")


if __name__ == "__main__":
    main()
