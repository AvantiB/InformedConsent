#!/usr/bin/env python
"""Train the final deployment meaning-preservation classifier on all labeled rows.

This script is for scoring new Union V0 / individual-model round trips after the
classifier model family has already been selected from split-based experiments.
It trains on all original human-labeled data and saves a reusable joblib bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
MEANING_DIR = REPO_ROOT / "meaning_preservation"
sys.path.insert(0, str(MEANING_DIR))

import run_classifier_experiments as base  # noqa: E402

CAT_COLS = ["modal_orig", "modal_recon"]
EXCLUDE = {"roundtrip_id", "llm", "information_model"}


def load_cue_dictionary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open() as f:
        data = json.load(f)
    if "cue_groups" not in data:
        raise ValueError("Cue dictionary must contain cue_groups.")
    cue_groups = {str(k): [str(vv).lower() for vv in v] for k, v in data["cue_groups"].items()}
    base.CUE_GROUPS = cue_groups
    base.PERMISSION = cue_groups.get("permission", base.PERMISSION)
    base.OBLIGATION = cue_groups.get("obligation", base.OBLIGATION)
    base.PROHIBITION = cue_groups.get("prohibition", base.PROHIBITION)
    return data


def add_optional_semantics(df: pd.DataFrame, feats: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    feats = base.add_embeddings(df, feats, args.embedding_model, args.embedding_batch_size, args.embedding_backend, args.embedding_device)
    feats = base.add_nli(df, feats, args.nli_model, args.nli_batch_size, args.nli_device)
    return feats


def choose_columns(feats: pd.DataFrame, feature_set: str) -> tuple[list[str], list[str], list[str]]:
    semantic = [c for c in feats.columns if c.startswith("embedding_") or c.startswith("nli_")]
    cue_cols = [c for c in feats.columns if any(c.endswith(s) for s in ["_count", "_jaccard", "_missing_count", "_added_count", "_presence_preserved"])]
    cue_cols += [c for c in ["modal_category_changed"] if c in feats.columns]
    lexical_mapping = [
        "orig_len_tokens", "recon_len_tokens", "length_ratio", "abs_length_diff", "token_jaccard", "tfidf_cosine",
        "annotation_count", "unique_element_count", "mapping_len_chars", "mapping_bracket_count", "mapping_paren_count",
    ]
    if feature_set == "dictionary_modal":
        num = cue_cols
    elif feature_set == "semantic":
        num = semantic
    elif feature_set == "dictionary_modal_semantic":
        num = cue_cols + semantic
    elif feature_set == "engineered_semantic":
        num = lexical_mapping + cue_cols + semantic
    elif feature_set == "engineered_all":
        num = lexical_mapping + cue_cols
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")
    num = [c for c in dict.fromkeys(num) if c in feats.columns and pd.api.types.is_numeric_dtype(feats[c])]
    cat = [c for c in CAT_COLS if c in feats.columns and feature_set != "semantic"]
    cols = num + cat
    if not cols:
        raise ValueError(f"No usable columns for feature_set={feature_set}.")
    return cols, num, cat


def make_model(num_cols: list[str], cat_cols: list[str], seed: int) -> Pipeline:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_roundtrips_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cue_dictionary", default="meaning_preservation/literature_informed_consent_cues.json")
    ap.add_argument("--feature_set", default="engineered_semantic", choices=["engineered_all", "dictionary_modal", "semantic", "dictionary_modal_semantic", "engineered_semantic"])
    ap.add_argument("--embedding_model", default=None)
    ap.add_argument("--embedding_backend", default="auto", choices=["auto", "sentence_transformers", "hf"])
    ap.add_argument("--embedding_device", default=None)
    ap.add_argument("--embedding_batch_size", type=int, default=64)
    ap.add_argument("--nli_model", default=None)
    ap.add_argument("--nli_device", default=None)
    ap.add_argument("--nli_batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cue_dict = load_cue_dictionary(Path(args.cue_dictionary) if args.cue_dictionary else None)
    df = base.build_dataset(Path(args.labeled_roundtrips_csv), out)
    feats = base.extract_features(df)
    feats = add_optional_semantics(df, feats, args)
    cols, num_cols, cat_cols = choose_columns(feats, args.feature_set)

    X = feats[cols].copy()
    y = df["meaning_preserved"].astype(int).values
    model = make_model(num_cols, cat_cols, args.seed)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_columns": cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "feature_set": args.feature_set,
        "embedding_model": args.embedding_model,
        "embedding_backend": args.embedding_backend,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "nli_model": args.nli_model,
        "nli_device": args.nli_device,
        "nli_batch_size": args.nli_batch_size,
        "cue_dictionary": cue_dict,
    }
    joblib.dump(bundle, out / "final_meaning_preservation_classifier.joblib")
    feats.to_csv(out / "training_features.csv", index=False)
    summary = {
        "n_training_rows": int(len(df)),
        "positive_labels": int(df["meaning_preserved"].sum()),
        "negative_labels": int((df["meaning_preserved"] == 0).sum()),
        "feature_set": args.feature_set,
        "n_features": len(cols),
        "embedding_model": args.embedding_model,
        "nli_model": args.nli_model,
    }
    (out / "final_classifier_training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote classifier bundle to {out / 'final_meaning_preservation_classifier.joblib'}")


if __name__ == "__main__":
    main()
