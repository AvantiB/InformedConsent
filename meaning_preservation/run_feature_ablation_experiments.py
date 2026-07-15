#!/usr/bin/env python
"""Feature-set ablation runner for meaning-preservation classification.

This complements run_classifier_experiments.py by separating feature-source
questions from classifier questions.

Feature sets:
- bow_tfidf_lr_baseline: lexical baseline only
- engineered_*: hand-engineered/dictionary consent features only
- semantic_*: embedding/NLI similarity features only
- engineered_semantic_*: engineered + semantic features
- full_hybrid_tfidf_engineered_semantic_lr: TF-IDF + engineered + semantic, ablation only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_classifier_experiments import (
    _resolve_hf_model_name,
    add_embeddings,
    add_nli,
    build_dataset,
    extract_features,
    metrics,
    scores,
    splits,
)

CAT_COLS = ["modal_orig", "modal_recon"]
META_COLS = {"roundtrip_id", "llm", "information_model"}


def get_feature_columns(feats: pd.DataFrame):
    semantic = [c for c in feats.columns if c.startswith("embedding_") or c.startswith("nli_")]
    excluded = META_COLS | set(semantic) | set(CAT_COLS)
    engineered_num = [
        c for c in feats.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(feats[c])
    ]
    engineered_cat = [c for c in CAT_COLS if c in feats.columns]
    return engineered_num, engineered_cat, semantic


def make_preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(parts)


def lr_model(num_cols, cat_cols):
    return Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])


def rf_model(num_cols, cat_cols, seed):
    return Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )),
    ])


def xgb_model(num_cols, cat_cols, seed, y_train):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        print(f"[WARN] xgboost unavailable; skipping XGBoost model: {e}")
        return None
    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    return Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            scale_pos_weight=neg / pos,
        )),
    ])


def add_threshold_rows(rows, split, model, y_true, p):
    for t in [0.5, 0.7, 0.8, 0.9]:
        pred = p >= t
        rows.append({
            "split": split,
            "model": model,
            "threshold": t,
            "n_retained": int(pred.sum()),
            "retained_fraction": float(pred.mean()),
            "precision_at_threshold": precision_score(y_true, pred, zero_division=0),
            "recall_at_threshold": recall_score(y_true, pred, zero_division=0),
        })


def eval_model(rows, threshold_rows, split, model_name, model, X, y, tr, te):
    fit = model.fit(X.iloc[tr], y[tr])
    p = scores(fit, X.iloc[te])
    row = {"split": split, "model": model_name, "n_train": len(tr), "n_test": len(te)}
    row.update(metrics(y[te], p))
    rows.append(row)
    add_threshold_rows(threshold_rows, split, model_name, y[te], p)
    return fit


def write_coefficients(model, path):
    try:
        imp = pd.DataFrame({
            "feature": model.named_steps["pre"].get_feature_names_out(),
            "coefficient": model.named_steps["clf"].coef_.ravel(),
        })
        imp["abs_coefficient"] = imp["coefficient"].abs()
        imp.sort_values("abs_coefficient", ascending=False).to_csv(path, index=False)
    except Exception as e:
        print(f"[WARN] coefficients unavailable for {path.name}: {e}")


def run_ablation_experiments(df, feats, out_dir, seed, n_folds, skip_rf, skip_xgb, skip_full_hybrid):
    y = df["meaning_preserved"].astype(int).values
    eng_num, eng_cat, sem_cols = get_feature_columns(feats)
    X_eng = feats[eng_num + eng_cat]
    X_sem = feats[sem_cols] if sem_cols else None
    X_eng_sem = feats[eng_num + sem_cols + eng_cat] if sem_cols else None

    pd.DataFrame([
        {"feature_set": "bow_tfidf", "features": "TF-IDF ngrams", "role": "lexical baseline only"},
        {"feature_set": "engineered", "features": ", ".join(eng_num + eng_cat), "role": "hand-engineered/dictionary features only"},
        {"feature_set": "semantic", "features": ", ".join(sem_cols) if sem_cols else "none", "role": "embedding/NLI features only"},
        {"feature_set": "engineered_semantic", "features": ", ".join(eng_num + sem_cols + eng_cat) if sem_cols else "none", "role": "main candidate feature set"},
        {"feature_set": "full_hybrid", "features": "TF-IDF + engineered + semantic", "role": "ablation only"},
    ]).to_csv(out_dir / "feature_sets.csv", index=False)

    rows, threshold_rows = [], []
    for split_name, tr, te in splits(df, seed, n_folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        eval_model(rows, threshold_rows, split_name, "majority", DummyClassifier(strategy="most_frequent"), X_eng, y, tr, te)

        bow = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        bow.fit(df.iloc[tr]["pair_text"], y[tr])
        p = scores(bow, df.iloc[te]["pair_text"])
        row = {"split": split_name, "model": "bow_tfidf_lr_baseline", "n_train": len(tr), "n_test": len(te)}
        row.update(metrics(y[te], p))
        rows.append(row)
        add_threshold_rows(threshold_rows, split_name, "bow_tfidf_lr_baseline", y[te], p)

        eval_model(rows, threshold_rows, split_name, "engineered_lr", lr_model(eng_num, eng_cat), X_eng, y, tr, te)
        if not skip_rf:
            eval_model(rows, threshold_rows, split_name, "engineered_rf", rf_model(eng_num, eng_cat, seed), X_eng, y, tr, te)
        if not skip_xgb:
            xgb = xgb_model(eng_num, eng_cat, seed, y[tr])
            if xgb is not None:
                eval_model(rows, threshold_rows, split_name, "engineered_xgb", xgb, X_eng, y, tr, te)

        if X_sem is not None:
            eval_model(rows, threshold_rows, split_name, "semantic_lr", lr_model(sem_cols, []), X_sem, y, tr, te)
            if not skip_rf:
                eval_model(rows, threshold_rows, split_name, "semantic_rf", rf_model(sem_cols, [], seed), X_sem, y, tr, te)
            if not skip_xgb:
                xgb = xgb_model(sem_cols, [], seed, y[tr])
                if xgb is not None:
                    eval_model(rows, threshold_rows, split_name, "semantic_xgb", xgb, X_sem, y, tr, te)

            eval_model(rows, threshold_rows, split_name, "engineered_semantic_lr", lr_model(eng_num + sem_cols, eng_cat), X_eng_sem, y, tr, te)
            if not skip_rf:
                eval_model(rows, threshold_rows, split_name, "engineered_semantic_rf", rf_model(eng_num + sem_cols, eng_cat, seed), X_eng_sem, y, tr, te)
            if not skip_xgb:
                xgb = xgb_model(eng_num + sem_cols, eng_cat, seed, y[tr])
                if xgb is not None:
                    eval_model(rows, threshold_rows, split_name, "engineered_semantic_xgb", xgb, X_eng_sem, y, tr, te)

            if not skip_full_hybrid:
                tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
                Xb_tr = tf.fit_transform(df.iloc[tr]["pair_text"])
                Xb_te = tf.transform(df.iloc[te]["pair_text"])
                pre = make_preprocessor(eng_num + sem_cols, eng_cat)
                Xs_tr = pre.fit_transform(X_eng_sem.iloc[tr], y[tr])
                Xs_te = pre.transform(X_eng_sem.iloc[te])
                clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
                clf.fit(sparse.hstack([Xb_tr, Xs_tr]).tocsr(), y[tr])
                p = scores(clf, sparse.hstack([Xb_te, Xs_te]).tocsr())
                name = "full_hybrid_tfidf_engineered_semantic_lr"
                row = {"split": split_name, "model": name, "n_train": len(tr), "n_test": len(te)}
                row.update(metrics(y[te], p))
                rows.append(row)
                add_threshold_rows(threshold_rows, split_name, name, y[te], p)

    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_split.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out_dir / "threshold_metrics.csv", index=False)

    final = lr_model(eng_num, eng_cat).fit(X_eng, y)
    joblib.dump(final, out_dir / "final_engineered_lr.joblib")
    write_coefficients(final, out_dir / "engineered_lr_coefficients.csv")
    if X_eng_sem is not None:
        final = lr_model(eng_num + sem_cols, eng_cat).fit(X_eng_sem, y)
        joblib.dump(final, out_dir / "final_engineered_semantic_lr.joblib")
        write_coefficients(final, out_dir / "engineered_semantic_lr_coefficients.csv")


def write_report(processed, results, metadata):
    audit = pd.read_csv(processed / "dataset_audit.csv")
    met = pd.read_csv(results / "metrics_by_split.csv")
    th = pd.read_csv(results / "threshold_metrics.csv")
    fs = pd.read_csv(results / "feature_sets.csv")
    avg = met.groupby("model", as_index=False)[["auroc", "auprc", "accuracy", "precision", "recall", "f1"]].mean(numeric_only=True)
    lines = [
        "# Meaning-Preservation Feature Ablation Report",
        "",
        "## Run metadata",
        pd.DataFrame([{"field": k, "value": v} for k, v in metadata.items()]).to_markdown(index=False),
        "",
        "## Feature sets",
        fs.to_markdown(index=False),
        "",
        "## Dataset audit",
        audit.to_markdown(index=False),
        "",
        "## Metrics",
        met.sort_values(["split", "f1"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Average across splits",
        avg.sort_values("f1", ascending=False).to_markdown(index=False),
        "",
        "## Threshold metrics",
        th.to_markdown(index=False),
        "",
        "## Note",
        "BOW is retained as a lexical baseline. The main candidate feature set is engineered_semantic, which combines consent-aware engineered features with embedding/NLI similarity features when available.",
    ]
    (results / "meaning_preservation_classifier_summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--output_dir", default="meaning_preservation/outputs/feature_ablation")
    ap.add_argument("--embedding_model", default=None)
    ap.add_argument("--embedding_backend", default="auto", choices=["auto", "sentence_transformers", "hf"])
    ap.add_argument("--embedding_device", default=None)
    ap.add_argument("--nli_model", default=None)
    ap.add_argument("--nli_device", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_cv_splits", type=int, default=3)
    ap.add_argument("--skip_rf", action="store_true")
    ap.add_argument("--skip_xgb", action="store_true")
    ap.add_argument("--skip_full_hybrid", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    processed, features, results = out / "processed", out / "features", out / "results"
    for d in [processed, features, results]:
        d.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building dataset")
    df = build_dataset(Path(args.roundtrips_csv), processed)
    print(df.shape, df["meaning_preserved"].value_counts().to_dict())

    print("[2/4] Extracting features")
    feats = extract_features(df)
    feats = add_embeddings(df, feats, args.embedding_model, args.batch_size, backend=args.embedding_backend, device=args.embedding_device)
    feats = add_nli(df, feats, args.nli_model, max(1, args.batch_size // 2), args.nli_device)
    feats.to_csv(features / "features.csv", index=False)

    eng_num, eng_cat, sem_cols = get_feature_columns(feats)
    metadata = {
        "embedding_model_requested": args.embedding_model or "none",
        "embedding_model_resolved": _resolve_hf_model_name(args.embedding_model) if args.embedding_model else "none",
        "embedding_backend": args.embedding_backend,
        "embedding_device": args.embedding_device or "auto",
        "semantic_features": ", ".join(sem_cols) if sem_cols else "none",
        "engineered_numeric_feature_count": len(eng_num),
        "engineered_categorical_features": ", ".join(eng_cat),
        "classifiers": "logistic regression, random forest, XGBoost if installed",
    }
    (results / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("[3/4] Running feature-set ablations")
    run_ablation_experiments(df, feats, results, args.seed, args.group_cv_splits, args.skip_rf, args.skip_xgb, args.skip_full_hybrid)

    print("[4/4] Writing report")
    write_report(processed, results, metadata)
    print(f"Done: {results / 'meaning_preservation_classifier_summary.md'}")


if __name__ == "__main__":
    main()
