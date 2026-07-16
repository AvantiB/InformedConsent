#!/usr/bin/env python
"""Literature-informed feature subgroup ablations for meaning preservation.

This runner is intended for the consent/data-sharing contribution of the
project. It separates broad "engineered" features into interpretable subgroups:

1. BOW TF-IDF baseline
2. lexical similarity only
3. mapping/annotation complexity only
4. dictionary/modal cue features only
5. semantic embedding/NLI features only
6. dictionary/modal + semantic
7. all engineered features
8. all engineered + semantic
9. TF-IDF + all engineered + semantic as a full-hybrid ablation

The cue dictionary is external JSON so the seed vocabulary can be edited,
expanded from literature, or replaced without touching code.
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

import run_classifier_experiments as base


ID_COLS = {"roundtrip_id"}
METADATA_COLS = {"llm", "information_model"}
MODAL_CATEGORICAL_COLS = ["modal_orig", "modal_recon"]

LEXICAL_COLS = [
    "orig_len_tokens",
    "recon_len_tokens",
    "length_ratio",
    "abs_length_diff",
    "token_jaccard",
    "tfidf_cosine",
]

MAPPING_COMPLEXITY_COLS = [
    "annotation_count",
    "unique_element_count",
    "mapping_len_chars",
    "mapping_bracket_count",
    "mapping_paren_count",
]


def load_cue_dictionary(path: Path | None) -> dict:
    if path is None:
        return {"cue_groups": base.CUE_GROUPS, "name": "built_in_seed_cues", "references": []}
    with path.open() as f:
        data = json.load(f)
    if "cue_groups" not in data or not isinstance(data["cue_groups"], dict):
        raise ValueError("Cue dictionary JSON must contain an object field named 'cue_groups'.")
    return data


def apply_cue_dictionary(data: dict) -> None:
    """Patch run_classifier_experiments cue groups before feature extraction."""
    cue_groups = {str(k): [str(vv).lower() for vv in v] for k, v in data["cue_groups"].items()}
    base.CUE_GROUPS = cue_groups

    # extract_features uses these three lists to derive modal_orig/modal_recon.
    # Keep the original lists if a custom dictionary omits the corresponding key.
    base.PERMISSION = cue_groups.get("permission", base.PERMISSION)
    base.OBLIGATION = cue_groups.get("obligation", base.OBLIGATION)
    base.PROHIBITION = cue_groups.get("prohibition", base.PROHIBITION)


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X).astype(float)


def make_preprocessor(num_cols: list[str], cat_cols: list[str]):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def lr_model(num_cols: list[str], cat_cols: list[str]):
    return Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])


def rf_model(num_cols: list[str], cat_cols: list[str], seed: int):
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


def xgb_model(num_cols: list[str], cat_cols: list[str], seed: int, y_train: np.ndarray):
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


def cue_feature_columns(feats: pd.DataFrame, cue_groups: dict) -> tuple[list[str], list[str]]:
    numeric = []
    for group in cue_groups:
        for pattern in [
            f"orig_{group}_count",
            f"recon_{group}_count",
            f"{group}_jaccard",
            f"{group}_missing_count",
            f"{group}_added_count",
            f"{group}_presence_preserved",
        ]:
            if pattern in feats.columns:
                numeric.append(pattern)
    if "modal_category_changed" in feats.columns:
        numeric.append("modal_category_changed")
    cat = [c for c in MODAL_CATEGORICAL_COLS if c in feats.columns]
    return sorted(set(numeric)), cat


def present_numeric(cols: list[str], feats: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in feats.columns and pd.api.types.is_numeric_dtype(feats[c])]


def build_feature_sets(feats: pd.DataFrame, cue_groups: dict) -> dict[str, tuple[pd.DataFrame, list[str], list[str], str]]:
    lexical = present_numeric(LEXICAL_COLS, feats)
    mapping = present_numeric(MAPPING_COMPLEXITY_COLS, feats)
    cue_num, cue_cat = cue_feature_columns(feats, cue_groups)
    semantic = [c for c in feats.columns if c.startswith("embedding_") or c.startswith("nli_")]

    feature_sets: dict[str, tuple[pd.DataFrame, list[str], list[str], str]] = {}

    def add(name: str, num_cols: list[str], cat_cols: list[str], role: str):
        num_cols = [c for c in dict.fromkeys(num_cols) if c in feats.columns]
        cat_cols = [c for c in dict.fromkeys(cat_cols) if c in feats.columns]
        if not num_cols and not cat_cols:
            return
        feature_sets[name] = (feats[num_cols + cat_cols].copy(), num_cols, cat_cols, role)

    add("lexical_similarity", lexical, [], "generic lexical/surface similarity features only")
    add("mapping_complexity", mapping, [], "annotation/mapping complexity features only")
    add("dictionary_modal", cue_num, cue_cat, "literature-informed deontic/privacy/biobank cue features only")
    add("engineered_all", lexical + mapping + cue_num, cue_cat, "all non-embedding hand-engineered features")
    if semantic:
        add("semantic", semantic, [], "embedding/NLI semantic similarity features only")
        add("dictionary_modal_semantic", cue_num + semantic, cue_cat, "dictionary/modal cues plus semantic similarity")
        add("engineered_semantic", lexical + mapping + cue_num + semantic, cue_cat, "all engineered features plus semantic similarity")
    return feature_sets


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
    row.update(base.metrics(y[te], p))
    rows.append(row)
    add_threshold_rows(threshold_rows, split, model_name, y[te], p)
    return fit


def write_lr_coefficients(model, path: Path):
    try:
        imp = pd.DataFrame({
            "feature": model.named_steps["pre"].get_feature_names_out(),
            "coefficient": model.named_steps["clf"].coef_.ravel(),
        })
        imp["abs_coefficient"] = imp["coefficient"].abs()
        imp.sort_values("abs_coefficient", ascending=False).to_csv(path, index=False)
    except Exception as e:
        print(f"[WARN] coefficients unavailable for {path.name}: {e}")


def write_rf_importance(model, path: Path):
    try:
        feature_names = model.named_steps["pre"].get_feature_names_out()
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.named_steps["clf"].feature_importances_,
        })
        imp.sort_values("importance", ascending=False).to_csv(path, index=False)
    except Exception as e:
        print(f"[WARN] RF importances unavailable for {path.name}: {e}")


def write_cue_dictionary_tables(cue_data: dict, feats: pd.DataFrame, df: pd.DataFrame, out_dir: Path):
    cue_groups = cue_data["cue_groups"]

    pd.DataFrame([
        {
            "cue_group": group,
            "n_terms": len(terms),
            "terms": ", ".join(terms),
        }
        for group, terms in cue_groups.items()
    ]).to_csv(out_dir / "cue_dictionary_terms.csv", index=False)

    rows = []
    labels = df["meaning_preserved"].astype(int).values
    for group in cue_groups:
        cols = {
            "orig_count": f"orig_{group}_count",
            "recon_count": f"recon_{group}_count",
            "jaccard": f"{group}_jaccard",
            "missing_count": f"{group}_missing_count",
            "added_count": f"{group}_added_count",
            "presence_preserved": f"{group}_presence_preserved",
        }
        if cols["orig_count"] not in feats.columns:
            continue
        for label in [0, 1]:
            mask = labels == label
            row = {
                "cue_group": group,
                "label": label,
                "n_rows": int(mask.sum()),
                "orig_present_fraction": float((feats.loc[mask, cols["orig_count"]] > 0).mean()),
                "recon_present_fraction": float((feats.loc[mask, cols["recon_count"]] > 0).mean()),
            }
            for stat_name, col in cols.items():
                if col in feats.columns:
                    row[f"mean_{stat_name}"] = float(feats.loc[mask, col].mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "cue_group_frequency_by_label.csv", index=False)


def run_ablation(df, feats, cue_data, out_dir: Path, seed: int, n_folds: int, skip_rf: bool, skip_xgb: bool, skip_full_hybrid: bool):
    y = df["meaning_preserved"].astype(int).values
    cue_groups = cue_data["cue_groups"]
    feature_sets = build_feature_sets(feats, cue_groups)

    fs_rows = [
        {"feature_set": "bow_tfidf", "role": "lexical baseline only", "n_numeric": np.nan, "n_categorical": np.nan, "features": "TF-IDF ngrams"}
    ]
    for name, (_, num_cols, cat_cols, role) in feature_sets.items():
        fs_rows.append({
            "feature_set": name,
            "role": role,
            "n_numeric": len(num_cols),
            "n_categorical": len(cat_cols),
            "features": ", ".join(num_cols + cat_cols),
        })
    if "engineered_semantic" in feature_sets:
        fs_rows.append({
            "feature_set": "full_hybrid_tfidf_engineered_semantic",
            "role": "ablation only; tests whether adding sparse BOW helps beyond engineered+semantic",
            "n_numeric": np.nan,
            "n_categorical": np.nan,
            "features": "TF-IDF + engineered_semantic",
        })
    pd.DataFrame(fs_rows).to_csv(out_dir / "feature_sets.csv", index=False)

    rows, threshold_rows = [], []
    for split_name, tr, te in base.splits(df, seed, n_folds):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        # Baselines
        first_X = next(iter(feature_sets.values()))[0]
        eval_model(rows, threshold_rows, split_name, "majority", DummyClassifier(strategy="most_frequent"), first_X, y, tr, te)

        bow = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ])
        bow.fit(df.iloc[tr]["pair_text"], y[tr])
        p = scores(bow, df.iloc[te]["pair_text"])
        row = {"split": split_name, "model": "bow_tfidf_lr_baseline", "n_train": len(tr), "n_test": len(te)}
        row.update(base.metrics(y[te], p))
        rows.append(row)
        add_threshold_rows(threshold_rows, split_name, "bow_tfidf_lr_baseline", y[te], p)

        for fs_name, (X, num_cols, cat_cols, _) in feature_sets.items():
            eval_model(rows, threshold_rows, split_name, f"{fs_name}_lr", lr_model(num_cols, cat_cols), X, y, tr, te)
            if not skip_rf:
                eval_model(rows, threshold_rows, split_name, f"{fs_name}_rf", rf_model(num_cols, cat_cols, seed), X, y, tr, te)
            if not skip_xgb:
                xgb = xgb_model(num_cols, cat_cols, seed, y[tr])
                if xgb is not None:
                    eval_model(rows, threshold_rows, split_name, f"{fs_name}_xgb", xgb, X, y, tr, te)

        if not skip_full_hybrid and "engineered_semantic" in feature_sets:
            X_es, num_cols, cat_cols, _ = feature_sets["engineered_semantic"]
            tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
            Xb_tr = tf.fit_transform(df.iloc[tr]["pair_text"])
            Xb_te = tf.transform(df.iloc[te]["pair_text"])
            pre = make_preprocessor(num_cols, cat_cols)
            Xs_tr = pre.fit_transform(X_es.iloc[tr], y[tr])
            Xs_te = pre.transform(X_es.iloc[te])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
            clf.fit(sparse.hstack([Xb_tr, Xs_tr]).tocsr(), y[tr])
            p = scores(clf, sparse.hstack([Xb_te, Xs_te]).tocsr())
            model_name = "full_hybrid_tfidf_engineered_semantic_lr"
            row = {"split": split_name, "model": model_name, "n_train": len(tr), "n_test": len(te)}
            row.update(base.metrics(y[te], p))
            rows.append(row)
            add_threshold_rows(threshold_rows, split_name, model_name, y[te], p)

    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_split.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out_dir / "threshold_metrics.csv", index=False)

    # Train final interpretable models on the full data for feature analysis.
    for fs_name in ["dictionary_modal", "engineered_all", "dictionary_modal_semantic", "engineered_semantic"]:
        if fs_name not in feature_sets:
            continue
        X, num_cols, cat_cols, _ = feature_sets[fs_name]
        final_lr = lr_model(num_cols, cat_cols).fit(X, y)
        joblib.dump(final_lr, out_dir / f"final_{fs_name}_lr.joblib")
        write_lr_coefficients(final_lr, out_dir / f"{fs_name}_lr_coefficients.csv")
        if not skip_rf:
            final_rf = rf_model(num_cols, cat_cols, seed).fit(X, y)
            joblib.dump(final_rf, out_dir / f"final_{fs_name}_rf.joblib")
            write_rf_importance(final_rf, out_dir / f"{fs_name}_rf_importances.csv")


def write_report(processed: Path, results: Path, metadata: dict):
    audit = pd.read_csv(processed / "dataset_audit.csv")
    met = pd.read_csv(results / "metrics_by_split.csv")
    th = pd.read_csv(results / "threshold_metrics.csv")
    fs = pd.read_csv(results / "feature_sets.csv")
    avg = met.groupby("model", as_index=False)[["auroc", "auprc", "accuracy", "precision", "recall", "f1"]].mean(numeric_only=True)
    lines = [
        "# Meaning-Preservation Literature-Informed Feature Subgroup Report",
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
        "## Feature analysis outputs",
        "- cue_dictionary_terms.csv",
        "- cue_group_frequency_by_label.csv",
        "- *_lr_coefficients.csv",
        "- *_rf_importances.csv",
        "",
        "## Note",
        "The cue dictionary is a literature-informed seed resource, not a complete validated ontology. Use the subgroup ablations to test whether deontic, privacy/data-practice, and biobank-specific language categories carry predictive signal for meaning preservation.",
    ]
    (results / "meaning_preservation_feature_subgroup_summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--output_dir", default="meaning_preservation/outputs/feature_subgroup_ablation")
    ap.add_argument("--cue_dictionary", default="meaning_preservation/literature_informed_consent_cues.json")
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

    cue_path = Path(args.cue_dictionary) if args.cue_dictionary else None
    cue_data = load_cue_dictionary(cue_path)
    apply_cue_dictionary(cue_data)

    print("[1/4] Building dataset")
    df = base.build_dataset(Path(args.roundtrips_csv), processed)
    print(df.shape, df["meaning_preserved"].value_counts().to_dict())

    print("[2/4] Extracting literature-informed features")
    feats = base.extract_features(df)
    feats = base.add_embeddings(df, feats, args.embedding_model, args.batch_size, backend=args.embedding_backend, device=args.embedding_device)
    feats = base.add_nli(df, feats, args.nli_model, max(1, args.batch_size // 2), args.nli_device)
    feats.to_csv(features / "features.csv", index=False)
    write_cue_dictionary_tables(cue_data, feats, df, results)

    semantic_cols = [c for c in feats.columns if c.startswith("embedding_") or c.startswith("nli_")]
    metadata = {
        "cue_dictionary": str(cue_path) if cue_path else "built-in",
        "cue_dictionary_name": cue_data.get("name", "unknown"),
        "n_cue_groups": len(cue_data["cue_groups"]),
        "embedding_model_requested": args.embedding_model or "none",
        "embedding_model_resolved": base._resolve_hf_model_name(args.embedding_model) if args.embedding_model else "none",
        "embedding_backend": args.embedding_backend,
        "embedding_device": args.embedding_device or "auto",
        "semantic_features": ", ".join(semantic_cols) if semantic_cols else "none",
        "classifiers": "logistic regression, random forest, XGBoost if installed",
        "literature_basis": "deontic modality; privacy/data-practice categories; DUO/data-use restrictions; biobank consent concepts",
    }
    (results / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("[3/4] Running feature subgroup ablations")
    run_ablation(df, feats, cue_data, results, args.seed, args.group_cv_splits, args.skip_rf, args.skip_xgb, args.skip_full_hybrid)

    print("[4/4] Writing report")
    write_report(processed, results, metadata)
    print(f"Done: {results / 'meaning_preservation_feature_subgroup_summary.md'}")


if __name__ == "__main__":
    main()
