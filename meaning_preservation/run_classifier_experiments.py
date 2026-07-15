#!/usr/bin/env python
"""MVP meaning-preservation classifier experiments.

Runs lexical, consent-aware, optional embedding, optional NLI, and hybrid
classifiers for binary meaning-preservation labels from LLM round trips.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PERMISSION = ["may", "can", "could", "allowed", "allow", "permitted", "permit", "authorized", "agree", "consent"]
OBLIGATION = ["must", "should", "required", "require", "need to", "have to", "responsible", "obligated", "duty"]
PROHIBITION = ["may not", "cannot", "can't", "will not", "not allowed", "not permitted", "prohibited", "restricted", "no "]
NEGATION = ["not", "no", "never", "cannot", "can't", "without", "neither", "nor"]
CONDITION = ["if", "when", "unless", "only if", "as long as", "provided that", "until", "before", "after", "during"]
EXCEPTION = ["however", "except", "but", "although", "nevertheless", "already", "prior", "except that"]
RESTRICTION = ["only", "limited", "limit", "restriction", "restricted", "commercial", "non-commercial", "identifiable", "de-identified", "geographic", "institution", "approved", "irb", "ethics", "no expiration", "at any time"]
WITHDRAWAL = ["withdraw", "revoke", "quit", "stop", "withdrawal"]
ACTIONS = ["use", "used", "store", "stored", "share", "shared", "disclose", "disclosed", "collect", "collected", "withdraw", "revoke", "destroy", "retain", "contact", "return", "access", "sell", "distribute", "retrieve", "study", "analyze", "learn"]
RESOURCES = ["data", "information", "health information", "medical record", "records", "dna", "sample", "samples", "specimen", "specimens", "biospecimen", "blood", "urine", "saliva", "results", "database", "databases"]
ACTORS = ["researcher", "researchers", "doctor", "doctors", "study team", "institution", "sponsor", "company", "biobank", "irb", "university", "clinic", "hospital", "all of us", "mayo"]
PURPOSES = ["research", "future research", "cancer", "genetic", "genomic", "commercial", "clinical care", "public health", "study", "studies"]

CUE_GROUPS = {
    "permission": PERMISSION,
    "obligation": OBLIGATION,
    "prohibition": PROHIBITION,
    "negation": NEGATION,
    "condition": CONDITION,
    "exception": EXCEPTION,
    "restriction": RESTRICTION,
    "withdrawal": WITHDRAWAL,
    "action": ACTIONS,
    "resource": RESOURCES,
    "actor": ACTORS,
    "purpose": PURPOSES,
}


def norm(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def low(x) -> str:
    return norm(x).lower()


def infer_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column; tried {candidates}. Available: {list(df.columns)}")
    return None


def parse_label(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        if int(x) in (0, 1):
            return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "preserved", "meaning preserved", "pass"}:
        return 1
    if s in {"0", "false", "no", "n", "not preserved", "not_preserved", "meaning not preserved", "fail"}:
        return 0
    return np.nan


def cue_set(text: str, cues: Sequence[str]) -> set[str]:
    t = low(text)
    found = set()
    for cue in cues:
        c = cue.lower()
        pat = re.escape(c) if " " in c or not c.isalnum() else r"\b" + re.escape(c) + r"\b"
        if re.search(pat, t):
            found.add(c)
    return found


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", low(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_dataset(roundtrips_csv: Path, out_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(roundtrips_csv)
    id_col = infer_col(raw, ["roundtrip_id", "id", "source_id"], False)
    sent_col = infer_col(raw, ["sentence_id", "sent_id", "source_sentence_id"], False)
    form_col = infer_col(raw, ["form_id", "form_key", "source_file"], False)
    orig_col = infer_col(raw, ["canonical_full_text", "full_text_original", "original_sentence", "full_text"])
    recon_col = infer_col(raw, ["reconstructed_sentence", "backward_mapping", "backward_reconstruction", "reconstruction"])
    map_col = infer_col(raw, ["annotations_serialized", "forward_mapping", "annotations_combined", "mapping"], False)
    llm_col = infer_col(raw, ["llm", "model", "llm_name"])
    info_col = infer_col(raw, ["information_model", "info_model", "model_family"])
    label_col = infer_col(raw, ["meaning_preserved", "human_meaning_preserved", "label"])
    ann_col = infer_col(raw, ["annotation_count", "n_annotations"], False)
    uniq_col = infer_col(raw, ["unique_element_count", "n_unique_elements"], False)
    df = pd.DataFrame({
        "roundtrip_id": raw[id_col].astype(str) if id_col else [f"RT_{i}" for i in range(len(raw))],
        "sentence_id": raw[sent_col].astype(str) if sent_col else [f"SENT_{i}" for i in range(len(raw))],
        "form_id": raw[form_col].astype(str) if form_col else "FORM_UNKNOWN",
        "original_text": raw[orig_col].map(norm),
        "reconstructed_text": raw[recon_col].map(norm),
        "forward_mapping": raw[map_col].map(norm) if map_col else "",
        "llm": raw[llm_col].astype(str),
        "information_model": raw[info_col].astype(str),
        "meaning_preserved": raw[label_col].map(parse_label),
        "annotation_count": raw[ann_col] if ann_col else np.nan,
        "unique_element_count": raw[uniq_col] if uniq_col else np.nan,
    })
    df = df.dropna(subset=["meaning_preserved"])
    df = df[df["reconstructed_text"].str.len() > 0].copy()
    df["meaning_preserved"] = df["meaning_preserved"].astype(int)
    df["pair_text"] = "ORIGINAL: " + df["original_text"] + "\nRECONSTRUCTION: " + df["reconstructed_text"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "roundtrip_dataset.csv", index=False)
    audit = pd.DataFrame([
        {"metric": "rows", "value": len(df)},
        {"metric": "positive_labels", "value": int(df["meaning_preserved"].sum())},
        {"metric": "negative_labels", "value": int((df["meaning_preserved"] == 0).sum())},
        {"metric": "llms", "value": df["llm"].nunique()},
        {"metric": "information_models", "value": df["information_model"].nunique()},
        {"metric": "sentences", "value": df["sentence_id"].nunique()},
        {"metric": "forms", "value": df["form_id"].nunique()},
    ])
    audit.to_csv(out_dir / "dataset_audit.csv", index=False)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        orig, rec = low(r["original_text"]), low(r["reconstructed_text"])
        otok, rtok = tokens(orig), tokens(rec)
        row = {
            "roundtrip_id": r["roundtrip_id"],
            "orig_len_tokens": len(otok),
            "recon_len_tokens": len(rtok),
            "length_ratio": len(rtok) / max(1, len(otok)),
            "abs_length_diff": abs(len(otok) - len(rtok)),
            "token_jaccard": jaccard(otok, rtok),
            "annotation_count": r.get("annotation_count", np.nan),
            "unique_element_count": r.get("unique_element_count", np.nan),
            "mapping_len_chars": len(low(r.get("forward_mapping", ""))),
            "mapping_bracket_count": low(r.get("forward_mapping", "")).count("["),
            "mapping_paren_count": low(r.get("forward_mapping", "")).count("("),
            "llm": r["llm"],
            "information_model": r["information_model"],
        }
        for name, cues in CUE_GROUPS.items():
            a, b = cue_set(orig, cues), cue_set(rec, cues)
            row[f"orig_{name}_count"] = len(a)
            row[f"recon_{name}_count"] = len(b)
            row[f"{name}_jaccard"] = jaccard(a, b)
            row[f"{name}_missing_count"] = len(a - b)
            row[f"{name}_added_count"] = len(b - a)
            row[f"{name}_presence_preserved"] = float(bool(a) == bool(b))
        om = "prohibition" if cue_set(orig, PROHIBITION) else "obligation" if cue_set(orig, OBLIGATION) else "permission" if cue_set(orig, PERMISSION) else "none"
        rm = "prohibition" if cue_set(rec, PROHIBITION) else "obligation" if cue_set(rec, OBLIGATION) else "permission" if cue_set(rec, PERMISSION) else "none"
        row["modal_orig"] = om
        row["modal_recon"] = rm
        row["modal_category_changed"] = float(om != rm)
        rows.append(row)
    feats = pd.DataFrame(rows)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(pd.concat([df["original_text"], df["reconstructed_text"]], ignore_index=True).fillna(""))
    n = len(df)
    feats["tfidf_cosine"] = np.asarray(X[:n].multiply(X[n:]).sum(axis=1)).ravel()
    return feats


def _resolve_hf_model_name(model_name: str) -> str:
    """Allow shorthand names like all-MiniLM-L6-v2 while preserving local paths."""
    if Path(model_name).exists() or "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def _encode_with_hf_transformers(texts: list[str], model_name: str, batch_size: int, device: str | None) -> np.ndarray:
    """Mean-pool a Hugging Face encoder without importing sentence-transformers.

    This fallback avoids sentence-transformers import failures that can occur on
    HPC environments with incompatible transformers/torchvision stacks.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"Could not import torch/transformers for HF embedding fallback: {e}") from e

    hf_name = _resolve_hf_model_name(model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(device).eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = ["" if pd.isna(t) else str(t) for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            outputs.append(pooled.cpu().numpy())

    return _l2_normalize(np.vstack(outputs))


def add_embeddings(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    model_name: str | None,
    batch_size: int,
    backend: str = "auto",
    device: str | None = None,
) -> pd.DataFrame:
    if not model_name:
        return feats

    backend = backend.lower()
    if backend not in {"auto", "sentence_transformers", "hf"}:
        raise ValueError("--embedding_backend must be one of: auto, sentence_transformers, hf")

    eo = er = None
    st_error = None

    if backend in {"auto", "sentence_transformers"}:
        try:
            from sentence_transformers import SentenceTransformer
            st_name = _resolve_hf_model_name(model_name)
            print(f"[INFO] Computing embeddings with sentence-transformers: {st_name}")
            model = SentenceTransformer(st_name, device=device)
            eo = model.encode(
                df["original_text"].tolist(),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            er = model.encode(
                df["reconstructed_text"].tolist(),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        except Exception as e:
            st_error = e
            if backend == "sentence_transformers":
                print(f"[WARN] sentence-transformers embedding failed; skipping embeddings: {e}")
                return feats
            print(f"[WARN] sentence-transformers embedding failed; trying HF transformers fallback: {e}")

    if eo is None or er is None:
        try:
            print(f"[INFO] Computing embeddings with HF transformers mean pooling: {_resolve_hf_model_name(model_name)}")
            eo = _encode_with_hf_transformers(df["original_text"].tolist(), model_name, batch_size, device)
            er = _encode_with_hf_transformers(df["reconstructed_text"].tolist(), model_name, batch_size, device)
        except Exception as e:
            print(f"[WARN] HF embedding fallback failed; skipping embeddings: {e}")
            if st_error is not None:
                print(f"[WARN] Original sentence-transformers error was: {st_error}")
            return feats

    feats = feats.copy()
    feats["embedding_cosine"] = (np.asarray(eo) * np.asarray(er)).sum(axis=1)
    feats["embedding_distance"] = 1 - feats["embedding_cosine"]
    return feats


def add_nli(df: pd.DataFrame, feats: pd.DataFrame, model_name: str | None, batch_size: int, device: str | None) -> pd.DataFrame:
    if not model_name:
        return feats
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        print(f"[WARN] transformers/torch unavailable; skipping NLI: {e}")
        return feats
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    def idx(term, default):
        return next((i for i, v in id2label.items() if term in v), default)

    ent, neu, con = idx("entail", 2), idx("neutral", 1), idx("contrad", 0)

    def score(a, b):
        outs = []
        with torch.no_grad():
            for i in range(0, len(a), batch_size):
                enc = tok(
                    a[i:i + batch_size],
                    b[i:i + batch_size],
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                outs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
        return np.vstack(outs)

    o = df["original_text"].fillna("").astype(str).tolist()
    r = df["reconstructed_text"].fillna("").astype(str).tolist()
    por = score(o, r)
    pro = score(r, o)
    feats = feats.copy()
    feats["nli_entail_o2r"] = por[:, ent]
    feats["nli_contra_o2r"] = por[:, con]
    feats["nli_neutral_o2r"] = por[:, neu]
    feats["nli_entail_r2o"] = pro[:, ent]
    feats["nli_contra_r2o"] = pro[:, con]
    feats["nli_neutral_r2o"] = pro[:, neu]
    feats["nli_min_bidirectional_entail"] = np.minimum(por[:, ent], pro[:, ent])
    feats["nli_max_contradiction"] = np.maximum(por[:, con], pro[:, con])
    return feats


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "auprc": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
    }


def splits(df: pd.DataFrame, seed: int, n_folds: int):
    y = df["meaning_preserved"].values
    out = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    out += [("random", tr, te) for tr, te in sss.split(np.zeros(len(df)), y)]
    groups = df["sentence_id"].astype(str).values
    k = min(n_folds, len(np.unique(groups)))
    if k >= 2:
        gkf = GroupKFold(n_splits=k)
        out += [(f"leave_sentence_fold{i + 1}", tr, te) for i, (tr, te) in enumerate(gkf.split(np.zeros(len(df)), y, groups))]
    for col, prefix in [("llm", "leave_llm"), ("information_model", "leave_info_model")]:
        vals = df[col].astype(str)
        for v in sorted(vals.unique()):
            tr, te = np.where(vals.values != v)[0], np.where(vals.values == v)[0]
            if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", v)[:60]
                out.append((f"{prefix}_{safe}", tr, te))
            else:
                print(f"[WARN] skipping {prefix}={v}: only one class in train/test")
    return out


def make_preprocessor(feats: pd.DataFrame, include_metadata=True):
    cat = ["modal_orig", "modal_recon"]
    if include_metadata:
        cat += ["llm", "information_model"]
    cat = [c for c in cat if c in feats.columns]
    num = [c for c in feats.columns if c not in {"roundtrip_id"} | set(cat) and pd.api.types.is_numeric_dtype(feats[c])]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return pre, feats[num + cat]


def run_experiments(df: pd.DataFrame, feats: pd.DataFrame, out_dir: Path, seed: int, n_folds: int, run_rf: bool):
    y = df["meaning_preserved"].astype(int).values
    sp = splits(df, seed, n_folds)
    pre, Xf = make_preprocessor(feats, True)
    pre_no, Xf_no = make_preprocessor(feats, False)
    rows, th_rows = [], []
    for name, tr, te in sp:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        models = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "feature_lr": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))]),
            "feature_lr_no_metadata": Pipeline([("pre", pre_no), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))]),
        }
        if run_rf:
            models["feature_rf"] = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced"))])
        for mname, model in models.items():
            X = Xf_no if mname.endswith("no_metadata") else Xf
            fit = model.fit(X.iloc[tr], y[tr])
            p = scores(fit, X.iloc[te])
            row = {"split": name, "model": mname, "n_train": len(tr), "n_test": len(te)}
            row.update(metrics(y[te], p))
            rows.append(row)
            for t in [0.5, 0.7, 0.8, 0.9]:
                pred = p >= t
                th_rows.append({
                    "split": name,
                    "model": mname,
                    "threshold": t,
                    "n_retained": int(pred.sum()),
                    "retained_fraction": pred.mean(),
                    "precision_at_threshold": precision_score(y[te], pred, zero_division=0),
                    "recall_at_threshold": recall_score(y[te], pred, zero_division=0),
                })
        bow = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ])
        bow.fit(df.iloc[tr]["pair_text"], y[tr])
        p = scores(bow, df.iloc[te]["pair_text"])
        row = {"split": name, "model": "bow_tfidf_lr", "n_train": len(tr), "n_test": len(te)}
        row.update(metrics(y[te], p))
        rows.append(row)
        tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        Xb_tr, Xb_te = tf.fit_transform(df.iloc[tr]["pair_text"]), tf.transform(df.iloc[te]["pair_text"])
        Xc_tr, Xc_te = pre.fit_transform(Xf.iloc[tr], y[tr]), pre.transform(Xf.iloc[te])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
        clf.fit(sparse.hstack([Xb_tr, Xc_tr]).tocsr(), y[tr])
        p = scores(clf, sparse.hstack([Xb_te, Xc_te]).tocsr())
        row = {"split": name, "model": "hybrid_bow_features_lr", "n_train": len(tr), "n_test": len(te)}
        row.update(metrics(y[te], p))
        rows.append(row)
        for t in [0.5, 0.7, 0.8, 0.9]:
            pred = p >= t
            th_rows.append({
                "split": name,
                "model": "hybrid_bow_features_lr",
                "threshold": t,
                "n_retained": int(pred.sum()),
                "retained_fraction": pred.mean(),
                "precision_at_threshold": precision_score(y[te], pred, zero_division=0),
                "recall_at_threshold": recall_score(y[te], pred, zero_division=0),
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_split.csv", index=False)
    pd.DataFrame(th_rows).to_csv(out_dir / "threshold_metrics.csv", index=False)
    final = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ]).fit(Xf, y)
    joblib.dump(final, out_dir / "final_feature_lr.joblib")
    try:
        imp = pd.DataFrame({
            "feature": final.named_steps["pre"].get_feature_names_out(),
            "coefficient": final.named_steps["clf"].coef_.ravel(),
        })
        imp["abs_coefficient"] = imp["coefficient"].abs()
        imp.sort_values("abs_coefficient", ascending=False).to_csv(out_dir / "feature_lr_coefficients.csv", index=False)
    except Exception as e:
        print(f"[WARN] feature coefficients unavailable: {e}")


def write_report(processed: Path, results: Path):
    audit = pd.read_csv(processed / "dataset_audit.csv")
    met = pd.read_csv(results / "metrics_by_split.csv")
    th = pd.read_csv(results / "threshold_metrics.csv") if (results / "threshold_metrics.csv").exists() else pd.DataFrame()
    lines = [
        "# Meaning-Preservation Classifier MVP Report",
        "",
        "## Dataset audit",
        audit.to_markdown(index=False),
        "",
        "## Metrics",
        met.sort_values(["split", "f1"], ascending=[True, False]).to_markdown(index=False),
        "",
    ]
    avg = met.groupby("model", as_index=False)[["auroc", "auprc", "accuracy", "precision", "recall", "f1"]].mean(numeric_only=True)
    lines += ["## Average across splits", avg.sort_values("f1", ascending=False).to_markdown(index=False), ""]
    if not th.empty:
        lines += ["## Threshold metrics", th.to_markdown(index=False), ""]
    lines += [
        "## Note",
        "Use this classifier as a proxy evaluator for binary human meaning-preservation labels, not as definitive ontology-correctness validation.",
    ]
    (results / "meaning_preservation_classifier_summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--output_dir", default="meaning_preservation/outputs/mvp")
    ap.add_argument("--embedding_model", default=None)
    ap.add_argument(
        "--embedding_backend",
        default="auto",
        choices=["auto", "sentence_transformers", "hf"],
        help="Embedding backend. auto tries sentence-transformers first, then HF transformers mean-pooling fallback.",
    )
    ap.add_argument("--embedding_device", default=None, help="Device for embeddings, e.g. cuda, cuda:0, or cpu.")
    ap.add_argument("--nli_model", default=None)
    ap.add_argument("--nli_device", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_cv_splits", type=int, default=3)
    ap.add_argument("--run_rf", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    processed = out / "processed"
    features = out / "features"
    results = out / "results"
    for d in [processed, features, results]:
        d.mkdir(parents=True, exist_ok=True)
    print("[1/4] Building dataset")
    df = build_dataset(Path(args.roundtrips_csv), processed)
    print(df.shape, df["meaning_preserved"].value_counts().to_dict())
    print("[2/4] Extracting features")
    feats = extract_features(df)
    feats = add_embeddings(
        df,
        feats,
        args.embedding_model,
        args.batch_size,
        backend=args.embedding_backend,
        device=args.embedding_device,
    )
    feats = add_nli(df, feats, args.nli_model, max(1, args.batch_size // 2), args.nli_device)
    feats.to_csv(features / "features.csv", index=False)
    print("[3/4] Running experiments")
    run_experiments(df, feats, results, args.seed, args.group_cv_splits, args.run_rf)
    print("[4/4] Writing report")
    write_report(processed, results)
    print(f"Done: {results / 'meaning_preservation_classifier_summary.md'}")


if __name__ == "__main__":
    main()
