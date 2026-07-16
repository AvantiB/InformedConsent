#!/usr/bin/env python
"""Cluster phrase/source-node evidence units into candidate meta-model units.

The clustering is data/language-driven: evidence-unit text is embedded, then grouped
without manually predefining the reduced schema. Cluster summaries include source-
model coverage, cue-group composition, and preservation behavior.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def norm(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def resolve_hf_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    if model_name == "all-MiniLM-L6-v2":
        return "sentence-transformers/all-MiniLM-L6-v2"
    return model_name


def embed_with_sentence_transformers(texts: list[str], model_name: str, device: str | None, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(resolve_hf_name(model_name), device=device)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def embed_with_hf(texts: list[str], model_name: str, device: str | None, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved = resolve_hf_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModel.from_pretrained(resolved)
    if device:
        model = model.to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            if device:
                enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


def embed_with_tfidf_svd(texts: list[str], n_components: int = 128) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD | None]:
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    if min(X.shape) <= 2:
        return normalize(X).toarray(), vec, None
    n_components = min(n_components, max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    return normalize(Z), vec, svd


def get_embeddings(texts: list[str], model_name: str | None, backend: str, device: str | None, batch_size: int):
    if model_name:
        if backend in {"auto", "sentence_transformers"}:
            try:
                return embed_with_sentence_transformers(texts, model_name, device, batch_size), "sentence_transformers"
            except Exception as e:
                if backend == "sentence_transformers":
                    raise
                print(f"[WARN] sentence-transformers embedding failed, falling back to HF: {e}")
        if backend in {"auto", "hf"}:
            try:
                return embed_with_hf(texts, model_name, device, batch_size), "hf"
            except Exception as e:
                if backend == "hf":
                    raise
                print(f"[WARN] HF embedding failed, falling back to TF-IDF/SVD: {e}")
    Z, _, _ = embed_with_tfidf_svd(texts)
    return Z, "tfidf_svd"


def cluster_embeddings(Z: np.ndarray, method: str, distance_threshold: float, n_clusters: int | None, seed: int) -> np.ndarray:
    if method == "kmeans":
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(len(Z))))
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(Z)

    kwargs = {"linkage": "average"}
    # sklearn changed affinity->metric; support both.
    try:
        if n_clusters is None:
            return AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", **kwargs).fit_predict(Z)
        return AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", **kwargs).fit_predict(Z)
    except TypeError:
        if n_clusters is None:
            return AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity="cosine", **kwargs).fit_predict(Z)
        return AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", **kwargs).fit_predict(Z)


def top_terms_for_cluster(texts: list[str], labels: np.ndarray, cluster_id: int, n: int = 12) -> str:
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return ""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    try:
        X = vec.fit_transform([texts[i] for i in idx])
    except ValueError:
        return ""
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1][:n]
    return "; ".join(terms[order])


def split_cues(series: pd.Series) -> list[str]:
    vals = []
    for x in series.dropna().astype(str):
        vals.extend([v for v in x.split(";") if v])
    return vals


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray, Z: np.ndarray, out_dir: Path) -> None:
    df = df.copy()
    df["cluster_id"] = labels
    texts = df["unit_text_for_embedding"].fillna("").astype(str).tolist()
    df.to_csv(out_dir / "cluster_assignments.csv", index=False)

    rows = []
    for cid, g in df.groupby("cluster_id"):
        idx = g.index.to_numpy()
        centroid = Z[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Z[idx], centroid).ravel()
        exemplar_i = idx[int(np.argmax(sims))]
        cue_counts = Counter(split_cues(g.get("cue_groups", pd.Series(dtype=str))))
        model_counts = g["information_model"].value_counts().head(10).to_dict() if "information_model" in g else {}
        label_counts = g["source_element_label"].dropna().astype(str).value_counts().head(10).to_dict()
        rows.append({
            "cluster_id": int(cid),
            "n_evidence_units": int(len(g)),
            "n_roundtrips": int(g["roundtrip_id"].nunique()) if "roundtrip_id" in g else 0,
            "n_sentences": int(g["sentence_id"].nunique()) if "sentence_id" in g else 0,
            "preserved_rate": float(pd.to_numeric(g["meaning_preserved"], errors="coerce").mean()) if "meaning_preserved" in g else np.nan,
            "top_terms": top_terms_for_cluster(texts, labels, cid),
            "exemplar_text": texts[exemplar_i],
            "top_cue_groups": json.dumps(dict(cue_counts.most_common(10))),
            "source_model_counts": json.dumps(model_counts),
            "top_source_labels": json.dumps(label_counts),
        })
    pd.DataFrame(rows).sort_values("n_evidence_units", ascending=False).to_csv(out_dir / "cluster_summary.csv", index=False)

    coverage = (
        df.groupby(["cluster_id", "information_model"], dropna=False)
        .agg(n_evidence_units=("evidence_unit_id", "count"), n_sentences=("sentence_id", "nunique"))
        .reset_index()
        .sort_values(["cluster_id", "n_evidence_units"], ascending=[True, False])
    )
    coverage.to_csv(out_dir / "cluster_source_model_coverage.csv", index=False)

    pair_counts = Counter()
    for _, g in df.groupby("roundtrip_id"):
        cids = sorted(set(g["cluster_id"].astype(int)))
        for a_i, a in enumerate(cids):
            for b in cids[a_i + 1 :]:
                pair_counts[(a, b)] += 1
    pd.DataFrame([
        {"cluster_a": a, "cluster_b": b, "n_roundtrips": n}
        for (a, b), n in pair_counts.items()
    ]).sort_values("n_roundtrips", ascending=False).to_csv(out_dir / "cluster_pair_cooccurrence.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence_units_csv", required=True)
    ap.add_argument("--output_dir", default="meta_model/outputs/clusters")
    ap.add_argument("--embedding_model", default=None, help="Optional embedding model, e.g. all-MiniLM-L6-v2")
    ap.add_argument("--embedding_backend", default="auto", choices=["auto", "sentence_transformers", "hf", "tfidf_svd"])
    ap.add_argument("--embedding_device", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--cluster_method", default="agglomerative", choices=["agglomerative", "kmeans"])
    ap.add_argument("--distance_threshold", type=float, default=0.35)
    ap.add_argument("--n_clusters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.evidence_units_csv)
    if "unit_text_for_embedding" not in df.columns:
        raise ValueError("evidence_units_csv must contain unit_text_for_embedding")
    texts = df["unit_text_for_embedding"].fillna("").astype(str).tolist()

    backend = args.embedding_backend
    model = args.embedding_model
    if backend == "tfidf_svd":
        model = None
    Z, actual_backend = get_embeddings(texts, model, backend, args.embedding_device, args.batch_size)
    labels = cluster_embeddings(Z, args.cluster_method, args.distance_threshold, args.n_clusters, args.seed)

    summarize_clusters(df, labels, Z, out_dir)
    np.save(out_dir / "evidence_unit_embeddings.npy", Z)
    (out_dir / "clustering_metadata.json").write_text(json.dumps({
        "n_evidence_units": int(len(df)),
        "n_clusters": int(len(set(labels.tolist()))),
        "embedding_backend": actual_backend,
        "embedding_model": resolve_hf_name(args.embedding_model) if args.embedding_model else "tfidf_svd",
        "cluster_method": args.cluster_method,
        "distance_threshold": args.distance_threshold,
        "n_clusters_requested": args.n_clusters,
    }, indent=2))

    print(f"Wrote cluster assignments for {len(df):,} evidence units to {out_dir}")
    print(f"Induced {len(set(labels.tolist())):,} clusters")


if __name__ == "__main__":
    main()
