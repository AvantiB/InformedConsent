#!/usr/bin/env python
"""Retrieve Union V0 candidate source elements for each consent sentence.

This script operationalizes the large Union V0 inventory as a retrievable
controlled vocabulary. It does not reduce the model; it only selects a compact
candidate set from the full union inventory for each sentence.

Use this when the full combined ICO+DUO+FHIR+ODRL dictionary is too large or too
noisy to include directly in every prompt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TEXT_COL_CANDIDATES = [
    "canonical_full_text",
    "full_text_original",
    "original_sentence",
    "full_text",
    "sentence",
    "text",
]
ID_COL_CANDIDATES = ["roundtrip_id", "sentence_id", "id", "source_sentence_id"]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns: {candidates}. Available: {list(df.columns)}")
    return None


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())


def retrieve_candidates(
    inventory: pd.DataFrame,
    sentences: pd.DataFrame,
    text_col: str,
    id_col: str,
    top_k: int,
    min_per_source: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_text = inventory["searchable_text"].fillna("").map(normalize_text).tolist()
    sent_text = sentences[text_col].fillna("").map(normalize_text).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    X_inv = vectorizer.fit_transform(inv_text)
    X_sent = vectorizer.transform(sent_text)

    sim = linear_kernel(X_sent, X_inv)
    rows = []
    for i, sent_row in sentences.reset_index(drop=True).iterrows():
        scores = sim[i]
        ranked = np.argsort(-scores)
        selected: list[int] = []

        # Global top-k lexical candidates.
        for idx in ranked[:top_k]:
            if scores[idx] <= 0 and len(selected) >= top_k:
                break
            selected.append(int(idx))

        # Source-model diversity guard: keep at least min_per_source from each model.
        if min_per_source > 0:
            for source_model, group in inventory.groupby("source_model"):
                group_idx = group.index.to_numpy()
                group_scores = scores[group_idx]
                best = group_idx[np.argsort(-group_scores)[:min_per_source]]
                selected.extend([int(x) for x in best])

        selected = list(dict.fromkeys(selected))
        selected = sorted(selected, key=lambda x: scores[x], reverse=True)

        for rank, idx in enumerate(selected, start=1):
            inv = inventory.iloc[idx]
            rows.append({
                "sentence_row_index": i,
                "sentence_id": sent_row[id_col],
                "sentence_text": sent_row[text_col],
                "rank": rank,
                "retrieval_method": "tfidf_union_v0",
                "retrieval_score": float(scores[idx]),
                "union_element_id": inv["union_element_id"],
                "source_model": inv["source_model"],
                "source_element_id": inv["source_element_id"],
                "source_element_label": inv["source_element_label"],
                "source_element_definition": inv.get("source_element_definition", ""),
            })

    candidates = pd.DataFrame(rows)
    summary = candidates.groupby("sentence_id", as_index=False).agg(
        n_candidates=("union_element_id", "nunique"),
        max_score=("retrieval_score", "max"),
        mean_score=("retrieval_score", "mean"),
        source_models=("source_model", lambda x: ",".join(sorted(set(map(str, x))))),
    )
    return candidates, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_csv", required=True)
    ap.add_argument("--sentences_csv", required=True)
    ap.add_argument("--output_csv", default="meta_model/v0_union/sentence_candidate_elements.csv")
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--min_per_source", type=int, default=2)
    ap.add_argument("--max_sentences", type=int, default=None, help="Optional debug limit.")
    args = ap.parse_args()

    inventory = pd.read_csv(args.inventory_csv)
    sentences = pd.read_csv(args.sentences_csv)
    if args.max_sentences:
        sentences = sentences.head(args.max_sentences).copy()

    text_col = args.text_col or pick_col(sentences, TEXT_COL_CANDIDATES)
    id_col = args.id_col or pick_col(sentences, ID_COL_CANDIDATES)

    candidates, summary = retrieve_candidates(
        inventory=inventory,
        sentences=sentences,
        text_col=text_col,
        id_col=id_col,
        top_k=args.top_k,
        min_per_source=args.min_per_source,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_csv, index=False)

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_csv.parent / "retrieval_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote {len(candidates):,} candidate rows to {output_csv}")
    print(f"Wrote retrieval summary to {summary_csv}")
    print(f"Text column: {text_col}; ID column: {id_col}")


if __name__ == "__main__":
    main()
