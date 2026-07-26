#!/usr/bin/env python
"""Build data-driven evidence cards for LLM-assisted schema induction.

Input is the fold output from script 23. Each evidence card represents one
near-equivalence seed cluster and includes only training-fold evidence derived
from human-valid original annotations:

- source-element-sense nodes included in the seed cluster;
- source models, LLMs, forms, and sentences represented;
- top spans, polarity/decision patterns, and example sentences;
- same-span/overlap/nested evidence supporting equivalence;
- complementary/proximity neighbors kept separate from merge evidence;
- unsafe-merge warnings and quality flags.

No manual schema and no baseline round-trip reconstructions are used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def load_json_list(x: Any) -> list[str]:
    s = norm(x)
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [norm(a) for a in v if norm(a)]
        return []
    except Exception:
        return [s]


def load_json_dict(x: Any) -> dict[str, Any]:
    s = norm(x)
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def sample_examples(df: pd.DataFrame, n: int) -> list[dict[str, str]]:
    if df.empty:
        return []
    cols = [
        "form_id", "sentence_id", "sentence_text", "span_text", "source_element_key",
        "source_element_label", "information_model", "llm", "decision_value", "linguistic_polarity",
    ]
    cols = [c for c in cols if c in df.columns]
    out = []
    for _, r in df.drop_duplicates(subset=[c for c in ["sentence_context_id", "span_text"] if c in df.columns]).head(n).iterrows():
        out.append({c: norm(r.get(c)) for c in cols})
    return out


def relation_rows(rel: pd.DataFrame, sense_ids: set[str], rel_types: set[str], max_n: int) -> list[dict[str, Any]]:
    if rel.empty or not {"sense_id_1", "sense_id_2", "relationship_type"}.issubset(rel.columns):
        return []
    mask = (rel["sense_id_1"].astype(str).isin(sense_ids) | rel["sense_id_2"].astype(str).isin(sense_ids)) & rel["relationship_type"].astype(str).isin(rel_types)
    use = rel[mask].copy()
    sort_cols = [c for c in ["relationship_confidence", "positive_preserved_support", "same_span_count", "overlap_count", "proximity_weighted_cooccurrence"] if c in use.columns]
    if sort_cols:
        use = use.sort_values(sort_cols, ascending=False)
    rows = []
    keep_cols = [
        "sense_id_1", "sense_id_2", "relationship_type", "relationship_confidence", "relationship_reason",
        "same_span_count", "overlap_count", "nested_span_count", "same_sentence_cooccurrence",
        "proximity_weighted_cooccurrence", "mean_token_distance", "cross_model_support",
        "cross_llm_support", "span_text_similarity", "embedding_similarity", "pmi", "lift",
        "positive_preserved_support", "negative_failure_support", "example_span_pairs_json",
    ]
    for _, r in use.head(max_n).iterrows():
        item = {c: r.get(c) for c in keep_cols if c in use.columns}
        if "example_span_pairs_json" in item:
            item["example_span_pairs"] = load_json_list(item.pop("example_span_pairs_json"))[:5]
        rows.append(item)
    return rows


def node_rows(nodes: pd.DataFrame, sense_ids: set[str]) -> list[dict[str, Any]]:
    if nodes.empty or "source_element_sense_id" not in nodes.columns:
        return []
    out = []
    for _, r in nodes[nodes["source_element_sense_id"].astype(str).isin(sense_ids)].iterrows():
        out.append({
            "source_element_sense_id": norm(r.get("source_element_sense_id")),
            "source_element_key": norm(r.get("source_element_key")),
            "source_element_label": norm(r.get("source_element_label")),
            "dominant_role_signature": norm(r.get("dominant_role_signature")),
            "n_mentions": int(float(r.get("n_mentions", 0) or 0)),
            "n_forms": int(float(r.get("n_forms", 0) or 0)),
            "n_sentences": int(float(r.get("n_sentences", 0) or 0)),
            "top_spans": load_json_list(r.get("top_spans_json"))[:12],
            "polarity_counts": load_json_dict(r.get("polarity_counts_json")),
            "decision_counts": load_json_dict(r.get("decision_counts_json")),
        })
    return out


def quality_flags(card: dict[str, Any]) -> list[str]:
    flags = []
    if card.get("n_source_models", 0) <= 1:
        flags.append("single_source_model_support")
    if card.get("n_llms", 0) <= 1:
        flags.append("single_llm_support")
    if card.get("n_forms", 0) <= 1:
        flags.append("single_form_support")
    if card.get("unsafe_merge_warnings"):
        flags.append("unsafe_merge_warning_present")
    if not card.get("equivalence_support_edges"):
        flags.append("singleton_or_no_explicit_equivalence_edge")
    if len(card.get("source_element_sense_nodes", [])) > 5:
        flags.append("large_seed_cluster_review_boundary")
    return flags


def build_fold_cards(fold_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    clusters = read_csv(fold_dir / "seed_concept_clusters.csv")
    mentions = read_csv(fold_dir / "source_element_sense_mentions_valid_train.csv")
    nodes = read_csv(fold_dir / "source_element_sense_nodes.csv")
    rel = read_csv(fold_dir / "typed_relationship_edges.csv")
    bundle = read_csv(fold_dir / "provision_bundle_edges.csv")
    unsafe = read_csv(fold_dir / "unsafe_merge_edges.csv")
    if clusters.empty:
        return []

    cards = []
    for _, r in clusters.iterrows():
        seed_id = norm(r.get("seed_cluster_id"))
        senses = set(load_json_list(r.get("sense_ids_json")))
        m = mentions[mentions["source_element_sense_id"].astype(str).isin(senses)].copy() if not mentions.empty and senses else pd.DataFrame()
        source_elements = []
        source_models = set(load_json_list(r.get("source_models_json")))
        llms = set(load_json_list(r.get("llms_json")))
        if not m.empty:
            source_elements = [x for x, _ in Counter(m["source_element_key"].astype(str)).most_common(25)]
            source_models.update(str(x) for x in m.get("information_model", pd.Series(dtype=str)).astype(str).unique() if str(x))
            llms.update(str(x) for x in m.get("llm", pd.Series(dtype=str)).astype(str).unique() if str(x))
        eq_edges = relation_rows(rel, senses, {"near_equivalent", "broader_narrower"}, args.max_edges_per_card)
        comp_edges = relation_rows(bundle if not bundle.empty else rel, senses, {"complementary"}, args.max_edges_per_card)
        unsafe_edges = relation_rows(unsafe if not unsafe.empty else rel, senses, {"unsafe_to_merge"}, args.max_edges_per_card)
        card = {
            "fold_id": fold_dir.name,
            "seed_cluster_id": seed_id,
            "sense_ids": sorted(senses),
            "source_element_sense_nodes": node_rows(nodes, senses),
            "source_elements_included": source_elements,
            "source_models_represented": sorted(source_models),
            "llms_represented": sorted(llms),
            "n_sense_nodes": int(float(r.get("n_sense_nodes", 0) or 0)),
            "n_mentions": int(float(r.get("n_mentions", 0) or 0)),
            "n_forms": int(float(r.get("n_forms", 0) or 0)),
            "n_sentences": int(float(r.get("n_sentences", 0) or 0)),
            "n_source_models": len(source_models),
            "n_llms": len(llms),
            "top_spans": load_json_list(r.get("top_spans_json"))[:args.max_spans_per_card],
            "suggested_terms": load_json_list(r.get("suggested_terms_json"))[:15],
            "source_element_labels": load_json_list(r.get("source_element_labels_json"))[:20],
            "equivalence_support_edges": eq_edges,
            "complementary_or_proximity_neighbors": comp_edges,
            "unsafe_merge_warnings": unsafe_edges,
            "polarity_patterns": dict(Counter(m.get("linguistic_polarity", pd.Series(dtype=str)).astype(str))) if not m.empty else {},
            "decision_value_patterns": dict(Counter(m.get("decision_value", pd.Series(dtype=str)).astype(str))) if not m.empty else {},
            "example_sentences": sample_examples(m, args.example_sentences_per_card),
        }
        card["quality_flags"] = quality_flags(card)
        cards.append(card)
    cards.sort(key=lambda c: (c["n_forms"], c["n_sentences"], c["n_mentions"]), reverse=True)
    return cards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold_root", required=True, help="Root containing fold_*/ outputs from script 23.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--example_sentences_per_card", type=int, default=4)
    ap.add_argument("--max_edges_per_card", type=int, default=12)
    ap.add_argument("--max_spans_per_card", type=int, default=25)
    args = ap.parse_args()

    root = Path(args.fold_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_cards = []
    for fold_dir in sorted(root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        cards = build_fold_cards(fold_dir, args)
        fold_out = out / fold_dir.name
        fold_out.mkdir(parents=True, exist_ok=True)
        with (fold_out / "schema_induction_evidence_cards.jsonl").open("w") as f:
            for c in cards:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (fold_out / "schema_induction_evidence_cards.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False))
        all_cards.extend(cards)
    with (out / "schema_induction_evidence_cards_all_folds.jsonl").open("w") as f:
        for c in all_cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "schema_induction_metadata.json").write_text(json.dumps({
        "n_cards": len(all_cards),
        "fold_root": str(root),
        "method": "Evidence cards are built from valid original annotation evidence, source-element-sense clusters, typed relation edges, and complementarity graphs. No manual schema or baseline reconstructions are used.",
    }, indent=2))
    print(f"Wrote {len(all_cards)} evidence cards to {out}")


if __name__ == "__main__":
    main()
