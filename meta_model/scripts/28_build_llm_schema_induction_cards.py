#!/usr/bin/env python
"""Build evidence cards for LLM-assisted functional schema induction.

The cards are the only input to the LLM inducer. They summarize the data-driven
seed evidence from form-level CV: selected stable clusters, source models,
evidence spans, selected source elements, typed near-equivalence/complementarity,
and optional source-model-to-functional crosswalk hints.

The manual Functional V1 schema is intentionally not used here, so the induced
schema arm can be treated as an independently induced condition.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def load_json_list(x: Any) -> list[str]:
    s = norm(x)
    if not s:
        return []
    try:
        v = json.loads(s)
        return [norm(a) for a in v if norm(a)] if isinstance(v, list) else []
    except Exception:
        return [s]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def sample_rows(df: pd.DataFrame, n: int) -> list[dict[str, str]]:
    rows = []
    if df.empty:
        return rows
    cols = [c for c in ["sentence_text", "span_text", "union_element_id", "source_element_label", "information_model", "meaning_preserved"] if c in df.columns]
    for _, r in df.head(n).iterrows():
        rows.append({c: norm(r.get(c)) for c in cols})
    return rows


def build_cards(args: argparse.Namespace) -> None:
    root = Path(args.fold_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = read_csv(Path(args.selected_fields_csv))
    if selection.empty:
        raise SystemExit(f"No selected fields found: {args.selected_fields_csv}")

    crosswalk = read_csv(Path(args.crosswalk_csv)) if args.crosswalk_csv else pd.DataFrame()
    cw_by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not crosswalk.empty and "union_element_id" in crosswalk.columns:
        for _, r in crosswalk.iterrows():
            cw_by_uid[norm(r.get("union_element_id"))].append({
                "candidate_v1_field": norm(r.get("target_v1_field")),
                "mapping_type": norm(r.get("mapping_type")),
                "rationale": norm(r.get("rationale")),
            })

    all_cards = []
    for fold, fsel in selection.groupby("fold", dropna=False):
        fold_dir = root / str(fold)
        mentions = read_csv(fold_dir / "source_element_sense_mentions_train.csv")
        rel = read_csv(fold_dir / "typed_relationship_edges.csv")
        bundle = read_csv(fold_dir / "provision_bundle_edges.csv")
        fold_cards = []
        for _, r in fsel.iterrows():
            if norm(r.get("selection_tier")) not in {"core_candidate", "extension_candidate"}:
                continue
            cid = norm(r.get("candidate_field_id"))
            sense_ids = set(load_json_list(r.get("sense_ids_json")))
            if not sense_ids and "source_element_sense_id" in mentions.columns:
                # fallback: selected file may not carry sense_ids_json; use candidate id only as identifier.
                sense_ids = set()
            m = mentions[mentions.get("source_element_sense_id", pd.Series(dtype=str)).astype(str).isin(sense_ids)].copy() if sense_ids else pd.DataFrame()
            source_elements = []
            spans = []
            source_models = []
            examples = []
            if not m.empty:
                source_elements = [x for x, _ in Counter(m["union_element_id"].astype(str)).most_common(20)] if "union_element_id" in m.columns else []
                spans = [x for x, _ in Counter(m["span_text"].astype(str)).most_common(25)] if "span_text" in m.columns else []
                source_models = sorted(set(m["information_model"].astype(str))) if "information_model" in m.columns else []
                examples = sample_rows(m.drop_duplicates(subset=["sentence_text"]) if "sentence_text" in m.columns else m, int(args.example_sentences_per_card))
            if not spans:
                spans = load_json_list(r.get("top_spans_json"))[:25]
            if not source_models:
                source_models = load_json_list(r.get("source_models_json"))
            terms = load_json_list(r.get("suggested_terms_json"))[:15]
            cw = []
            for uid in source_elements[:10]:
                for item in cw_by_uid.get(uid, [])[:3]:
                    if item not in cw:
                        cw.append(item)
            related_edges = []
            if not rel.empty and {"source", "target", "relationship_type"}.issubset(rel.columns) and sense_ids:
                mask = rel["source"].astype(str).isin(sense_ids) | rel["target"].astype(str).isin(sense_ids)
                for _, e in rel[mask].head(int(args.max_edges_per_card)).iterrows():
                    related_edges.append({
                        "source": norm(e.get("source")),
                        "target": norm(e.get("target")),
                        "relationship_type": norm(e.get("relationship_type")),
                        "weight": norm(e.get("weight")),
                    })
            complementary = []
            if not bundle.empty and sense_ids:
                possible_cols = [c for c in ["source", "target", "source_sense_id", "target_sense_id"] if c in bundle.columns]
                if len(possible_cols) >= 2:
                    a, b = possible_cols[:2]
                    mask = bundle[a].astype(str).isin(sense_ids) | bundle[b].astype(str).isin(sense_ids)
                    for _, e in bundle[mask].head(int(args.max_edges_per_card)).iterrows():
                        complementary.append({c: norm(e.get(c)) for c in bundle.columns[:8]})
            card = {
                "fold": str(fold),
                "candidate_field_id": cid,
                "selection_tier": norm(r.get("selection_tier")),
                "stability_group_id": norm(r.get("stability_group_id")),
                "stability_n_folds": int(float(r.get("stability_n_folds", 0) or 0)),
                "n_positive_mentions": int(float(r.get("n_positive_mentions", 0) or 0)),
                "n_mentions": int(float(r.get("n_mentions", 0) or 0)),
                "n_source_models": int(float(r.get("n_source_models", 0) or 0)),
                "suggested_terms": terms,
                "top_spans": spans,
                "source_models": source_models,
                "top_source_elements": source_elements[:20],
                "crosswalk_hints_from_source_elements": cw[:15],
                "near_equivalence_or_related_edges": related_edges,
                "complementary_cooccurrence_edges": complementary,
                "example_sentences": examples,
            }
            fold_cards.append(card)
            all_cards.append(card)
        fold_out = out / str(fold)
        fold_out.mkdir(parents=True, exist_ok=True)
        with (fold_out / "schema_induction_evidence_cards.jsonl").open("w") as f:
            for c in fold_cards:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (fold_out / "schema_induction_evidence_cards.json").write_text(json.dumps(fold_cards, indent=2, ensure_ascii=False))
    with (out / "schema_induction_evidence_cards_all_folds.jsonl").open("w") as f:
        for c in all_cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "schema_induction_metadata.json").write_text(json.dumps({
        "n_cards": len(all_cards),
        "fold_root": str(root),
        "selected_fields_csv": args.selected_fields_csv,
        "crosswalk_csv": args.crosswalk_csv,
        "note": "These evidence cards seed LLM-assisted schema induction; the manual schema is not used as input.",
    }, indent=2))
    print(f"Wrote {len(all_cards)} evidence cards to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold_root", required=True)
    ap.add_argument("--selected_fields_csv", required=True)
    ap.add_argument("--crosswalk_csv", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--example_sentences_per_card", type=int, default=3)
    ap.add_argument("--max_edges_per_card", type=int, default=12)
    args = ap.parse_args()
    build_cards(args)


if __name__ == "__main__":
    main()
