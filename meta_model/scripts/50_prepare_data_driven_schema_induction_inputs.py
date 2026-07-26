#!/usr/bin/env python
"""Prepare fold-specific data-driven schema induction inputs.

This script builds the data-driven LLM schema-induction packets from corrected
round-trip outputs. For each held-out fold, it uses only training-fold evidence
from baseline source-model/Union runs.

Expected input is the scored CSV produced by:
  49_compile_schema_strategy_roundtrips.py -> standardized_roundtrips.csv
  09_score_roundtrip_outputs.py -> scored_roundtrips.csv

The script joins rows back to the original roundtrips CSV by sentence text to
recover form_key/fold membership, then excludes any evidence from held-out forms.
If a sentence appears in both training and held-out forms, it is excluded for
that held-out fold to avoid text-level leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COL_CANDIDATES = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
FORM_KEY_CANDIDATES = ["form_key", "form_id", "document_id", "source_form", "file_name"]
DEFAULT_CONDITIONS = ["individual_source_model_json", "union_v0_full_dictionary"]
SENTENCE_DECISIONS = ["permit", "deny", "mixed", "unclear"]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def text_key(x: Any) -> str:
    return norm(x).casefold()


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available columns={list(df.columns)}")
    return None


def stable_fold(key: str, n_folds: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_folds


def parse_jsonish(text: Any) -> Any:
    t = norm(text)
    if not t:
        return None
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|yaml|csv)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    for l, r in [("{", "}"), ("[", "]")]:
        a, b = t.find(l), t.rfind(r)
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except Exception:
                pass
    return None


def source_text_col(df: pd.DataFrame) -> str:
    for c in ["original_text", "source_text", "original_sentence", "sentence_text"]:
        if c in df.columns:
            return c
    return pick_col(df, TEXT_COL_CANDIDATES) or ""


def load_form_sentence_map(roundtrips_csv: Path, n_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(roundtrips_csv).fillna("")
    form_col = pick_col(df, FORM_KEY_CANDIDATES)
    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    m = df[[form_col, text_col]].copy()
    m.columns = ["form_key", "sentence_text"]
    m["form_key"] = m["form_key"].map(norm)
    m["sentence_text"] = m["sentence_text"].map(norm)
    m["text_key"] = m["sentence_text"].map(text_key)
    m = m[(m["form_key"] != "") & (m["text_key"] != "")].drop_duplicates(subset=["form_key", "text_key"])
    forms = sorted(m["form_key"].unique())
    folds = pd.DataFrame([{"form_key": f, "fold_id": f"fold_{stable_fold(f, n_folds):02d}"} for f in forms])
    return m.merge(folds, on="form_key", how="left"), folds


def attach_form_keys(scored: pd.DataFrame, form_sentence_map: pd.DataFrame) -> pd.DataFrame:
    text_col = source_text_col(scored)
    out = scored.copy()
    out["_source_text"] = out[text_col].map(norm)
    out["text_key"] = out["_source_text"].map(text_key)
    # Aggregate possible form keys by exact text. If any heldout form has the same text,
    # that row is excluded for that fold to avoid sentence-text leakage.
    grouped = form_sentence_map.groupby("text_key").agg(
        form_keys=("form_key", lambda x: sorted(set(map(str, x)))),
        fold_ids=("fold_id", lambda x: sorted(set(map(str, x)))),
    ).reset_index()
    out = out.merge(grouped, on="text_key", how="left")
    out["form_keys"] = out["form_keys"].apply(lambda x: x if isinstance(x, list) else [])
    out["fold_ids"] = out["fold_ids"].apply(lambda x: x if isinstance(x, list) else [])
    return out


def annotation_list(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        anns = obj.get("annotations") or obj.get("valid_annotations") or []
    elif isinstance(obj, list):
        anns = obj
    else:
        anns = []
    return [a for a in anns if isinstance(a, dict)]


def extract_ann_fields(a: dict[str, Any], default_info_model: str) -> dict[str, str]:
    label_id = norm(
        a.get("union_element_id")
        or a.get("label_id")
        or a.get("source_element_id")
        or a.get("field_id")
        or a.get("element_id")
        or a.get("id")
    )
    label_name = norm(
        a.get("source_element_label")
        or a.get("label_name")
        or a.get("field_name")
        or a.get("label")
        or a.get("element_label")
        or a.get("name")
    )
    label_definition = norm(a.get("label_definition") or a.get("source_element_definition") or a.get("field_definition") or a.get("definition"))
    span = norm(a.get("span_text") or a.get("evidence_span_text") or a.get("span") or a.get("text") or a.get("value"))
    source_model = norm(a.get("source_model") or default_info_model)
    return {
        "label_id": label_id,
        "label_name": label_name,
        "label_definition": label_definition,
        "span_text": span,
        "source_model": source_model,
    }


def label_key(x: dict[str, str]) -> str:
    sid = x["label_id"] or x["label_name"]
    return f"{x['source_model']}::{sid}"


def compact_examples(values: list[str], max_n: int, max_chars: int) -> list[str]:
    out = []
    seen = set()
    for v in values:
        s = norm(v)
        if not s or s.casefold() in seen:
            continue
        seen.add(s.casefold())
        if len(s) > max_chars:
            s = s[: max_chars - 1].rstrip() + "…"
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def build_fold_input(
    scored: pd.DataFrame,
    folds: pd.DataFrame,
    fold_id: str,
    include_conditions: set[str],
    min_score_for_supported: float,
    max_profiles: int,
    max_edges: int,
    max_examples: int,
) -> dict[str, Any]:
    heldout_forms = set(folds.loc[folds["fold_id"] == fold_id, "form_key"].map(norm))
    train = scored[scored["condition"].isin(include_conditions)].copy()
    # Training-fold only. Exclude rows with the heldout fold among possible text matches.
    train = train[~train["fold_ids"].apply(lambda xs: fold_id in set(xs or []))].copy()

    profile = {}
    span_examples: dict[str, list[str]] = defaultdict(list)
    sentence_examples: dict[str, list[str]] = defaultdict(list)
    model_support: dict[str, set[str]] = defaultdict(set)
    llm_support: dict[str, set[str]] = defaultdict(set)
    score_values: dict[str, list[float]] = defaultdict(list)
    supported_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_support_sentences: dict[tuple[str, str], set[str]] = defaultdict(set)

    row_examples_high = []
    row_examples_low = []

    for _, row in train.iterrows():
        obj = parse_jsonish(row.get("forward_mapping"))
        anns_raw = annotation_list(obj)
        keys_for_row = []
        score = row.get("classifier_preservation_score", None)
        try:
            score_f = float(score)
        except Exception:
            score_f = None
        info_model = norm(row.get("information_model"))
        llm = norm(row.get("llm"))
        source_text = norm(row.get("original_text") or row.get("source_text") or row.get("_source_text"))
        reconstructed = norm(row.get("reconstructed_text") or row.get("reconstructed_sentence"))
        source_id = norm(row.get("source_id") or row.get("roundtrip_id"))

        for ann in anns_raw:
            x = extract_ann_fields(ann, info_model)
            if not (x["label_id"] or x["label_name"]) or not x["span_text"]:
                continue
            k = label_key(x)
            keys_for_row.append(k)
            raw_counts[k] += 1
            model_support[k].add(x["source_model"] or info_model)
            llm_support[k].add(llm)
            if score_f is not None:
                score_values[k].append(score_f)
                if score_f >= min_score_for_supported:
                    supported_counts[k] += 1
            span_examples[k].append(x["span_text"])
            sentence_examples[k].append(source_text)
            if k not in profile:
                profile[k] = {
                    "source_element_key": k,
                    "source_model": x["source_model"] or info_model,
                    "source_element_id": x["label_id"],
                    "source_element_label": x["label_name"],
                    "source_element_definition": x["label_definition"],
                }

        uniq = sorted(set(keys_for_row))
        for a, b in itertools.combinations(uniq, 2):
            edge = tuple(sorted((a, b)))
            edge_counts[edge] += 1
            if source_id:
                edge_support_sentences[edge].add(source_id)

        if score_f is not None and source_text:
            rec = {"source_text": source_text[:350], "reconstructed_text": reconstructed[:350], "score": score_f, "condition": norm(row.get("condition")), "llm": llm}
            if score_f >= min_score_for_supported and len(row_examples_high) < max_examples:
                row_examples_high.append(rec)
            elif score_f < 0.5 and len(row_examples_low) < max_examples:
                row_examples_low.append(rec)

    profile_rows = []
    for k, base in profile.items():
        scores = score_values.get(k, [])
        profile_rows.append({
            **base,
            "raw_mention_count": int(raw_counts[k]),
            "supported_mention_count": int(supported_counts[k]),
            "mean_preservation_score": sum(scores) / len(scores) if scores else None,
            "source_model_support": sorted(model_support[k]),
            "llm_support": sorted(llm_support[k]),
            "example_spans": compact_examples(span_examples[k], 6, 80),
            "example_sentences": compact_examples(sentence_examples[k], 3, 220),
        })
    profile_rows.sort(key=lambda r: (r["supported_mention_count"], r["raw_mention_count"]), reverse=True)
    profile_rows = profile_rows[:max_profiles]
    kept = {r["source_element_key"] for r in profile_rows}

    edge_rows = []
    for (a, b), c in edge_counts.most_common():
        if a not in kept or b not in kept:
            continue
        edge_rows.append({
            "source_element_key_a": a,
            "source_element_key_b": b,
            "cooccurrence_count": int(c),
            "support_sentence_count": len(edge_support_sentences.get((a, b), set())),
        })
        if len(edge_rows) >= max_edges:
            break

    return {
        "schema_induction_arm": "data_driven_llm_training_fold_evidence",
        "fold_id": fold_id,
        "heldout_form_keys": sorted(heldout_forms),
        "evidence_policy": {
            "training_only": True,
            "heldout_fold_excluded": fold_id,
            "included_conditions": sorted(include_conditions),
            "min_score_for_supported": min_score_for_supported,
            "sentence_decision": {"field": "sentence_decision", "allowed_values": SENTENCE_DECISIONS, "not_a_dictionary_label": True},
            "schema_shape": "flat span-level dictionary with optional model-chosen modifiers",
        },
        "training_evidence_counts": {
            "n_training_roundtrip_rows": int(len(train)),
            "n_source_element_profiles": len(profile_rows),
            "n_cooccurrence_edges": len(edge_rows),
        },
        "source_element_profiles": profile_rows,
        "cooccurrence_edges": edge_rows,
        "high_preservation_examples": row_examples_high,
        "low_preservation_examples": row_examples_low,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored_roundtrips_csv", required=True)
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--fold_assignments_csv", default=None, help="Optional existing form_key/fold_id CSV. If omitted, folds are reconstructed by hash.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--include_conditions", default=",".join(DEFAULT_CONDITIONS))
    ap.add_argument("--min_score_for_supported", type=float, default=0.7)
    ap.add_argument("--max_profiles", type=int, default=120)
    ap.add_argument("--max_edges", type=int, default=160)
    ap.add_argument("--max_examples", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    form_map, folds_hash = load_form_sentence_map(Path(args.roundtrips_csv), args.n_folds)
    if args.fold_assignments_csv:
        folds = pd.read_csv(args.fold_assignments_csv).fillna("")
        if "form_key" not in folds.columns or "fold_id" not in folds.columns:
            raise ValueError("fold_assignments_csv must contain form_key and fold_id")
        folds = folds[["form_key", "fold_id"]].copy()
        folds["form_key"] = folds["form_key"].map(norm)
        folds["fold_id"] = folds["fold_id"].map(norm)
        form_map = form_map.drop(columns=["fold_id"], errors="ignore").merge(folds, on="form_key", how="left")
    else:
        folds = folds_hash

    scored = pd.read_csv(args.scored_roundtrips_csv).fillna("")
    scored = attach_form_keys(scored, form_map)
    include_conditions = {x.strip() for x in args.include_conditions.split(",") if x.strip()}

    folds.to_csv(out / "data_driven_fold_assignments.csv", index=False)
    for fold_id in sorted(folds["fold_id"].unique()):
        payload = build_fold_input(
            scored,
            folds,
            fold_id,
            include_conditions,
            args.min_score_for_supported,
            args.max_profiles,
            args.max_edges,
            args.max_examples,
        )
        fold_dir = out / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "data_driven_induction_input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    manifest = {
        "scored_roundtrips_csv": args.scored_roundtrips_csv,
        "roundtrips_csv": args.roundtrips_csv,
        "fold_assignments_csv": args.fold_assignments_csv,
        "include_conditions": sorted(include_conditions),
        "min_score_for_supported": args.min_score_for_supported,
        "max_profiles": args.max_profiles,
        "max_edges": args.max_edges,
        "max_examples": args.max_examples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote data-driven induction inputs under {out}")


if __name__ == "__main__":
    main()
