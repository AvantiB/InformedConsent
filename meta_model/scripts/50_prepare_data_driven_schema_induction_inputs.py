#!/usr/bin/env python
"""Prepare fold-specific inputs for data-driven schema induction.

This data-driven arm uses the original annotation dataset only, excluding NA-only
annotation rows. It does not use baseline round-trip outputs, classifier scores,
or reconstructed sentences.

For each held-out fold, the script builds an induction packet from training-fold
annotation evidence:

- source element profiles from original annotations;
- span examples and sentence examples;
- within-sentence co-occurrence edges;
- representative training sentences.

The source dictionaries may be used only to attach static metadata/definitions to
labels that already occur in the original annotation dataset. Held-out forms are
excluded from the packet for their fold. When the same sentence text appears in a
held-out form, matching training rows are excluded for that fold as a conservative
anti-leakage rule.
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
ANNOTATION_COL_CANDIDATES = ["annotations_serialized", "annotations_json", "annotations", "forward_mapping"]
INFO_MODEL_CANDIDATES = ["information_model", "info_model", "source_model"]
LLM_CANDIDATES = ["llm", "model"]
SENTENCE_DECISIONS = ["permit", "deny", "mixed", "unclear"]
NA_VALUES = {"", "na", "n/a", "none", "null", "unknown", "not applicable", "no annotation"}
SENTENCE_LEVEL_LABELS = {
    "odrl::rule_testsentence",
    "rule_testsentence",
    "fhir_consent::consent.provision.type",
    "fhir::consent.provision.type",
    "consent.provision.type",
    "fhir_consent::consent.decision",
    "fhir::consent.decision",
    "consent.decision",
    "duo::duo.decision",
    "ico::ico.decision",
}


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


def source_model_aliases(model: str) -> set[str]:
    m = norm(model)
    return {m, m.replace("_Consent", ""), m.replace("_", "")}


def load_folds(roundtrips: pd.DataFrame, form_col: str, n_folds: int, fold_assignments_csv: str | None) -> pd.DataFrame:
    if fold_assignments_csv:
        folds = pd.read_csv(fold_assignments_csv).fillna("")
        if "form_key" not in folds.columns or "fold_id" not in folds.columns:
            raise ValueError("fold_assignments_csv must contain form_key and fold_id columns")
        folds = folds[["form_key", "fold_id"]].copy()
        folds["form_key"] = folds["form_key"].map(norm)
        folds["fold_id"] = folds["fold_id"].map(norm)
        return folds.drop_duplicates(subset=["form_key"])
    forms = sorted(roundtrips[form_col].map(norm).dropna().unique())
    return pd.DataFrame([{"form_key": f, "fold_id": f"fold_{stable_fold(f, n_folds):02d}"} for f in forms if f])


def load_inventory(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    inv = pd.read_csv(p).fillna("")
    required = {"source_model", "source_element_id", "source_element_label"}
    if not required.issubset(set(inv.columns)):
        return pd.DataFrame()
    if "source_element_definition" not in inv.columns:
        inv["source_element_definition"] = ""
    if "element_scope" not in inv.columns:
        inv["element_scope"] = "span"
    inv = inv[~inv["element_scope"].astype(str).str.casefold().str.contains("sentence", na=False)].copy()
    return inv


def build_inventory_lookup(inv: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    if inv.empty:
        return lookup
    for _, r in inv.iterrows():
        model = norm(r.get("source_model"))
        meta = {
            "source_model": model,
            "source_element_id": norm(r.get("source_element_id")),
            "source_element_label": norm(r.get("source_element_label")),
            "source_element_definition": norm(r.get("source_element_definition")),
        }
        for alias in source_model_aliases(model):
            for key in [meta["source_element_id"], meta["source_element_label"]]:
                if key:
                    lookup[(alias.casefold(), key.casefold())] = meta
    return lookup


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


def is_na_value(x: Any) -> bool:
    s = norm(x).casefold().strip("[](){} .;:,\t\n")
    return s in NA_VALUES


def is_sentence_level_label(source_model: str, label: str) -> bool:
    raw = norm(label).casefold()
    model = norm(source_model).casefold()
    candidates = {raw, f"{model}::{raw}"}
    return bool(candidates & SENTENCE_LEVEL_LABELS)


def parse_annotations(value: Any, source_model: str, inventory_lookup: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, str]]:
    obj = parse_jsonish(value)
    raw_annotations: list[Any]
    if isinstance(obj, dict):
        raw_annotations = obj.get("annotations") or obj.get("valid_annotations") or []
    elif isinstance(obj, list):
        raw_annotations = obj
    else:
        raw_annotations = []

    parsed: list[dict[str, str]] = []
    if raw_annotations:
        for a in raw_annotations:
            if not isinstance(a, dict):
                continue
            span = norm(a.get("span_text") or a.get("evidence_span_text") or a.get("span") or a.get("text") or a.get("value"))
            label = norm(a.get("source_element_label") or a.get("label_name") or a.get("field_name") or a.get("label") or a.get("element_label") or a.get("name"))
            element_id = norm(a.get("source_element_id") or a.get("label_id") or a.get("element_id") or a.get("field_id") or a.get("id"))
            decision = norm(a.get("decision") or a.get("sentence_decision") or a.get("polarity"))
            parsed.append(make_annotation(span, label or element_id, decision, source_model, inventory_lookup))
    else:
        text = norm(value)
        # Compact format examples: span [label] (decision); span [label] (decision)
        pattern = re.compile(r"([^\[\]\n;]+?)\s*\[([^\[\]]+)\]\s*(?:\(([^)]*)\))?")
        for m in pattern.finditer(text):
            span, label, decision = m.group(1), m.group(2), m.group(3) or ""
            parsed.append(make_annotation(span, label, decision, source_model, inventory_lookup))

    out = []
    for a in parsed:
        if not a:
            continue
        if is_na_value(a.get("span_text")) and is_na_value(a.get("source_element_label") or a.get("source_element_id")):
            continue
        if is_na_value(a.get("source_element_label")) or is_sentence_level_label(source_model, a.get("source_element_label", "")):
            continue
        if not norm(a.get("span_text")) or not (norm(a.get("source_element_label")) or norm(a.get("source_element_id"))):
            continue
        out.append(a)
    return out


def make_annotation(span: Any, label: Any, decision: Any, source_model: str, inventory_lookup: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    span_s = norm(span)
    label_s = norm(label)
    source_model_s = norm(source_model)
    meta = None
    for alias in source_model_aliases(source_model_s):
        meta = inventory_lookup.get((alias.casefold(), label_s.casefold()))
        if meta:
            break
    if meta:
        source_model_s = meta["source_model"] or source_model_s
        element_id = meta["source_element_id"]
        element_label = meta["source_element_label"] or label_s
        definition = meta["source_element_definition"]
    else:
        element_id = ""
        element_label = label_s
        definition = ""
    return {
        "span_text": span_s,
        "source_model": source_model_s,
        "source_element_id": element_id,
        "source_element_label": element_label,
        "source_element_definition": definition,
        "decision_or_tag": norm(decision),
    }


def element_key(a: dict[str, str]) -> str:
    ident = a.get("source_element_id") or a.get("source_element_label")
    return f"{a.get('source_model')}::{ident}"


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


def build_fold_payload(df: pd.DataFrame, folds: pd.DataFrame, fold_id: str, max_profiles: int, max_edges: int, max_sentences: int, exclude_heldout_text_overlap: bool) -> dict[str, Any]:
    heldout_forms = set(folds.loc[folds["fold_id"] == fold_id, "form_key"].map(norm))
    heldout_text_keys = set(df.loc[df["form_key"].isin(heldout_forms), "text_key"])
    train = df[~df["form_key"].isin(heldout_forms)].copy()
    if exclude_heldout_text_overlap:
        train = train[~train["text_key"].isin(heldout_text_keys)].copy()

    raw_counts: Counter[str] = Counter()
    sentence_counts: Counter[str] = Counter()
    form_sets: dict[str, set[str]] = defaultdict(set)
    llm_support: dict[str, set[str]] = defaultdict(set)
    source_model_support: dict[str, set[str]] = defaultdict(set)
    span_examples: dict[str, list[str]] = defaultdict(list)
    sentence_examples: dict[str, list[str]] = defaultdict(list)
    decision_examples: dict[str, list[str]] = defaultdict(list)
    profile_base: dict[str, dict[str, Any]] = {}
    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_form_sets: dict[tuple[str, str], set[str]] = defaultdict(set)

    for _, row in train.iterrows():
        anns = row["valid_annotations"]
        keys_this_row = []
        for a in anns:
            k = element_key(a)
            keys_this_row.append(k)
            raw_counts[k] += 1
            form_sets[k].add(row["form_key"])
            llm_support[k].add(row["llm"])
            source_model_support[k].add(a["source_model"])
            span_examples[k].append(a["span_text"])
            sentence_examples[k].append(row["sentence_text"])
            if a.get("decision_or_tag"):
                decision_examples[k].append(a["decision_or_tag"])
            if k not in profile_base:
                profile_base[k] = {
                    "source_element_key": k,
                    "source_model": a["source_model"],
                    "source_element_id": a["source_element_id"],
                    "source_element_label": a["source_element_label"],
                    "source_element_definition": a["source_element_definition"],
                }
        for k in set(keys_this_row):
            sentence_counts[k] += 1
        for a, b in itertools.combinations(sorted(set(keys_this_row)), 2):
            edge = tuple(sorted((a, b)))
            edge_counts[edge] += 1
            edge_form_sets[edge].add(row["form_key"])

    profiles = []
    for k, base in profile_base.items():
        profiles.append({
            **base,
            "raw_mention_count": int(raw_counts[k]),
            "sentence_count": int(sentence_counts[k]),
            "form_count": len(form_sets[k]),
            "source_model_support": sorted(source_model_support[k]),
            "llm_support": sorted(llm_support[k]),
            "example_spans": compact_examples(span_examples[k], 8, 90),
            "example_sentences": compact_examples(sentence_examples[k], 4, 240),
            "decision_or_tag_examples": compact_examples(decision_examples[k], 6, 60),
        })
    profiles.sort(key=lambda r: (r["form_count"], r["sentence_count"], r["raw_mention_count"]), reverse=True)
    profiles = profiles[:max_profiles]
    kept = {p["source_element_key"] for p in profiles}

    edges = []
    for (a, b), c in edge_counts.most_common():
        if a not in kept or b not in kept:
            continue
        edges.append({
            "source_element_key_a": a,
            "source_element_key_b": b,
            "cooccurrence_count": int(c),
            "form_count": len(edge_form_sets[(a, b)]),
        })
        if len(edges) >= max_edges:
            break

    representative_sentences = []
    sent_df = train.drop_duplicates(subset=["form_key", "sentence_text"]).head(max_sentences)
    for _, r in sent_df.iterrows():
        representative_sentences.append({"form_key": r["form_key"], "sentence_text": r["sentence_text"]})

    return {
        "schema_induction_arm": "data_driven_original_annotation_dataset",
        "fold_id": fold_id,
        "heldout_form_keys": sorted(heldout_forms),
        "evidence_policy": {
            "source": "original annotation dataset only",
            "na_only_annotation_rows_excluded": True,
            "baseline_roundtrip_outputs_used": False,
            "classifier_scores_used": False,
            "reconstructions_used": False,
            "training_only": True,
            "heldout_fold_excluded": fold_id,
            "heldout_text_overlap_excluded": bool(exclude_heldout_text_overlap),
            "sentence_decision": {"field": "sentence_decision", "allowed_values": SENTENCE_DECISIONS, "not_a_dictionary_label": True},
            "schema_shape": "flat span-level dictionary with optional model-chosen modifiers",
        },
        "training_evidence_counts": {
            "n_training_rows_after_na_filter": int(len(train)),
            "n_source_element_profiles": len(profiles),
            "n_cooccurrence_edges": len(edges),
        },
        "source_element_profiles": profiles,
        "cooccurrence_edges": edges,
        "representative_training_sentences": representative_sentences,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrips_csv", required=True, help="Original annotation dataset / roundtrips CSV.")
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv", help="Optional source dictionary metadata; only labels present in original annotations are used.")
    ap.add_argument("--fold_assignments_csv", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--max_profiles", type=int, default=140)
    ap.add_argument("--max_edges", type=int, default=180)
    ap.add_argument("--max_sentences", type=int, default=50)
    ap.add_argument("--keep_heldout_text_overlap", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df0 = pd.read_csv(args.roundtrips_csv).fillna("")
    form_col = pick_col(df0, FORM_KEY_CANDIDATES)
    text_col = pick_col(df0, TEXT_COL_CANDIDATES)
    ann_col = pick_col(df0, ANNOTATION_COL_CANDIDATES)
    info_col = pick_col(df0, INFO_MODEL_CANDIDATES, required=False)
    llm_col = pick_col(df0, LLM_CANDIDATES, required=False)

    folds = load_folds(df0, form_col, args.n_folds, args.fold_assignments_csv)
    inv = load_inventory(args.inventory_csv)
    lookup = build_inventory_lookup(inv)

    rows = []
    excluded_na = 0
    for _, r in df0.iterrows():
        form_key = norm(r.get(form_col))
        sentence = norm(r.get(text_col))
        source_model = norm(r.get(info_col)) if info_col else ""
        llm = norm(r.get(llm_col)) if llm_col else ""
        anns = parse_annotations(r.get(ann_col), source_model, lookup)
        if not anns:
            excluded_na += 1
            continue
        rows.append({
            "form_key": form_key,
            "sentence_text": sentence,
            "text_key": text_key(sentence),
            "source_model": source_model,
            "llm": llm,
            "valid_annotations": anns,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid non-NA annotations were parsed from the original dataset.")
    df = df.merge(folds, on="form_key", how="left")

    folds.to_csv(out / "data_driven_fold_assignments.csv", index=False)
    for fold_id in sorted(folds["fold_id"].unique()):
        payload = build_fold_payload(
            df,
            folds,
            fold_id,
            args.max_profiles,
            args.max_edges,
            args.max_sentences,
            exclude_heldout_text_overlap=not args.keep_heldout_text_overlap,
        )
        fold_dir = out / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "data_driven_induction_input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    manifest = {
        "roundtrips_csv": args.roundtrips_csv,
        "inventory_csv": args.inventory_csv,
        "fold_assignments_csv": args.fold_assignments_csv,
        "n_original_rows": int(len(df0)),
        "n_valid_annotation_rows": int(len(df)),
        "n_excluded_na_or_empty_annotation_rows": int(excluded_na),
        "max_profiles": args.max_profiles,
        "max_edges": args.max_edges,
        "max_sentences": args.max_sentences,
        "baseline_roundtrip_outputs_used": False,
        "classifier_scores_used": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote data-driven original-annotation induction inputs under {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
