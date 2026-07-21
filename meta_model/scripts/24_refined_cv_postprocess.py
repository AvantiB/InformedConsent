#!/usr/bin/env python
"""Post-process refined CV outputs for form matching and field selection.

This helper keeps the main induction script conservative while adding two
paper-facing safeguards:

1. repair-fold-assignments: add explicit form-ID aliases when the main
   roundtrips.csv and expert workbooks use different punctuation for the same
   consent form, e.g. ``Alzheimer_s`` versus ``Alzheimer's``.
2. select-fields: reduce raw fold candidate fields into selected fold-specific
   schemas using cross-fold term-profile stability, positive support, and
   source-model support. This does not assign final human-readable names.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

TEXT_COLS = ["canonical_full_text", "full_text_original", "full_text", "sentence_text", "sentence", "text"]
FORM_COLS = ["form_key", "form_id", "source_file", "source_file_original", "source_id", "input_workbook"]
STOP = set(
    "the a an and or of to in for with without by from on at as is are be been this that it its your you we our may can will shall my i they their them us all about into if then than".split()
)
ARTIFACT_STOP = set(
    "permit deny permission prohibition obligation mixed unclear ico fhir duo odrl provision provisions directive role actor action data consent rule".split()
)


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates}; available={list(df.columns)}")
    return None


def strip_workflow_suffixes(v: str) -> str:
    v = re.sub(r"\.(txt|csv|xlsx?)$", "", v, flags=re.I)
    v = re.sub(r"_annotated$", "", v, flags=re.I)
    v = re.sub(r"_output$", "", v, flags=re.I)
    v = re.sub(r"\s+annotated$", "", v, flags=re.I)
    v = re.sub(r"\s+output$", "", v, flags=re.I)
    v = re.sub(r"\s+copy(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    v = re.sub(r"_copy(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    return re.sub(r"\s+", " ", v).strip(" _-")


def display_form_value(raw: Any) -> str:
    """Mimic current induction display IDs without losing apostrophes."""
    v = norm(raw)
    if not v:
        return ""
    v = strip_workflow_suffixes(v)
    if not v or v.lower() in {"nan", "none", "null"}:
        return ""
    if v.startswith("FORM_") and "e3b0c442" in v:
        return ""
    return v


def form_value_from_row(row: pd.Series) -> str:
    for c in FORM_COLS:
        if c in row.index and norm(row.get(c)):
            v = display_form_value(row.get(c))
            if v:
                return v
    return ""


def form_match_key(raw: Any) -> str:
    """Punctuation-insensitive key for alias matching.

    ``Alzheimer_s Disease`` and ``Alzheimer's Disease`` both become
    ``alzheimer s disease``. This key is used only for crosswalk/audit; the
    original display IDs are preserved in fold assignment outputs.
    """
    v = display_form_value(raw).lower()
    v = v.replace("’", "'").replace("‘", "'").replace("`", "'")
    v = re.sub(r"[^a-z0-9]+", " ", v)
    return " ".join(v.split())


def expert_forms(expert_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(expert_csv).fillna("")
    text_col = pick_col(df, TEXT_COLS, required=True)
    rows = []
    seen = set()
    for i, r in df.iterrows():
        text = norm(r.get(text_col))
        form = form_value_from_row(r)
        if not text or not form or form in seen:
            continue
        seen.add(form)
        rows.append({"expert_form_id": form, "expert_form_match_key": form_match_key(form), "first_expert_source_row": int(i) + 2})
    return pd.DataFrame(rows)


def repair_fold_assignments(args: argparse.Namespace) -> None:
    folds = pd.read_csv(args.fold_assignments_csv).fillna("")
    fold_col = "canonical_form_id" if "canonical_form_id" in folds.columns else "form_id"
    exp = expert_forms(Path(args.expert_roundtrips_csv))

    out_rows: list[dict[str, Any]] = []
    for _, r in folds.iterrows():
        d = r.to_dict()
        d.setdefault("canonical_form_id", norm(r.get(fold_col)))
        d.setdefault("form_id", norm(r.get(fold_col)))
        d["assignment_source"] = "original"
        d["alias_for_canonical_form_id"] = ""
        d["form_match_key"] = form_match_key(d.get("canonical_form_id") or d.get("form_id"))
        out_rows.append(d)

    key_to_fold_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_ids = {norm(r.get("canonical_form_id") or r.get("form_id")) for r in out_rows}
    for r in out_rows:
        key_to_fold_rows[r["form_match_key"]].append(r)

    audit_rows = []
    for _, e in exp.iterrows():
        eform = norm(e["expert_form_id"])
        ekey = norm(e["expert_form_match_key"])
        if eform in exact_ids:
            audit_rows.append({**e.to_dict(), "match_status": "exact", "matched_canonical_form_id": eform, "fold_id": ""})
            continue
        matches = key_to_fold_rows.get(ekey, [])
        if len(matches) == 1:
            m = matches[0]
            alias = dict(m)
            alias["form_id"] = eform
            alias["canonical_form_id"] = eform
            alias["assignment_source"] = "expert_alias"
            alias["alias_for_canonical_form_id"] = m.get("canonical_form_id") or m.get("form_id")
            alias["form_match_key"] = ekey
            out_rows.append(alias)
            exact_ids.add(eform)
            audit_rows.append({**e.to_dict(), "match_status": "matched_by_punctuation_insensitive_key", "matched_canonical_form_id": alias["alias_for_canonical_form_id"], "fold_id": m.get("fold_id", "")})
        elif len(matches) > 1:
            audit_rows.append({**e.to_dict(), "match_status": "ambiguous_match", "matched_canonical_form_id": json.dumps([m.get("canonical_form_id") for m in matches], ensure_ascii=False), "fold_id": ""})
        else:
            audit_rows.append({**e.to_dict(), "match_status": "unmatched", "matched_canonical_form_id": "", "fold_id": ""})

    out = pd.DataFrame(out_rows).drop_duplicates(subset=["canonical_form_id"], keep="first")
    out.to_csv(args.output_csv, index=False)
    pd.DataFrame(audit_rows).to_csv(args.audit_csv, index=False)
    meta = {
        "input_fold_assignments_csv": args.fold_assignments_csv,
        "expert_roundtrips_csv": args.expert_roundtrips_csv,
        "output_csv": args.output_csv,
        "n_original_fold_rows": int(len(folds)),
        "n_output_fold_rows_with_aliases": int(len(out)),
        "n_expert_forms": int(len(exp)),
        "audit_csv": args.audit_csv,
    }
    Path(args.output_csv).with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote repaired fold assignments to {args.output_csv}")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", norm(text).lower()) if t not in STOP and t not in ARTIFACT_STOP]


def term_set_from_row(r: pd.Series, k: int) -> set[str]:
    try:
        terms = json.loads(r.get("suggested_terms_json", "[]") or "[]")[:k]
    except Exception:
        terms = []
    if not terms:
        try:
            spans = json.loads(r.get("top_spans_json", "[]") or "[]")
        except Exception:
            spans = []
        terms = [x for x, _ in Counter(t for s in spans for t in tokenize(s)).most_common(k)]
    return {t for t in terms if t and t not in ARTIFACT_STOP}


def jaccard_set(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def schema_field_from_row(r: pd.Series, tier: str) -> dict[str, Any]:
    return {
        "name": f"field_{r['candidate_field_id']}",
        "status": tier,
        "candidate_field_id": r["candidate_field_id"],
        "description": "Selected data-derived candidate field from strict near-equivalence among context-specific source-element senses; final naming requires consensus audit.",
        "suggested_terms": json.loads(r.get("suggested_terms_json", "[]") or "[]"),
        "positive_span_examples": json.loads(r.get("top_spans_json", "[]") or "[]")[:10],
        "source_model_support": json.loads(r.get("source_models_json", "[]") or "[]"),
        "evidence": {
            "n_sense_nodes": int(r.get("n_sense_nodes", 0)),
            "n_mentions": int(r.get("n_mentions", 0)),
            "n_positive_mentions": int(r.get("n_positive_mentions", 0)),
            "positive_fraction": float(r.get("positive_fraction", 0)),
        },
    }


def write_selected_schema(fold_dir: Path, fold_id: int, selected: pd.DataFrame) -> None:
    fields = [schema_field_from_row(r, str(r.get("selection_tier", "selected_candidate"))) for _, r in selected.iterrows()]
    schema = {
        "meta_model_id": f"refined_selected_consent_metamodel_fold_{fold_id}",
        "status": "selected_fold_specific_candidate_for_heldout_evaluation",
        "derivation_split": {"fold_id": fold_id, "training_forms_only": True, "test_forms_excluded_from_schema_development": True},
        "method": {
            "unit_of_analysis": "source-element-in-context mention",
            "selection_rule": "raw candidate fields are selected by cross-fold term-profile stability, positive support, and source-model support",
            "merge_rule": "candidate fields originate only from strict near-equivalence edges; co-occurrence is not merge evidence",
        },
        "decision": {"scope": "sentence_or_provision_level", "allowed_values": ["permit", "deny", "obligation", "mixed", "unclear"]},
        "fields": fields,
        "residual_important_content": {"description": "Meaning-critical content not captured by selected fields."},
        "provenance": {"required": True, "note": "Field IDs remain fold-specific candidates; final names require cross-fold consensus and audit."},
    }
    out = fold_dir / "refined_selected_candidate_schema.yaml"
    out.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True))
    out.with_suffix(".json").write_text(json.dumps(schema, indent=2, ensure_ascii=False))


def select_fields(args: argparse.Namespace) -> None:
    root = Path(args.fold_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted(root.glob("fold_*/candidate_field_clusters.csv")):
        fold = p.parent.name
        fold_id = int(re.sub(r"\D", "", fold) or 0)
        df = pd.read_csv(p).fillna("")
        for _, r in df.iterrows():
            source_models = json.loads(r.get("source_models_json", "[]") or "[]")
            records.append({
                "fold": fold,
                "fold_id": fold_id,
                "fold_dir": str(p.parent),
                "candidate_field_id": str(r.get("candidate_field_id")),
                "n_positive_mentions": int(r.get("n_positive_mentions", 0)),
                "n_mentions": int(r.get("n_mentions", 0)),
                "n_sense_nodes": int(r.get("n_sense_nodes", 0)),
                "positive_fraction": float(r.get("positive_fraction", 0)),
                "n_source_models": len(source_models),
                "source_models_json": r.get("source_models_json", "[]"),
                "suggested_terms_json": r.get("suggested_terms_json", "[]"),
                "top_spans_json": r.get("top_spans_json", "[]"),
                "term_set": term_set_from_row(r, int(args.signature_terms)),
            })
    if not records:
        raise SystemExit(f"No fold candidate clusters found under {root}")

    uf = UnionFind()
    for r in records:
        uf.find(f"{r['fold']}::{r['candidate_field_id']}")
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a["fold"] == b["fold"]:
                continue
            if a["n_positive_mentions"] < args.min_select_positive_mentions or b["n_positive_mentions"] < args.min_select_positive_mentions:
                continue
            sim = jaccard_set(a["term_set"], b["term_set"])
            if sim >= args.stability_jaccard:
                uf.union(f"{a['fold']}::{a['candidate_field_id']}", f"{b['fold']}::{b['candidate_field_id']}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[uf.find(f"{r['fold']}::{r['candidate_field_id']}")].append(r)

    rows = []
    for idx, members in enumerate(sorted(grouped.values(), key=lambda xs: (-len({m['fold'] for m in xs}), -sum(m['n_positive_mentions'] for m in xs))), start=1):
        group_id = f"SG{idx:03d}"
        n_folds = len({m["fold"] for m in members})
        total_pos = sum(m["n_positive_mentions"] for m in members)
        max_source_models = max(m["n_source_models"] for m in members)
        top_terms = [x for x, _ in Counter(t for m in members for t in m["term_set"]).most_common(12)]
        if n_folds >= args.core_min_folds and total_pos >= args.core_min_total_positive_mentions and max_source_models >= args.min_source_models:
            tier = "core_candidate"
        elif n_folds >= args.extension_min_folds and total_pos >= args.extension_min_total_positive_mentions and max_source_models >= args.min_source_models:
            tier = "extension_candidate"
        else:
            tier = "audit_only"
        for m in members:
            row = {k: v for k, v in m.items() if k not in {"term_set"}}
            row.update({
                "stability_group_id": group_id,
                "selection_tier": tier,
                "stability_n_folds": n_folds,
                "stability_total_positive_mentions": total_pos,
                "stability_max_source_models": max_source_models,
                "stability_top_terms_json": json.dumps(top_terms, ensure_ascii=False),
            })
            rows.append(row)

    stab = pd.DataFrame(rows)
    tier_order = {"core_candidate": 0, "extension_candidate": 1, "audit_only": 2}
    stab["tier_order"] = stab["selection_tier"].map(tier_order).fillna(9)
    stab = stab.sort_values(["tier_order", "stability_n_folds", "stability_total_positive_mentions", "fold"], ascending=[True, False, False, True]).drop(columns=["tier_order"])
    stab.to_csv(out / "cross_fold_field_stability_groups.csv", index=False)

    summary = stab.groupby(["stability_group_id", "selection_tier"], dropna=False).agg(
        n_folds=("fold", "nunique"),
        total_positive_mentions=("n_positive_mentions", "sum"),
        max_source_models=("n_source_models", "max"),
        top_terms_json=("stability_top_terms_json", "first"),
        folds=("fold", lambda x: json.dumps(sorted(set(x)), ensure_ascii=False)),
        member_fields=("candidate_field_id", lambda x: json.dumps(list(x), ensure_ascii=False)),
    ).reset_index().sort_values(["selection_tier", "n_folds", "total_positive_mentions"], ascending=[True, False, False])
    summary.to_csv(out / "selected_field_stability_summary.csv", index=False)

    selected = stab[stab["selection_tier"].isin(["core_candidate", "extension_candidate"]) & (stab["n_positive_mentions"] >= args.min_select_positive_mentions)].copy()
    selected.to_csv(out / "selected_fields_long.csv", index=False)
    for fold, g in selected.groupby("fold"):
        fold_dir = Path(g["fold_dir"].iloc[0])
        fold_id = int(g["fold_id"].iloc[0])
        write_selected_schema(fold_dir, fold_id, g)

    meta = {
        "selection_method": "cross-fold term-profile stability with positive-support and source-model thresholds",
        "signature_terms": int(args.signature_terms),
        "stability_jaccard": float(args.stability_jaccard),
        "core_min_folds": int(args.core_min_folds),
        "extension_min_folds": int(args.extension_min_folds),
        "min_source_models": int(args.min_source_models),
        "min_select_positive_mentions": int(args.min_select_positive_mentions),
        "core_min_total_positive_mentions": int(args.core_min_total_positive_mentions),
        "extension_min_total_positive_mentions": int(args.extension_min_total_positive_mentions),
        "selected_field_counts_by_fold": selected.groupby("fold").size().to_dict() if not selected.empty else {},
    }
    (out / "field_selection_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote field selection outputs to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("repair-fold-assignments")
    p.add_argument("--fold_assignments_csv", required=True)
    p.add_argument("--expert_roundtrips_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--audit_csv", required=True)

    p = sub.add_parser("select-fields")
    p.add_argument("--fold_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--signature_terms", type=int, default=10)
    p.add_argument("--stability_jaccard", type=float, default=0.45)
    p.add_argument("--core_min_folds", type=int, default=3)
    p.add_argument("--extension_min_folds", type=int, default=2)
    p.add_argument("--min_source_models", type=int, default=2)
    p.add_argument("--min_select_positive_mentions", type=int, default=20)
    p.add_argument("--core_min_total_positive_mentions", type=int, default=80)
    p.add_argument("--extension_min_total_positive_mentions", type=int, default=40)

    args = ap.parse_args()
    if args.cmd == "repair-fold-assignments":
        repair_fold_assignments(args)
    elif args.cmd == "select-fields":
        select_fields(args)


if __name__ == "__main__":
    main()
