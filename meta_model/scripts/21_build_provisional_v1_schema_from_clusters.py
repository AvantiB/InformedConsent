#!/usr/bin/env python
"""Build a provisional empirical Reduced V1 schema directly from discovered clusters.

This is a PI-facing evaluation schema, not the final audited V1. It lets us run
the data-driven clusters as-is and compare meaning preservation against
individual source models and Union V0 before expert naming/organization.

Fields are named semantic_cluster_C### so we do not pretend that final expert
field names have already been assigned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc


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


def read_optional_csv(path: str | None) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path and Path(path).exists() else pd.DataFrame()


def build_schema(args: argparse.Namespace) -> dict[str, Any]:
    summary = pd.read_csv(args.semantic_cluster_summary_csv).fillna("")
    clusters = pd.read_csv(args.semantic_clusters_csv).fillna("")
    decisions = read_optional_csv(args.decision_summary_csv)

    statuses = {x.strip() for x in args.include_statuses.split(",") if x.strip()}
    if statuses and "selection_status" in summary.columns:
        keep = summary[summary["selection_status"].astype(str).isin(statuses)].copy()
    else:
        keep = summary.copy()
    keep = keep.sort_values(["n_positive_source_sentences_max", "n_positive_information_models_max"], ascending=[False, False])
    if args.max_clusters:
        keep = keep.head(args.max_clusters).copy()
    if keep.empty:
        raise SystemExit("No semantic clusters selected. Check --include_statuses or discovery outputs.")

    fields: list[dict[str, Any]] = []
    decision_support: list[str] = []
    if not decisions.empty and "sentence_level_element_id" in decisions.columns:
        decision_support = decisions["sentence_level_element_id"].astype(str).head(20).tolist()
    fields.append({
        "name": "decision",
        "status": "core_sentence_level",
        "description": "Sentence/provision decision type derived from decision evidence and roundtrip decision fields.",
        "values": ["permit", "deny", "obligation", "mixed", "unclear"],
        "selection_evidence": {"decision_element_support": decision_support},
    })

    cluster_map = {cid: g for cid, g in clusters.groupby("semantic_cluster_id")} if not clusters.empty else {}
    for _, row in keep.iterrows():
        cid = norm(row.get("semantic_cluster_id", ""))
        if not cid:
            continue
        g = cluster_map.get(cid, pd.DataFrame())
        if not g.empty:
            g = g.sort_values(["n_positive_source_sentences", "n_positive_mentions"], ascending=[False, False])
            source_elements = g["union_element_id"].astype(str).head(args.max_source_elements_per_field).tolist()
            span_examples: list[str] = []
            for raw in g.get("top_positive_span_examples_json", pd.Series(dtype=str)).head(10):
                span_examples.extend(load_json_list(raw)[:2])
        else:
            source_elements = load_json_list(row.get("top_source_elements_json", ""))[: args.max_source_elements_per_field]
            span_examples = load_json_list(row.get("top_positive_span_examples_json", ""))[:12]
        fields.append({
            "name": f"semantic_cluster_{cid}",
            "status": "provisional_empirical_cluster",
            "semantic_cluster_id": cid,
            "description": "Un-audited empirical semantic cluster discovered from expert-preserved same/overlapping-span evidence. PI should inspect source elements and assign a final field name or split/merge decision.",
            "value_type": "normalized_value_with_evidence",
            "allow_multiple": True,
            "selection_evidence": {
                "selection_status": norm(row.get("selection_status", "")),
                "n_source_elements": int(float(row.get("n_source_elements", 0) or 0)),
                "n_positive_source_sentences_max": float(row.get("n_positive_source_sentences_max", 0) or 0),
                "n_positive_information_models_max": float(row.get("n_positive_information_models_max", 0) or 0),
                "n_positive_llms_max": float(row.get("n_positive_llms_max", 0) or 0),
                "mean_expert_positive_rate": float(row.get("mean_expert_positive_rate", 0) or 0),
                "name_suggestion_terms": load_json_list(row.get("name_suggestion_terms_json", "")),
            },
            "source_element_support": source_elements,
            "positive_span_examples": span_examples[:12],
        })

    fields += [
        {"name": "residual_important_content", "status": "audit", "description": "Meaning-critical content not captured by provisional clusters.", "value_type": "short_evidence_phrase"},
        {"name": "provenance", "status": "audit", "description": "Source sentence, evidence spans, cluster IDs, and rationale for audit.", "value_type": "audit_metadata"},
    ]

    return {
        "meta_model_id": "reduced_consent_metamodel_v1_provisional_empirical_clusters",
        "version": args.version,
        "status": "provisional_unreviewed_pi_facing_evaluation_schema",
        "design_goal": "Run the data-driven Reduced V1 semantic clusters as-is to evaluate meaning preservation before final expert naming/organization.",
        "selection_method": {
            "derivation_corpus": "original researcher annotation workbooks cleaned into expert_roundtrips_clean.csv",
            "cluster_source": "script 17 semantic-equivalence clusters",
            "field_selection": "selected by empirical support status and coverage thresholds, not by manual role names",
            "field_naming": "provisional semantic_cluster_C### IDs; final names deferred to PI/expert audit",
            "human_role_at_this_stage": "none for schema generation; PI review follows performance comparison",
        },
        "fields": fields,
        "provision_structure": {
            "rule_type": "decision",
            "selected_cluster_fields": [f["name"] for f in fields if f.get("semantic_cluster_id")],
            "audit_fields": ["residual_important_content", "provenance"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic_cluster_summary_csv", required=True)
    ap.add_argument("--semantic_clusters_csv", required=True)
    ap.add_argument("--decision_summary_csv")
    ap.add_argument("--output_yaml", required=True)
    ap.add_argument("--output_json")
    ap.add_argument("--include_statuses", default="high_support_equivalence_cluster,context_specific_equivalence_cluster")
    ap.add_argument("--max_clusters", type=int, default=0)
    ap.add_argument("--max_source_elements_per_field", type=int, default=25)
    ap.add_argument("--version", default="0.1-provisional-empirical")
    args = ap.parse_args()

    schema = build_schema(args)
    out = Path(args.output_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    print(f"Wrote provisional empirical V1 schema to {out}")


if __name__ == "__main__":
    main()
