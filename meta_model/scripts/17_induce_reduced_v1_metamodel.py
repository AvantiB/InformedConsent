#!/usr/bin/env python
"""Induce a candidate reduced V1 informed-consent meta-model from round-trip evidence.

Inputs come from 15_analyze_roundtrip_scored_outputs.py after sentence-level
rule/decision fields have been separated from span-level evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

ROLE_SPECS: dict[str, dict[str, Any]] = {
    "decision": {"core": True, "description": "Provision-level permit/deny/obligation/mixed decision.", "keywords": ["decision", "permission", "prohibition", "permit", "deny", "type"]},
    "action": {"core": True, "description": "Governed action: collect, use, store, share, disclose, access, analyze, contact, withdraw, destroy, return results.", "keywords": ["action", "verb", "operation", "collect", "use", "store", "share", "disclose", "access", "analy", "contact", "withdraw", "destroy", "return"]},
    "resource": {"core": True, "description": "Governed data/specimen/resource.", "keywords": ["resource", "asset", "object", "data", "information", "record", "specimen", "sample", "tissue", "blood", "dna", "genetic", "genomic", "image", "audio", "video"]},
    "actor": {"core": True, "description": "Party who acts or is authorized/required to act.", "keywords": ["actor", "agent", "party", "assigner", "performer", "researcher", "doctor", "investigator", "institution", "team"]},
    "recipient_or_grantee": {"core": True, "description": "Party receiving access, data, specimens, results, or permission.", "keywords": ["recipient", "grantee", "target", "assignee", "shared with", "available to", "accessor", "receiver"]},
    "purpose": {"core": True, "description": "Purpose or use context: current study, future research, disease-specific research, commercial development, clinical care.", "keywords": ["purpose", "research", "study", "future", "commercial", "clinical", "care", "objective"]},
    "condition": {"core": True, "description": "Conditional trigger or prerequisite.", "keywords": ["condition", "if", "when", "unless", "only if", "precondition", "approval", "governance", "irb", "law"]},
    "constraint_or_exception": {"core": True, "description": "Restrictions, exceptions, limitations, or boundaries.", "keywords": ["constraint", "restriction", "exception", "limitation", "limited", "only", "except", "not", "without", "prohibition"]},
    "temporal_scope": {"core": True, "description": "Time or duration of permission/restriction.", "keywords": ["time", "temporal", "duration", "future", "after", "before", "during", "until", "year", "month", "day", "ongoing", "long-term"]},
    "privacy_identifiability": {"core": True, "description": "Identifiability, privacy, coding, de-identification, anonymity, confidentiality.", "keywords": ["privacy", "identifiable", "identifier", "identified", "de-identified", "coded", "anonymous", "confidential", "name", "contact"]},
    "choice_structure": {"core": False, "description": "Choice architecture: optional, yes/no, separate permission, decline, join, consent, withdrawal right.", "keywords": ["choice", "consent", "agree", "decline", "yes", "no", "optional", "join", "participate", "withdraw"]},
    "lifecycle_effect": {"core": False, "description": "Operational effect over time: retain, destroy, continue using already shared data, stop future use, return/no return results.", "keywords": ["retain", "destroy", "delete", "store", "withdraw", "withdrawal effect", "return result", "result", "incidental", "continue"]},
    "risk_benefit_or_results": {"core": False, "description": "Risks, benefits, incidental findings, and return of results.", "keywords": ["risk", "benefit", "harm", "result", "finding", "incidental", "return"]},
    "residual_important_content": {"core": False, "description": "Short uncaptured but meaning-critical phrase.", "keywords": ["residual", "unmatched", "other"]},
    "provenance": {"core": True, "description": "Source/evidence/provenance metadata for audit.", "keywords": ["provenance", "evidence", "source", "span"]},
}
ROLE_ORDER = list(ROLE_SPECS)


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def json_list(x: Any) -> list[Any]:
    try:
        y = json.loads(norm(x))
        return y if isinstance(y, list) else []
    except Exception:
        return []


def blob(row: pd.Series) -> str:
    parts = [row.get(c, "") for c in ["union_element_id", "source_model", "source_element_id", "source_element_label", "source_element_definition", "top_original_cue_groups_json"]]
    parts += [str(v) for v in json_list(row.get("top_span_examples_json", ""))[:8]]
    parts += [str(v) for v in json_list(row.get("top_original_cue_groups_json", ""))[:8]]
    return " ".join(norm(p).lower() for p in parts if norm(p))


def kscore(text: str, keywords: list[str]) -> int:
    s = 0
    for kw in keywords:
        k = kw.lower()
        s += (2 if k in text else 0) if " " in k else len(re.findall(rf"\b{re.escape(k)}", text))
    return s


def assign_role(row: pd.Series) -> tuple[str, str, int, dict[str, int]]:
    uid = norm(row.get("union_element_id"))
    tail = uid.split("::")[-1].lower()
    text = blob(row)
    overrides = [
        ("action", ["action", "verb", "operation"]),
        ("resource", ["asset", "resource", "object", "specimen", "sample", "data category", "information category"]),
        ("actor", ["actor", "agent", "assigner", "performer"]),
        ("recipient_or_grantee", ["recipient", "grantee", "assignee", "target"]),
        ("purpose", ["purpose", "research category", "disease"]),
        ("privacy_identifiability", ["identifier", "identifiable", "coded", "anonymous", "privacy", "confidential"]),
        ("temporal_scope", ["duration", "temporal", "time", "period"]),
        ("condition", ["condition", "precondition", "approval", "governance"]),
        ("constraint_or_exception", ["prohibition", "exception", "restriction", "limitation"]),
        ("risk_benefit_or_results", ["result", "finding", "risk", "benefit"]),
        ("choice_structure", ["consent", "choice", "withdraw"]),
        ("lifecycle_effect", ["destroy", "retain", "withdrawal effect"]),
    ]
    for role, needles in overrides:
        hits = [n for n in needles if n in tail or n in text]
        if hits:
            return role, f"keyword_override:{','.join(hits[:3])}", 99, {}
    scores = {r: kscore(text, spec["keywords"]) for r, spec in ROLE_SPECS.items()}
    scores["provenance"] = min(scores.get("provenance", 0), 1)
    scores["residual_important_content"] = min(scores.get("residual_important_content", 0), 1)
    role, score = max(scores.items(), key=lambda kv: (kv[1], -ROLE_ORDER.index(kv[0])))
    return (role if score > 0 else "residual_important_content"), ("max_keyword_score" if score > 0 else "no_role_keywords_matched"), int(score), scores


def weight(row: pd.Series) -> float:
    def f(c: str) -> float:
        try:
            return float(row.get(c, 0) or 0)
        except Exception:
            return 0.0
    return f("n_source_sentences") * (1 + 0.15 * f("n_llms")) * max(0.2, f("mean_classifier_preservation_score")) * max(0.2, f("mean_cue_group_recall")) * max(0.2, f("mean_content_token_recall"))


def merge_inventory(evidence: pd.DataFrame, inventory_csv: str | None) -> pd.DataFrame:
    out = evidence.copy()
    if inventory_csv:
        inv = pd.read_csv(inventory_csv).fillna("")
        if "union_element_id" in inv.columns:
            drop = [c for c in inv.columns if c in out.columns and c != "union_element_id"]
            out = out.drop(columns=drop) if drop else out
            out = out.merge(inv, on="union_element_id", how="left")
    return out.fillna("")


def build_assignments(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in evidence.iterrows():
        role, reason, score, scores = assign_role(r)
        rec = r.to_dict()
        rec.update({"candidate_role": role, "role_assignment_reason": reason, "role_assignment_score": score, "role_score_json": json.dumps(scores, ensure_ascii=False), "evidence_weight": weight(r), "recommended_core": ROLE_SPECS[role]["core"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_roles(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ROLE_ORDER:
        g = assignments[assignments["candidate_role"].eq(role)]
        if g.empty:
            continue
        top = []
        for _, r in g.sort_values("evidence_weight", ascending=False).head(10).iterrows():
            top.append({"union_element_id": r.get("union_element_id"), "label": r.get("source_element_label", ""), "source_model": r.get("source_model", "")})
        rows.append({
            "candidate_role": role,
            "description": ROLE_SPECS[role]["description"],
            "recommended_core": ROLE_SPECS[role]["core"],
            "n_source_elements": len(g),
            "n_source_sentences_sum": pd.to_numeric(g.get("n_source_sentences"), errors="coerce").sum(),
            "mean_classifier_preservation_score": pd.to_numeric(g.get("mean_classifier_preservation_score"), errors="coerce").mean(),
            "mean_content_token_recall": pd.to_numeric(g.get("mean_content_token_recall"), errors="coerce").mean(),
            "mean_cue_group_recall": pd.to_numeric(g.get("mean_cue_group_recall"), errors="coerce").mean(),
            "evidence_weight_sum": pd.to_numeric(g.get("evidence_weight"), errors="coerce").sum(),
            "top_source_elements_json": json.dumps(top, ensure_ascii=False),
        })
    return pd.DataFrame(rows).sort_values(["recommended_core", "evidence_weight_sum"], ascending=[False, False])


def role_pairs(assignments: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty or pairs.empty:
        return pd.DataFrame()
    role_map = dict(zip(assignments["union_element_id"], assignments["candidate_role"]))
    rows = []
    for _, p in pairs.iterrows():
        a, b = norm(p.get("union_element_id_a")), norm(p.get("union_element_id_b"))
        if a not in role_map or b not in role_map:
            continue
        x, y = sorted([role_map[a], role_map[b]])
        rows.append({"role_a": x, "role_b": y, "source_element_a": a, "source_element_b": b, "n_cooccurrences": p.get("n_cooccurrences", 0), "n_source_sentences": p.get("n_source_sentences", 0), "mean_classifier_preservation_score": p.get("mean_classifier_preservation_score", None)})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    agg = []
    for keys, g in raw.groupby(["role_a", "role_b"]):
        agg.append({"role_a": keys[0], "role_b": keys[1], "n_source_element_pairs": len(g), "n_cooccurrences_sum": pd.to_numeric(g["n_cooccurrences"], errors="coerce").sum(), "n_source_sentences_max": pd.to_numeric(g["n_source_sentences"], errors="coerce").max(), "mean_classifier_preservation_score": pd.to_numeric(g["mean_classifier_preservation_score"], errors="coerce").mean(), "top_source_pairs_json": json.dumps(g.sort_values("n_source_sentences", ascending=False).head(8)[["source_element_a", "source_element_b", "n_source_sentences"]].to_dict(orient="records"), ensure_ascii=False)})
    return pd.DataFrame(agg).sort_values(["n_source_sentences_max", "n_cooccurrences_sum"], ascending=[False, False])


def schema_yaml(role_summary: pd.DataFrame, assignments: pd.DataFrame, decision_summary: pd.DataFrame) -> dict[str, Any]:
    fields = []
    present = set(role_summary["candidate_role"])
    for role in ROLE_ORDER:
        if role not in present:
            continue
        spec = ROLE_SPECS[role]
        g = assignments[assignments["candidate_role"].eq(role)].sort_values("evidence_weight", ascending=False)
        fields.append({"name": role, "status": "core" if spec["core"] else "extension", "description": spec["description"], "value_type": "normalized_value_with_evidence", "allow_multiple": role not in {"decision", "privacy_identifiability"}, "source_element_support": g["union_element_id"].head(12).tolist()})
    c = Counter()
    if not decision_summary.empty and "top_values_json" in decision_summary.columns:
        for x in decision_summary["top_values_json"].dropna().astype(str):
            try:
                for item in json.loads(x):
                    c[str(item.get("value", "")).lower()] += int(item.get("n", 0))
            except Exception:
                pass
    return {"meta_model_id": "reduced_consent_metamodel_v1_candidate", "version": "0.1", "status": "candidate_data_driven_requires_audit", "design_goal": "compact functional consent representation for LLM round-trip meaning preservation", "evaluation_variants": {"compact_evidence": "short evidence phrases; no full-clause copying", "permissive_evidence": "same reduced schema but permits longer evidence phrases when needed"}, "decision_values_observed": [v for v, _ in c.most_common(12) if v], "fields": fields, "provision_structure": {"primary_bundle": ["decision", "action", "resource", "actor", "recipient_or_grantee", "purpose"], "modifier_bundle": ["condition", "constraint_or_exception", "temporal_scope", "privacy_identifiability", "choice_structure", "lifecycle_effect"], "audit_bundle": ["residual_important_content", "provenance"]}}


def write_md(schema: dict[str, Any], role_summary: pd.DataFrame, role_cooc: pd.DataFrame, path: Path) -> None:
    lines = ["# Candidate Reduced V1 Consent Meta-Model", "", "This candidate was induced from corrected span-level source-element evidence, cue preservation, lexical/content preservation, and co-occurrence.", "", "## Evidence variants", "", "- **compact_evidence**: short evidence phrases only; tests reduced role abstraction.", "- **permissive_evidence**: same schema, longer evidence phrases allowed; controls evidence-span granularity.", "", "## Candidate fields", ""]
    for f in schema["fields"]:
        lines += [f"### {f['name']} ({f['status']})", "", f["description"], "", f"Source support examples: {', '.join(f.get('source_element_support', [])[:8])}", ""]
    if not role_summary.empty:
        lines += ["## Role evidence summary", "", role_summary.to_markdown(index=False), ""]
    if not role_cooc.empty:
        lines += ["## Role co-occurrence summary", "", role_cooc.head(30).to_markdown(index=False), ""]
    lines += ["## Human audit scope", "", "Audit field names, unsafe merges, high-stakes extensions, and frequent residual content. The candidate schema itself is generated from evidence tables.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", required=True)
    ap.add_argument("--inventory_csv", default=None)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    analysis, out = Path(args.analysis_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence_path = analysis / "source_element_evidence_summary.csv"
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing {evidence_path}. Rerun script 15 after sentence-level decision fix.")
    evidence = merge_inventory(pd.read_csv(evidence_path).fillna(""), args.inventory_csv)
    pairs = pd.read_csv(analysis / "source_element_cooccurrence_pairs.csv").fillna("") if (analysis / "source_element_cooccurrence_pairs.csv").exists() else pd.DataFrame()
    decision_summary = pd.read_csv(analysis / "sentence_level_decision_summary.csv").fillna("") if (analysis / "sentence_level_decision_summary.csv").exists() else pd.DataFrame()
    assignments = build_assignments(evidence)
    role_summary = summarize_roles(assignments)
    role_cooc = role_pairs(assignments, pairs)
    schema = schema_yaml(role_summary, assignments, decision_summary)
    assignments.to_csv(out / "candidate_role_assignments.csv", index=False)
    role_summary.to_csv(out / "candidate_role_evidence_summary.csv", index=False)
    role_cooc.to_csv(out / "candidate_role_cooccurrence_summary.csv", index=False)
    (out / "reduced_metamodel_v1_candidate.yaml").write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True))
    (out / "reduced_metamodel_v1_candidate.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    write_md(schema, role_summary, role_cooc, out / "reduced_metamodel_v1_candidate.md")
    print(f"Wrote candidate V1 meta-model evidence to {out}")


if __name__ == "__main__":
    main()
