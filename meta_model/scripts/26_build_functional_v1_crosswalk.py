#!/usr/bin/env python
"""Build a source-model to Reduced Functional V1 crosswalk.

Inputs:
- source_element_inventory.csv from the Union V0/source-model inventory.
- reduced_functional_v1_candidate.yaml.

Outputs:
- functional_v1_crosswalk.csv: one or more target-field rows per source element.
- functional_v1_crosswalk_summary.csv: source-model by V1-field counts.
- functional_v1_model_field_matrix.csv: plot-ready model-field matrix.
- functional_v1_context_dependent_review.csv: broad elements needing expert/context review.

The goal is transparency, not final automatic adjudication. Broad elements such as
ODRL Party and Constraint are intentionally marked context_dependent_split.
"""
from __future__ import annotations

import argparse
import json
import re
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


def infer_source_model(uid: str, row: pd.Series) -> str:
    for c in ["information_model", "source_model", "model", "canonical_information_model"]:
        if c in row.index and norm(row.get(c)):
            return norm(row.get(c))
    return uid.split("::", 1)[0] if "::" in uid else "unknown"


def source_tail(uid: str) -> str:
    return uid.split("::", 1)[1] if "::" in uid else uid


def add(rows: list[dict[str, Any]], row: pd.Series, target: str, mapping_type: str, rationale: str, secondary: list[str] | None = None) -> None:
    uid = norm(row.get("union_element_id") or row.get("source_element_id") or row.get("element_id") or row.get("id"))
    source_model = infer_source_model(uid, row)
    rows.append({
        "information_model": source_model,
        "union_element_id": uid,
        "source_element_label": norm(row.get("source_element_label") or row.get("label") or row.get("name")),
        "source_element_definition": norm(row.get("source_element_definition") or row.get("definition") or row.get("description")),
        "v1_field": target,
        "secondary_v1_fields_json": json.dumps(secondary or [], ensure_ascii=False),
        "mapping_type": mapping_type,
        "mapping_direction": "source_to_v1",
        "requires_context_review": mapping_type in {"context_dependent_split", "partial", "no_direct_mapping", "needs_expert_review"},
        "rationale": rationale,
    })


def explicit_mapping(row: pd.Series) -> list[tuple[str, str, str, list[str]]]:
    uid = norm(row.get("union_element_id") or row.get("source_element_id") or row.get("element_id") or row.get("id"))
    info = infer_source_model(uid, row)
    tail = source_tail(uid)
    tl = tail.lower()
    label = norm(row.get("source_element_label") or row.get("label") or row.get("name")).lower()
    definition = norm(row.get("source_element_definition") or row.get("definition") or row.get("description")).lower()
    text = " ".join([uid.lower(), tl, label, definition])

    # Sentence/provision decisions.
    if re.search(r"(provision\.type|consent\.decision|rule_testsentence|duo\.decision|ico\.decision|decision$)", text):
        return [("sentence_decision", "exact_decision", "source element encodes provision-level permit/deny/obligation polarity", [])]

    # ODRL core primitives.
    if info.upper() == "ODRL":
        if tl in {"party", "party_1", "party_2"} or "party" in tl:
            return [("participant_or_subject", "context_dependent_split", "ODRL Party may denote participant, actor, institution, or repository depending on span", ["authorized_actor", "institution_or_custodian", "repository_or_registry"])]
        if "asset" in tl:
            return [("governed_resource", "near_equivalent", "ODRL Asset denotes the governed data/specimen/object", [])]
        if "action" in tl:
            return [("governed_action", "near_equivalent_or_action_subtype", "ODRL Action denotes the governed act", ["choice_or_withdrawal_right", "contact_or_request"])]
        if "constraint" in tl:
            return [("condition_or_exception", "context_dependent_split", "ODRL Constraint may denote temporal, purpose, privacy, exception, or restriction scope", ["temporal_scope", "purpose_or_use_context", "restriction_or_prohibition", "privacy_or_identifiability"])]
        if "permission" in tl:
            return [("sentence_decision", "exact_decision_plus_cue", "ODRL Permission encodes permit polarity; textual cues map separately", ["decision_cue_or_consent_act"])]
        if "prohibition" in tl:
            return [("sentence_decision", "exact_decision_plus_restriction", "ODRL Prohibition encodes deny polarity and often supports restrictions", ["restriction_or_prohibition"])]
        if "duty" in tl or "obligation" in tl:
            return [("sentence_decision", "exact_obligation_plus_action", "ODRL Duty encodes obligation-like provision polarity", ["governed_action", "consequence_or_protection"])]

    # FHIR Consent common elements.
    if info.lower().startswith("fhir") or "consent." in tl:
        fhir_map = {
            "consent.subject": ("participant_or_subject", "near_equivalent", []),
            "consent.grantor": ("participant_or_subject", "near_equivalent_or_consent_actor", ["decision_cue_or_consent_act"]),
            "consent.grantee": ("authorized_actor", "context_dependent_split", ["institution_or_custodian"]),
            "consent.controller": ("institution_or_custodian", "context_dependent_split", ["authorized_actor", "repository_or_registry"]),
            "consent.manager": ("institution_or_custodian", "context_dependent_split", ["authorized_actor"]),
            "consent.provision.actor": ("authorized_actor", "context_dependent_split", ["institution_or_custodian"]),
            "consent.provision.actor.role": ("authorized_actor", "context_dependent_split", ["participant_or_subject"]),
            "consent.provision.data": ("governed_resource", "near_equivalent", ["privacy_or_identifiability"]),
            "consent.provision.action": ("governed_action", "near_equivalent_or_action_subtype", ["choice_or_withdrawal_right", "contact_or_request"]),
            "consent.action": ("governed_action", "near_equivalent_or_action_subtype", ["choice_or_withdrawal_right"]),
            "consent.provision.purpose": ("purpose_or_use_context", "near_equivalent", ["research_domain_or_study_topic"]),
            "consent.provision.period": ("temporal_scope", "near_equivalent", ["temporal_target"]),
            "consent.period": ("temporal_scope", "near_equivalent", ["temporal_target"]),
            "consent.provision.dataperiod": ("temporal_scope", "near_equivalent", ["temporal_target", "study_or_data_lifecycle"]),
            "consent.provision.securitylabel": ("privacy_or_identifiability", "near_equivalent", ["restriction_or_prohibition"]),
            "consent.category": ("purpose_or_use_context", "partial", ["study_or_data_lifecycle"]),
            "consent.provision": ("residual_important_content", "container_no_direct_mapping", []),
            "consent": ("residual_important_content", "container_no_direct_mapping", []),
        }
        key = tl.lower()
        if key in fhir_map:
            target, mtype, sec = fhir_map[key]
            return [(target, mtype, f"FHIR {tail} maps to the corresponding functional V1 role", sec)]

    # DUO common concepts are often purpose/domain/restriction labels.
    if info.upper() == "DUO":
        duo_tail = tail.upper()
        if duo_tail in {"GRU", "HMB", "PS"} or "general research" in text or "health" in text:
            return [("purpose_or_use_context", "near_equivalent_or_partial", "DUO data-use purpose/research-use category", ["research_domain_or_study_topic", "condition_or_exception"])]
        if duo_tail in {"DS", "GSO", "RS", "GS", "NRES"} or "disease" in text or "population" in text:
            return [("research_domain_or_study_topic", "partial", "DUO category constrains research topic/domain or allowed use scope", ["purpose_or_use_context", "restriction_or_prohibition"])]
        if duo_tail in {"TS", "RTN"} or "time" in text or "retention" in text:
            return [("temporal_scope", "partial", "DUO temporal/retention category", ["temporal_target", "study_or_data_lifecycle"])]
        if duo_tail in {"COL", "NMDS", "PUB", "IRB"} or "restriction" in text:
            return [("restriction_or_prohibition", "partial", "DUO restriction or governance condition", ["condition_or_exception", "privacy_or_identifiability"])]

    # ICO patterns.
    if info.upper() == "ICO" or uid.lower().startswith("ico"):
        if any(x in text for x in ["withdraw", "refus", "declin", "choice", "voluntar"]):
            return [("choice_or_withdrawal_right", "near_equivalent", "ICO participation/withdrawal/choice element", ["decision_cue_or_consent_act", "governed_action"])]
        if any(x in text for x in ["consent", "permission", "authorize", "agree", "informed consenting"]):
            return [("decision_cue_or_consent_act", "near_equivalent", "ICO consent/authorization act or cue", ["sentence_decision"])]
        if any(x in text for x in ["participant", "patient", "subject", "consenter"]):
            return [("participant_or_subject", "near_equivalent", "ICO person role maps to participant/subject", [])]
        if any(x in text for x in ["researcher", "investigator", "doctor", "physician"]):
            return [("authorized_actor", "near_equivalent", "ICO actor role maps to authorized actor", [])]
        if any(x in text for x in ["specimen", "sample", "biospecimen", "data", "information", "record"]):
            return [("governed_resource", "near_equivalent", "ICO data/specimen/information element", ["privacy_or_identifiability"])]
        if any(x in text for x in ["use", "collect", "share", "disclos", "access", "return", "destroy", "store"]):
            return [("governed_action", "near_equivalent_or_action_subtype", "ICO act element maps to governed action", [])]

    # Keyword fallback. These are intentionally low-confidence and require review.
    keyword_rules = [
        ("participant_or_subject", ["participant", "subject", "child", "patient", "consenter", "grantor"], ["authorized_actor"]),
        ("authorized_actor", ["researcher", "investigator", "doctor", "physician", "staff", "team", "grantee"], ["institution_or_custodian"]),
        ("institution_or_custodian", ["institution", "clinic", "hospital", "university", "custodian", "controller", "manager"], ["repository_or_registry"]),
        ("repository_or_registry", ["database", "registry", "repository", "biobank", "bank"], ["institution_or_custodian"]),
        ("governed_resource", ["data", "sample", "specimen", "dna", "blood", "urine", "saliva", "record", "information", "result"], ["privacy_or_identifiability"]),
        ("governed_action", ["use", "collect", "store", "share", "disclose", "access", "destroy", "return", "donate", "contact"], []),
        ("purpose_or_use_context", ["purpose", "research", "clinical", "public health"], ["research_domain_or_study_topic"]),
        ("temporal_scope", ["period", "duration", "expire", "expiration", "until", "time", "future", "indefinite"], ["temporal_target"]),
        ("condition_or_exception", ["condition", "exception", "unless", "except", "only if"], ["restriction_or_prohibition"]),
        ("restriction_or_prohibition", ["prohibit", "restriction", "deny", "not", "cannot"], ["condition_or_exception"]),
        ("privacy_or_identifiability", ["privacy", "confidential", "identifiable", "de-identified", "anonymous", "security"], []),
        ("consequence_or_protection", ["care", "penalty", "benefit", "affect", "protection"], []),
    ]
    for target, keys, sec in keyword_rules:
        if any(k in text for k in keys):
            return [(target, "needs_expert_review", "keyword fallback based on source label/definition; review required", sec)]

    return [("residual_important_content", "no_direct_mapping", "no clear V1 functional mapping from available source metadata", [])]


def load_schema_fields(path: Path) -> set[str]:
    obj = yaml.safe_load(path.read_text())
    fields = {norm(f.get("name")) for f in obj.get("fields", []) if isinstance(f, dict)}
    fields.add("sentence_decision")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory_csv", required=True)
    ap.add_argument("--schema_yaml", default="meta_model/schemas/reduced_functional_v1_candidate.yaml")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    inv = pd.read_csv(args.inventory_csv).fillna("")
    valid_fields = load_schema_fields(Path(args.schema_yaml))
    rows: list[dict[str, Any]] = []
    for _, r in inv.iterrows():
        uid = norm(r.get("union_element_id") or r.get("source_element_id") or r.get("element_id") or r.get("id"))
        if not uid:
            continue
        for target, mtype, rationale, secondary in explicit_mapping(r):
            if target not in valid_fields:
                target = "residual_important_content"
                mtype = "needs_expert_review"
                rationale = "mapping target not present in schema; routed to residual for review"
            add(rows, r, target, mtype, rationale, [s for s in secondary if s in valid_fields])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xw = pd.DataFrame(rows)
    xw.to_csv(out / "functional_v1_crosswalk.csv", index=False)

    summary = xw.groupby(["information_model", "v1_field", "mapping_type"], dropna=False).size().reset_index(name="n_source_elements")
    summary.to_csv(out / "functional_v1_crosswalk_summary.csv", index=False)

    matrix = xw.pivot_table(index="v1_field", columns="information_model", values="union_element_id", aggfunc="count", fill_value=0).reset_index()
    matrix.to_csv(out / "functional_v1_model_field_matrix.csv", index=False)

    review = xw[xw["requires_context_review"] == True].copy()
    review.to_csv(out / "functional_v1_context_dependent_review.csv", index=False)

    # Plot/network-ready edge list.
    edges = xw[["information_model", "union_element_id", "v1_field", "mapping_type", "requires_context_review"]].copy()
    edges.rename(columns={"information_model": "source_model", "union_element_id": "source_element", "v1_field": "target_v1_field"}, inplace=True)
    edges.to_csv(out / "functional_v1_source_to_field_edges.csv", index=False)

    meta = {
        "inventory_csv": args.inventory_csv,
        "schema_yaml": args.schema_yaml,
        "n_source_elements": int(xw["union_element_id"].nunique()),
        "n_crosswalk_rows": int(len(xw)),
        "n_context_dependent_or_review_rows": int(len(review)),
        "outputs": [
            "functional_v1_crosswalk.csv",
            "functional_v1_crosswalk_summary.csv",
            "functional_v1_model_field_matrix.csv",
            "functional_v1_context_dependent_review.csv",
            "functional_v1_source_to_field_edges.csv",
        ],
    }
    (out / "functional_v1_crosswalk_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote functional V1 crosswalk outputs to {out}")


if __name__ == "__main__":
    main()
