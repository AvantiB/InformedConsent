#!/usr/bin/env python
"""Build the final Reduced V1 YAML schema from audited semantic clusters.

This script intentionally requires human-audited cluster names/selections. It does
not infer final V1 fields from hard-coded role buckets. Fill in
semantic_cluster_audit_template.csv from script 17, then run this script.
"""
from __future__ import annotations

import argparse, json, re
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

YES={"1","true","yes","y","include","included"}


def norm(x:Any)->str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return " ".join(str(x).split())

def truthy(x:Any)->bool:
    return norm(x).lower() in YES

def safe_name(x:str)->str:
    s=re.sub(r"[^a-zA-Z0-9]+","_",norm(x).lower()).strip("_")
    return s or "unnamed_field"

def load_json_list(x:Any)->list[str]:
    s=norm(x)
    if not s: return []
    try:
        v=json.loads(s)
        return [norm(a) for a in v if norm(a)] if isinstance(v,list) else []
    except Exception:
        return [norm(s)]

def build_schema(audit:pd.DataFrame, clusters:pd.DataFrame, decision_summary:pd.DataFrame|None, args)->dict[str,Any]:
    needed={"semantic_cluster_id","include_in_v1","final_field_name"}
    missing=needed-set(audit.columns)
    if missing: raise ValueError(f"Audit CSV missing required columns: {sorted(missing)}")
    include=audit[audit["include_in_v1"].map(truthy)].copy()
    include=include[include["final_field_name"].map(lambda x: bool(norm(x)))].copy()
    if include.empty:
        raise SystemExit("No audited clusters selected. Fill include_in_v1=yes and final_field_name before building schema.")
    cmap={cid:g for cid,g in clusters.groupby("semantic_cluster_id")} if not clusters.empty else {}
    fields=[]
    decision_support=[]
    if decision_summary is not None and not decision_summary.empty and "sentence_level_element_id" in decision_summary.columns:
        decision_support=decision_summary["sentence_level_element_id"].astype(str).head(20).tolist()
    fields.append({
        "name":"decision",
        "status":"core",
        "source":"sentence_level_decision_fields",
        "description":"Provision rule type derived from sentence-level decision evidence and/or roundtrip_decision.",
        "values":["permit","deny","obligation","mixed","unclear"],
        "selection_evidence":{"decision_element_support":decision_support},
    })
    used=Counter()
    for _,r in include.iterrows():
        cid=norm(r.semantic_cluster_id); base=safe_name(r.final_field_name); used[base]+=1; name=base if used[base]==1 else f"{base}_{used[base]}"
        g=cmap.get(cid,pd.DataFrame())
        support=g.sort_values(["n_positive_source_sentences","n_positive_mentions"],ascending=[False,False]) if not g.empty else pd.DataFrame()
        source_elements=support["union_element_id"].astype(str).head(25).tolist() if not support.empty else load_json_list(r.get("top_source_elements_json",""))
        spans=[]
        if not support.empty and "top_positive_span_examples_json" in support.columns:
            for raw in support["top_positive_span_examples_json"].head(10):
                spans.extend(load_json_list(raw)[:3])
        if not spans: spans=load_json_list(r.get("top_positive_span_examples_json",""))[:12]
        fields.append({
            "name":name,
            "semantic_cluster_id":cid,
            "status":norm(r.get("field_status","")) or "core_or_context_from_audit",
            "description":norm(r.get("field_description","")) or f"Audited field from empirical semantic cluster {cid}.",
            "value_type":"normalized_value_with_evidence",
            "allow_multiple":True,
            "audit_decision":norm(r.get("audit_decision","")),
            "unsafe_merge_notes":norm(r.get("unsafe_merge_notes","")),
            "split_or_merge_notes":norm(r.get("split_or_merge_notes","")),
            "selection_evidence":{
                "n_source_elements":int(r.get("n_source_elements",len(source_elements)) or 0),
                "n_positive_source_sentences_max":float(r.get("n_positive_source_sentences_max",0) or 0),
                "n_positive_information_models_max":float(r.get("n_positive_information_models_max",0) or 0),
                "n_positive_llms_max":float(r.get("n_positive_llms_max",0) or 0),
                "mean_expert_positive_rate":float(r.get("mean_expert_positive_rate",0) or 0),
            },
            "source_element_support":source_elements,
            "positive_span_examples":spans[:12],
        })
    fields += [
        {"name":"residual_important_content","status":"audit","description":"Short meaning-critical content not captured by selected audited fields.","value_type":"short_evidence_phrase"},
        {"name":"provenance","status":"audit","description":"Source sentence, evidence spans, semantic cluster IDs, and source elements used for audit.","value_type":"audit_metadata"},
    ]
    return {
        "meta_model_id":"reduced_consent_metamodel_v1_audited",
        "version":args.version,
        "status":"audited_schema_ready_for_validation",
        "design_goal":"Reduced functional consent representation derived from expert-preserved semantic-equivalence clusters and finalized by naming/unsafe-merge audit.",
        "selection_method":{
            "derivation_corpus":"original expert-evaluated round-trip workbooks cleaned by script 12",
            "cluster_discovery":"script 17 semantic-equivalence graph; provision-bundle graph kept separate from merge evidence",
            "field_selection":"audited semantic_cluster_audit_template.csv; no hard-coded role bucket selection",
            "human_role":"naming, include/exclude decisions, and unsafe-merge/split review",
        },
        "fields":fields,
        "provision_structure":{"rule_type":"decision","selected_cluster_fields":[f["name"] for f in fields if f.get("semantic_cluster_id")],"audit_fields":["residual_important_content","provenance"]},
    }

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit_csv",required=True)
    ap.add_argument("--clusters_csv",required=True)
    ap.add_argument("--decision_summary_csv")
    ap.add_argument("--output_yaml",required=True)
    ap.add_argument("--output_json")
    ap.add_argument("--version",default="1.0-audit-draft")
    args=ap.parse_args()
    audit=pd.read_csv(args.audit_csv).fillna(""); clusters=pd.read_csv(args.clusters_csv).fillna("")
    decision=pd.read_csv(args.decision_summary_csv).fillna("") if args.decision_summary_csv and Path(args.decision_summary_csv).exists() else pd.DataFrame()
    schema=build_schema(audit,clusters,decision,args)
    out=Path(args.output_yaml); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(schema,sort_keys=False,allow_unicode=True))
    if args.output_json: Path(args.output_json).write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    print(f"Wrote audited Reduced V1 schema to {out}")

if __name__=="__main__": main()
