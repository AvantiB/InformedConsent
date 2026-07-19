#!/usr/bin/env python
"""Induce a compact candidate V1 consent meta-model from corrected evidence.

Inputs: script-15 outputs after sentence-level decision fields are separated.
Outputs: role assignments, role summaries, role co-occurrences, and a candidate
YAML schema used by the V1 round-trip runner.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any
import pandas as pd
try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

ROLES = ["decision","action","resource","actor","recipient_or_grantee","purpose","condition","constraint_or_exception","temporal_scope","privacy_identifiability","choice_structure","lifecycle_effect","risk_benefit_or_results","residual_important_content","provenance"]
CORE = {"decision","action","resource","actor","recipient_or_grantee","purpose","condition","constraint_or_exception","temporal_scope","privacy_identifiability","provenance"}
DESC = {
"decision":"Provision-level rule type: permit, deny, obligation, mixed, unclear.",
"action":"Governed action: collect, use, store, share, disclose, access, analyze, contact, withdraw, destroy.",
"resource":"Governed object/resource: data, health information, record, specimen, tissue, blood, DNA/genomic data, image, contact information.",
"actor":"Party who acts or is authorized/required to act.",
"recipient_or_grantee":"Party receiving data, specimens, results, access, or permission.",
"purpose":"Purpose/use context: current study, future research, disease-specific research, commercial development, clinical care.",
"condition":"Conditional trigger or prerequisite.",
"constraint_or_exception":"Restriction, exception, limitation, prohibition, or boundary.",
"temporal_scope":"Time or duration: future, after withdrawal, at any time, years, indefinite, ongoing.",
"privacy_identifiability":"Identifiability/privacy state: identifiable, coded, de-identified, anonymous, confidential.",
"choice_structure":"Choice architecture: optional, yes/no, decline, consent, join, withdrawal right.",
"lifecycle_effect":"Operational effect over time: retain, destroy, stop future use, continue prior use, return/no return results.",
"risk_benefit_or_results":"Risk, benefit, incidental finding, or result-return content.",
"residual_important_content":"Short meaning-critical content not captured elsewhere.",
"provenance":"Source/evidence metadata for audit."}
KW = {
"action":"action verb operation collect use store share disclose access analyze contact withdraw destroy return",
"resource":"resource asset object data information record specimen sample tissue blood dna genetic genomic image audio video contact",
"actor":"actor agent party assigner performer researcher doctor investigator institution team",
"recipient_or_grantee":"recipient grantee target assignee receiver access shared available",
"purpose":"purpose research study future commercial clinical care objective disease",
"condition":"condition if when unless approval governance irb law precondition",
"constraint_or_exception":"constraint restriction exception limitation limited only except not without prohibit",
"temporal_scope":"time temporal duration future after before during until year month day ongoing long-term",
"privacy_identifiability":"privacy identifiable identifier identified de-identified coded anonymous confidential name",
"choice_structure":"choice consent agree decline yes no optional join participate withdraw",
"lifecycle_effect":"retain destroy delete withdrawal effect continue return result",
"risk_benefit_or_results":"risk benefit harm result finding incidental",
"decision":"decision permission prohibition permit deny type",
"residual_important_content":"residual unmatched other",
"provenance":"provenance evidence source span"}

def norm(x:Any)->str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return " ".join(str(x).split())

def jlist(x:Any)->list[Any]:
    try:
        y=json.loads(norm(x)); return y if isinstance(y,list) else []
    except Exception: return []

def blob(r:pd.Series)->str:
    bits=[norm(r.get(c,"")) for c in ["union_element_id","source_model","source_element_id","source_element_label","source_element_definition","top_original_cue_groups_json"]]
    bits += [str(v) for v in jlist(r.get("top_span_examples_json",""))[:6]]
    bits += [str(v) for v in jlist(r.get("top_original_cue_groups_json",""))[:6]]
    return " ".join(bits).lower()

def rscore(text:str, role:str)->int:
    return sum((2 if " " in k and k in text else len(re.findall(rf"\b{re.escape(k)}",text))) for k in KW[role].split())

def assign(r:pd.Series)->tuple[str,str,int]:
    uid=norm(r.get("union_element_id")); tail=uid.split("::")[-1].lower(); text=blob(r)
    ov=[("action","action verb operation"),("resource","asset resource object specimen sample data information"),("actor","actor agent assigner performer"),("recipient_or_grantee","recipient grantee assignee target"),("purpose","purpose research disease"),("privacy_identifiability","identifier identifiable coded anonymous privacy confidential"),("temporal_scope","duration temporal time period"),("condition","condition approval governance"),("constraint_or_exception","prohibition exception restriction limitation"),("risk_benefit_or_results","result finding risk benefit"),("choice_structure","consent choice withdraw"),("lifecycle_effect","destroy retain withdrawal")]
    for role,words in ov:
        hits=[w for w in words.split() if w in tail or w in text]
        if hits: return role,"override:"+",".join(hits[:3]),99
    scores={role:rscore(text,role) for role in ROLES}; scores["provenance"]=min(scores["provenance"],1); scores["residual_important_content"]=min(scores["residual_important_content"],1)
    role=max(scores,key=lambda k:(scores[k],-ROLES.index(k))); sc=scores[role]
    return (role if sc>0 else "residual_important_content"),("keyword" if sc>0 else "unmatched"),int(sc)

def fnum(r:pd.Series,c:str)->float:
    try: return float(r.get(c,0) or 0)
    except Exception: return 0.0

def weight(r:pd.Series)->float:
    return fnum(r,"n_source_sentences")*(1+.15*fnum(r,"n_llms"))*max(.2,fnum(r,"mean_classifier_preservation_score"))*max(.2,fnum(r,"mean_cue_group_recall"))*max(.2,fnum(r,"mean_content_token_recall"))

def summarize(a:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for role,g in a.groupby("candidate_role"):
        top=g.sort_values("evidence_weight",ascending=False).head(10)
        rows.append({"candidate_role":role,"description":DESC[role],"recommended_core":role in CORE,"n_source_elements":len(g),"n_source_sentences_sum":pd.to_numeric(g.get("n_source_sentences"),errors="coerce").sum(),"mean_classifier_preservation_score":pd.to_numeric(g.get("mean_classifier_preservation_score"),errors="coerce").mean(),"mean_content_token_recall":pd.to_numeric(g.get("mean_content_token_recall"),errors="coerce").mean(),"mean_cue_group_recall":pd.to_numeric(g.get("mean_cue_group_recall"),errors="coerce").mean(),"evidence_weight_sum":pd.to_numeric(g.get("evidence_weight"),errors="coerce").sum(),"top_source_elements_json":json.dumps(top[["union_element_id"]].to_dict("records"),ensure_ascii=False)})
    return pd.DataFrame(rows).sort_values(["recommended_core","evidence_weight_sum"],ascending=[False,False])

def role_cooc(a:pd.DataFrame,pairs:pd.DataFrame)->pd.DataFrame:
    if pairs.empty: return pd.DataFrame()
    m=dict(zip(a.union_element_id,a.candidate_role)); rows=[]
    for _,p in pairs.iterrows():
        x,y=norm(p.get("union_element_id_a")),norm(p.get("union_element_id_b"))
        if x in m and y in m:
            r1,r2=sorted([m[x],m[y]]); rows.append({"role_a":r1,"role_b":r2,"n_source_sentences":p.get("n_source_sentences",0),"n_cooccurrences":p.get("n_cooccurrences",0)})
    raw=pd.DataFrame(rows)
    if raw.empty: return raw
    return raw.groupby(["role_a","role_b"],as_index=False).agg(n_source_sentences_max=("n_source_sentences","max"),n_cooccurrences_sum=("n_cooccurrences","sum")).sort_values(["n_source_sentences_max","n_cooccurrences_sum"],ascending=[False,False])

def make_schema(s:pd.DataFrame,a:pd.DataFrame)->dict[str,Any]:
    fields=[]
    for role in ROLES:
        if role not in set(s.candidate_role): continue
        g=a[a.candidate_role.eq(role)].sort_values("evidence_weight",ascending=False)
        fields.append({"name":role,"status":"core" if role in CORE else "extension","description":DESC[role],"value_type":"normalized_value_with_evidence","allow_multiple":role not in {"decision","privacy_identifiability"},"source_element_support":g.union_element_id.head(12).tolist()})
    return {"meta_model_id":"reduced_consent_metamodel_v1_candidate","version":"0.1","status":"candidate_data_driven_requires_audit","design_goal":"compact functional consent representation for LLM round-trip meaning preservation","evaluation_variants":{"compact_evidence":"short evidence phrases; no full-clause copying","permissive_evidence":"same reduced schema but permits longer evidence phrases when needed"},"fields":fields,"provision_structure":{"primary_bundle":["decision","action","resource","actor","recipient_or_grantee","purpose"],"modifier_bundle":["condition","constraint_or_exception","temporal_scope","privacy_identifiability","choice_structure","lifecycle_effect"],"audit_bundle":["residual_important_content","provenance"]}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--analysis_dir",required=True); ap.add_argument("--inventory_csv"); ap.add_argument("--output_dir",required=True); args=ap.parse_args(); adir,out=Path(args.analysis_dir),Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    ev=pd.read_csv(adir/"source_element_evidence_summary.csv").fillna("")
    if args.inventory_csv:
        inv=pd.read_csv(args.inventory_csv).fillna("")
        if "union_element_id" in inv.columns:
            ev=ev.drop(columns=[c for c in inv.columns if c in ev.columns and c!="union_element_id"],errors="ignore").merge(inv,on="union_element_id",how="left").fillna("")
    pairs=pd.read_csv(adir/"source_element_cooccurrence_pairs.csv").fillna("") if (adir/"source_element_cooccurrence_pairs.csv").exists() else pd.DataFrame()
    rows=[]
    for _,r in ev.iterrows():
        role,reason,sc=assign(r); d=r.to_dict(); d.update(candidate_role=role,role_assignment_reason=reason,role_assignment_score=sc,evidence_weight=weight(r),recommended_core=role in CORE); rows.append(d)
    a=pd.DataFrame(rows); s=summarize(a); c=role_cooc(a,pairs); schema=make_schema(s,a)
    a.to_csv(out/"candidate_role_assignments.csv",index=False); s.to_csv(out/"candidate_role_evidence_summary.csv",index=False); c.to_csv(out/"candidate_role_cooccurrence_summary.csv",index=False)
    (out/"reduced_metamodel_v1_candidate.yaml").write_text(yaml.safe_dump(schema,sort_keys=False,allow_unicode=True)); (out/"reduced_metamodel_v1_candidate.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    md=["# Candidate Reduced V1 Consent Meta-Model","","Compact and permissive evidence variants use the same reduced schema.","","## Candidate fields"]
    for f in schema["fields"]: md += [f"### {f['name']} ({f['status']})","",f["description"],"","Source support: "+", ".join(f.get("source_element_support",[])[:8]),""]
    md += ["## Role evidence summary","",s.to_markdown(index=False),""]; (out/"reduced_metamodel_v1_candidate.md").write_text("\n".join(md)); print(f"Wrote candidate V1 meta-model evidence to {out}")
if __name__=="__main__": main()
