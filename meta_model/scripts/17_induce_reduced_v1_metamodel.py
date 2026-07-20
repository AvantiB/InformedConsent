#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model from the expert-validated corpus.

This is the main derivation path. Expert-preserved round trips are positive
functional evidence; expert-failed rows are boundary evidence that weakens
merge evidence. New LLM runs are validation/stress-test data, not induction data.

Input is usually produced by 12_build_expert_roundtrip_corpus.py.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import pandas as pd
try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

TEXT_COLS=["canonical_full_text","full_text_original","original_sentence","source_text","full_text","sentence","text"]
ID_COLS=["source_id","ID","id","roundtrip_id","sentence_id","source_sentence_id"]
LLM_COLS=["llm","model","llm_name"]
INFO_COLS=["information_model","info_model","source_model","model_family"]
ANN_COLS=["annotations_json","annotations_serialized","annotations_raw","annotations_combined","forward_mapping","mapping","annotation","forward_raw"]
LABEL_COLS=["meaning_preserved","human_meaning_preserved","expert_meaning_preserved","preserved","label","human_label"]
ELIG_COLS=["eligible_element_analysis","eligible"]
ELEM_KEYS=["union_element_id","element_id","source_element_id","label","source_element_label","field","role","concept","class","property","duo_label","ico_label","odrl_label","fhir_label","element_label"]
SPAN_KEYS=["span_text","evidence_span_text","evidence_text","text_span","phrase","annotated_text","text"]
DECISION_TAILS={"DUO.decision","ICO.decision","Rule_TestSentence","Consent.provision.type","roundtrip_decision"}
DECISION_EXACT={"DUO::DUO.decision","ICO::ICO.decision","ODRL::Rule_TestSentence","FHIR_Consent::Consent.provision.type","FHIR::Consent.provision.type","ROUNDTRIP::roundtrip_decision"}
ROLE_KEYWORDS={
 "action":"action verb operation collect use store share disclose access analyze contact withdraw destroy return provide send release",
 "resource":"resource asset object data information record specimen sample tissue blood dna genetic genomic image audio video contact identifier",
 "actor_or_party":"actor agent party grantor grantee assigner assignee performer researcher doctor investigator institution team participant subject",
 "purpose":"purpose research study future commercial clinical care objective disease cancer genetic genomic",
 "condition_or_governance":"condition if when unless approval governance irb law precondition require allowed review committee",
 "constraint_or_prohibition":"constraint restriction exception limitation limited only except not without prohibit prohibition deny refuse",
 "temporal_scope":"time temporal duration future after before during until year month day ongoing long-term indefinite period",
 "privacy_identifiability":"privacy identifiable identifier identified de-identified deidentified coded anonymous confidential name contact",
 "choice_or_consent":"choice consent agree decline yes no optional join participate withdraw permission decision",
 "lifecycle_or_results":"retain destroy delete withdrawal effect continue return result finding incidental benefit risk harm",
 "residual_or_other":"residual unmatched other note rationale"}
CORE_HINTS={"action","resource","actor_or_party","purpose","condition_or_governance","constraint_or_prohibition","temporal_scope","privacy_identifiability","choice_or_consent"}
POS_DEFAULT="1,true,yes,y,preserved,meaning preserved,pass,passed,positive,match,matches"
NEG_VALUES={"0","false","no","n","negative","not preserved","not_preserved","fail","failed","mismatch","does not match"}
CODE_RE=re.compile(r"\b[A-Za-z][A-Za-z0-9_-]*:\d{3,}\b")
BRACKET_RE=re.compile(r"(?P<span>[^\[]*?)\s*\[(?P<label>[^\]]+)\](?:\s*\((?P<decision>[^)]*)\))?")

def norm(x:Any)->str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return " ".join(str(x).replace("\r","\n").split())

def pick(df:pd.DataFrame, cols:list[str], required:bool=False)->str|None:
    lower={str(c).strip().lower():c for c in df.columns}
    for c in cols:
        if c.lower() in lower: return lower[c.lower()]
    if required: raise ValueError(f"Missing required column from {cols}; available={list(df.columns)}")
    return None

def truthy(x:Any,pos:set[str])->bool|None:
    v=norm(x).lower()
    if v in pos: return True
    if v in NEG_VALUES: return False
    try:
        f=float(v); return f>=0.5
    except Exception: return None

def sid(text:str)->str: return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
def toks(x:str)->set[str]: return set(re.findall(r"[a-z0-9]+",x.lower()))
def jac(a:set[str],b:set[str])->float: return len(a&b)/max(1,len(a|b))
def tail(uid:str)->str: return str(uid).split("::")[-1]
def src_model(uid:str,info:str)->str: return uid.split("::",1)[0] if "::" in uid else (info or "unknown")
def is_decision(uid:str)->bool: return norm(uid) in DECISION_EXACT or tail(norm(uid)) in DECISION_TAILS

def parse_jsonish(x:Any)->Any|None:
    s=norm(x)
    if not s: return None
    if s.startswith("```"):
        s=re.sub(r"^```(?:json|csv|yaml)?\s*","",s,flags=re.I); s=re.sub(r"\s*```$","",s)
    try: return json.loads(s)
    except Exception: pass
    for l,r in [("{","}"),("[","]")]:
        a,b=s.find(l),s.rfind(r)
        if a>=0 and b>a:
            try: return json.loads(s[a:b+1])
            except Exception: pass
    return None

def walk(o:Any):
    if isinstance(o,dict):
        yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o,list):
        for v in o: yield from walk(v)

def first(d:dict[str,Any], keys:list[str])->str:
    for k in keys:
        if norm(d.get(k,"")): return norm(d.get(k,""))
    return ""

def canon(raw:str, info:str)->str:
    raw=norm(raw).replace("***","").strip()
    if not raw or raw.upper()=="NA": return ""
    if "::" in raw: return raw
    m=CODE_RE.search(raw); core=m.group(0) if m else raw.split()[0]
    return f"{info}::{core}" if info else core

def load_inv(path:str|None)->dict[str,dict[str,str]]:
    if not path or not Path(path).exists(): return {}
    df=pd.read_csv(path).fillna(""); out={}
    for _,r in df.iterrows():
        uid=norm(r.get("union_element_id",""))
        if not uid: continue
        rec={c:norm(r.get(c,"")) for c in df.columns}; out[uid]=rec
        sm,sid0,lab=norm(r.get("source_model","")),norm(r.get("source_element_id","")),norm(r.get("source_element_label",""))
        if sm and sid0: out.setdefault(f"{sm}::{sid0}",rec)
        if sm and lab: out.setdefault(f"{sm}::{lab.split()[0]}",rec)
    return out

def annotations_from_cell(raw:Any, info:str)->list[dict[str,Any]]:
    obj=parse_jsonish(raw); anns=[]
    if isinstance(obj,dict):
        if isinstance(obj.get("annotations"),list): anns += [a for a in obj["annotations"] if isinstance(a,dict)]
        anns += [d for d in walk(obj) if isinstance(d,dict) and first(d,ELEM_KEYS)]
    elif isinstance(obj,list): anns += [a for a in obj if isinstance(a,dict)]
    if not anns:
        text=str(raw or "").replace("\r","\n")
        for m in BRACKET_RE.finditer(text):
            label=norm(m.group("label")); uid=canon(label,info)
            if uid:
                anns.append({"union_element_id":uid,"source_element_label":label,"span_text":norm(m.group("span")),"decision_value":norm(m.groupdict().get("decision","")),"raw_annotation_text":norm(m.group(0))})
    ded=[]; seen=set()
    for a in anns:
        k=json.dumps(a,sort_keys=True,default=str)
        if k not in seen: seen.add(k); ded.append(a)
    return ded

def build_mentions(df:pd.DataFrame,args:argparse.Namespace)->tuple[pd.DataFrame,pd.DataFrame]:
    pos={x.strip().lower() for x in args.positive_label_values.split(",") if x.strip()}
    tc=pick(df,TEXT_COLS,True); ic=pick(df,ID_COLS); lc=pick(df,LLM_COLS); mc=pick(df,INFO_COLS); ac=pick(df,ANN_COLS,True)
    yc=args.label_col or pick(df,LABEL_COLS,True); ec=pick(df,ELIG_COLS)
    span_rows=[]; dec_rows=[]
    for idx,r in df.iterrows():
        if ec and norm(r.get(ec,"")).lower() in {"0","false","no","n"}: continue
        ok=truthy(r.get(yc,""),pos)
        if ok is None: continue
        text=norm(r.get(tc,"")); source_id=norm(r.get(ic,"")) if ic else sid(text)
        llm=norm(r.get(lc,"")) if lc else "unknown"; info=norm(r.get(mc,"")) if mc else "unknown"
        context_id=f"{source_id}|{info}|{llm}|{idx}"
        for j,a in enumerate(annotations_from_cell(r.get(ac,""),info)):
            raw=first(a,ELEM_KEYS); uid=canon(raw,info)
            if not uid: continue
            rec={"row_index":idx,"context_id":context_id,"source_id":source_id,"source_text":text,"llm":llm,"information_model":info,"expert_meaning_preserved":bool(ok),"union_element_id":uid,"source_model_inferred":src_model(uid,info),"span_text":first(a,SPAN_KEYS),"raw_element_label":raw,"annotation_index":j}
            (dec_rows if is_decision(uid) else span_rows).append(rec)
        if "roundtrip_decision" in df.columns and norm(r.get("roundtrip_decision","")):
            dec_rows.append({"row_index":idx,"context_id":context_id,"source_id":source_id,"source_text":text,"llm":llm,"information_model":info,"expert_meaning_preserved":bool(ok),"union_element_id":"ROUNDTRIP::roundtrip_decision","source_model_inferred":"ROUNDTRIP","span_text":norm(r.get("roundtrip_decision","")),"raw_element_label":"roundtrip_decision","annotation_index":-1})
    return pd.DataFrame(span_rows),pd.DataFrame(dec_rows)

def profiles(mentions:pd.DataFrame, inv:dict[str,dict[str,str]])->pd.DataFrame:
    if mentions.empty: return pd.DataFrame()
    rows=[]
    for uid,g in mentions.groupby("union_element_id"):
        p=g[g.expert_meaning_preserved.astype(bool)]; n=g[~g.expert_meaning_preserved.astype(bool)]; meta=inv.get(uid,{})
        pc=p.groupby("context_id").size() if not p.empty else pd.Series(dtype=int)
        rows.append({"union_element_id":uid,"source_model":meta.get("source_model",src_model(uid,"")),"source_element_id":meta.get("source_element_id",tail(uid)),"source_element_label":meta.get("source_element_label",tail(uid)),"source_element_definition":meta.get("source_element_definition",""),"n_raw_mentions":len(g),"n_positive_mentions":len(p),"n_negative_mentions":len(n),"n_positive_contexts":p.context_id.nunique(),"n_negative_contexts":n.context_id.nunique(),"n_positive_source_sentences":p.source_id.nunique(),"n_negative_source_sentences":n.source_id.nunique(),"n_positive_llms":p.llm.nunique(),"n_positive_information_models":p.information_model.nunique(),"expert_positive_rate":len(p)/max(1,len(g)),"mean_mentions_per_positive_context":float(pc.mean()) if len(pc) else 0.0,"max_mentions_in_single_positive_context":int(pc.max()) if len(pc) else 0,"top_span_examples_json":json.dumps([x for x,_ in Counter(g.span_text.astype(str)).most_common(10) if x],ensure_ascii=False),"top_positive_span_examples_json":json.dumps([x for x,_ in Counter(p.span_text.astype(str)).most_common(10) if x],ensure_ascii=False)})
    out=pd.DataFrame(rows)
    out["profile_text"]=out.apply(lambda r:" ".join([norm(r.get(c,"")) for c in ["union_element_id","source_model","source_element_id","source_element_label","source_element_definition","top_positive_span_examples_json"]]).lower(),axis=1)
    return out.sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])

def edge_table(prof:pd.DataFrame, mentions:pd.DataFrame, min_w:float)->pd.DataFrame:
    if prof.empty or mentions.empty: return pd.DataFrame()
    pmap={r.union_element_id:r for _,r in prof.iterrows()}; bucket=defaultdict(lambda:{"pos":set(),"neg":set(),"span":set(),"raw_pos":0})
    for (cid,ok),g in mentions.groupby(["context_id","expert_meaning_preserved"],dropna=False):
        counts=Counter(g.union_element_id.astype(str)); ids=sorted(counts)
        for a,b in itertools.combinations(ids,2):
            rec=bucket[tuple(sorted((a,b)))]
            rec["pos" if bool(ok) else "neg"].add(str(cid))
            if bool(ok): rec["raw_pos"] += min(counts[a],counts[b])
    pm=mentions[mentions.expert_meaning_preserved.astype(bool)]
    for (cid,span),g in pm.groupby(["context_id","span_text"],dropna=False):
        if not norm(span): continue
        for a,b in itertools.combinations(sorted(set(g.union_element_id.astype(str))),2): bucket[tuple(sorted((a,b)))]["span"].add(str(cid))
    rows=[]
    for a,b in itertools.combinations(prof.union_element_id.astype(str),2):
        rec=bucket.get(tuple(sorted((a,b))),{"pos":set(),"neg":set(),"span":set(),"raw_pos":0}); pa,pb=pmap[a],pmap[b]
        sim=jac(toks(str(pa.profile_text)),toks(str(pb.profile_text)))
        cross=1.0 if norm(pa.source_model) and norm(pb.source_model) and norm(pa.source_model)!=norm(pb.source_model) else 0.0
        posn,negn,spann,raw=len(rec["pos"]),len(rec["neg"]),len(rec["span"]),rec["raw_pos"]
        intensity=math.log1p(raw)/math.log1p(20) if raw else 0.0
        w=.40*min(1,posn/8)+.20*min(1,spann/3)+.20*sim+.10*cross+.10*min(1,intensity)-.30*min(1,negn/8)
        if w>=min_w: rows.append({"union_element_id_a":a,"union_element_id_b":b,"edge_weight":float(w),"positive_cooccurrence_contexts":posn,"negative_cooccurrence_contexts":negn,"same_span_positive_contexts":spann,"raw_joint_positive_mention_intensity":raw,"profile_similarity":float(sim),"cross_source_model_bonus":cross,"edge_interpretation":"same_function_or_merge_candidate" if spann or sim>.25 else "bundle_or_complement_candidate"})
    return pd.DataFrame(rows).sort_values("edge_weight",ascending=False) if rows else pd.DataFrame()

def cluster(prof:pd.DataFrame, edges:pd.DataFrame)->pd.DataFrame:
    if prof.empty: return prof.copy()
    ids=prof.union_element_id.astype(str).tolist(); assign={}; method="singleton"
    try:
        import networkx as nx
        G=nx.Graph(); G.add_nodes_from(ids)
        for _,e in edges.iterrows(): G.add_edge(str(e.union_element_id_a),str(e.union_element_id_b),weight=float(e.edge_weight))
        if G.number_of_edges():
            for i,c in enumerate(sorted(nx.algorithms.community.greedy_modularity_communities(G,weight="weight"),key=lambda z:(-len(z),sorted(z)[0]))):
                for uid in c: assign[uid]=f"C{i+1:02d}"
            method="networkx_greedy_modularity"
    except Exception: pass
    if not assign:
        parent={i:i for i in ids}
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra
        for _,e in edges.iterrows(): union(str(e.union_element_id_a),str(e.union_element_id_b))
        comps=defaultdict(list)
        for uid in ids: comps[find(uid)].append(uid)
        for i,c in enumerate(sorted(comps.values(),key=lambda z:(-len(z),sorted(z)[0]))):
            for uid in c: assign[uid]=f"C{i+1:02d}"
        method="threshold_connected_components"
    out=prof.copy(); out["cluster_id"]=out.union_element_id.map(assign); out["clustering_method"]=method; return out

def label_cluster(g:pd.DataFrame)->tuple[str,str]:
    text=" ".join(g.profile_text.astype(str)).lower(); ts=toks(text); scores={r:len(ts&toks(words)) for r,words in ROLE_KEYWORDS.items()}; role=max(scores,key=scores.get)
    return (role,",".join(sorted(ts&toks(ROLE_KEYWORDS[role]))[:10])) if scores[role] else ("residual_or_other","no keyword label evidence")

def cluster_summary(clust:pd.DataFrame, min_core:int)->pd.DataFrame:
    if clust.empty: return pd.DataFrame()
    rows=[]; rank={"core_shared":0,"context_module":1,"failure_boundary_audit":2,"audit_or_extension":3}
    for cid,g in clust.groupby("cluster_id"):
        lab,ev=label_cluster(g); sms=sorted(set(g.source_model.astype(str)))
        ps=pd.to_numeric(g.n_positive_source_sentences,errors="coerce").max(); ns=pd.to_numeric(g.n_negative_source_sentences,errors="coerce").max(); pll=pd.to_numeric(g.n_positive_llms,errors="coerce").max(); rate=pd.to_numeric(g.expert_positive_rate,errors="coerce").mean()
        cat="core_shared" if ps>=min_core and len(sms)>=2 and pll>=2 and lab in CORE_HINTS and rate>=.5 else "context_module" if ps>=max(3,min_core//3) and pll>=2 and rate>=.5 else "audit_or_extension"
        if ns>ps and cat!="core_shared": cat="failure_boundary_audit"
        top=g.sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])
        rows.append({"cluster_id":cid,"candidate_field_name":lab,"field_label_evidence":ev,"selection_category":cat,"n_source_elements":len(g),"n_source_models":len(sms),"source_models_json":json.dumps(sms,ensure_ascii=False),"n_positive_source_sentences_max":ps,"n_negative_source_sentences_max":ns,"n_positive_llms_max":pll,"mean_expert_positive_rate":rate,"top_source_elements_json":json.dumps(top.union_element_id.head(15).tolist(),ensure_ascii=False),"top_positive_span_examples_json":json.dumps([x for v in top.top_positive_span_examples_json.head(8) for x in json.loads(v or "[]")[:2]][:12],ensure_ascii=False)})
    out=pd.DataFrame(rows); out["_rank"]=out.selection_category.map(rank).fillna(9); return out.sort_values(["_rank","n_positive_source_sentences_max"],ascending=[True,False]).drop(columns=["_rank"])

def decisions_summary(dec:pd.DataFrame)->pd.DataFrame:
    if dec.empty: return pd.DataFrame()
    rows=[]
    for uid,g in dec.groupby("union_element_id"):
        p=g[g.expert_meaning_preserved.astype(bool)]
        rows.append({"sentence_level_element_id":uid,"n_mentions":len(g),"n_positive_mentions":len(p),"n_positive_source_sentences":p.source_id.nunique(),"n_positive_llms":p.llm.nunique(),"top_values_or_spans_json":json.dumps([x for x,_ in Counter(g.span_text.astype(str)).most_common(8) if x],ensure_ascii=False)})
    return pd.DataFrame(rows).sort_values("n_positive_source_sentences",ascending=False)

def make_schema(cs:pd.DataFrame, clust:pd.DataFrame, ds:pd.DataFrame)->dict[str,Any]:
    fields=[{"name":"decision","status":"core","source":"sentence_level_decision_fields","description":"Provision rule type derived from sentence-level decision fields and/or roundtrip_decision.","values":["permit","deny","obligation","mixed","unclear"],"selection_evidence":{"n_decision_elements":int(len(ds)),"decision_element_support":ds.get("sentence_level_element_id",pd.Series(dtype=str)).head(10).tolist()}}]
    used=Counter(); keep=cs[cs.selection_category.isin(["core_shared","context_module"])] if not cs.empty else pd.DataFrame()
    for _,c in keep.iterrows():
        base=str(c.candidate_field_name); used[base]+=1; name=base if used[base]==1 else f"{base}_{used[base]}"; support=clust[clust.cluster_id.eq(c.cluster_id)].sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])
        fields.append({"name":name,"cluster_id":c.cluster_id,"status":"core" if c.selection_category=="core_shared" else "context_module","description":f"Expert-positive evidence cluster labeled as {base}; selected as {c.selection_category}.","value_type":"normalized_value_with_evidence","allow_multiple":True,"selection_evidence":{"n_source_elements":int(c.n_source_elements),"n_source_models":int(c.n_source_models),"n_positive_source_sentences_max":float(c.n_positive_source_sentences_max),"n_positive_llms_max":float(c.n_positive_llms_max),"mean_expert_positive_rate":float(c.mean_expert_positive_rate),"label_evidence":c.field_label_evidence},"source_element_support":support.union_element_id.head(12).tolist(),"positive_span_examples":[x for v in support.top_positive_span_examples_json.head(6) for x in json.loads(v or "[]")[:2]][:8]})
    fields += [{"name":"residual_important_content","status":"audit","description":"Short meaning-critical content not captured by selected fields.","value_type":"short_evidence_phrase"},{"name":"provenance","status":"audit","description":"Source sentence, evidence spans, selected clusters, and source elements used for audit.","value_type":"audit_metadata"}]
    return {"meta_model_id":"reduced_consent_metamodel_v1_expert_induced_candidate","version":"0.1","status":"expert_validated_derivation_candidate_requires_audit_and_validation","design_goal":"Reduced functional consent representation induced from expert-preserved round trips and penalized/flagged by expert-failed round trips.","selection_method":{"derivation_corpus":"original expert-evaluated round-trip dataset","positive_evidence":"rows judged meaning-preserving by experts","boundary_evidence":"rows judged not meaning-preserving by experts","node_input":"span-level source-model elements; sentence-level decision fields are separated","edge_evidence":["expert-positive co-occurrence contexts","same-span expert-positive use","label/definition/span similarity","cross-source-model support","raw repeated mentions retained as salience evidence"],"edge_penalty":"expert-failed co-occurrence weakens merge evidence","clustering":"weighted graph community detection with connected-component fallback","field_selection":"core/shared clusters require expert-positive sentence coverage, multi-LLM support, cross-source support, positive rate, and functional label evidence"},"evaluation_variants":{"compact_evidence":"same schema; short evidence phrases; no full-clause copying","permissive_evidence":"same schema; longer evidence allowed when needed"},"fields":fields,"provision_structure":{"rule_type":"decision","selected_cluster_fields":[f["name"] for f in fields if f.get("cluster_id")],"audit_fields":["residual_important_content","provenance"]}}

def write_method(out:Path,args:argparse.Namespace)->None:
    txt=f"""# Expert-validated reduced V1 induction methodology

The reduced V1 candidate is induced from the original expert-evaluated round-trip dataset.

Rows judged meaning-preserving are positive functional evidence. Rows judged not meaning-preserving are boundary evidence that weakens proposed merges and flags unsafe simplifications.

Processing steps:
1. Parse clean expert round-trip rows into raw source-element mentions.
2. Separate sentence-level decision fields from span-level graph nodes.
3. Preserve raw repeated mentions as salience/frequency evidence.
4. Build context-level co-occurrence edges so one row does not create Cartesian-product edge inflation.
5. Add same-span evidence, profile similarity, and cross-source-model support.
6. Penalize edges seen mainly in expert-failed contexts.
7. Cluster the weighted graph and select core/context/audit fields.

Parameters: min_edge_weight={args.min_edge_weight}, min_core_positive_sentences={args.min_core_positive_sentences}
"""
    (out/"expert_validated_induction_methodology.md").write_text(txt)

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--expert_roundtrips_csv",required=True); ap.add_argument("--output_dir",required=True); ap.add_argument("--inventory_csv"); ap.add_argument("--label_col"); ap.add_argument("--positive_label_values",default=POS_DEFAULT); ap.add_argument("--min_edge_weight",type=float,default=.22); ap.add_argument("--min_core_positive_sentences",type=int,default=15); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.expert_roundtrips_csv).fillna(""); inv=load_inv(args.inventory_csv)
    ment,dec=build_mentions(df,args); prof=profiles(ment,inv); edges=edge_table(prof,ment,args.min_edge_weight); clust=cluster(prof,edges); cs=cluster_summary(clust,args.min_core_positive_sentences); ds=decisions_summary(dec); schema=make_schema(cs,clust,ds)
    ment.to_csv(out/"expert_element_mentions_long.csv",index=False); dec.to_csv(out/"expert_sentence_level_decision_mentions_long.csv",index=False); prof.to_csv(out/"expert_element_profiles.csv",index=False); edges.to_csv(out/"expert_element_relationship_edges.csv",index=False); clust.to_csv(out/"expert_element_clusters.csv",index=False); cs.to_csv(out/"expert_cluster_evidence_summary.csv",index=False); ds.to_csv(out/"expert_sentence_level_decision_summary.csv",index=False)
    (out/"reduced_metamodel_v1_candidate.yaml").write_text(yaml.safe_dump(schema,sort_keys=False,allow_unicode=True)); (out/"reduced_metamodel_v1_candidate.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    md=["# Expert-induced Candidate Reduced V1 Consent Meta-Model","","This candidate is induced from expert-preserved round trips and uses expert-failed rows as boundary evidence.","","## Selected fields"]
    for f in schema["fields"]:
        md += [f"### {f['name']} ({f.get('status','')})","",f.get("description",""),""]
        if f.get("selection_evidence"): md += ["Selection evidence:","","```json",json.dumps(f["selection_evidence"],indent=2),"```",""]
        if f.get("source_element_support"): md += ["Source-element support: "+", ".join(f["source_element_support"][:10]),""]
        if f.get("positive_span_examples"): md += ["Positive span examples: "+"; ".join(f["positive_span_examples"][:6]),""]
    if not cs.empty: md += ["## Cluster evidence summary","",cs.to_markdown(index=False),""]
    (out/"reduced_metamodel_v1_candidate.md").write_text("\n".join(md)); write_method(out,args); print(f"Wrote expert-validated V1 induction outputs to {out}")

if __name__=="__main__": main()
