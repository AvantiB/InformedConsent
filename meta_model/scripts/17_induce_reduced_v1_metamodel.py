#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model from the original expert-validated corpus.

This is the main derivation path. Expert-preserved round trips are treated as
functionally validated positive evidence; expert-failed round trips are boundary
evidence that weakens merge evidence and flags unsafe simplifications. Newer LLM
runs are validation/stress-test data, not the primary induction corpus.
"""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import pandas as pd
try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

TEXT_COLS=["canonical_full_text","full_text_original","original_sentence","source_text","full_text","sentence","text"]
ID_COLS=["roundtrip_id","source_id","sentence_id","source_sentence_id","id"]
LLM_COLS=["llm","model","llm_name"]
INFO_COLS=["information_model","info_model","source_model","model_family"]
FWD_COLS=["forward_mapping","annotations_serialized","annotations_json","mapping","annotation","forward_raw"]
LABEL_COLS=["meaning_preserved","human_meaning_preserved","expert_meaning_preserved","preserved","label","human_label"]
ELEM_KEYS=["union_element_id","element_id","source_element_id","label","source_element_label","field","role","concept","class","property","duo_label","ico_label","odrl_label","fhir_label"]
SPAN_KEYS=["span_text","evidence_span_text","evidence_text","text_span","phrase","annotated_text","text"]
DECISION_TAILS={"DUO.decision","ICO.decision","Rule_TestSentence","Consent.provision.type"}
DECISION_EXACT={"DUO.decision","ICO.decision","Rule_TestSentence","Consent.provision.type","DUO::DUO.decision","ICO::ICO.decision","ODRL::Rule_TestSentence","FHIR_Consent::Consent.provision.type","FHIR::Consent.provision.type"}
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

def norm(x:Any)->str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return " ".join(str(x).split())

def sid(text:str)->str: return hashlib.sha1(text.encode()).hexdigest()[:12]
def toks(x:str)->set[str]: return set(re.findall(r"[a-z0-9]+",x.lower()))
def jac(a:set[str],b:set[str])->float: return len(a&b)/max(1,len(a|b))
def tail(uid:str)->str: return str(uid).split("::")[-1]
def is_decision(uid:str)->bool: return norm(uid) in DECISION_EXACT or tail(norm(uid)) in DECISION_TAILS

def pick(df:pd.DataFrame, cols:list[str], required=False)->str|None:
    lower={c.lower():c for c in df.columns}
    for c in cols:
        if c.lower() in lower: return lower[c.lower()]
    if required: raise ValueError(f"Missing required column from {cols}; available={list(df.columns)}")
    return None

def truthy(x:Any,pos:set[str])->bool:
    v=norm(x).lower()
    if v in pos: return True
    if v in {"0","false","no","n","negative","not preserved","fail","failed"}: return False
    try: return float(v)>0.5
    except Exception: return False

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

def first(d:dict[str,Any],keys:list[str])->str:
    for k in keys:
        v=d.get(k)
        if norm(v): return norm(v)
    return ""

def canon(raw:str,info:str)->str:
    raw=norm(raw)
    if not raw: return ""
    return raw if "::" in raw else f"{info}::{raw}" if info else raw

def src_model(uid:str,info:str)->str: return uid.split("::",1)[0] if "::" in uid else (info or "unknown")

def csv_anns(x:Any)->list[dict[str,str]]:
    s=norm(x)
    if not s or "\n" not in s: return []
    try: rows=[[norm(c) for c in r] for r in csv.reader(s.splitlines()) if any(norm(c) for c in r)]
    except Exception: return []
    if not rows: return []
    head=[c.lower() for c in rows[0]]; has_head=any(h in head for h in ["span","span_text","annotation","label","duo_label","ico_label","odrl_label","fhir_label"])
    out=[]
    for r in (rows[1:] if has_head else rows):
        if has_head:
            d={head[i]:r[i] for i in range(min(len(head),len(r)))}; lab=first(d,ELEM_KEYS); sp=first(d,SPAN_KEYS)
        else:
            cells=[c for c in r if c]; lab=cells[-1] if cells else ""; sp=max(cells[:-1],key=len) if len(cells)>1 else ""
        if lab: out.append({"element_label":lab,"span_text":sp})
    return out

def extract_anns(raw:Any)->list[dict[str,Any]]:
    obj=parse_jsonish(raw); out=[]
    if isinstance(obj,dict):
        if isinstance(obj.get("annotations"),list): out += [a for a in obj["annotations"] if isinstance(a,dict)]
        out += [d for d in walk(obj) if isinstance(d,dict) and first(d,ELEM_KEYS)]
    elif isinstance(obj,list): out += [a for a in obj if isinstance(a,dict)]
    if not out: return csv_anns(raw)
    seen=set(); ded=[]
    for d in out:
        k=json.dumps(d,sort_keys=True,default=str)
        if k not in seen: seen.add(k); ded.append(d)
    return ded

def load_inv(path:str|None)->dict[str,dict[str,str]]:
    if not path or not Path(path).exists(): return {}
    df=pd.read_csv(path).fillna(""); look={}
    for _,r in df.iterrows():
        uid=norm(r.get("union_element_id",""))
        if not uid: continue
        rec={c:norm(r.get(c,"")) for c in df.columns}; look[uid]=rec
        sm,sid0=norm(r.get("source_model","")),norm(r.get("source_element_id",""))
        if sm and sid0: look.setdefault(f"{sm}::{sid0}",rec)
    return look

def build_mentions(df:pd.DataFrame,pos:set[str],label_col:str|None)->tuple[pd.DataFrame,pd.DataFrame]:
    tc=pick(df,TEXT_COLS,True); ic=pick(df,ID_COLS); lc=pick(df,LLM_COLS); mc=pick(df,INFO_COLS); fc=pick(df,FWD_COLS,True); yc=label_col or pick(df,LABEL_COLS,True)
    span,decision=[],[]
    for idx,r in df.iterrows():
        text=norm(r.get(tc,"")); source_id=norm(r.get(ic,"")) if ic else sid(text); llm=norm(r.get(lc,"")) if lc else ""; info=norm(r.get(mc,"")) if mc else ""; ok=truthy(r.get(yc,""),pos)
        for a in extract_anns(r.get(fc,"")):
            raw=first(a,ELEM_KEYS); uid=canon(raw,info)
            if not uid: continue
            rec={"row_index":idx,"source_id":source_id,"source_text":text,"llm":llm,"information_model":info,"expert_meaning_preserved":bool(ok),"union_element_id":uid,"source_model_inferred":src_model(uid,info),"span_text":first(a,SPAN_KEYS),"raw_element_label":raw}
            (decision if is_decision(uid) else span).append(rec)
    return pd.DataFrame(span),pd.DataFrame(decision)

def profiles(mentions:pd.DataFrame,inv:dict[str,dict[str,str]])->pd.DataFrame:
    rows=[]
    if mentions.empty: return pd.DataFrame()
    for uid,g in mentions.groupby("union_element_id"):
        p=g[g.expert_meaning_preserved.astype(bool)]; n=g[~g.expert_meaning_preserved.astype(bool)]; meta=inv.get(uid,{})
        spans=[s for s in g.span_text.astype(str) if s]; pspans=[s for s in p.span_text.astype(str) if s]
        rows.append({"union_element_id":uid,"source_model":meta.get("source_model",src_model(uid,"")),"source_element_id":meta.get("source_element_id",tail(uid)),"source_element_label":meta.get("source_element_label",tail(uid)),"source_element_definition":meta.get("source_element_definition",""),"n_mentions":len(g),"n_positive_mentions":len(p),"n_negative_mentions":len(n),"n_positive_source_sentences":p.source_id.nunique(),"n_negative_source_sentences":n.source_id.nunique(),"n_positive_llms":p.llm.nunique(),"n_negative_llms":n.llm.nunique(),"n_positive_information_models":p.information_model.nunique(),"expert_positive_rate":len(p)/max(1,len(g)),"top_span_examples_json":json.dumps([x for x,_ in Counter(spans).most_common(10)],ensure_ascii=False),"top_positive_span_examples_json":json.dumps([x for x,_ in Counter(pspans).most_common(10)],ensure_ascii=False)})
    out=pd.DataFrame(rows)
    out["profile_text"]=out.apply(lambda r:" ".join([norm(r.get(c,"")) for c in ["union_element_id","source_model","source_element_id","source_element_label","source_element_definition","top_positive_span_examples_json"]]).lower(),axis=1)
    return out.sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])

def edge_table(prof:pd.DataFrame,mentions:pd.DataFrame,min_w:float)->pd.DataFrame:
    ids=prof.union_element_id.astype(str).tolist(); pmap={r.union_element_id:r for _,r in prof.iterrows()}; bucket=defaultdict(lambda:{"pos":set(),"neg":set(),"span":set()})
    for ok,g0 in mentions.groupby("expert_meaning_preserved"):
        for key,g in g0.groupby(["source_id","llm","information_model"],dropna=False):
            es=sorted(set(g.union_element_id.astype(str)))
            for a,b in itertools.combinations(es,2): (bucket[tuple(sorted([a,b]))]["pos" if bool(ok) else "neg"]).add(str(key))
    pm=mentions[mentions.expert_meaning_preserved.astype(bool)]
    for key,g in pm.groupby(["source_id","llm","information_model","span_text"],dropna=False):
        if not norm(key[-1] if isinstance(key,tuple) else ""): continue
        es=sorted(set(g.union_element_id.astype(str)))
        for a,b in itertools.combinations(es,2): bucket[tuple(sorted([a,b]))]["span"].add(str(key))
    rows=[]
    for a,b in itertools.combinations(ids,2):
        rec=bucket.get(tuple(sorted([a,b])),{"pos":set(),"neg":set(),"span":set()}); pa,pb=pmap[a],pmap[b]
        sim=jac(toks(str(pa.profile_text)),toks(str(pb.profile_text))); cross=1.0 if norm(pa.source_model) and norm(pb.source_model) and norm(pa.source_model)!=norm(pb.source_model) else 0.0
        posn,negn,spann=len(rec["pos"]),len(rec["neg"]),len(rec["span"])
        w=.45*min(1,posn/8)+.25*min(1,spann/3)+.20*sim+.10*cross-.30*min(1,negn/8)
        if w>=min_w: rows.append({"union_element_id_a":a,"union_element_id_b":b,"edge_weight":float(w),"positive_cooccurrence_rows":posn,"negative_cooccurrence_rows":negn,"same_span_positive_rows":spann,"profile_similarity":float(sim),"cross_source_model_bonus":cross,"edge_interpretation":"same_function_or_merge_candidate" if spann or sim>.25 else "bundle_or_complement_candidate"})
    return pd.DataFrame(rows).sort_values("edge_weight",ascending=False) if rows else pd.DataFrame()

def cluster(prof:pd.DataFrame,edges:pd.DataFrame)->pd.DataFrame:
    ids=prof.union_element_id.astype(str).tolist(); assign={}; method="singleton"
    try:
        import networkx as nx
        G=nx.Graph(); G.add_nodes_from(ids)
        for _,e in edges.iterrows(): G.add_edge(str(e.union_element_id_a),str(e.union_element_id_b),weight=float(e.edge_weight))
        if G.number_of_edges():
            comm=list(nx.algorithms.community.greedy_modularity_communities(G,weight="weight"))
            for i,c in enumerate(sorted(comm,key=lambda z:(-len(z),sorted(z)[0]))):
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
    text=" ".join(g.profile_text.astype(str)).lower(); ts=toks(text); scores={r:len(ts&toks(w)) for r,w in ROLE_KEYWORDS.items()}; role=max(scores,key=scores.get)
    return (role, ",".join(sorted(ts&toks(ROLE_KEYWORDS[role]))[:10])) if scores[role] else ("residual_or_other","no keyword label evidence")

def cluster_summary(clust:pd.DataFrame,min_core:int)->pd.DataFrame:
    rows=[]
    for cid,g in clust.groupby("cluster_id"):
        lab,ev=label_cluster(g); sms=sorted(set(g.source_model.astype(str))); ps=pd.to_numeric(g.n_positive_source_sentences,errors="coerce").max(); ns=pd.to_numeric(g.n_negative_source_sentences,errors="coerce").max(); pll=pd.to_numeric(g.n_positive_llms,errors="coerce").max(); rate=pd.to_numeric(g.expert_positive_rate,errors="coerce").mean()
        cat="core_shared" if ps>=min_core and len(sms)>=2 and pll>=2 and lab in CORE_HINTS else "context_module" if ps>=max(3,min_core//3) and pll>=2 else "audit_or_extension"
        if ns>ps and cat!="core_shared": cat="failure_boundary_audit"
        top=g.sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])
        rows.append({"cluster_id":cid,"candidate_field_name":lab,"field_label_evidence":ev,"selection_category":cat,"n_source_elements":len(g),"n_source_models":len(sms),"source_models_json":json.dumps(sms,ensure_ascii=False),"n_positive_source_sentences_max":ps,"n_negative_source_sentences_max":ns,"n_positive_llms_max":pll,"mean_expert_positive_rate":rate,"top_source_elements_json":json.dumps(top.union_element_id.head(15).tolist(),ensure_ascii=False),"top_positive_span_examples_json":json.dumps([x for v in top.top_positive_span_examples_json.head(8) for x in json.loads(v or "[]")[:2]][:12],ensure_ascii=False)})
    return pd.DataFrame(rows).sort_values(["selection_category","n_positive_source_sentences_max"],ascending=[True,False])

def decisions_summary(dec:pd.DataFrame)->pd.DataFrame:
    rows=[]
    if dec.empty: return pd.DataFrame()
    for uid,g in dec.groupby("union_element_id"):
        p=g[g.expert_meaning_preserved.astype(bool)]; rows.append({"sentence_level_element_id":uid,"n_mentions":len(g),"n_positive_mentions":len(p),"n_positive_source_sentences":p.source_id.nunique(),"n_positive_llms":p.llm.nunique(),"top_values_or_spans_json":json.dumps([x for x,_ in Counter(g.span_text.astype(str)).most_common(8) if x],ensure_ascii=False)})
    return pd.DataFrame(rows).sort_values("n_positive_source_sentences",ascending=False)

def make_schema(cs:pd.DataFrame,clust:pd.DataFrame,ds:pd.DataFrame)->dict[str,Any]:
    fields=[{"name":"decision","status":"core","source":"sentence_level_decision_fields","description":"Provision rule type derived from DUO.decision, ICO.decision, ODRL Rule_TestSentence, and FHIR Consent.provision.type.","values":["permit","deny","obligation","mixed","unclear"],"selection_evidence":{"n_decision_elements":int(len(ds)),"decision_element_support":ds.get("sentence_level_element_id",pd.Series(dtype=str)).head(10).tolist()}}]
    used=Counter(); keep=cs[cs.selection_category.isin(["core_shared","context_module"])]
    for _,c in keep.iterrows():
        base=str(c.candidate_field_name); used[base]+=1; name=base if used[base]==1 else f"{base}_{used[base]}"; support=clust[clust.cluster_id.eq(c.cluster_id)].sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])
        fields.append({"name":name,"cluster_id":c.cluster_id,"status":"core" if c.selection_category=="core_shared" else "context_module","description":f"Expert-positive evidence cluster labeled as {base}; selected as {c.selection_category}.","value_type":"normalized_value_with_evidence","allow_multiple":True,"selection_evidence":{"n_source_elements":int(c.n_source_elements),"n_source_models":int(c.n_source_models),"n_positive_source_sentences_max":float(c.n_positive_source_sentences_max),"n_positive_llms_max":float(c.n_positive_llms_max),"mean_expert_positive_rate":float(c.mean_expert_positive_rate),"label_evidence":c.field_label_evidence},"source_element_support":support.union_element_id.head(12).tolist(),"positive_span_examples":[x for v in support.top_positive_span_examples_json.head(6) for x in json.loads(v or "[]")[:2]][:8]})
    fields += [{"name":"residual_important_content","status":"audit","description":"Short meaning-critical content not captured by selected fields.","value_type":"short_evidence_phrase"},{"name":"provenance","status":"audit","description":"Source sentence, evidence spans, selected clusters, and source elements used for audit.","value_type":"audit_metadata"}]
    return {"meta_model_id":"reduced_consent_metamodel_v1_expert_induced_candidate","version":"0.1","status":"expert_validated_derivation_candidate_requires_audit_and_validation","design_goal":"Reduced functional consent representation induced from expert-preserved round trips and penalized/flagged by expert-failed round trips.","selection_method":{"derivation_corpus":"original expert-evaluated round-trip dataset","positive_evidence":"rows judged meaning-preserving by experts","boundary_evidence":"rows judged not meaning-preserving by experts","node_input":"span-level source-model elements from forward mappings; sentence-level decision fields are separated","edge_evidence":["expert-positive co-occurrence","same-span expert-positive use","label/definition/span similarity","cross-source-model support"],"edge_penalty":"expert-failed co-occurrence weakens merge evidence and flags unsafe simplifications","clustering":"weighted graph community detection with fallback to connected components","field_selection":"core/shared clusters require expert-positive sentence coverage, multi-LLM support, cross-source-model support, and functional label evidence"},"evaluation_variants":{"compact_evidence":"same schema; short evidence phrases; no full-clause copying","permissive_evidence":"same schema; longer evidence allowed when needed to preserve condition, exception, temporal, or privacy meaning"},"fields":fields,"provision_structure":{"rule_type":"decision","selected_cluster_fields":[f["name"] for f in fields if f.get("cluster_id")],"audit_fields":["residual_important_content","provenance"]}}

def write_method(out:Path,args:argparse.Namespace):
    txt=f"""# Expert-validated reduced V1 induction methodology

The reduced V1 candidate is induced from the original expert-evaluated round-trip dataset.

Rows whose backward reconstructions were judged meaning-preserving are treated as functionally validated positive evidence: the forward representation contained sufficient structured information to preserve the sentence meaning. Rows judged not meaning-preserving are used as boundary evidence and weaken or flag proposed merges.

## Processing

1. Parse forward mappings into span-level source-element mentions.
2. Separate sentence-level decision fields: DUO.decision, ICO.decision, ODRL Rule_TestSentence, and FHIR Consent.provision.type.
3. Build positive element profiles from expert-preserved rows.
4. Build boundary profiles from expert-failed rows.
5. Construct weighted element-element edges from expert-positive co-occurrence, same-span evidence, profile similarity, and cross-source support.
6. Penalize edges that occur primarily in expert-failed rows.
7. Cluster the weighted graph.
8. Select clusters as core/shared, context modules, or audit/failure-boundary extensions.

Human review is limited to naming/audit/unsafe-merge checks. The candidate schema is validated on new LLM runs.

Parameters: min_edge_weight={args.min_edge_weight}, min_core_positive_sentences={args.min_core_positive_sentences}
"""
    (out/"expert_validated_induction_methodology.md").write_text(txt)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--expert_roundtrips_csv",required=True); ap.add_argument("--output_dir",required=True); ap.add_argument("--inventory_csv"); ap.add_argument("--label_col"); ap.add_argument("--positive_label_values",default="1,true,yes,y,preserved,meaning preserved,pass,positive"); ap.add_argument("--min_edge_weight",type=float,default=.22); ap.add_argument("--min_core_positive_sentences",type=int,default=15); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); pos={x.strip().lower() for x in args.positive_label_values.split(",") if x.strip()}
    df=pd.read_csv(args.expert_roundtrips_csv).fillna(""); inv=load_inv(args.inventory_csv); ment,dec=build_mentions(df,pos,args.label_col); prof=profiles(ment,inv); edges=edge_table(prof,ment,args.min_edge_weight); clust=cluster(prof,edges); cs=cluster_summary(clust,args.min_core_positive_sentences); ds=decisions_summary(dec); schema=make_schema(cs,clust,ds)
    ment.to_csv(out/"expert_element_mentions_long.csv",index=False); dec.to_csv(out/"expert_sentence_level_decision_mentions_long.csv",index=False); prof.to_csv(out/"expert_element_profiles.csv",index=False); edges.to_csv(out/"expert_element_relationship_edges.csv",index=False); clust.to_csv(out/"expert_element_clusters.csv",index=False); cs.to_csv(out/"expert_cluster_evidence_summary.csv",index=False); ds.to_csv(out/"expert_sentence_level_decision_summary.csv",index=False)
    (out/"reduced_metamodel_v1_candidate.yaml").write_text(yaml.safe_dump(schema,sort_keys=False,allow_unicode=True)); (out/"reduced_metamodel_v1_candidate.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    md=["# Expert-induced Candidate Reduced V1 Consent Meta-Model","","This candidate is induced from expert-preserved round trips and uses expert-failed rows as boundary evidence.","","## Selected fields"]
    for f in schema["fields"]:
        md += [f"### {f['name']} ({f.get('status','')})","",f.get("description",""),""]
        if f.get("selection_evidence"): md += ["Selection evidence:","","```json",json.dumps(f["selection_evidence"],indent=2),"```",""]
        if f.get("source_element_support"): md += ["Source-element support: "+", ".join(f["source_element_support"][:10]),""]
        if f.get("positive_span_examples"): md += ["Positive span examples: "+"; ".join(f["positive_span_examples"][:6]),""]
    md += ["## Cluster evidence summary","",cs.to_markdown(index=False),""]; (out/"reduced_metamodel_v1_candidate.md").write_text("\n".join(md)); write_method(out,args); print(f"Wrote expert-validated V1 induction outputs to {out}")
if __name__=="__main__": main()
