#!/usr/bin/env python
"""Discover Reduced V1 evidence from expert-validated round trips.

This is the empirical discovery step. It does not hard-code the reduced V1
fields and it does not turn co-occurrence bundles into fields. It writes:

- semantic-equivalence graph outputs: candidate source-element merges based on
  same/overlapping evidence spans, cross-model support, expert-positive support,
  and profile similarity;
- provision-bundle graph outputs: co-occurrence/composition evidence only, not
  merge evidence;
- an audit template to name/select clusters before a final V1 schema is built.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import pandas as pd

TEXT_COLS=["canonical_full_text","full_text_original","original_sentence","source_text","full_text","sentence","text"]
ID_COLS=["source_id","ID","id","roundtrip_id","sentence_id","source_sentence_id"]
LLM_COLS=["llm","model","llm_name"]
INFO_COLS=["information_model","info_model","source_model","model_family"]
ANN_COLS=["annotations_json","annotations_serialized","annotations_raw","annotations_combined","forward_mapping","mapping","annotation","forward_raw"]
LABEL_COLS=["meaning_preserved","human_meaning_preserved","expert_meaning_preserved","preserved","label","human_label"]
ELIG_COLS=["eligible_element_analysis","eligible"]
ELEM_KEYS=["union_element_id","element_id","source_element_id","label","source_element_label","field","role","concept","class","property","duo_label","ico_label","odrl_label","fhir_label","element_label"]
SPAN_KEYS=["span_text","evidence_span_text","evidence_text","text_span","phrase","annotated_text","text"]
DECISION_IDS={"DUO::DUO.decision","ICO::ICO.decision","ODRL::Rule_TestSentence","FHIR_Consent::Consent.provision.type","FHIR::Consent.provision.type","ROUNDTRIP::roundtrip_decision"}
DECISION_TAILS={"DUO.decision","ICO.decision","Rule_TestSentence","Consent.provision.type","roundtrip_decision"}
POS_DEFAULT="1,true,yes,y,preserved,meaning preserved,pass,passed,positive,match,matches"
NEG_VALUES={"0","false","no","n","negative","not preserved","not_preserved","fail","failed","mismatch","does not match"}
STOP=set("the a an and or of to in for with without by from on at as is are be been this that it its your you we our may can will shall consent data information use used using".split())
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
    try: return float(v)>=0.5
    except Exception: return None

def sid(text:str)->str: return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
def toks(x:str)->set[str]: return set(re.findall(r"[a-z0-9]+",str(x).lower()))-STOP
def jac(a:set[str],b:set[str])->float: return len(a&b)/max(1,len(a|b))
def tail(uid:str)->str: return str(uid).split("::")[-1]
def src_model(uid:str,info:str)->str: return uid.split("::",1)[0] if "::" in uid else (info or "unknown")
def is_decision(uid:str)->bool: return uid in DECISION_IDS or tail(uid) in DECISION_TAILS

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

def canon(raw:str,info:str)->str:
    raw=norm(raw).replace("***","").strip()
    if not raw or raw.upper()=="NA": return ""
    if "::" in raw: return raw
    m=CODE_RE.search(raw); core=m.group(0) if m else raw.split()[0]
    return f"{info}::{core}" if info else core

def annotations_from_cell(raw:Any, info:str)->list[dict[str,Any]]:
    obj=parse_jsonish(raw); anns=[]
    if isinstance(obj,dict):
        if isinstance(obj.get("annotations"),list): anns += [a for a in obj["annotations"] if isinstance(a,dict)]
        anns += [d for d in walk(obj) if isinstance(d,dict) and first(d,ELEM_KEYS)]
    elif isinstance(obj,list): anns += [a for a in obj if isinstance(a,dict)]
    if not anns:
        for m in BRACKET_RE.finditer(str(raw or "")):
            uid=canon(m.group("label"),info)
            if uid:
                anns.append({"union_element_id":uid,"source_element_label":norm(m.group("label")),"span_text":norm(m.group("span")),"decision_value":norm(m.groupdict().get("decision","")),"raw_annotation_text":norm(m.group(0))})
    out=[]; seen=set()
    for a in anns:
        uid=canon(first(a,ELEM_KEYS),info)
        if not uid: continue
        rec={"union_element_id":uid,"source_element_label":first(a,["source_element_label","label","element_label","source_element_id"]),"span_text":first(a,SPAN_KEYS),"decision_value":norm(a.get("decision_value","")),"raw_annotation_text":norm(a.get("raw_annotation_text",""))}
        key=json.dumps(rec,sort_keys=True)
        if key not in seen: seen.add(key); out.append(rec)
    return out

def load_inventory(path:str|None)->dict[str,dict[str,str]]:
    if not path or not Path(path).exists(): return {}
    df=pd.read_csv(path).fillna(""); inv={}
    for _,r in df.iterrows():
        uid=norm(r.get("union_element_id",""))
        if not uid: continue
        rec={c:norm(r.get(c,"")) for c in df.columns}; inv[uid]=rec
        sm,sid0,lab=norm(r.get("source_model","")),norm(r.get("source_element_id","")),norm(r.get("source_element_label",""))
        if sm and sid0: inv.setdefault(f"{sm}::{sid0}",rec)
        if sm and lab: inv.setdefault(f"{sm}::{lab.split()[0]}",rec)
    return inv

def build_mentions(df:pd.DataFrame,args)->tuple[pd.DataFrame,pd.DataFrame]:
    pos={x.strip().lower() for x in args.positive_label_values.split(",") if x.strip()}
    tc=pick(df,TEXT_COLS,True); ic=pick(df,ID_COLS); lc=pick(df,LLM_COLS); mc=pick(df,INFO_COLS); ac=pick(df,ANN_COLS,True)
    yc=args.label_col or pick(df,LABEL_COLS,True); ec=pick(df,ELIG_COLS)
    spans=[]; decisions=[]
    for idx,r in df.iterrows():
        if ec and norm(r.get(ec,"")).lower() in {"0","false","no","n"}: continue
        ok=truthy(r.get(yc,""),pos)
        if ok is None: continue
        text=norm(r.get(tc,"")); sentence_key=sid(text); source_id=norm(r.get(ic,"")) if ic else sentence_key
        info=norm(r.get(mc,"")) if mc else "unknown"; llm=norm(r.get(lc,"")) if lc else "unknown"; context_id=f"{sentence_key}|{info}|{llm}|{idx}"
        for j,a in enumerate(annotations_from_cell(r.get(ac,""),info)):
            uid=norm(a["union_element_id"])
            rec={"row_index":idx,"context_id":context_id,"sentence_key":sentence_key,"source_id":source_id,"source_text":text,"information_model":info,"llm":llm,"expert_meaning_preserved":bool(ok),"union_element_id":uid,"source_model_inferred":src_model(uid,info),"span_text":norm(a.get("span_text","")),"span_token_signature":" ".join(sorted(toks(a.get("span_text","")))),"raw_element_label":norm(a.get("source_element_label","")),"decision_value":norm(a.get("decision_value","")),"annotation_index":j}
            (decisions if is_decision(uid) else spans).append(rec)
        if "roundtrip_decision" in df.columns and norm(r.get("roundtrip_decision","")):
            decisions.append({"row_index":idx,"context_id":context_id,"sentence_key":sentence_key,"source_id":source_id,"source_text":text,"information_model":info,"llm":llm,"expert_meaning_preserved":bool(ok),"union_element_id":"ROUNDTRIP::roundtrip_decision","source_model_inferred":"ROUNDTRIP","span_text":norm(r.get("roundtrip_decision","")),"span_token_signature":norm(r.get("roundtrip_decision","")),"raw_element_label":"roundtrip_decision","decision_value":"","annotation_index":-1})
    return pd.DataFrame(spans),pd.DataFrame(decisions)

def profiles(m:pd.DataFrame,inv:dict[str,dict[str,str]])->pd.DataFrame:
    if m.empty: return pd.DataFrame()
    rows=[]
    for uid,g in m.groupby("union_element_id"):
        p=g[g.expert_meaning_preserved.astype(bool)]; n=g[~g.expert_meaning_preserved.astype(bool)]; meta=inv.get(uid,{})
        pc=p.groupby("context_id").size() if not p.empty else pd.Series(dtype=int)
        rows.append({"union_element_id":uid,"source_model":meta.get("source_model",src_model(uid,"")),"source_element_id":meta.get("source_element_id",tail(uid)),"source_element_label":meta.get("source_element_label",tail(uid)),"source_element_definition":meta.get("source_element_definition",""),"n_raw_mentions":len(g),"n_positive_mentions":len(p),"n_negative_mentions":len(n),"n_positive_contexts":p.context_id.nunique(),"n_negative_contexts":n.context_id.nunique(),"n_positive_source_sentences":p.sentence_key.nunique(),"n_negative_source_sentences":n.sentence_key.nunique(),"n_positive_llms":p.llm.nunique(),"n_positive_information_models":p.information_model.nunique(),"expert_positive_rate":len(p)/max(1,len(g)),"mean_mentions_per_positive_context":float(pc.mean()) if len(pc) else 0.0,"max_mentions_in_single_positive_context":int(pc.max()) if len(pc) else 0,"top_positive_span_examples_json":json.dumps([x for x,_ in Counter(p.span_text.astype(str)).most_common(12) if x],ensure_ascii=False)})
    out=pd.DataFrame(rows)
    out["profile_text"]=out.apply(lambda r:" ".join(norm(r.get(c,"")) for c in ["union_element_id","source_model","source_element_id","source_element_label","source_element_definition","top_positive_span_examples_json"]).lower(),axis=1)
    return out.sort_values(["n_positive_source_sentences","expert_positive_rate"],ascending=[False,False])

def semantic_edges(m:pd.DataFrame,prof:pd.DataFrame,min_w:float,span_jacc:float)->pd.DataFrame:
    if m.empty or prof.empty: return pd.DataFrame()
    pmap={r.union_element_id:r for _,r in prof.iterrows()}; bucket=defaultdict(lambda:{"same":set(),"overlap":set(),"fail":set(),"llms":set(),"infos":set()})
    for sk,g in m[m.expert_meaning_preserved.astype(bool)].groupby("sentence_key"):
        for a,b in itertools.combinations(g.to_dict("records"),2):
            if a["union_element_id"]==b["union_element_id"]: continue
            ta,tb=toks(a["span_text"]),toks(b["span_text"])
            same=norm(a["span_text"]).lower()==norm(b["span_text"]).lower() and bool(norm(a["span_text"]))
            over=bool(ta and tb and jac(ta,tb)>=span_jacc)
            if not (same or over): continue
            rec=bucket[tuple(sorted((a["union_element_id"],b["union_element_id"])))]
            rec["same" if same else "overlap"].add(sk); rec["llms"].update([a["llm"],b["llm"]]); rec["infos"].update([a["information_model"],b["information_model"]])
    for sk,g in m[~m.expert_meaning_preserved.astype(bool)].groupby("sentence_key"):
        for a,b in itertools.combinations(g.to_dict("records"),2):
            if a["union_element_id"]!=b["union_element_id"] and norm(a["span_text"]).lower()==norm(b["span_text"]).lower() and norm(a["span_text"]):
                bucket[tuple(sorted((a["union_element_id"],b["union_element_id"])))] ["fail"].add(sk)
    rows=[]; ids=prof.union_element_id.astype(str).tolist()
    for a,b in itertools.combinations(ids,2):
        rec=bucket.get(tuple(sorted((a,b))),{"same":set(),"overlap":set(),"fail":set(),"llms":set(),"infos":set()})
        sim=jac(toks(str(pmap[a].profile_text)),toks(str(pmap[b].profile_text))); same,over,fail=len(rec["same"]),len(rec["overlap"]),len(rec["fail"])
        cross_info=1.0 if len([x for x in rec["infos"] if x])>=2 else 0.0; cross_llm=1.0 if len([x for x in rec["llms"] if x])>=2 else 0.0
        w=.42*min(1,same/3)+.18*min(1,over/3)+.15*cross_info+.10*cross_llm+.20*sim-.20*min(1,fail/3)
        if w>=min_w:
            rows.append({"union_element_id_a":a,"union_element_id_b":b,"semantic_edge_weight":float(w),"same_span_positive_sentences":same,"overlapping_span_positive_sentences":over,"failure_same_span_sentences":fail,"profile_similarity":float(sim),"cross_information_model_support":cross_info,"cross_llm_support":cross_llm,"edge_type":"semantic_equivalence_candidate"})
    return pd.DataFrame(rows).sort_values("semantic_edge_weight",ascending=False) if rows else pd.DataFrame()

def provision_bundle_edges(m:pd.DataFrame)->pd.DataFrame:
    if m.empty: return pd.DataFrame()
    bucket=defaultdict(lambda:{"pos":set(),"neg":set(),"raw_pos":0})
    for (cid,ok),g in m.groupby(["context_id","expert_meaning_preserved"],dropna=False):
        counts=Counter(g.union_element_id.astype(str)); ids=sorted(counts)
        for a,b in itertools.combinations(ids,2):
            rec=bucket[tuple(sorted((a,b)))]; rec["pos" if bool(ok) else "neg"].add(str(cid))
            if bool(ok): rec["raw_pos"]+=min(counts[a],counts[b])
    rows=[]
    for (a,b),rec in bucket.items(): rows.append({"union_element_id_a":a,"union_element_id_b":b,"positive_cooccurrence_contexts":len(rec["pos"]),"negative_cooccurrence_contexts":len(rec["neg"]),"raw_joint_positive_mention_intensity":rec["raw_pos"],"edge_type":"provision_bundle_not_merge"})
    return pd.DataFrame(rows).sort_values(["positive_cooccurrence_contexts","raw_joint_positive_mention_intensity"],ascending=[False,False]) if rows else pd.DataFrame()

def cluster_semantic(prof:pd.DataFrame,edges:pd.DataFrame)->pd.DataFrame:
    if prof.empty: return prof.copy()
    ids=prof.union_element_id.astype(str).tolist(); assign={}; method="singleton"
    try:
        import networkx as nx
        G=nx.Graph(); G.add_nodes_from(ids)
        for _,e in edges.iterrows(): G.add_edge(str(e.union_element_id_a),str(e.union_element_id_b),weight=float(e.semantic_edge_weight))
        if G.number_of_edges():
            for i,c in enumerate(sorted(nx.algorithms.community.greedy_modularity_communities(G,weight="weight"),key=lambda z:(-len(z),sorted(z)[0]))):
                for uid in c: assign[uid]=f"C{i+1:03d}"
            method="networkx_greedy_modularity_on_semantic_edges"
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
            for uid in c: assign[uid]=f"C{i+1:03d}"
        method="threshold_connected_components_on_semantic_edges"
    out=prof.copy(); out["semantic_cluster_id"]=out.union_element_id.map(assign); out["clustering_method"]=method; return out

def suggest_terms(g:pd.DataFrame)->list[str]:
    text=" ".join(" ".join(norm(r.get(c,"")) for c in ["source_element_id","source_element_label","top_positive_span_examples_json"]) for _,r in g.iterrows()).lower()
    return [w for w,_ in Counter(toks(text)).most_common(8)]

def cluster_summary(clust:pd.DataFrame,min_sent:int)->pd.DataFrame:
    if clust.empty: return pd.DataFrame()
    rows=[]; rank={"high_support_equivalence_cluster":0,"context_specific_equivalence_cluster":1,"single_source_or_low_support_cluster":2,"failure_boundary_audit":3}
    for cid,g in clust.groupby("semantic_cluster_id"):
        ps=pd.to_numeric(g.n_positive_source_sentences,errors="coerce").max(); ns=pd.to_numeric(g.n_negative_source_sentences,errors="coerce").max(); pll=pd.to_numeric(g.n_positive_llms,errors="coerce").max(); pim=pd.to_numeric(g.n_positive_information_models,errors="coerce").max(); rate=pd.to_numeric(g.expert_positive_rate,errors="coerce").mean()
        status="high_support_equivalence_cluster" if ps>=min_sent and pim>=2 and pll>=2 and rate>=.5 else "context_specific_equivalence_cluster" if ps>=max(3,min_sent//3) and rate>=.5 else "single_source_or_low_support_cluster"
        if ns>ps and status!="high_support_equivalence_cluster": status="failure_boundary_audit"
        top=g.sort_values(["n_positive_source_sentences","n_positive_mentions"],ascending=[False,False])
        rows.append({"semantic_cluster_id":cid,"selection_status":status,"n_source_elements":len(g),"n_positive_source_sentences_max":ps,"n_negative_source_sentences_max":ns,"n_positive_llms_max":pll,"n_positive_information_models_max":pim,"mean_expert_positive_rate":rate,"name_suggestion_terms_json":json.dumps(suggest_terms(g),ensure_ascii=False),"top_source_elements_json":json.dumps(top.union_element_id.head(20).tolist(),ensure_ascii=False),"top_positive_span_examples_json":json.dumps([x for v in top.top_positive_span_examples_json.head(10) for x in json.loads(v or "[]")[:2]][:15],ensure_ascii=False)})
    out=pd.DataFrame(rows); out["_rank"]=out.selection_status.map(rank).fillna(9); return out.sort_values(["_rank","n_positive_source_sentences_max"],ascending=[True,False]).drop(columns="_rank")

def decision_summary(dec:pd.DataFrame)->pd.DataFrame:
    if dec.empty: return pd.DataFrame()
    rows=[]
    for uid,g in dec.groupby("union_element_id"):
        p=g[g.expert_meaning_preserved.astype(bool)]
        rows.append({"sentence_level_element_id":uid,"n_mentions":len(g),"n_positive_mentions":len(p),"n_positive_source_sentences":p.sentence_key.nunique(),"n_positive_llms":p.llm.nunique(),"top_values_or_spans_json":json.dumps([x for x,_ in Counter(g.span_text.astype(str)).most_common(10) if x],ensure_ascii=False)})
    return pd.DataFrame(rows).sort_values("n_positive_source_sentences",ascending=False)

def bundle_by_cluster(bundle:pd.DataFrame,clust:pd.DataFrame)->pd.DataFrame:
    if bundle.empty or clust.empty: return pd.DataFrame()
    cmap=dict(zip(clust.union_element_id,clust.semantic_cluster_id)); rows=[]
    for _,e in bundle.iterrows():
        ca,cb=cmap.get(e.union_element_id_a,""),cmap.get(e.union_element_id_b,"")
        if ca and cb and ca!=cb: rows.append({"semantic_cluster_id_a":ca,"semantic_cluster_id_b":cb,"element_a":e.union_element_id_a,"element_b":e.union_element_id_b,"positive_cooccurrence_contexts":e.positive_cooccurrence_contexts,"negative_cooccurrence_contexts":e.negative_cooccurrence_contexts,"raw_joint_positive_mention_intensity":e.raw_joint_positive_mention_intensity})
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    return df.groupby(["semantic_cluster_id_a","semantic_cluster_id_b"],as_index=False).agg(positive_cooccurrence_contexts=("positive_cooccurrence_contexts","sum"),negative_cooccurrence_contexts=("negative_cooccurrence_contexts","sum"),raw_joint_positive_mention_intensity=("raw_joint_positive_mention_intensity","sum"),example_element_pairs=("element_a",lambda x:"; ".join(list(x.head(5))))).sort_values("positive_cooccurrence_contexts",ascending=False)

def audit_template(cs:pd.DataFrame)->pd.DataFrame:
    if cs.empty: return pd.DataFrame()
    out=cs.copy(); out.insert(1,"include_in_v1",""); out.insert(2,"final_field_name",""); out.insert(3,"audit_decision",""); out.insert(4,"unsafe_merge_notes",""); out.insert(5,"split_or_merge_notes",""); return out

def write_reports(out:Path,args,cs:pd.DataFrame)->None:
    method=f"""# Expert V1 discovery methodology\n\nThis directory is a discovery output, not a final schema.\n\nReduced V1 is derived from expert-evaluated original round trips. Expert-preserved rows are positive functional evidence; expert-failed rows are boundary evidence.\n\nTwo graphs are built separately:\n\n1. Semantic-equivalence graph: candidate merges are supported by same or overlapping evidence spans, cross-model support, cross-LLM support, expert-positive evidence, and element-profile similarity. This graph is clustered.\n2. Provision-bundle graph: co-occurrence within a sentence/model run. This graph describes composition and should not be used directly to merge fields.\n\nRaw repeated annotations are retained as node salience features. Co-occurrence bundle edges are counted at the context level to avoid Cartesian-product inflation.\n\nHuman review is limited to naming, unsafe-merge checks, and include/exclude decisions in semantic_cluster_audit_template.csv.\n\nParameters: min_semantic_edge_weight={args.min_semantic_edge_weight}, span_overlap_jaccard={args.span_overlap_jaccard}, min_core_positive_sentences={args.min_core_positive_sentences}\n"""
    (out/"expert_v1_discovery_methodology.md").write_text(method)
    lines=["# Expert V1 Semantic Cluster Discovery","","This is not the final V1 schema. Use `semantic_cluster_audit_template.csv` for naming and unsafe-merge review.",""]
    for _,r in cs.iterrows():
        lines += [f"## {r.semantic_cluster_id}: {r.selection_status}","",f"Suggested terms: {r.name_suggestion_terms_json}",f"Support: positive sentences={r.n_positive_source_sentences_max}, information models={r.n_positive_information_models_max}, LLMs={r.n_positive_llms_max}","","Source elements:",r.top_source_elements_json,"","Span examples:",r.top_positive_span_examples_json,""]
    (out/"semantic_cluster_discovery_report.md").write_text("\n".join(lines))

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--expert_roundtrips_csv",required=True); ap.add_argument("--output_dir",required=True); ap.add_argument("--inventory_csv"); ap.add_argument("--label_col"); ap.add_argument("--positive_label_values",default=POS_DEFAULT); ap.add_argument("--min_semantic_edge_weight",type=float,default=.28); ap.add_argument("--span_overlap_jaccard",type=float,default=.5); ap.add_argument("--min_core_positive_sentences",type=int,default=15); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.expert_roundtrips_csv).fillna(""); inv=load_inventory(args.inventory_csv)
    mentions,decisions=build_mentions(df,args); prof=profiles(mentions,inv); sedges=semantic_edges(mentions,prof,args.min_semantic_edge_weight,args.span_overlap_jaccard); bundles=provision_bundle_edges(mentions); clust=cluster_semantic(prof,sedges); cs=cluster_summary(clust,args.min_core_positive_sentences); ds=decision_summary(decisions); bc=bundle_by_cluster(bundles,clust); audit=audit_template(cs)
    mentions.to_csv(out/"expert_element_mentions_long.csv",index=False); decisions.to_csv(out/"expert_sentence_level_decision_mentions_long.csv",index=False); prof.to_csv(out/"expert_element_profiles.csv",index=False); sedges.to_csv(out/"semantic_equivalence_edges.csv",index=False); bundles.to_csv(out/"provision_bundle_edges.csv",index=False); clust.to_csv(out/"semantic_equivalence_clusters.csv",index=False); cs.to_csv(out/"semantic_cluster_evidence_summary.csv",index=False); audit.to_csv(out/"semantic_cluster_audit_template.csv",index=False); ds.to_csv(out/"expert_sentence_level_decision_summary.csv",index=False); bc.to_csv(out/"provision_bundle_summary_by_semantic_cluster.csv",index=False)
    write_reports(out,args,cs); print(f"Wrote empirical V1 discovery outputs to {out}")

if __name__=="__main__": main()
