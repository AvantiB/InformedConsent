#!/usr/bin/env python
"""Rebuild a corrected PI expert-review package (v2).

This script fixes the review-package issues identified after first generation:
1. example sentences are a fixed random sample of source IDs and reused;
2. all available rows for each sampled source sentence are shown across LLMs and
   modeling strategies;
3. parsed annotations are highlighted and clickable/hoverable with node/field
   details when available;
4. crosswalks are restricted to DUO/ICO/ODRL/FHIR -> Manual V1 / LLM-induced V1;
5. source-model element IDs/labels/definitions are preserved in crosswalk tables;
6. README includes classifier-score summaries by modeling strategy.
"""
from __future__ import annotations

import argparse, hashlib, html, json, random, re, shutil
from pathlib import Path
from typing import Any
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

SCORE_CANDIDATES = [
    "classifier_preservation_score", "meaning_preserved_score", "meaning_preservation_score",
    "classifier_score", "predicted_probability", "probability", "score",
    "meaning_preserved_pred_proba", "meaning_preserved_pred", "mean_classifier_score",
]
CONDITION_ORDER = [
    "individual_source_model_json", "union_v0_full_dictionary",
    "functional_v1_manual", "functional_v1_llm_induced",
    "functional_v1_llm_induced_consensus",
]
SOURCE_MODELS = ["DUO", "ICO", "ODRL", "FHIR_Consent", "FHIR"]
SPAN_KEYS = ["span_text","span","text","value","evidence_span_text","verbatim","quote"]
FIELD_KEYS = ["field_id","field_name","label","element_id","union_element_id","cluster_id","source_element","source_element_id","role","type","name","id"]


def norm(x: Any) -> str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def read_csv(p: str | Path) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False).fillna("") if Path(p).exists() else pd.DataFrame()


def score_col(df: pd.DataFrame) -> str | None:
    return next((c for c in SCORE_CANDIDATES if c in df.columns), None)


def tail(x: str) -> str:
    return x.split("::",1)[1] if "::" in x else x


def infer_model(uid: str, fallback: str = "") -> str:
    if "::" in uid: return uid.split("::",1)[0]
    return norm(fallback)


def pick(r: pd.Series, cols: list[str]) -> str:
    for c in cols:
        if c in r.index and norm(r.get(c)): return norm(r.get(c))
    return ""


def safe_float(x: Any) -> float | None:
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None


def parse_jsonish(x: Any) -> Any:
    s = norm(x)
    if not s: return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|csv|yaml)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try: return json.loads(s)
    except Exception: pass
    for l,r in [("{","}"),("[","]")]:
        a,b=s.find(l),s.rfind(r)
        if a>=0 and b>a:
            try: return json.loads(s[a:b+1])
            except Exception: pass
    return None


def first_text(d: dict[str,Any], keys: list[str]) -> str:
    for k in keys:
        if norm(d.get(k)): return norm(d.get(k))
    return ""


def ann_from_dict(d: dict[str,Any]) -> dict[str,str] | None:
    span = first_text(d, SPAN_KEYS)
    lab = first_text(d, FIELD_KEYS)
    dec = first_text(d, ["decision","polarity","permission","rule_type","sentence_decision","value"])
    if span and lab:
        return {"span_text":span,"label":lab,"decision":dec,"parse_source":"json"}
    for k,v in d.items():
        if isinstance(v,dict):
            span2=first_text(v,SPAN_KEYS)
            if span2:
                return {"span_text":span2,"label":str(k),"decision":first_text(v,["polarity","decision","value"]),"parse_source":"json_nested"}
    return None


def dedup_anns(anns):
    seen=set(); out=[]
    for a in anns:
        key=(a.get("span_text","").lower(),a.get("label","").lower(),a.get("decision","").lower())
        if a.get("span_text") and key not in seen:
            seen.add(key); out.append(a)
    return out


def collect_anns(obj: Any) -> list[dict[str,str]]:
    out=[]
    if obj is None: return out
    if isinstance(obj,list):
        for it in obj:
            if isinstance(it,dict):
                a=ann_from_dict(it)
                if a: out.append(a)
                out += collect_anns(it.get("annotations"))
        return dedup_anns(out)
    if isinstance(obj,dict):
        for key in ["annotations","span_annotations","elements","fields","mapped_elements"]:
            if isinstance(obj.get(key),list): out += collect_anns(obj.get(key))
        for ck in ["interpretation_units","provisions","sentence_level_elements"]:
            if isinstance(obj.get(ck),list):
                for unit in obj.get(ck):
                    if not isinstance(unit,dict): continue
                    for k,v in unit.items():
                        if k in {"annotations","evidence","source"}: continue
                        vals = v if isinstance(v,list) else [v]
                        for item in vals:
                            if isinstance(item,dict):
                                sp=first_text(item,SPAN_KEYS)
                                if sp: out.append({"span_text":sp,"label":k,"decision":first_text(item,["polarity","decision","value"]),"parse_source":ck})
                            elif norm(item) and len(norm(item).split()) <= 18:
                                out.append({"span_text":norm(item),"label":k,"decision":"","parse_source":ck})
        a=ann_from_dict(obj)
        if a: out.append(a)
    return dedup_anns(out)


def compact_parse(text: str) -> list[dict[str,str]]:
    pat = re.compile(r"(?P<span>[^\[\]\n]{2,260}?)\s*\[(?P<label>[^\[\]]{1,220})\]\s*(?:\((?P<decision>[^)]{1,100})\))?", re.S)
    return [{"span_text":norm(m.group("span")).strip(" ;,.-"),"label":norm(m.group("label")),"decision":norm(m.group("decision")),"parse_source":"compact"} for m in pat.finditer(norm(text)) if norm(m.group("span")) and norm(m.group("label"))]


def parse_anns(raw: Any) -> list[dict[str,str]]:
    obj=parse_jsonish(raw)
    anns=collect_anns(obj)
    return anns if anns else compact_parse(norm(raw))


def normalize_manual_crosswalk(path: Path, out_dir: Path) -> pd.DataFrame:
    df=read_csv(path)
    rows=[]
    for _,r in df.iterrows():
        sid=pick(r,["union_element_id","source_element_id","source_element","element_id","id","element"])
        sm=infer_model(sid,pick(r,["information_model","source_model","model","canonical_information_model"]))
        rows.append({
            "source_model": "FHIR_Consent" if sm=="FHIR" else sm,
            "source_element_id": sid,
            "source_element_label": pick(r,["source_element_label","source_label","label","element_label","canonical_label","name"]) or tail(sid),
            "source_element_definition": pick(r,["source_element_definition","source_definition","definition","description"]),
            "manual_v1_field": pick(r,["v1_field","manual_v1_field","functional_field","proposed_v1_field","field_id","target_field"]),
            "secondary_manual_v1_fields": pick(r,["secondary_v1_fields_json","secondary_v1_fields","secondary_fields"]),
            "manual_mapping_type": pick(r,["mapping_type","relationship","mapping_relation"]) or "candidate_mapping",
            "manual_rationale": pick(r,["rationale","context_rule","rule","notes"]),
            "expert_review_status": "", "expert_notes": "",
        })
    out=pd.DataFrame(rows).drop_duplicates()
    out.to_csv(out_dir/"manual_v1_source_model_crosswalk_for_review.csv",index=False)
    return out


def read_jsonl(p: Path) -> list[dict[str,Any]]:
    if not p.exists(): return []
    rows=[]
    for line in p.read_text().splitlines():
        if line.strip():
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


def listify(x: Any) -> list[Any]:
    if x is None or (isinstance(x,float) and pd.isna(x)): return []
    if isinstance(x,list): return x
    if isinstance(x,str):
        s=x.strip()
        if not s: return []
        try:
            j=json.loads(s)
            if isinstance(j,list): return j
        except Exception: pass
        return [v.strip() for v in re.split(r"[;|,]",s) if v.strip()]
    return [x]


def card_id(c: dict[str,Any]) -> str:
    for k in ["candidate_field_id","card_id","evidence_card_id","stability_group_id","selected_field_id","field_id","id"]:
        if norm(c.get(k)): return norm(c.get(k))
    return ""


def load_cards(root: Path) -> dict[str,dict[str,Any]]:
    cards={}
    for p in root.glob("fold_*/schema_induction_evidence_cards.jsonl"):
        fold=p.parent.name
        for c in read_jsonl(p):
            c=dict(c); c["fold"]=fold
            for key in {card_id(c), f"{fold}::{card_id(c)}", norm(c.get("stability_group_id"))}:
                if key: cards[key]=c
    return cards


def source_rows_from_card(c: dict[str,Any]) -> list[dict[str,str]]:
    rows=[]
    for item in listify(c.get("top_source_elements")) + listify(c.get("source_elements")) + listify(c.get("member_elements")):
        if isinstance(item,dict):
            sid=pick(pd.Series(item),["union_element_id","source_element","source_element_id","element_id","id","label"])
            sm=infer_model(sid,pick(pd.Series(item),["source_model","information_model","model"]))
            rows.append({"source_model":sm,"source_element_id":sid,"source_element_label":pick(pd.Series(item),["source_element_label","source_label","label","name"]) or tail(sid),"source_element_definition":pick(pd.Series(item),["source_element_definition","definition","description"])})
        elif norm(item):
            sid=norm(item); rows.append({"source_model":infer_model(sid),"source_element_id":sid,"source_element_label":tail(sid),"source_element_definition":""})
    for ex in listify(c.get("example_sentences")):
        if isinstance(ex,dict) and norm(ex.get("union_element_id")):
            sid=norm(ex.get("union_element_id"))
            rows.append({"source_model":infer_model(sid,norm(ex.get("information_model"))),"source_element_id":sid,"source_element_label":norm(ex.get("source_element_label")) or tail(sid),"source_element_definition":""})
    seen=set(); out=[]
    for r in rows:
        sid=norm(r["source_element_id"]); sm=norm(r["source_model"])
        if not sid: continue
        key=(sm,sid)
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def field_eids(f: dict[str,Any]) -> list[str]:
    ids=[]
    for k in ["evidence_card_ids","assigned_evidence_cards","supporting_evidence_cards","evidence_cards","source_cards","cluster_ids","stability_group_ids"]:
        ids += [norm(x) for x in listify(f.get(k)) if norm(x)]
    for k in ["rationale","evidence","notes","definition","description"]:
        ids += re.findall(r"(?:C\d{3,}|SG[_-]?\d+|field[_-]?\d+|cluster[_-]?\d+|candidate[_-]?field[_-]?\d+)",norm(f.get(k)),flags=re.I)
    return sorted(set(ids))


def tokens(x: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]+",x.lower()) if len(t)>2}


def fallback_cards(f: dict[str,Any], cards: dict[str,dict[str,Any]], fold: str) -> list[dict[str,Any]]:
    ft=tokens(" ".join(norm(f.get(k)) for k in ["field_id","name","definition","description"]))
    scored=[]
    for c in cards.values():
        if c.get("fold")!=fold: continue
        ct=tokens(" ".join(map(str,listify(c.get("suggested_terms"))+listify(c.get("top_spans"))+listify(c.get("top_source_elements")))))
        ov=len(ft & ct)
        if ov: scored.append((ov,card_id(c),c))
    return [c for _,_,c in sorted(scored,reverse=True,key=lambda x:(x[0],x[1]))[:5]]


def schema_fields(path: Path) -> list[dict[str,Any]]:
    if not path.exists(): return []
    obj=json.loads(path.read_text()) if path.suffix==".json" else yaml.safe_load(path.read_text())
    fs=obj.get("fields") or obj.get("schema_fields") or obj.get("functional_fields") or []
    if isinstance(fs,dict): return [{"field_id":k, **(v if isinstance(v,dict) else {"definition":v})} for k,v in fs.items()]
    return [f if isinstance(f,dict) else {"field_id":str(f)} for f in fs]


def build_llm_crosswalk(schema_root: Path, cards_root: Path, manual: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cards=load_cards(cards_root)
    lookup={}
    for _,r in manual.iterrows():
        sid=norm(r.source_element_id); lookup[sid.lower()]=r.to_dict(); lookup[tail(sid).lower()]=r.to_dict()
    rows=[]
    for sp in sorted(schema_root.glob("fold_*/llm_induced_functional_v1_candidate.*")):
        if sp.suffix.lower() not in {".yaml",".yml",".json"}: continue
        fold=sp.parent.name
        for f in schema_fields(sp):
            fid=norm(f.get("field_id") or f.get("id") or f.get("name"))
            fname=norm(f.get("name") or fid)
            definition=norm(f.get("definition") or f.get("description"))
            eids=field_eids(f)
            used=[]
            for eid in eids:
                for key in [eid,f"{fold}::{eid}"]:
                    if key in cards: used.append(cards[key])
            if not used: used=fallback_cards(f,cards,fold)
            srows=[]
            for c in used: srows += source_rows_from_card(c)
            if not srows:
                srows=[{"source_model":"","source_element_id":"","source_element_label":"","source_element_definition":""}]
            for sr in srows:
                sid=norm(sr["source_element_id"])
                ml=lookup.get(sid.lower()) or lookup.get(tail(sid).lower()) or {}
                rows.append({
                    "fold":fold, "source_model":norm(sr["source_model"]) or norm(ml.get("source_model")),
                    "source_element_id":sid, "source_element_label":norm(sr["source_element_label"]) or norm(ml.get("source_element_label")),
                    "source_element_definition":norm(sr["source_element_definition"]) or norm(ml.get("source_element_definition")),
                    "llm_induced_v1_field":fid, "llm_induced_v1_field_name":fname,
                    "llm_induced_v1_definition":definition, "llm_mapping_basis":"evidence_card" if eids else "field_text_to_evidence_card_fallback",
                    "evidence_card_ids":"; ".join(eids or [card_id(c) for c in used if card_id(c)]),
                    "manual_v1_fields_linked_by_source_element": norm(ml.get("manual_v1_field")),
                    "expert_review_status":"", "expert_notes":"",
                })
    out=pd.DataFrame(rows).drop_duplicates()
    out.to_csv(out_dir/"llm_induced_v1_source_model_crosswalk_by_fold_for_review.csv",index=False)
    return out


def build_combined(manual: pd.DataFrame, llm: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    keys=["source_model","source_element_id","source_element_label","source_element_definition"]
    m=manual.groupby(keys,dropna=False).agg(manual_v1_fields=("manual_v1_field",lambda x:"; ".join(sorted(set(norm(v) for v in x if norm(v))))), manual_mapping_types=("manual_mapping_type",lambda x:"; ".join(sorted(set(norm(v) for v in x if norm(v)))))).reset_index() if not manual.empty else pd.DataFrame()
    l=llm.groupby(keys,dropna=False).agg(llm_induced_v1_fields=("llm_induced_v1_field",lambda x:"; ".join(sorted(set(norm(v) for v in x if norm(v))))), llm_induced_folds=("fold",lambda x:"; ".join(sorted(set(norm(v) for v in x if norm(v))))), llm_mapping_basis=("llm_mapping_basis",lambda x:"; ".join(sorted(set(norm(v) for v in x if norm(v)))))).reset_index() if not llm.empty else pd.DataFrame()
    out=m.merge(l,on=keys,how="outer") if not m.empty and not l.empty else (m if not m.empty else l)
    if not out.empty:
        out["expert_review_status"]=""; out["expert_notes"]=""
        out.to_csv(out_dir/"source_model_to_manual_and_llm_v1_crosswalk_for_review.csv",index=False)
    return out


def write_crosswalk_html(manual,llm,combined,out):
    def table(df,n=40): return "<p><i>Not available.</i></p>" if df.empty else df.head(n).to_html(index=False,escape=True)
    h=f"""<!doctype html><html><head><meta charset='utf-8'><style>body{{font-family:Arial;margin:28px}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{border:1px solid #ddd;padding:5px;vertical-align:top}}th{{background:#eff6ff}}h1{{color:#0f172a}}h2{{color:#0f766e}}.note{{background:#f8fafc;border-left:4px solid #0ea5e9;padding:12px}}</style></head><body><h1>Source model crosswalks to Manual V1 and LLM-induced V1</h1><div class='note'>These tables map source information model elements (DUO/ICO/ODRL/FHIR) to Manual V1 and LLM-induced V1 fields. LLM-induced mappings are evidence-derived and should be expert reviewed.</div><h2>Combined source-model → V1 crosswalk</h2>{table(combined)}<h2>Manual V1 crosswalk</h2>{table(manual)}<h2>LLM-induced V1 crosswalk by fold</h2>{table(llm)}</body></html>"""
    out.write_text(h,encoding="utf-8")


def node_info(manual: pd.DataFrame, llm: pd.DataFrame) -> dict[str,str]:
    d={}
    def add(k,v):
        if norm(k) and norm(v):
            d.setdefault(norm(k).lower(), norm(v)); d.setdefault(tail(norm(k)).lower(), norm(v))
    for df in [manual,llm]:
        for _,r in df.iterrows():
            sid=pick(r,["source_element_id"]); lab=pick(r,["source_element_label"]); definition=pick(r,["source_element_definition"])
            target=pick(r,["manual_v1_field","llm_induced_v1_field","llm_induced_v1_field_name"])
            detail=" | ".join(x for x in [sid,lab,definition, f"Maps to: {target}" if target else ""] if x)
            add(sid,detail); add(lab,detail)
    return d


def highlight(text: str, anns: list[dict[str,str]], nd: dict[str,str]) -> str:
    low=text.lower(); hits=[]; used=[]
    for a in anns:
        sp=norm(a.get("span_text"))
        if not sp: continue
        st=low.find(sp.lower())
        if st<0: continue
        en=st+len(sp)
        if any(not(en<=u[0] or st>=u[1]) for u in used): continue
        used.append((st,en)); hits.append((st,en,a))
    hits=sorted(hits)
    colors=["c1","c2","c3","c4","c5","c6","c7","c8"]; parts=[]; pos=0
    for i,(st,en,a) in enumerate(hits):
        lab=norm(a.get("label")); detail=" | ".join(x for x in [f"Span: {a.get('span_text')}",f"Label: {lab}",f"Decision/polarity: {a.get('decision')}" if norm(a.get('decision')) else "", nd.get(lab.lower()) or nd.get(tail(lab).lower()) or ""] if x)
        enc=html.escape(detail,quote=True)
        parts.append(html.escape(text[pos:st]))
        parts.append(f'<mark class="ann {colors[i%len(colors)]}" tabindex="0" title="{enc}" data-detail="{enc}" onclick="showDetail(this)">{html.escape(text[st:en])}<sup>{html.escape(tail(lab)[:18])}</sup></mark>')
        pos=en
    parts.append(html.escape(text[pos:]))
    return "".join(parts) if hits else html.escape(text)


def fixed_source_sample(df: pd.DataFrame, out_dir: Path, n: int, seed: int, refresh: bool) -> pd.DataFrame:
    orig_col=next(c for c in ["original_text","source_text","canonical_full_text","full_text_original"] if c in df.columns)
    if "source_id" not in df.columns:
        df=df.copy(); df["source_id"]=[hashlib.sha1(norm(x).encode()).hexdigest()[:12] for x in df[orig_col]]
    sample_path=out_dir/"fixed_example_source_ids.csv"
    if sample_path.exists() and not refresh:
        ids=pd.read_csv(sample_path)["source_id"].astype(str).tolist()
    else:
        cand=df.groupby("source_id").agg(n_rows=("source_id","size"),n_conditions=("condition","nunique"),n_llms=("llm","nunique"),original_text=(orig_col,"first")).reset_index()
        cand=cand[(cand.n_conditions>=2)&(cand.n_llms>=2)] if not cand.empty else cand
        ids=cand["source_id"].tolist(); random.Random(seed).shuffle(ids); ids=ids[:n]
        cand[cand.source_id.isin(ids)].to_csv(sample_path,index=False)
    return df[df.source_id.astype(str).isin(ids)].copy()


def build_examples(diag: pd.DataFrame, out_dir: Path, nd: dict[str,str], n: int, seed: int, refresh: bool):
    out_dir.mkdir(parents=True,exist_ok=True)
    df=fixed_source_sample(diag,out_dir,n,seed,refresh)
    orig_col=next(c for c in ["original_text","source_text","canonical_full_text","full_text_original"] if c in df.columns)
    rec_col=next(c for c in ["reconstructed_text","reconstructed_sentence","backward_mapping","reconstruction"] if c in df.columns)
    map_col=next((c for c in ["forward_mapping","annotations_serialized","annotations_combined","mapping"] if c in df.columns),None)
    sc=score_col(df)
    rows=[]; cards=[]
    order={c:i for i,c in enumerate(CONDITION_ORDER)}
    df["_ord"]=df.condition.map(order).fillna(99) if "condition" in df.columns else 0
    df=df.sort_values(["source_id","_ord","llm","information_model"])
    for _,r in df.iterrows():
        anns=parse_anns(r.get(map_col,"") if map_col else "")
        row={k:norm(r.get(k)) for k in ["source_id","roundtrip_id","condition","information_model","llm"]}
        row.update({"classifier_score":safe_float(r.get(sc)) if sc else None, "annotation_count":safe_float(r.get("annotation_count")), "unique_element_count":safe_float(r.get("unique_element_count")), "content_word_recall":safe_float(r.get("content_word_recall")), "important_category_presence_recall":safe_float(r.get("important_category_presence_recall")), "modal_word_change_ratio":safe_float(r.get("modal_word_change_ratio")), "suspected_error_flags":norm(r.get("suspected_error_flags")), "original_text":norm(r.get(orig_col)), "reconstructed_text":norm(r.get(rec_col)), "parsed_annotations_json":json.dumps(anns,ensure_ascii=False), "n_parsed_annotations":len(anns), "expert_meaning_preserved":"", "expert_notes":""})
        rows.append(row); cards.append((row,anns,highlight(row["original_text"],anns,nd)))
    pd.DataFrame(rows).to_csv(out_dir/"expert_review_examples.csv",index=False)
    try: pd.DataFrame(rows).to_excel(out_dir/"expert_review_examples.xlsx",index=False)
    except Exception: pass
    css="""body{font-family:Arial;margin:28px;background:#f8fafc;color:#1f2937}h1{color:#0f172a}h2{color:#0f766e}.src{background:white;border:2px solid #bae6fd;border-radius:14px;padding:16px;margin:20px 0}.card{background:white;border:1px solid #dbeafe;border-radius:12px;padding:12px;margin:12px 0}.ann{padding:1px 3px;border-radius:4px;cursor:pointer}.ann:hover{outline:2px solid #0ea5e9}.c1{background:#bfdbfe}.c2{background:#99f6e4}.c3{background:#ddd6fe}.c4{background:#fecaca}.c5{background:#fde68a}.c6{background:#bbf7d0}.c7{background:#fbcfe8}.c8{background:#bae6fd}.txt{background:#f8fafc;border-left:4px solid #0ea5e9;padding:10px;border-radius:8px;line-height:1.6}.rec{border-left-color:#14b8a6}.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:3px 8px;margin:2px;font-size:12px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}th{background:#eff6ff}#detail{position:sticky;top:0;background:#ecfeff;border:1px solid #67e8f9;padding:10px;border-radius:10px}"""
    js="<script>function showDetail(el){document.getElementById('detail').innerHTML='<b>Annotation detail</b><br>'+el.getAttribute('data-detail');}</script>"
    parts=[f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style>{js}</head><body><h1>Fixed expert-review examples</h1><p>Examples are a fixed random sample of source sentences. For each source sentence, all available LLM and modeling-strategy rows are shown.</p><div id='detail'><b>Annotation detail</b><br>Click or hover highlighted spans.</div>"]
    for sid,group in pd.DataFrame([c[0] for c in cards]).groupby("source_id",sort=False):
        original=group.iloc[0]["original_text"]; parts.append(f"<section class='src'><h2>Source: {html.escape(sid)}</h2><div class='txt'><b>Original:</b> {html.escape(original)}</div>")
        for row,anns,hlt in [c for c in cards if c[0]["source_id"]==sid]:
            badges=" ".join([f"<span class='badge'><b>{k}:</b> {row[k]:.3f}</span>" for k in ["classifier_score","content_word_recall","important_category_presence_recall","modal_word_change_ratio"] if isinstance(row.get(k),float)])
            parts.append(f"<div class='card'><b>{html.escape(row['condition'])}</b> | LLM: {html.escape(row['llm'])} | information model: {html.escape(row['information_model'])}<br>{badges}<div class='txt'>{hlt}</div><div class='txt rec'><b>Reconstruction:</b> {html.escape(row['reconstructed_text'])}</div><details open><summary>All parsed annotations ({len(anns)})</summary><table><tr><th>Span</th><th>Label/field</th><th>Decision/polarity</th><th>Node details</th></tr>")
            for a in anns:
                lab=norm(a.get("label")); detail=nd.get(lab.lower()) or nd.get(tail(lab).lower()) or ""
                parts.append(f"<tr><td>{html.escape(norm(a.get('span_text')))}</td><td>{html.escape(lab)}</td><td>{html.escape(norm(a.get('decision')))}</td><td>{html.escape(detail)}</td></tr>")
            parts.append("</table></details></div>")
        parts.append("</section>")
    parts.append("</body></html>")
    (out_dir/"expert_review_examples.html").write_text("".join(parts),encoding="utf-8")


def summarize_and_plot(diag: pd.DataFrame, package_dir: Path):
    comp=package_dir/"comparison"; plots=package_dir/"plots"; comp.mkdir(parents=True,exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    sc=score_col(diag); agg={}
    if sc: agg[sc]="mean"
    for c in ["content_word_recall","important_category_presence_recall","important_cue_exact_recall","modal_word_change_ratio","annotation_count","unique_element_count","forward_parse_ok","backward_parse_ok","unmatched_language_rate","suspected_error_count"]:
        if c in diag.columns: agg[c]="mean"
    overall=diag.groupby("condition",dropna=False).agg(agg).reset_index()
    if sc: overall=overall.rename(columns={sc:"mean_classifier_score"})
    overall.to_csv(comp/"schema_condition_overall.csv",index=False)
    if {"condition","llm"}.issubset(diag.columns):
        by=diag.groupby(["condition","llm"],dropna=False).agg(agg).reset_index()
        if sc: by=by.rename(columns={sc:"mean_classifier_score"})
        by.to_csv(comp/"schema_condition_by_llm.csv",index=False)
        if sc:
            piv=diag.pivot_table(index="condition",columns="llm",values=sc,aggfunc="mean").reindex([c for c in CONDITION_ORDER if c in diag.condition.unique()])
            plt.figure(figsize=(9,4.5)); im=plt.imshow(piv.values,aspect="auto"); plt.colorbar(im); plt.xticks(range(len(piv.columns)),piv.columns,rotation=45,ha="right"); plt.yticks(range(len(piv.index)),piv.index); plt.title("Meaning-preservation classifier score by condition and LLM"); plt.tight_layout(); plt.savefig(plots/"01_condition_x_llm_classifier_score_heatmap.png",dpi=250,bbox_inches="tight"); plt.close()
    if sc:
        plotdf=diag.groupby("condition")[sc].mean().reset_index()
        plt.figure(figsize=(8,4)); plt.bar(range(len(plotdf)),plotdf[sc]); plt.xticks(range(len(plotdf)),plotdf.condition,rotation=35,ha="right"); plt.ylabel("Mean classifier score"); plt.title("Mean classifier score by schema condition"); plt.tight_layout(); plt.savefig(plots/"02_classifier_score_by_condition.png",dpi=250,bbox_inches="tight"); plt.close()
    return overall


def write_readmes(pkg: Path, snapshot: pd.DataFrame, seed: int, n_examples: int):
    lines=["# PI Expert Review Package v2","","This corrected package uses a fixed random sample of source sentences for annotation examples and restricts crosswalks to source information model elements mapped to Manual V1 and LLM-induced V1.","","## Key files","","- `expert_review_examples/expert_review_examples.html`: fixed examples with clickable/hoverable highlighted annotations.","- `expert_review_examples/fixed_example_source_ids.csv`: source IDs sampled with a fixed seed for reproducibility.","- `crosswalks/source_model_to_manual_and_llm_v1_crosswalk_for_review.csv`: combined DUO/ICO/ODRL/FHIR -> Manual/LLM-induced V1 crosswalk.","- `comparison/schema_condition_overall.csv`: performance metrics by modeling strategy.","- `plots/01_condition_x_llm_classifier_score_heatmap.png`: condition x LLM classifier score heatmap.","","## Fixed sampling","",f"Random seed: `{seed}`. Number of sampled source sentences requested: `{n_examples}`.","","## Meaning-preservation classifier and strategy-level metrics",""]
    lines.append(snapshot.to_markdown(index=False) if not snapshot.empty else "Metric snapshot not available.")
    lines += ["","## Expert review questions","","- Are Manual V1 fields too broad, too narrow, redundant, or missing important consent roles?","- Do LLM-induced V1 fields identify useful additional boundaries or unsafe merges?","- Are source-model element mappings into Manual/LLM-induced V1 correct?","- Do highlighted examples preserve consent meaning after reconstruction?"]
    (pkg/"README_PI_REVIEW_PACKAGE.md").write_text("\n".join(lines),encoding="utf-8")
    (pkg/"summary_for_pi.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root",default="meta_model/functional_v1_experiments")
    ap.add_argument("--package_dir",default="")
    ap.add_argument("--diagnostic_csv",default="")
    ap.add_argument("--manual_crosswalk_csv",default="meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv")
    ap.add_argument("--llm_induced_schema_root",default="meta_model/functional_v1/llm_induced")
    ap.add_argument("--evidence_cards_root",default="meta_model/functional_v1/llm_induction_cards")
    ap.add_argument("--n_source_examples",type=int,default=12)
    ap.add_argument("--random_seed",type=int,default=17)
    ap.add_argument("--refresh_example_sample",action="store_true")
    ap.add_argument("--zip",action="store_true")
    args=ap.parse_args()
    out_root=Path(args.out_root); pkg=Path(args.package_dir) if args.package_dir else out_root/"pi_expert_review_package_v2"; pkg.mkdir(parents=True,exist_ok=True)
    diag_path=Path(args.diagnostic_csv) if args.diagnostic_csv else out_root/"diagnostics"/"roundtrip_diagnostic_metrics.csv"
    diag=read_csv(diag_path)
    if diag.empty: raise SystemExit(f"Missing or empty diagnostic CSV: {diag_path}")
    for folder in ["diagnostics","plots"]:
        src=out_root/folder; dst=pkg/folder
        if src.exists():
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(src,dst)
    snapshot=summarize_and_plot(diag,pkg)
    cross=pkg/"crosswalks"; cross.mkdir(exist_ok=True)
    manual=normalize_manual_crosswalk(Path(args.manual_crosswalk_csv),cross)
    llm=build_llm_crosswalk(Path(args.llm_induced_schema_root),Path(args.evidence_cards_root),manual,cross)
    combined=build_combined(manual,llm,cross)
    write_crosswalk_html(manual,llm,combined,cross/"v1_crosswalk_review_summary.html")
    build_examples(diag,pkg/"expert_review_examples",node_info(manual,llm),args.n_source_examples,args.random_seed,args.refresh_example_sample)
    write_readmes(pkg,snapshot,args.random_seed,args.n_source_examples)
    email="Subject: Informed consent meta-model expert review package v2\n\nDear [PI/team],\n\nI prepared a corrected review package with fixed source-sentence examples, clickable highlighted annotations, DUO/ICO/ODRL/FHIR-to-V1 crosswalks, and strategy-level meaning-preservation metrics.\n\nBest,\n[Your Name]\n"
    (pkg/"email_draft_to_pi.md").write_text(email,encoding="utf-8")
    if args.zip:
        z=shutil.make_archive(str(pkg),"zip",root_dir=pkg)
        print(f"Wrote zip archive: {z}")
    print(f"PI review package v2 ready: {pkg}")


if __name__=="__main__":
    main()
