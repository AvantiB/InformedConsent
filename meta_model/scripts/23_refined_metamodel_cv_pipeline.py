#!/usr/bin/env python
"""Refined informed-consent meta-model cross-validation pipeline.

Paper-facing workflow:
- create form-level cross-validation splits using stable form_key when available;
- build provenance-preserving mention evidence;
- split broad source elements into context-specific sense nodes;
- separate near-equivalence from broader/narrower, related-distinct, and complementary evidence;
- merge only strict near-equivalence edges into candidate fields;
- retain co-occurrence as provision-bundle/complementarity evidence;
- compute multi-layer lexical/cue preservation summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

TEXT_COLS = ["canonical_full_text", "full_text_original", "full_text", "sentence_text", "sentence", "text"]
FORM_COLS = ["form_key", "form_id", "source_file", "source_file_original", "source_id", "input_workbook"]
ID_COLS = ["sentence_id", "source_sentence_id", "roundtrip_id", "source_id", "ID", "id"]
ANNOTATION_COLS = ["annotations_json", "annotations_serialized", "annotations_raw", "annotations_combined"]
PRESERVED_COLS = ["meaning_preserved", "Results", "results", "roundtrip_decision", "expert_meaning_preserved"]
RECON_COLS = ["backward_mapping", "reconstructed_sentence", "reconstructed_text"]
INFO_COLS = ["information_model", "info_model", "canonical_information_model"]
LLM_COLS = ["llm", "model", "model_key"]
STOP = set("the a an and or of to in for with without by from on at as is are be been this that it its your you we our may can will shall my i they their them us all about into if then than".split())
CUE_WORDS = set("agree allow authorize consent permit decline refuse withdraw quit choose decide may can will cannot not optional required".split())


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def stable_id(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates}; available={list(df.columns)}")
    return None


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", norm(text).lower()) if t not in STOP]


def lexical_head(span: str) -> str:
    toks = tokenize(span)
    return toks[-1] if toks else ""


def canonical_span(span: str) -> str:
    return " ".join(tokenize(span))


def entropy(values: list[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    c = Counter(vals)
    total = sum(c.values())
    return -sum((n / total) * math.log2(n / total) for n in c.values())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def canonical_form_value(raw: str) -> str:
    v = norm(raw)
    if not v:
        return ""
    v = re.sub(r"\.(txt|csv|xlsx?)$", "", v, flags=re.I)
    v = re.sub(r"_annotated$", "", v, flags=re.I)
    v = re.sub(r"_output$", "", v, flags=re.I)
    v = re.sub(r"\s+annotated$", "", v, flags=re.I)
    v = re.sub(r"\s+output$", "", v, flags=re.I)
    v = re.sub(r"\s+copy(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    v = re.sub(r"_copy(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    v = re.sub(r"\s+", " ", v).strip(" _-")
    if not v or v.lower() in {"nan", "none", "null"}:
        return ""
    if v.startswith("FORM_") and "e3b0c442" in v:
        return ""
    return v


def form_value(row: pd.Series) -> str:
    for c in FORM_COLS:
        if c in row.index and norm(row.get(c)):
            v = canonical_form_value(row.get(c))
            if v:
                return v
    return ""


def source_sentence_id(row: pd.Series, text: str) -> str:
    for c in ID_COLS:
        if c in row.index and norm(row.get(c)):
            return norm(row.get(c))
    return "SENT_" + stable_id(text)


def parse_jsonish(text: str) -> Any:
    s = norm(text)
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|yaml)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    for l, r in [("[", "]"), ("{", "}")]:
        a, b = s.find(l), s.rfind(r)
        if a >= 0 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                pass
    return None


def parse_annotations(row: pd.Series) -> list[dict[str, Any]]:
    for c in ANNOTATION_COLS:
        if c in row.index and norm(row.get(c)):
            obj = parse_jsonish(norm(row.get(c)))
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                anns = obj.get("annotations") or obj.get("provisions") or []
                return [x for x in anns if isinstance(x, dict)] if isinstance(anns, list) else []
    return []


def positive_label(x: Any) -> int:
    s = norm(x).lower()
    if s in {"1", "1.0", "true", "yes", "y", "preserved", "pass", "passed", "meaning preserved"}:
        return 1
    if s in {"0", "0.0", "false", "no", "n", "failed", "fail", "not preserved"}:
        return 0
    return -1


def load_inventory(path: Path | None) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path and path.exists() else pd.DataFrame()


def inventory_lookup(inv: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if inv.empty:
        return out
    for _, r in inv.iterrows():
        uid = norm(r.get("union_element_id"))
        if uid:
            out[uid] = {k: norm(r.get(k)) for k in inv.columns}
    return out


def get_annotation_uid(ann: dict[str, Any], info_model: str) -> str:
    uid = norm(ann.get("union_element_id") or ann.get("source_element_id") or ann.get("element_id") or ann.get("label") or ann.get("source_element_label"))
    if uid and "::" not in uid and info_model:
        uid = f"{info_model}::{uid}"
    return uid


def get_annotation_span(ann: dict[str, Any]) -> str:
    for k in ["span_text", "evidence_span_text", "cue_span_text", "span", "text", "raw_span"]:
        if norm(ann.get(k)):
            return norm(ann.get(k))
    return ""


def make_folds(args: argparse.Namespace) -> None:
    source_path = Path(args.split_source_csv) if args.split_source_csv else Path(args.expert_roundtrips_csv)
    df = pd.read_csv(source_path).fillna("")
    text_col = pick_col(df, TEXT_COLS, required=True)
    groups: dict[str, dict[str, Any]] = {}
    audit_rows, excluded_rows = [], []
    for i, r in df.iterrows():
        text = norm(r.get(text_col))
        raw_form = ""
        for c in FORM_COLS:
            if c in df.columns and norm(r.get(c)):
                raw_form = norm(r.get(c)); break
        form = form_value(r)
        if not text or not form:
            excluded_rows.append({"source_row": int(i) + 2, "raw_form_value": raw_form, "reason": "empty_text_or_form"})
            continue
        h = stable_id(text)
        g = groups.setdefault(form, {"form_id": form, "canonical_form_id": form, "n_rows": 0, "sentence_hashes": set(), "raw_values": set()})
        g["n_rows"] += 1; g["sentence_hashes"].add(h); g["raw_values"].add(raw_form or form)
        audit_rows.append({"source_csv": source_path.name, "source_row": int(i) + 2, "raw_form_value": raw_form or form, "canonical_form_id": form, "sentence_id": source_sentence_id(r, text), "sentence_text_hash": h})
    forms = []
    for g in groups.values():
        forms.append({"form_id": g["form_id"], "canonical_form_id": g["canonical_form_id"], "n_rows": g["n_rows"], "n_unique_sentences": len(g["sentence_hashes"]), "n_raw_form_values": len(g["raw_values"]), "raw_form_values_json": json.dumps(sorted(g["raw_values"]), ensure_ascii=False)})
    forms = sorted(forms, key=lambda x: stable_id(x["canonical_form_id"] + str(args.seed)))
    for i, r in enumerate(forms):
        r["fold_id"] = i % int(args.n_folds)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forms).to_csv(out / "fold_assignments.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "form_grouping_audit.csv", index=False)
    if excluded_rows:
        pd.DataFrame(excluded_rows).to_csv(out / "excluded_empty_form_rows.csv", index=False)
    meta = {"n_forms": len(forms), "n_folds": int(args.n_folds), "split_unit": "canonical_consent_form", "seed": int(args.seed), "split_source_csv": source_path.name, "n_raw_form_values": int(sum(r["n_raw_form_values"] for r in forms)), "n_excluded_empty_or_unknown_rows": len(excluded_rows), "note": "Review form_grouping_audit.csv and excluded_empty_form_rows.csv before running fold induction."}
    (out / "fold_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out / 'fold_assignments.csv'}")


def build_mentions(df: pd.DataFrame, fold_map: dict[str, int], test_fold: int, inv: pd.DataFrame) -> pd.DataFrame:
    text_col = pick_col(df, TEXT_COLS, required=True)
    info_col = pick_col(df, INFO_COLS); llm_col = pick_col(df, LLM_COLS); preserved_col = pick_col(df, PRESERVED_COLS); recon_col = pick_col(df, RECON_COLS)
    lookup = inventory_lookup(inv); rows = []
    for row_idx, r in df.iterrows():
        text = norm(r.get(text_col)); form = form_value(r)
        if not text or not form:
            continue
        fold = fold_map.get(form)
        split = "unassigned" if fold is None else ("test" if fold == test_fold else "train")
        sent_id = source_sentence_id(r, text); context_id = f"{form}|{sent_id}"
        info = norm(r.get(info_col)) if info_col else ""; llm = norm(r.get(llm_col)) if llm_col else ""
        preserved = positive_label(r.get(preserved_col)) if preserved_col else -1; recon = norm(r.get(recon_col)) if recon_col else ""
        for j, ann in enumerate(parse_annotations(r), start=1):
            uid = get_annotation_uid(ann, info); span = get_annotation_span(ann)
            if not uid or not span or uid.lower() in {"na", "n/a", "none"}:
                continue
            invrow = lookup.get(uid, {})
            rows.append({"fold_id": test_fold, "split": split, "form_id": form, "source_row": int(row_idx) + 2, "sentence_context_id": context_id, "sentence_id": sent_id, "sentence_text": text, "information_model": info, "llm": llm, "annotation_index": j, "union_element_id": uid, "source_element_label": norm(ann.get("source_element_label") or invrow.get("source_element_label") or ann.get("label")), "source_element_definition": norm(invrow.get("source_element_definition")), "span_text": span, "span_canonical": canonical_span(span), "span_head": lexical_head(span), "span_token_count": len(tokenize(span)), "sentence_decision": norm(ann.get("decision_value") or ann.get("decision") or r.get("roundtrip_decision", "")), "meaning_preserved": preserved, "reconstructed_sentence": recon, "provenance_key": f"{form}|{sent_id}|{info}|{llm}|row{int(row_idx)+2}|ann{j}"})
    return pd.DataFrame(rows)


def sense_nodes(train: pd.DataFrame, min_support: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train.empty:
        return pd.DataFrame(), pd.DataFrame()
    head_counts = train.groupby(["union_element_id", "span_head"]).size().to_dict(); rows = []
    for _, r in train.iterrows():
        head = norm(r.get("span_head")) or "no_head"; uid = norm(r.get("union_element_id"))
        if head_counts.get((uid, head), 0) < min_support:
            head = "misc"
        x = r.to_dict(); x["sense_key"] = head; x["sense_id"] = f"{uid}__sense_{re.sub(r'[^A-Za-z0-9]+', '_', head).strip('_') or 'misc'}"; rows.append(x)
    mentions = pd.DataFrame(rows); node_rows = []
    for sid, g in mentions.groupby("sense_id"):
        spans = g["span_text"].astype(str).tolist(); uids = sorted(g["union_element_id"].astype(str).unique()); source_models = sorted({u.split("::", 1)[0] for u in uids if "::" in u})
        node_rows.append({"sense_id": sid, "union_element_ids": json.dumps(uids, ensure_ascii=False), "source_models": json.dumps(source_models, ensure_ascii=False), "n_mentions": len(g), "n_positive_mentions": int((g["meaning_preserved"] == 1).sum()), "n_forms": g["form_id"].nunique(), "n_sentences": g["sentence_context_id"].nunique(), "top_span_heads": json.dumps([x for x, _ in Counter(g["span_head"]).most_common(10) if x], ensure_ascii=False), "top_spans": json.dumps([x for x, _ in Counter(spans).most_common(15) if x], ensure_ascii=False), "span_head_entropy": entropy(g["span_head"].astype(str).tolist()), "source_element_labels": json.dumps([x for x, _ in Counter(g["source_element_label"]).most_common(10) if x], ensure_ascii=False)})
    return mentions, pd.DataFrame(node_rows)


def classify_pair(a: pd.Series, b: pd.Series, overlap_threshold: float) -> tuple[str, float]:
    sa, sb = norm(a.get("span_text")), norm(b.get("span_text")); ca, cb = norm(a.get("span_canonical")), norm(b.get("span_canonical")); ha, hb = norm(a.get("span_head")), norm(b.get("span_head"))
    if not sa or not sb or a.get("sense_id") == b.get("sense_id"):
        return "self_or_empty", 0.0
    jac = jaccard(sa, sb); toks_a, toks_b = set(tokenize(sa)), set(tokenize(sb))
    if ca and ca == cb:
        return "near_equivalent", 1.0
    if toks_a and toks_b and (toks_a < toks_b or toks_b < toks_a):
        return "broader_narrower", jac
    if jac >= overlap_threshold and ha and ha == hb:
        return "near_equivalent", jac
    if jac > 0:
        return "related_distinct", jac
    return "complementary", 0.0


def build_relationships(sense_mentions: pd.DataFrame, overlap_threshold: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_stats: dict[tuple[str, str, str], dict[str, Any]] = {}; bundle_stats: dict[tuple[str, str], dict[str, Any]] = {}
    if sense_mentions.empty:
        return pd.DataFrame(), pd.DataFrame()
    freq = sense_mentions["sense_id"].value_counts().to_dict()
    for _, g0 in sense_mentions.groupby("sentence_context_id"):
        g = g0.drop_duplicates(subset=["sense_id", "span_canonical", "information_model", "llm"])
        for ra, rb in combinations(g.to_dict("records"), 2):
            a, b = pd.Series(ra), pd.Series(rb); s1, s2 = sorted([str(a["sense_id"]), str(b["sense_id"])]); rel, ov = classify_pair(a, b, overlap_threshold)
            if rel == "self_or_empty":
                continue
            pos = 1 if int(a.get("meaning_preserved", -1)) == 1 or int(b.get("meaning_preserved", -1)) == 1 else 0
            d = edge_stats.setdefault((s1, s2, rel), {"sense_id_1": s1, "sense_id_2": s2, "relationship_type": rel, "n_contexts": 0, "n_positive_contexts": 0, "max_overlap": 0.0, "forms": set(), "information_models": set(), "llms": set(), "example_spans": []})
            d["n_contexts"] += 1; d["n_positive_contexts"] += pos; d["max_overlap"] = max(float(d["max_overlap"]), ov); d["forms"].add(str(a.get("form_id", ""))); d["information_models"].update([str(a.get("information_model", "")), str(b.get("information_model", ""))]); d["llms"].update([str(a.get("llm", "")), str(b.get("llm", ""))])
            if len(d["example_spans"]) < 5:
                d["example_spans"].append(f"{a.get('span_text')} || {b.get('span_text')}")
            bd = bundle_stats.setdefault((s1, s2), {"sense_id_1": s1, "sense_id_2": s2, "n_cooccurrence_contexts": 0, "n_positive_contexts": 0, "forms": set(), "example_spans": []})
            bd["n_cooccurrence_contexts"] += 1; bd["n_positive_contexts"] += pos; bd["forms"].add(str(a.get("form_id", "")))
            if len(bd["example_spans"]) < 5:
                bd["example_spans"].append(f"{a.get('span_text')} || {b.get('span_text')}")
    rel_rows = []
    for (s1, s2, rel), d in edge_stats.items():
        hub_penalty = math.sqrt(max(1, freq.get(s1, 1)) * max(1, freq.get(s2, 1))); strict = 1.0 if rel == "near_equivalent" else 0.0; support = float(d["n_positive_contexts"]) + 0.25 * float(d["n_contexts"] - d["n_positive_contexts"]); weight = strict * support * (1 + 0.05 * max(0, len(d["information_models"]) - 1)) / hub_penalty
        rel_rows.append({"sense_id_1": s1, "sense_id_2": s2, "relationship_type": rel, "n_contexts": d["n_contexts"], "n_positive_contexts": d["n_positive_contexts"], "n_forms": len(d["forms"]), "n_information_models": len([x for x in d["information_models"] if x]), "n_llms": len([x for x in d["llms"] if x]), "max_span_overlap": round(float(d["max_overlap"]), 4), "equivalence_weight": round(weight, 6), "example_span_pairs_json": json.dumps(d["example_spans"], ensure_ascii=False)})
    bundle_rows = [{"sense_id_1": s1, "sense_id_2": s2, "edge_type": "provision_bundle_complementarity", "n_cooccurrence_contexts": d["n_cooccurrence_contexts"], "n_positive_contexts": d["n_positive_contexts"], "n_forms": len(d["forms"]), "example_span_pairs_json": json.dumps(d["example_spans"], ensure_ascii=False)} for (s1, s2), d in bundle_stats.items()]
    return pd.DataFrame(rel_rows), pd.DataFrame(bundle_rows)


class UnionFind:
    def __init__(self):
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


def equivalence_clusters(nodes: pd.DataFrame, edges: pd.DataFrame, min_weight: float, min_contexts: int) -> pd.DataFrame:
    uf = UnionFind()
    for sid in nodes.get("sense_id", pd.Series(dtype=str)).astype(str):
        uf.find(sid)
    if not edges.empty:
        use = edges[(edges["relationship_type"] == "near_equivalent") & (pd.to_numeric(edges["equivalence_weight"], errors="coerce") >= min_weight) & (pd.to_numeric(edges["n_positive_contexts"], errors="coerce") >= min_contexts)]
        for _, r in use.iterrows():
            uf.union(str(r["sense_id_1"]), str(r["sense_id_2"]))
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in nodes.get("sense_id", pd.Series(dtype=str)).astype(str):
        groups[uf.find(sid)].append(sid)
    node_map = {str(r["sense_id"]): r.to_dict() for _, r in nodes.iterrows()}; rows = []
    for i, senses in enumerate(sorted(groups.values(), key=lambda xs: (-len(xs), xs[0])), start=1):
        cid = f"F{i:03d}"; sub = [node_map[s] for s in senses if s in node_map]; spans, labels = [], []
        for x in sub:
            spans += json.loads(x.get("top_spans", "[]") or "[]"); labels += json.loads(x.get("source_element_labels", "[]") or "[]")
        n_mentions = sum(int(x.get("n_mentions", 0)) for x in sub); n_pos = sum(int(x.get("n_positive_mentions", 0)) for x in sub); source_models = sorted({m for x in sub for m in json.loads(x.get("source_models", "[]") or "[]")})
        rows.append({"candidate_field_id": cid, "sense_ids_json": json.dumps(senses, ensure_ascii=False), "n_sense_nodes": len(senses), "n_mentions": n_mentions, "n_positive_mentions": n_pos, "positive_fraction": round(n_pos / n_mentions, 4) if n_mentions else 0, "source_models_json": json.dumps(source_models, ensure_ascii=False), "top_spans_json": json.dumps([x for x, _ in Counter(spans).most_common(20)], ensure_ascii=False), "suggested_terms_json": json.dumps([x for x, _ in Counter(t for s in spans + labels for t in tokenize(s)).most_common(12)], ensure_ascii=False)})
    return pd.DataFrame(rows)


def build_schema(clusters: pd.DataFrame, out_yaml: Path, fold_id: int, args: argparse.Namespace) -> None:
    fields = []
    for _, r in clusters.iterrows():
        if int(r.get("n_positive_mentions", 0)) < int(args.min_field_positive_mentions):
            continue
        fields.append({"name": f"field_{r['candidate_field_id']}", "status": "fold_specific_candidate", "candidate_field_id": r["candidate_field_id"], "description": "Data-derived candidate field from strict near-equivalence among context-specific source-element senses.", "suggested_terms": json.loads(r.get("suggested_terms_json", "[]") or "[]"), "positive_span_examples": json.loads(r.get("top_spans_json", "[]") or "[]")[:10], "source_model_support": json.loads(r.get("source_models_json", "[]") or "[]"), "evidence": {"n_sense_nodes": int(r.get("n_sense_nodes", 0)), "n_mentions": int(r.get("n_mentions", 0)), "n_positive_mentions": int(r.get("n_positive_mentions", 0)), "positive_fraction": float(r.get("positive_fraction", 0))}})
    schema = {"meta_model_id": f"refined_consent_metamodel_fold_{fold_id}", "status": "fold_specific_candidate_for_heldout_evaluation", "derivation_split": {"fold_id": fold_id, "training_forms_only": True, "test_forms_excluded_from_schema_development": True}, "method": {"unit_of_analysis": "source-element-in-context mention", "sense_induction": "source elements split by observed lexical head/evidence-span usage before graph construction", "merge_rule": "only strict near-equivalence edges can merge candidate fields", "cooccurrence_rule": "provision-bundle cooccurrence is retained as complementarity evidence and is not used directly for merging"}, "decision": {"scope": "sentence_or_provision_level", "allowed_values": ["permit", "deny", "mixed", "unclear"]}, "fields": fields, "residual_important_content": {"description": "Meaning-critical content not captured by candidate fields."}, "provenance": {"required": True, "note": "Preserve form, sentence, source model, LLM, span, and candidate field evidence."}}
    out_yaml.parent.mkdir(parents=True, exist_ok=True); out_yaml.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)); out_yaml.with_suffix(".json").write_text(json.dumps(schema, indent=2, ensure_ascii=False))


def run_fold(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.expert_roundtrips_csv).fillna(""); folds = pd.read_csv(args.fold_assignments_csv).fillna("")
    fold_col = "canonical_form_id" if "canonical_form_id" in folds.columns else "form_id"; fold_map = {norm(r[fold_col]): int(r["fold_id"]) for _, r in folds.iterrows()}
    inv = load_inventory(Path(args.inventory_csv)) if args.inventory_csv else pd.DataFrame(); out = Path(args.output_dir) / f"fold_{int(args.fold_id):02d}"; out.mkdir(parents=True, exist_ok=True)
    mentions = build_mentions(df, fold_map, int(args.fold_id), inv); mentions.to_csv(out / "evidence_mentions_all.csv", index=False)
    train = mentions[mentions["split"] == "train"].copy(); test = mentions[mentions["split"] == "test"].copy(); unassigned = mentions[mentions["split"] == "unassigned"].copy()
    train.to_csv(out / "evidence_mentions_train.csv", index=False); test.to_csv(out / "evidence_mentions_test_provenance_only.csv", index=False)
    if not unassigned.empty:
        unassigned.to_csv(out / "evidence_mentions_unassigned_review.csv", index=False)
    sm, nodes = sense_nodes(train, int(args.min_sense_support)); sm.to_csv(out / "source_element_sense_mentions_train.csv", index=False); nodes.to_csv(out / "source_element_sense_nodes.csv", index=False)
    rel, bundle = build_relationships(sm, float(args.span_overlap_threshold)); rel.to_csv(out / "typed_relationship_edges.csv", index=False); bundle.to_csv(out / "provision_bundle_edges.csv", index=False)
    clusters = equivalence_clusters(nodes, rel, float(args.min_equivalence_weight), int(args.min_equivalence_positive_contexts)); clusters.to_csv(out / "candidate_field_clusters.csv", index=False)
    build_schema(clusters, out / "refined_candidate_schema.yaml", int(args.fold_id), args)
    meta = {"fold_id": int(args.fold_id), "n_train_mentions": int(len(train)), "n_test_mentions_for_provenance": int(len(test)), "n_unassigned_mentions": int(len(unassigned)), "n_train_forms": int(train["form_id"].nunique()) if not train.empty else 0, "n_test_forms": int(test["form_id"].nunique()) if not test.empty else 0, "n_sense_nodes": int(len(nodes)), "n_relationship_edges": int(len(rel)), "n_bundle_edges": int(len(bundle)), "n_candidate_fields": int(len(clusters))}
    (out / "fold_run_metadata.json").write_text(json.dumps(meta, indent=2)); print(f"Wrote fold outputs to {out}")


def summarize_folds(args: argparse.Namespace) -> None:
    root = Path(args.fold_root); rows = []
    for p in sorted(root.glob("fold_*/candidate_field_clusters.csv")):
        fold = p.parent.name; df = pd.read_csv(p).fillna("")
        for _, r in df.iterrows():
            terms = tuple(json.loads(r.get("suggested_terms_json", "[]") or "[]")[:5])
            rows.append({"fold": fold, "candidate_field_id": r.get("candidate_field_id"), "signature_terms": " | ".join(terms), "n_positive_mentions": r.get("n_positive_mentions"), "n_sense_nodes": r.get("n_sense_nodes"), "source_models_json": r.get("source_models_json")})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(out / "fold_candidate_fields_long.csv", index=False)
    if not df.empty:
        rec = df.groupby("signature_terms", dropna=False).agg(n_folds=("fold", "nunique"), folds=("fold", lambda x: json.dumps(sorted(set(x)), ensure_ascii=False)), total_positive_mentions=("n_positive_mentions", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum()))).reset_index().sort_values(["n_folds", "total_positive_mentions"], ascending=[False, False]); rec.to_csv(out / "field_recurrence_across_folds.csv", index=False)
    print(f"Wrote fold stability summaries to {out}")


def content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if len(t) > 2]


def eval_pairs(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.standardized_roundtrips_csv).fillna(""); orig_col = pick_col(df, ["original_text", "source_text", "canonical_full_text", "full_text"], required=True); rec_col = pick_col(df, ["reconstructed_text", "reconstructed_sentence", "backward_mapping"], required=True)
    rows = []
    for _, r in df.iterrows():
        orig = norm(r.get(orig_col)); rec = norm(r.get(rec_col)); o, rr = set(content_words(orig)), set(content_words(rec)); inter = o & rr; cue_o = {x for x in o if x in CUE_WORDS}; cue_r = {x for x in rr if x in CUE_WORDS}
        rows.append({"roundtrip_id": norm(r.get("roundtrip_id", "")), "condition": norm(r.get("condition", "")), "information_model": norm(r.get("information_model", "")), "llm": norm(r.get("llm", "")), "content_word_recall": len(inter) / len(o) if o else 0, "content_word_precision": len(inter) / len(rr) if rr else 0, "content_words_original": len(o), "content_words_reconstructed": len(rr), "dropped_content_words_json": json.dumps(sorted(o - rr), ensure_ascii=False), "added_content_words_json": json.dumps(sorted(rr - o), ensure_ascii=False), "cue_recall": len(cue_o & cue_r) / len(cue_o) if cue_o else 1.0, "annotation_count": r.get("annotation_count", ""), "unique_element_count": r.get("unique_element_count", ""), "meaning_preserved": r.get("meaning_preserved", "")})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); res = pd.DataFrame(rows); res.to_csv(out / "lexical_cue_preservation_long.csv", index=False)
    if not res.empty:
        summary = res.groupby(["condition", "information_model", "llm"], dropna=False).agg(mean_content_recall=("content_word_recall", "mean"), mean_content_precision=("content_word_precision", "mean"), mean_cue_recall=("cue_recall", "mean"), mean_annotation_count=("annotation_count", lambda x: pd.to_numeric(x, errors="coerce").mean()), mean_unique_fields=("unique_element_count", lambda x: pd.to_numeric(x, errors="coerce").mean()), n=("roundtrip_id", "count")).reset_index(); summary.to_csv(out / "coverage_complexity_summary.csv", index=False)
    print(f"Wrote multi-layer evaluation metrics to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-folds"); p.add_argument("--expert_roundtrips_csv", required=True); p.add_argument("--split_source_csv", default="", help="Use main roundtrips.csv with stable form_key to create folds; defaults to expert_roundtrips_csv."); p.add_argument("--output_dir", required=True); p.add_argument("--n_folds", type=int, default=4); p.add_argument("--seed", type=int, default=17)
    p = sub.add_parser("run-fold"); p.add_argument("--expert_roundtrips_csv", required=True); p.add_argument("--fold_assignments_csv", required=True); p.add_argument("--inventory_csv", default=""); p.add_argument("--output_dir", required=True); p.add_argument("--fold_id", type=int, required=True); p.add_argument("--min_sense_support", type=int, default=2); p.add_argument("--span_overlap_threshold", type=float, default=0.75); p.add_argument("--min_equivalence_weight", type=float, default=0.02); p.add_argument("--min_equivalence_positive_contexts", type=int, default=1); p.add_argument("--min_field_positive_mentions", type=int, default=5)
    p = sub.add_parser("summarize-folds"); p.add_argument("--fold_root", required=True); p.add_argument("--output_dir", required=True)
    p = sub.add_parser("evaluate-roundtrips"); p.add_argument("--standardized_roundtrips_csv", required=True); p.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    if args.cmd == "make-folds": make_folds(args)
    elif args.cmd == "run-fold": run_fold(args)
    elif args.cmd == "summarize-folds": summarize_folds(args)
    elif args.cmd == "evaluate-roundtrips": eval_pairs(args)


if __name__ == "__main__":
    main()
