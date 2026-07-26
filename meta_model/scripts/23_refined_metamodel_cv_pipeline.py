#!/usr/bin/env python
"""Data-driven informed-consent meta-model induction with form-level CV.

Paper-facing workflow implemented here:

A. Build one provenance-preserving row per original annotation mention.
B. Retain only span-level annotations whose human round-trip meaning-preservation
   label is 1 as valid induction evidence.
C. Split broad source elements into context-specific source-element-sense nodes.
D. Compute recurrence, span overlap, nesting, co-occurrence, proximity,
   cross-model/cross-LLM support, lexical/embedding similarity, PMI/lift, and
   positive/negative preservation support for sense-node pairs.
E. Type pairwise relations as near_equivalent, broader_narrower,
   complementary, related_context_only, or unsafe_to_merge.
F. Cluster only near_equivalent edges. Preserve complementary/proximity edges as
   a separate functional/provision-bundle graph.

The evidence-card and LLM induction stages are handled by scripts 28 and 29.
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

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

TEXT_COLS = ["canonical_full_text", "full_text_original", "full_text", "sentence_text", "sentence", "text"]
FORM_COLS = ["form_key", "form_id", "source_file", "source_file_original", "input_workbook"]
ID_COLS = ["sentence_text_id", "sentence_id", "source_sentence_id", "canonical_source_key", "roundtrip_id", "source_id"]
ANNOTATION_COLS = ["annotations_json", "annotations_serialized", "annotations_raw", "annotations_combined", "forward_mapping"]
PRESERVED_COLS = ["meaning_preserved", "human_meaning_preserved", "expert_meaning_preserved", "Results", "results"]
INFO_COLS = ["information_model", "info_model", "canonical_information_model", "source_model"]
LLM_COLS = ["llm", "model", "model_key"]
ROUNDTRIP_COLS = ["roundtrip_id", "source_id"]

NA_VALUES = {"", "na", "n/a", "none", "null", "unknown", "not applicable", "no annotation"}
RESERVED_LABELS = {"unmatched_language", "unmatched", "interpretation_units", "rationale", "raw_response", "raw reconstruction text"}
SENTENCE_LEVEL_LABELS = {
    "odrl::rule_testsentence", "rule_testsentence",
    "fhir_consent::consent.provision.type", "fhir::consent.provision.type", "consent.provision.type",
    "fhir_consent::consent.decision", "fhir::consent.decision", "consent.decision",
    "duo::duo.decision", "ico::ico.decision",
}
STOP = set("the a an and or of to in for with without by from on at as is are be been this that it its your you we our may can will shall my i they their them us all about into if then than".split())
DECISION_ARTIFACT_TERMS = set("permit permits permitted permitting deny denies denied denying permission permissions prohibition prohibitions prohibit prohibited obligation obligations required requirement requirements unclear mixed".split())
NEGATION_TERMS = {"no", "not", "never", "without", "cannot", "can't", "won't", "will not", "may not"}
CONDITIONAL_TERMS = {"if", "unless", "when", "provided", "upon", "only if"}
UNCERTAINTY_TERMS = {"may", "might", "could", "possibly", "uncertain"}

ROLE_LEXICONS = {
    "participant_or_subject": {"i", "me", "my", "you", "your", "participant", "subject", "patient"},
    "researcher_or_authorized_actor": {"researcher", "researchers", "investigator", "investigators", "team", "staff", "scientist"},
    "institution_or_custodian": {"institution", "organization", "hospital", "university", "clinic", "sponsor", "custodian"},
    "repository_or_registry": {"database", "databases", "biobank", "repository", "registry", "archive"},
    "resource_or_specimen": {"data", "information", "record", "records", "sample", "samples", "specimen", "biospecimen", "blood", "dna", "urine", "saliva"},
    "action": {"collect", "store", "stored", "use", "used", "share", "shared", "disclose", "withdraw", "stop", "destroy", "retrieve", "sell", "sold", "contact"},
    "purpose_or_use_context": {"research", "study", "studies", "purpose", "topic", "care", "treatment", "commercial"},
    "temporal_scope": {"time", "future", "before", "after", "during", "date", "years", "expiration", "beginning"},
    "privacy_or_identifiability": {"identifiable", "deidentified", "de-identified", "anonymous", "anonymized", "confidential", "privacy"},
    "condition_or_restriction": {"if", "unless", "only", "except", "restriction", "limitation", "required", "approved"},
    "choice_or_right": {"agree", "consent", "choose", "choice", "yes", "no", "withdraw", "decline", "refuse", "permission"},
    "consequence_or_protection": {"penalty", "care", "benefit", "risk", "affect", "protection", "compensation"},
}


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
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates}; available={list(df.columns)}")
    return None


def canonical_form_value(raw: Any) -> str:
    v = norm(raw)
    if not v:
        return ""
    v = re.sub(r"\.(txt|csv|xlsx?)$", "", v, flags=re.I)
    v = re.sub(r"(_annotated|_output)$", "", v, flags=re.I)
    v = re.sub(r"\s+(annotated|output)$", "", v, flags=re.I)
    v = re.sub(r"(?:_copy|\s+copy)(?:[_\s-]*\d+)?$", "", v, flags=re.I)
    return v.strip(" _-")


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


def roundtrip_id(row: pd.Series, row_idx: int) -> str:
    for c in ROUNDTRIP_COLS:
        if c in row.index and norm(row.get(c)):
            return norm(row.get(c))
    return f"ROW_{row_idx + 2}"


def positive_label(x: Any) -> int:
    s = norm(x).lower()
    if s in {"1", "1.0", "true", "yes", "y", "preserved", "pass", "passed", "meaning preserved"}:
        return 1
    if s in {"0", "0.0", "false", "no", "n", "failed", "fail", "not preserved"}:
        return 0
    return -1


def normalize_decision_value(x: Any) -> str:
    s = norm(x).strip("()[]{} ").lower()
    if s in {"permit", "permitted", "permission", "allow", "allowed", "authorization", "authorize"}:
        return "permit"
    if s in {"deny", "denied", "prohibition", "prohibit", "prohibited", "disallow", "disallowed"}:
        return "deny"
    if s in {"mixed", "both"}:
        return "mixed"
    if s in {"unclear", "unknown", "na", "n/a", "none", "null"}:
        return "unclear"
    return s if s else ""


def tokenize(text: str, include_stop: bool = True) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", norm(text).lower())
    return toks if include_stop else [t for t in toks if t not in STOP and t not in DECISION_ARTIFACT_TERMS]


def lexical_head(span: str) -> str:
    toks = tokenize(span, include_stop=False)
    return toks[-1] if toks else ""


def canonical_span(span: str) -> str:
    return " ".join(tokenize(span, include_stop=False))


def jaccard_text(a: str, b: str) -> float:
    sa, sb = set(tokenize(a, include_stop=False)), set(tokenize(b, include_stop=False))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def linguistic_polarity(span: str) -> str:
    s = f" {norm(span).lower()} "
    if any(f" {x} " in s for x in CONDITIONAL_TERMS):
        return "conditional"
    if any(f" {x} " in s for x in NEGATION_TERMS):
        return "negated"
    if any(f" {x} " in s for x in UNCERTAINTY_TERMS):
        return "uncertain"
    return "affirmed"


def role_signature(span: str, local_context: str = "") -> str:
    toks = set(tokenize(f"{span} {local_context}", include_stop=True))
    scores = {role: len(toks & words) for role, words in ROLE_LEXICONS.items()}
    best = max(scores, key=scores.get) if scores else "other"
    return best if scores.get(best, 0) > 0 else "other"


def parse_jsonish(text: Any) -> Any:
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


def load_inventory(path: Path | None) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path and path.exists() else pd.DataFrame()


def inventory_lookup(inv: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    if inv.empty:
        return out
    for _, r in inv.iterrows():
        model = norm(r.get("source_model"))
        meta = {
            "source_model": model,
            "source_element_id": norm(r.get("source_element_id")),
            "source_element_label": norm(r.get("source_element_label")),
            "source_element_definition": norm(r.get("source_element_definition")),
            "element_scope": norm(r.get("element_scope")),
        }
        aliases = {model, model.replace("_Consent", ""), model.replace("_", "")}
        for alias in aliases:
            for key in [meta["source_element_id"], meta["source_element_label"]]:
                if key:
                    out[(alias.casefold(), key.casefold())] = meta
    return out


def is_na(x: Any) -> bool:
    return norm(x).casefold().strip("[](){} .;:,\t\n") in NA_VALUES


def is_sentence_level_label(model: str, label: str) -> bool:
    raw = norm(label).casefold()
    model_cf = norm(model).casefold()
    return raw in SENTENCE_LEVEL_LABELS or f"{model_cf}::{raw}" in SENTENCE_LEVEL_LABELS


def compact_annotation_parser(text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"(.+?)\s*\[([^\[\]]+)\]\s*\(([^)]*)\)(?=\s+.+?\s*\[[^\[\]]+\]\s*\([^)]*\)|\s*$)")
    return [{"span_text": norm(m.group(1)), "label": norm(m.group(2)), "decision_value": norm(m.group(3))} for m in pattern.finditer(norm(text))]


def parse_annotations(row: pd.Series, ann_col: str, info_model: str, lookup: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = norm(row.get(ann_col))
    obj = parse_jsonish(raw)
    raw_anns: list[Any] = []
    if isinstance(obj, list):
        raw_anns = obj
    elif isinstance(obj, dict):
        raw_anns = obj.get("annotations") or obj.get("valid_annotations") or []
    if not raw_anns:
        raw_anns = compact_annotation_parser(raw)

    valid, audit = [], []
    for i, ann in enumerate(raw_anns, start=1):
        if not isinstance(ann, dict):
            continue
        span = norm(ann.get("span_text") or ann.get("evidence_span_text") or ann.get("cue_span_text") or ann.get("span") or ann.get("text") or ann.get("value"))
        label = norm(ann.get("source_element_label") or ann.get("label_name") or ann.get("field_name") or ann.get("label") or ann.get("element_label") or ann.get("name"))
        element_id = norm(ann.get("source_element_id") or ann.get("union_element_id") or ann.get("label_id") or ann.get("element_id") or ann.get("field_id") or ann.get("id"))
        decision = normalize_decision_value(ann.get("decision_value") or ann.get("decision") or ann.get("sentence_decision") or ann.get("polarity"))
        key = label or element_id
        reason = ""
        if is_na(span) and is_na(key):
            reason = "na_only"
        elif not span or not key:
            reason = "empty_span_or_label"
        elif key.casefold() in RESERVED_LABELS:
            reason = "reserved_non_annotation_content"
        elif is_sentence_level_label(info_model, key):
            reason = "sentence_level_label"
        if reason:
            audit.append({"annotation_index": i, "span_text": span, "label": key, "reason": reason})
            continue

        meta = None
        aliases = {info_model, info_model.replace("_Consent", ""), info_model.replace("_", "")}
        for alias in aliases:
            meta = lookup.get((alias.casefold(), key.casefold()))
            if meta:
                break
        valid.append({
            "annotation_index": i,
            "span_text": span,
            "source_model": norm(meta.get("source_model")) if meta else info_model,
            "source_element_id": norm(meta.get("source_element_id")) if meta else element_id,
            "source_element_label": norm(meta.get("source_element_label")) if meta else label or key,
            "source_element_definition": norm(meta.get("source_element_definition")) if meta else "",
            "decision_value": decision,
        })
    return valid, audit


def find_span_offsets(sentence: str, span: str, occurrence_index: int = 0) -> tuple[int, int]:
    if not sentence or not span:
        return -1, -1
    s, q = sentence.casefold(), span.casefold()
    starts = [m.start() for m in re.finditer(re.escape(q), s)]
    if not starts:
        return -1, -1
    start = starts[min(occurrence_index, len(starts) - 1)]
    return start, start + len(span)


def token_offsets(sentence: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\b\w+(?:[-']\w+)*\b", sentence)]


def char_to_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    if start < 0 or end < 0:
        return -1, -1
    hits = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    return (min(hits), max(hits)) if hits else (-1, -1)


def build_mentions(df: pd.DataFrame, folds: pd.DataFrame, test_fold: str, inv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_col = pick_col(df, TEXT_COLS, required=True)
    ann_col = pick_col(df, ANNOTATION_COLS, required=True)
    info_col = pick_col(df, INFO_COLS)
    llm_col = pick_col(df, LLM_COLS)
    preserved_col = pick_col(df, PRESERVED_COLS, required=True)
    fold_map = {norm(r["form_key"]): norm(r["fold_id"]) for _, r in folds.iterrows()}
    lookup = inventory_lookup(inv)
    rows, audit_rows = [], []

    for row_idx, r in df.iterrows():
        text = norm(r.get(text_col))
        form = form_value(r)
        if not text or not form:
            continue
        fold = fold_map.get(form, "")
        split = "test" if fold == test_fold else ("train" if fold else "unassigned")
        sent_id = source_sentence_id(r, text)
        context_id = f"{form}|{sent_id}"
        info = norm(r.get(info_col)) if info_col else ""
        llm = norm(r.get(llm_col)) if llm_col else ""
        preserved = positive_label(r.get(preserved_col))
        anns, audit = parse_annotations(r, ann_col, info, lookup)
        for a in audit:
            audit_rows.append({"source_row": row_idx + 2, "roundtrip_id": roundtrip_id(r, row_idx), "form_id": form, "sentence_id": sent_id, "information_model": info, "llm": llm, **a})
        occurrence_counter: Counter[str] = Counter()
        toks = token_offsets(text)
        for ann in anns:
            span = ann["span_text"]
            occ = occurrence_counter[span.casefold()]
            occurrence_counter[span.casefold()] += 1
            start, end = find_span_offsets(text, span, occ)
            tok_start, tok_end = char_to_token_span(toks, start, end)
            left = text[max(0, start - 100):start] if start >= 0 else ""
            right = text[end:min(len(text), end + 100)] if end >= 0 else ""
            source_id = ann["source_element_id"] or ann["source_element_label"]
            source_key = f"{ann['source_model']}::{source_id}"
            validity = "valid_positive" if preserved == 1 else ("invalid_negative" if preserved == 0 else "unlabeled")
            rows.append({
                "test_fold": test_fold,
                "split": split,
                "roundtrip_id": roundtrip_id(r, row_idx),
                "sentence_context_id": context_id,
                "sentence_id": sent_id,
                "form_id": form,
                "source_row": row_idx + 2,
                "source_model": ann["source_model"],
                "information_model": info,
                "llm": llm,
                "source_element_id": ann["source_element_id"],
                "source_element_label": ann["source_element_label"],
                "source_element_definition": ann["source_element_definition"],
                "source_element_key": source_key,
                "span_text": span,
                "span_start": start,
                "span_end": end,
                "token_start": tok_start,
                "token_end": tok_end,
                "sentence_text": text,
                "local_context": norm(f"{left} || {right}"),
                "span_canonical": canonical_span(span),
                "span_head": lexical_head(span),
                "span_token_count": len(tokenize(span, include_stop=False)),
                "role_signature": role_signature(span, f"{left} {right}"),
                "linguistic_polarity": linguistic_polarity(span),
                "decision_value": ann["decision_value"],
                "roundtrip_meaning_preserved": preserved,
                "annotation_validity_status": validity,
                "provenance_key": f"{form}|{sent_id}|{info}|{llm}|row{row_idx + 2}|ann{ann['annotation_index']}",
            })
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def embed_texts(texts: list[str], model_name: str | None, device: str | None, batch_size: int) -> tuple[np.ndarray, str]:
    if model_name:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
            return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True), "sentence_transformers"
        except Exception as exc:
            print(f"[WARN] embedding model failed; using TF-IDF: {exc}", flush=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vec.fit_transform(texts)
    return normalize(X).toarray(), "tfidf_char"


def agglomerative_labels(Z: np.ndarray, distance_threshold: float) -> np.ndarray:
    if len(Z) <= 1:
        return np.zeros(len(Z), dtype=int)
    try:
        return AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average").fit_predict(Z)
    except TypeError:
        return AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity="cosine", linkage="average").fit_predict(Z)


def induce_senses(all_mentions: pd.DataFrame, min_support: int, distance_threshold: float, embedding_model: str | None, device: str | None, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if all_mentions.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    positive = all_mentions[(all_mentions["split"] == "train") & (all_mentions["roundtrip_meaning_preserved"] == 1)].copy()
    assigned = all_mentions.copy()
    assigned["source_element_sense_id"] = ""
    metadata: dict[str, Any] = {"embedding_backends": Counter(), "n_source_elements": 0}

    for source_key, gpos in positive.groupby("source_element_key"):
        metadata["n_source_elements"] += 1
        idx_pos = gpos.index.tolist()
        texts_pos = [norm(f"span={r.span_text} context={r.local_context} role={r.role_signature}") for r in gpos.itertuples()]
        Zpos, backend = embed_texts(texts_pos, embedding_model, device, batch_size)
        metadata["embedding_backends"][backend] += 1
        labels = np.zeros(len(gpos), dtype=int) if len(gpos) < min_support else agglomerative_labels(Zpos, distance_threshold)
        centroids: dict[int, np.ndarray] = {}
        names: dict[int, str] = {}
        for cid in sorted(set(labels.tolist())):
            mask = labels == cid
            centroid = Zpos[mask].mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            centroids[cid] = centroid
            roles = [gpos.iloc[i]["role_signature"] for i in np.where(mask)[0]]
            heads = [gpos.iloc[i]["span_head"] for i in np.where(mask)[0]]
            role = Counter(roles).most_common(1)[0][0] if roles else "other"
            head = Counter([h for h in heads if h]).most_common(1)[0][0] if any(heads) else "misc"
            names[cid] = f"{source_key}__sense_{re.sub(r'[^A-Za-z0-9]+', '_', role).strip('_')}_{re.sub(r'[^A-Za-z0-9]+', '_', head).strip('_') or 'misc'}_{cid:02d}"
        for local_i, row_i in enumerate(idx_pos):
            assigned.at[row_i, "source_element_sense_id"] = names[int(labels[local_i])]

        gall = assigned[assigned["source_element_key"] == source_key]
        other_idx = [i for i in gall.index if i not in idx_pos]
        if other_idx:
            texts_all = [norm(f"span={assigned.at[i, 'span_text']} context={assigned.at[i, 'local_context']} role={assigned.at[i, 'role_signature']}") for i in other_idx]
            if backend == "sentence_transformers" and embedding_model:
                Zall, _ = embed_texts(texts_all, embedding_model, device, batch_size)
            else:
                combined = texts_pos + texts_all
                Zcombined, _ = embed_texts(combined, None, None, batch_size)
                Zpos = Zcombined[:len(texts_pos)]
                Zall = Zcombined[len(texts_pos):]
                for cid in centroids:
                    mask = labels == cid
                    centroid = Zpos[mask].mean(axis=0)
                    centroids[cid] = centroid / max(np.linalg.norm(centroid), 1e-12)
            C = np.vstack([centroids[c] for c in sorted(centroids)])
            cids = sorted(centroids)
            sims = cosine_similarity(Zall, C)
            for k, row_i in enumerate(other_idx):
                assigned.at[row_i, "source_element_sense_id"] = names[cids[int(np.argmax(sims[k]))]]

    assigned = assigned[assigned["source_element_sense_id"].astype(str).ne("")].copy()
    nodes = []
    train_pos = assigned[(assigned["split"] == "train") & (assigned["roundtrip_meaning_preserved"] == 1)]
    for sid, g in train_pos.groupby("source_element_sense_id"):
        role_counts = Counter(g["role_signature"].astype(str))
        polarity_counts = Counter(g["linguistic_polarity"].astype(str))
        decision_counts = Counter(g["decision_value"].astype(str))
        nodes.append({
            "source_element_sense_id": sid,
            "source_element_key": g["source_element_key"].iloc[0],
            "source_element_id": g["source_element_id"].iloc[0],
            "source_element_label": g["source_element_label"].iloc[0],
            "source_element_definition": g["source_element_definition"].iloc[0],
            "source_models_json": json.dumps(sorted(set(g["source_model"].astype(str))), ensure_ascii=False),
            "information_models_json": json.dumps(sorted(set(g["information_model"].astype(str))), ensure_ascii=False),
            "llms_json": json.dumps(sorted(set(g["llm"].astype(str))), ensure_ascii=False),
            "n_mentions": len(g),
            "n_forms": g["form_id"].nunique(),
            "n_sentences": g["sentence_context_id"].nunique(),
            "dominant_role_signature": role_counts.most_common(1)[0][0] if role_counts else "other",
            "role_signature_entropy": entropy_from_counter(role_counts),
            "polarity_counts_json": json.dumps(dict(polarity_counts), ensure_ascii=False),
            "decision_counts_json": json.dumps(dict(decision_counts), ensure_ascii=False),
            "top_spans_json": json.dumps([x for x, _ in Counter(g["span_text"]).most_common(20)], ensure_ascii=False),
            "top_contexts_json": json.dumps([x for x, _ in Counter(g["local_context"]).most_common(8)], ensure_ascii=False),
        })
    metadata["embedding_backends"] = dict(metadata["embedding_backends"])
    return assigned, pd.DataFrame(nodes), metadata


def entropy_from_counter(c: Counter[str]) -> float:
    total = sum(c.values())
    return -sum((n / total) * math.log2(n / total) for n in c.values()) if total else 0.0


def span_relation(a: pd.Series, b: pd.Series) -> tuple[bool, bool, bool, float, float]:
    a0, a1, b0, b1 = int(a["span_start"]), int(a["span_end"]), int(b["span_start"]), int(b["span_end"])
    same = a0 >= 0 and b0 >= 0 and a0 == b0 and a1 == b1
    overlap = a0 >= 0 and b0 >= 0 and max(a0, b0) < min(a1, b1)
    nested = overlap and ((a0 <= b0 and a1 >= b1) or (b0 <= a0 and b1 >= a1)) and not same
    ta0, ta1, tb0, tb1 = int(a["token_start"]), int(a["token_end"]), int(b["token_start"]), int(b["token_end"])
    if min(ta0, ta1, tb0, tb1) < 0:
        dist = float("nan")
    elif ta1 < tb0:
        dist = float(tb0 - ta1 - 1)
    elif tb1 < ta0:
        dist = float(ta0 - tb1 - 1)
    else:
        dist = 0.0
    prox = 0.0 if math.isnan(dist) else 1.0 / (1.0 + dist)
    return same, overlap, nested, dist, prox


def node_embeddings(nodes: pd.DataFrame, mentions: pd.DataFrame, model_name: str | None, device: str | None, batch_size: int) -> tuple[dict[str, np.ndarray], str]:
    texts, ids = [], []
    for sid, g in mentions.groupby("source_element_sense_id"):
        texts.append(" || ".join([norm(x) for x in g["span_text"].head(20).tolist()] + [norm(x) for x in g["local_context"].head(5).tolist()]))
        ids.append(sid)
    if not texts:
        return {}, "none"
    Z, backend = embed_texts(texts, model_name, device, batch_size)
    return {sid: Z[i] for i, sid in enumerate(ids)}, backend


def build_pair_features(mentions: pd.DataFrame, nodes: pd.DataFrame, embedding_model: str | None, device: str | None, batch_size: int) -> tuple[pd.DataFrame, str]:
    if mentions.empty or nodes.empty:
        return pd.DataFrame(), "none"
    train = mentions[mentions["split"] == "train"].copy()
    pos = train[train["roundtrip_meaning_preserved"] == 1].copy()
    node_emb, backend = node_embeddings(nodes, pos, embedding_model, device, batch_size)
    node_ids = sorted(nodes["source_element_sense_id"].astype(str).unique())
    node_counts = pos.groupby("source_element_sense_id")["sentence_context_id"].nunique().to_dict()
    n_contexts = max(1, pos["sentence_context_id"].nunique())
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "same_span_count": 0, "overlap_count": 0, "nested_span_count": 0,
        "same_sentence_cooccurrence": 0, "positive_preserved_support": 0, "negative_failure_support": 0,
        "proximity_sum": 0.0, "distances": [], "span_sims": [], "forms": set(),
        "information_models": set(), "llms": set(), "cross_model_contexts": 0, "cross_llm_contexts": 0,
        "role_pairs": Counter(), "span_pairs": [],
    })
    for context_id, g0 in train.groupby("sentence_context_id"):
        g = g0.drop_duplicates(subset=["source_element_sense_id", "span_start", "span_end", "information_model", "llm", "roundtrip_meaning_preserved"])
        for ra, rb in combinations(g.to_dict("records"), 2):
            a, b = pd.Series(ra), pd.Series(rb)
            s1, s2 = sorted([str(a["source_element_sense_id"]), str(b["source_element_sense_id"])])
            if s1 == s2:
                continue
            d = stats[(s1, s2)]
            same, overlap, nested, dist, prox = span_relation(a, b)
            both_pos = int(a["roundtrip_meaning_preserved"]) == 1 and int(b["roundtrip_meaning_preserved"]) == 1
            any_neg = int(a["roundtrip_meaning_preserved"]) == 0 or int(b["roundtrip_meaning_preserved"]) == 0
            if both_pos:
                d["same_sentence_cooccurrence"] += 1
                d["positive_preserved_support"] += 1
                d["same_span_count"] += int(same)
                d["overlap_count"] += int(overlap)
                d["nested_span_count"] += int(nested)
                d["proximity_sum"] += prox
                if not math.isnan(dist):
                    d["distances"].append(dist)
                d["span_sims"].append(jaccard_text(str(a["span_text"]), str(b["span_text"])))
                d["forms"].add(str(a["form_id"]))
                d["information_models"].update([str(a["information_model"]), str(b["information_model"])])
                d["llms"].update([str(a["llm"]), str(b["llm"])])
                d["cross_model_contexts"] += int(str(a["information_model"]) != str(b["information_model"]))
                d["cross_llm_contexts"] += int(str(a["llm"]) != str(b["llm"]))
                d["role_pairs"][(str(a["role_signature"]), str(b["role_signature"]))] += 1
                if len(d["span_pairs"]) < 8:
                    d["span_pairs"].append(f"{a['span_text']} || {b['span_text']}")
            if any_neg:
                d["negative_failure_support"] += 1

    node_map = nodes.set_index("source_element_sense_id").to_dict("index")
    rows = []
    for s1, s2 in combinations(node_ids, 2):
        d = stats[(s1, s2)]
        cooc = int(d["same_sentence_cooccurrence"])
        p_a = node_counts.get(s1, 0) / n_contexts
        p_b = node_counts.get(s2, 0) / n_contexts
        p_ab = cooc / n_contexts
        lift = p_ab / (p_a * p_b) if p_a > 0 and p_b > 0 else 0.0
        pmi = math.log2(lift) if lift > 0 else 0.0
        emb_sim = float(np.dot(node_emb[s1], node_emb[s2])) if s1 in node_emb and s2 in node_emb else 0.0
        mean_dist = float(np.mean(d["distances"])) if d["distances"] else np.nan
        mean_span_sim = float(np.mean(d["span_sims"])) if d["span_sims"] else 0.0
        role1 = norm(node_map.get(s1, {}).get("dominant_role_signature"))
        role2 = norm(node_map.get(s2, {}).get("dominant_role_signature"))
        role_compatible = role1 == role2 and role1 not in {"", "other"}
        rows.append({
            "sense_id_1": s1, "sense_id_2": s2,
            "same_span_count": d["same_span_count"],
            "overlap_count": d["overlap_count"],
            "nested_span_count": d["nested_span_count"],
            "same_sentence_cooccurrence": cooc,
            "proximity_weighted_cooccurrence": round(float(d["proximity_sum"]), 6),
            "mean_token_distance": mean_dist,
            "cross_model_support": len([x for x in d["information_models"] if x]),
            "cross_llm_support": len([x for x in d["llms"] if x]),
            "cross_model_contexts": d["cross_model_contexts"],
            "cross_llm_contexts": d["cross_llm_contexts"],
            "form_support": len(d["forms"]),
            "span_text_similarity": round(mean_span_sim, 6),
            "embedding_similarity": round(emb_sim, 6),
            "pmi": round(pmi, 6),
            "lift": round(lift, 6),
            "positive_preserved_support": d["positive_preserved_support"],
            "negative_failure_support": d["negative_failure_support"],
            "dominant_role_1": role1,
            "dominant_role_2": role2,
            "role_signature_compatible": role_compatible,
            "role_pair_counts_json": json.dumps({f"{a}|{b}": n for (a, b), n in d["role_pairs"].items()}, ensure_ascii=False),
            "example_span_pairs_json": json.dumps(d["span_pairs"], ensure_ascii=False),
        })
    return pd.DataFrame(rows), backend


def type_relationships(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if features.empty:
        return features
    rows = []
    for _, r in features.iterrows():
        same = int(r["same_span_count"])
        overlap = int(r["overlap_count"])
        nested = int(r["nested_span_count"])
        cooc = int(r["same_sentence_cooccurrence"])
        pos = int(r["positive_preserved_support"])
        neg = int(r["negative_failure_support"])
        emb = float(r["embedding_similarity"])
        span_sim = float(r["span_text_similarity"])
        prox = float(r["proximity_weighted_cooccurrence"])
        mean_dist = r["mean_token_distance"]
        compatible = bool(r["role_signature_compatible"])
        role1, role2 = norm(r["dominant_role_1"]), norm(r["dominant_role_2"])
        role_conflict = role1 != role2 and role1 not in {"", "other"} and role2 not in {"", "other"}
        relationship = "related_context_only"
        reasons = []

        if role_conflict and (same > 0 or overlap > 0 or emb >= args.unsafe_similarity_threshold):
            relationship = "unsafe_to_merge"
            reasons.append("mixed functional role signatures")
        elif same >= args.min_same_span_support and pos >= args.min_positive_pair_support and emb >= args.near_equivalence_embedding_threshold and compatible:
            relationship = "near_equivalent"
            reasons.append("recurrent same-span support with compatible roles")
        elif overlap >= args.min_overlap_support and span_sim >= args.near_equivalence_span_threshold and pos >= args.min_positive_pair_support and compatible:
            relationship = "near_equivalent"
            reasons.append("recurrent overlapping spans with high lexical similarity")
        elif nested >= args.min_nested_support and pos >= args.min_positive_pair_support:
            relationship = "broader_narrower"
            reasons.append("recurrent nested spans")
        elif cooc >= args.min_complementary_support and prox >= args.min_proximity_weight and overlap == 0 and role_conflict:
            relationship = "complementary"
            reasons.append("recurrent proximal co-occurrence with distinct roles")
        elif cooc == 0 and emb >= args.unsafe_similarity_threshold and role_conflict:
            relationship = "unsafe_to_merge"
            reasons.append("semantic similarity conflicts with functional role signatures")
        elif cooc > 0:
            reasons.append("contextual co-occurrence without merge evidence")
        else:
            reasons.append("insufficient relational evidence")

        confidence = 0.0
        if relationship == "near_equivalent":
            confidence = min(1.0, 0.25 * same + 0.15 * overlap + 0.25 * emb + 0.20 * span_sim + 0.05 * int(r["cross_model_support"]) + 0.05 * int(r["cross_llm_support"]))
        elif relationship == "broader_narrower":
            confidence = min(1.0, nested / max(1, pos) + 0.25 * span_sim)
        elif relationship == "complementary":
            confidence = min(1.0, prox / max(1, cooc) + 0.1 * math.log1p(cooc))
        elif relationship == "unsafe_to_merge":
            confidence = min(1.0, 0.5 + 0.25 * emb + 0.1 * math.log1p(overlap + same))
        x = r.to_dict()
        x["relationship_type"] = relationship
        x["relationship_confidence"] = round(confidence, 6)
        x["relationship_reason"] = "; ".join(reasons)
        x["negative_support_ratio"] = round(neg / max(1, pos + neg), 6)
        rows.append(x)
    return pd.DataFrame(rows)


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


def build_equivalence_clusters(nodes: pd.DataFrame, relationships: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    uf = UnionFind()
    for sid in nodes.get("source_element_sense_id", pd.Series(dtype=str)).astype(str):
        uf.find(sid)
    if not relationships.empty:
        use = relationships[
            (relationships["relationship_type"] == "near_equivalent")
            & (pd.to_numeric(relationships["relationship_confidence"], errors="coerce") >= args.min_equivalence_confidence)
            & (pd.to_numeric(relationships["positive_preserved_support"], errors="coerce") >= args.min_positive_pair_support)
        ]
        for _, r in use.iterrows():
            uf.union(str(r["sense_id_1"]), str(r["sense_id_2"]))
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in nodes.get("source_element_sense_id", pd.Series(dtype=str)).astype(str):
        groups[uf.find(sid)].append(sid)
    node_map = nodes.set_index("source_element_sense_id").to_dict("index")
    rows = []
    for i, senses in enumerate(sorted(groups.values(), key=lambda xs: (-len(xs), xs[0])), start=1):
        sub = [node_map[s] for s in senses if s in node_map]
        spans, labels, models, llms = [], [], set(), set()
        for x in sub:
            spans += json.loads(x.get("top_spans_json", "[]") or "[]")
            labels.append(norm(x.get("source_element_label")))
            models.update(json.loads(x.get("information_models_json", "[]") or "[]"))
            llms.update(json.loads(x.get("llms_json", "[]") or "[]"))
        rows.append({
            "seed_cluster_id": f"SC{i:03d}",
            "sense_ids_json": json.dumps(senses, ensure_ascii=False),
            "n_sense_nodes": len(senses),
            "n_mentions": sum(int(x.get("n_mentions", 0)) for x in sub),
            "n_forms": sum(int(x.get("n_forms", 0)) for x in sub),
            "n_sentences": sum(int(x.get("n_sentences", 0)) for x in sub),
            "n_source_models": len(models),
            "n_llms": len(llms),
            "source_models_json": json.dumps(sorted(models), ensure_ascii=False),
            "llms_json": json.dumps(sorted(llms), ensure_ascii=False),
            "source_element_labels_json": json.dumps([x for x, _ in Counter(labels).most_common(20) if x], ensure_ascii=False),
            "top_spans_json": json.dumps([x for x, _ in Counter(spans).most_common(25)], ensure_ascii=False),
            "suggested_terms_json": json.dumps([x for x, _ in Counter(t for s in spans for t in tokenize(s, include_stop=False)).most_common(15)], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def make_folds(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv).fillna("")
    forms = sorted({form_value(r) for _, r in df.iterrows() if form_value(r)})
    ordered = sorted(forms, key=lambda x: stable_id(f"{x}|{args.seed}"))
    rows = [{"form_key": f, "fold_id": f"fold_{i % args.n_folds:02d}"} for i, f in enumerate(ordered)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_assignments.csv", index=False)
    (out / "fold_metadata.json").write_text(json.dumps({"n_forms": len(rows), "n_folds": args.n_folds, "split_unit": "canonical_consent_form", "seed": args.seed}, indent=2))


def run_fold(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv).fillna("")
    folds = pd.read_csv(args.fold_assignments_csv).fillna("")
    inv = load_inventory(Path(args.inventory_csv)) if args.inventory_csv else pd.DataFrame()
    out = Path(args.output_dir) / args.fold_id
    out.mkdir(parents=True, exist_ok=True)

    mentions, excluded = build_mentions(df, folds, args.fold_id, inv)
    mentions.to_csv(out / "annotation_evidence_mentions_all.csv", index=False)
    excluded.to_csv(out / "annotation_exclusion_audit.csv", index=False)
    valid_train = mentions[(mentions["split"] == "train") & (mentions["roundtrip_meaning_preserved"] == 1)].copy()
    valid_train.to_csv(out / "annotation_evidence_mentions_valid_train.csv", index=False)
    invalid_train = mentions[(mentions["split"] == "train") & (mentions["roundtrip_meaning_preserved"] == 0)].copy()
    invalid_train.to_csv(out / "annotation_evidence_mentions_invalid_train_audit.csv", index=False)
    test = mentions[mentions["split"] == "test"].copy()
    test.to_csv(out / "annotation_evidence_mentions_test_provenance_only.csv", index=False)

    sense_mentions, nodes, sense_meta = induce_senses(
        mentions, args.min_sense_support, args.sense_distance_threshold,
        args.embedding_model or None, args.embedding_device or None, args.batch_size,
    )
    sense_mentions.to_csv(out / "source_element_sense_mentions_all.csv", index=False)
    sense_mentions[(sense_mentions["split"] == "train") & (sense_mentions["roundtrip_meaning_preserved"] == 1)].to_csv(out / "source_element_sense_mentions_valid_train.csv", index=False)
    nodes.to_csv(out / "source_element_sense_nodes.csv", index=False)

    pair_features, pair_backend = build_pair_features(sense_mentions, nodes, args.embedding_model or None, args.embedding_device or None, args.batch_size)
    pair_features.to_csv(out / "pairwise_evidence_features.csv", index=False)
    relationships = type_relationships(pair_features, args)
    relationships.to_csv(out / "typed_relationship_edges.csv", index=False)
    relationships[relationships["relationship_type"] == "complementary"].to_csv(out / "provision_bundle_edges.csv", index=False)
    relationships[relationships["relationship_type"] == "unsafe_to_merge"].to_csv(out / "unsafe_merge_edges.csv", index=False)

    clusters = build_equivalence_clusters(nodes, relationships, args)
    clusters.to_csv(out / "seed_concept_clusters.csv", index=False)
    metadata = {
        "fold_id": args.fold_id,
        "n_mentions_all": int(len(mentions)),
        "n_valid_train_mentions": int(len(valid_train)),
        "n_invalid_train_mentions": int(len(invalid_train)),
        "n_test_mentions_provenance_only": int(len(test)),
        "n_excluded_annotation_items": int(len(excluded)),
        "n_sense_nodes": int(len(nodes)),
        "n_pairwise_features": int(len(pair_features)),
        "n_near_equivalent_edges": int((relationships.get("relationship_type", pd.Series(dtype=str)) == "near_equivalent").sum()),
        "n_complementary_edges": int((relationships.get("relationship_type", pd.Series(dtype=str)) == "complementary").sum()),
        "n_unsafe_edges": int((relationships.get("relationship_type", pd.Series(dtype=str)) == "unsafe_to_merge").sum()),
        "n_seed_clusters": int(len(clusters)),
        "sense_induction": sense_meta,
        "pair_embedding_backend": pair_backend,
        "human_validity_rule": "Only annotations from rows with human meaning_preserved == 1 are valid induction evidence.",
        "cluster_rule": "Only typed near_equivalent edges may merge source-element-sense nodes.",
        "bundle_rule": "Complementary/proximity edges are retained separately and never used for equivalence clustering.",
    }
    (out / "fold_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def summarize_folds(args: argparse.Namespace) -> None:
    root = Path(args.fold_root)
    rows = []
    for p in sorted(root.glob("fold_*/seed_concept_clusters.csv")):
        fold = p.parent.name
        df = pd.read_csv(p).fillna("")
        for _, r in df.iterrows():
            terms = json.loads(r.get("suggested_terms_json", "[]") or "[]")[:6]
            rows.append({"fold_id": fold, "seed_cluster_id": r.get("seed_cluster_id"), "signature_terms": " | ".join(terms), "n_mentions": r.get("n_mentions"), "n_sense_nodes": r.get("n_sense_nodes"), "source_models_json": r.get("source_models_json")})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(rows)
    long.to_csv(out / "seed_clusters_across_folds_long.csv", index=False)
    if not long.empty:
        rec = long.groupby("signature_terms", dropna=False).agg(
            n_folds=("fold_id", "nunique"),
            folds=("fold_id", lambda x: json.dumps(sorted(set(x)), ensure_ascii=False)),
            total_mentions=("n_mentions", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
        ).reset_index().sort_values(["n_folds", "total_mentions"], ascending=[False, False])
        rec.to_csv(out / "seed_cluster_recurrence_across_folds.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-folds")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)

    p = sub.add_parser("run-fold")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--fold_assignments_csv", required=True)
    p.add_argument("--inventory_csv", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fold_id", required=True)
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding_device", default="")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_sense_support", type=int, default=3)
    p.add_argument("--sense_distance_threshold", type=float, default=0.35)
    p.add_argument("--min_same_span_support", type=int, default=2)
    p.add_argument("--min_overlap_support", type=int, default=2)
    p.add_argument("--min_nested_support", type=int, default=2)
    p.add_argument("--min_positive_pair_support", type=int, default=2)
    p.add_argument("--min_complementary_support", type=int, default=2)
    p.add_argument("--min_proximity_weight", type=float, default=0.75)
    p.add_argument("--near_equivalence_embedding_threshold", type=float, default=0.72)
    p.add_argument("--near_equivalence_span_threshold", type=float, default=0.70)
    p.add_argument("--unsafe_similarity_threshold", type=float, default=0.68)
    p.add_argument("--min_equivalence_confidence", type=float, default=0.55)

    p = sub.add_parser("summarize-folds")
    p.add_argument("--fold_root", required=True)
    p.add_argument("--output_dir", required=True)

    args = ap.parse_args()
    if args.cmd == "make-folds":
        make_folds(args)
    elif args.cmd == "run-fold":
        run_fold(args)
    elif args.cmd == "summarize-folds":
        summarize_folds(args)


if __name__ == "__main__":
    main()
