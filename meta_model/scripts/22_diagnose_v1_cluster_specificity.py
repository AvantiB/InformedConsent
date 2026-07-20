#!/usr/bin/env python
"""Diagnose whether empirical V1 clusters are specific enough for a reduced schema.

This script is exploratory. It does not change cluster assignments. It summarizes
where discovered semantic clusters appear too broad, hub-like, or overlapping in
smoke-test outputs. The goal is to guide a second induction pass toward a
complementary schema whose fields have more specific, non-redundant roles.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

STOP = set(
    "the a an and or of to in for with without by from on at as is are be been this that it its your you we our may can will shall my i they their them us all".split()
)

SPAN_TYPE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("temporal", re.compile(r"\b(any time|at any time|year|years|month|months|day|days|before|after|until|during|long|period|future|ongoing)\b", re.I)),
    ("condition_or_exception", re.compile(r"\b(if|unless|except|only if|when|required|must|cannot|not|without)\b", re.I)),
    ("decision_cue", re.compile(r"\b(agree|allow|authorize|permission|consent|permit|decline|refuse|withdraw|quit|choose|decide)\b", re.I)),
    ("action", re.compile(r"\b(use|used|store|stored|share|shared|collect|collected|send|sent|access|link|contact|destroy|withdraw|quit|join)\b", re.I)),
    ("resource", re.compile(r"\b(data|information|record|records|sample|samples|specimen|specimens|dna|genetic|health|biospecimen)\b", re.I)),
    ("repository_or_system", re.compile(r"\b(database|databases|repository|registry|platform|system|biobank)\b", re.I)),
    ("organization_or_actor", re.compile(r"\b(researcher|researchers|doctor|doctors|team|staff|all of us|institution|mayo|nih|program|study)\b", re.I)),
    ("participant", re.compile(r"\b(i|me|my|you|your|participant|participants|person|people)\b", re.I)),
    ("purpose", re.compile(r"\b(research|study|studies|learn|understand|improve|discover|purpose)\b", re.I)),
    ("privacy_or_identifiability", re.compile(r"\b(private|privacy|confidential|identified|de-identified|anonymous|name|identity|secure|security)\b", re.I)),
]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def split_paths(x: str) -> list[Path]:
    return [Path(p.strip()) for p in x.split(",") if p.strip()]


def json_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    s = norm(x)
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", norm(text).lower()) if t not in STOP]


def lexical_head(span: str) -> str:
    ts = tokens(span)
    return ts[-1] if ts else ""


def entropy(values: list[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def span_types(span: str) -> list[str]:
    out = [name for name, pat in SPAN_TYPE_RULES if pat.search(norm(span))]
    return out or ["other"]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def cluster_from_annotation(ann: Any) -> str:
    if not isinstance(ann, dict):
        return ""
    return norm(ann.get("cluster_id") or ann.get("union_element_id") or ann.get("element_id") or ann.get("label"))


def parse_smoke_dir(path: Path) -> pd.DataFrame:
    csv_path = path / "reduced_v1_roundtrip_outputs.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path).fillna("")
    rows = []
    for _, r in df.iterrows():
        source_id = norm(r.get("source_id", ""))
        source_text = norm(r.get("source_text", ""))
        evidence_mode = norm(r.get("evidence_mode", path.name))
        llm = path.parent.name if path.name in {"compact", "permissive"} else path.name
        parsed = None
        for c in ["forward_mapping", "v1_mapping_json", "forward_raw"]:
            if c in df.columns and norm(r.get(c, "")):
                try:
                    parsed = json.loads(norm(r.get(c, "")))
                    break
                except Exception:
                    continue
        if not isinstance(parsed, dict):
            continue
        for ann in parsed.get("annotations") or []:
            if not isinstance(ann, dict):
                continue
            cid = cluster_from_annotation(ann)
            span = norm(ann.get("span_text", ""))
            if not cid or not span:
                continue
            rows.append({
                "llm": llm,
                "evidence_mode": evidence_mode,
                "source_id": source_id,
                "source_text": source_text,
                "cluster_id": cid,
                "span_text": span,
                "overlap_group_id": norm(ann.get("overlap_group_id", "")),
                "span_relation": norm(ann.get("span_relation", "")),
            })
    return pd.DataFrame(rows)


def cluster_membership_metrics(clusters: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if clusters.empty:
        return pd.DataFrame()
    for cid, g in clusters.groupby("semantic_cluster_id"):
        elements = g["union_element_id"].astype(str).tolist() if "union_element_id" in g.columns else []
        source_models = sorted({e.split("::", 1)[0] for e in elements if "::" in e})
        m = mentions[mentions["union_element_id"].astype(str).isin(elements)].copy() if not mentions.empty and "union_element_id" in mentions.columns else pd.DataFrame()
        spans = m["span_text"].astype(str).tolist() if not m.empty and "span_text" in m.columns else []
        heads = [lexical_head(s) for s in spans]
        stypes = [t for s in spans for t in span_types(s)]
        element_families = [e.split("::", 1)[0] for e in elements if "::" in e]
        hub_counts = Counter(elements)
        top_hub_frac = (hub_counts.most_common(1)[0][1] / max(1, len(elements))) if elements else 0.0
        rows.append({
            "semantic_cluster_id": cid,
            "n_source_elements": len(elements),
            "n_source_models": len(source_models),
            "source_models": ", ".join(source_models),
            "n_discovery_mentions": len(m),
            "n_unique_span_heads": len(set(h for h in heads if h)),
            "span_head_entropy": entropy(heads),
            "span_type_entropy": entropy(stypes),
            "source_model_entropy": entropy(element_families),
            "top_source_element_fraction": top_hub_frac,
            "top_span_heads": json.dumps([x for x, _ in Counter(heads).most_common(12) if x], ensure_ascii=False),
            "top_span_types": json.dumps([x for x, _ in Counter(stypes).most_common(12) if x], ensure_ascii=False),
            "top_spans": json.dumps([x for x, _ in Counter(spans).most_common(12) if x], ensure_ascii=False),
            "top_source_elements": json.dumps(elements[:25], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def smoke_overlap_metrics(smoke: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if smoke.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for (llm, mode, sid, span), g in smoke.groupby(["llm", "evidence_mode", "source_id", "span_text"], dropna=False):
        clusters = sorted(g["cluster_id"].astype(str).unique().tolist())
        if len(clusters) > 1:
            rows.append({"llm": llm, "evidence_mode": mode, "source_id": sid, "span_text": span, "n_clusters_same_span": len(clusters), "clusters": "; ".join(clusters)})
    overlap = pd.DataFrame(rows).sort_values(["n_clusters_same_span", "span_text"], ascending=[False, True]) if rows else pd.DataFrame()
    cm = []
    for cid, g in smoke.groupby("cluster_id"):
        spans = g["span_text"].astype(str).tolist()
        heads = [lexical_head(s) for s in spans]
        stypes = [t for s in spans for t in span_types(s)]
        same_span_overlap = 0
        if not overlap.empty:
            same_span_overlap = overlap[overlap["clusters"].astype(str).str.contains(re.escape(str(cid)), regex=True)].shape[0]
        cm.append({
            "semantic_cluster_id": cid,
            "n_smoke_annotations": len(g),
            "n_smoke_source_sentences": g["source_id"].nunique(),
            "n_unique_smoke_spans": len(set(spans)),
            "n_same_span_multi_cluster_cases": same_span_overlap,
            "smoke_span_head_entropy": entropy(heads),
            "smoke_span_type_entropy": entropy(stypes),
            "top_smoke_span_heads": json.dumps([x for x, _ in Counter(heads).most_common(12) if x], ensure_ascii=False),
            "top_smoke_span_types": json.dumps([x for x, _ in Counter(stypes).most_common(12) if x], ensure_ascii=False),
            "top_smoke_spans": json.dumps([x for x, _ in Counter(spans).most_common(12) if x], ensure_ascii=False),
        })
    return overlap, pd.DataFrame(cm)


def combine_and_flag(membership: pd.DataFrame, smoke_metrics: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame()
    out = membership.merge(smoke_metrics, on="semantic_cluster_id", how="left") if not smoke_metrics.empty else membership.copy()
    for col in ["n_same_span_multi_cluster_cases", "smoke_span_head_entropy", "smoke_span_type_entropy", "n_smoke_annotations"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    flags = []
    for _, r in out.iterrows():
        fs = []
        if float(r.get("span_type_entropy", 0)) >= 1.75 or float(r.get("smoke_span_type_entropy", 0)) >= 1.75:
            fs.append("high_span_type_entropy")
        if float(r.get("span_head_entropy", 0)) >= 2.75 or float(r.get("smoke_span_head_entropy", 0)) >= 2.75:
            fs.append("high_lexical_head_entropy")
        if int(float(r.get("n_source_elements", 0))) >= 20:
            fs.append("large_cluster")
        if int(float(r.get("n_same_span_multi_cluster_cases", 0))) >= 3:
            fs.append("frequent_same_span_overlap")
        if float(r.get("top_source_element_fraction", 0)) >= 0.35 and int(float(r.get("n_source_elements", 0))) >= 8:
            fs.append("hub_element_dominance")
        flags.append(";".join(fs))
    out["specificity_flags"] = flags
    out["needs_split_review"] = out["specificity_flags"].astype(str).str.len() > 0
    return out.sort_values(["needs_split_review", "n_same_span_multi_cluster_cases", "span_type_entropy", "span_head_entropy"], ascending=[False, False, False, False])


def write_report(out: Path, diag: pd.DataFrame, overlap: pd.DataFrame) -> None:
    lines = [
        "# V1 cluster specificity diagnostics",
        "",
        "This report diagnoses whether empirical clusters are specific enough to act as complementary reduced-schema fields.",
        "It is exploratory and should be used to guide the next induction pass, not as a final audit decision.",
        "",
        "## Interpretation",
        "",
        "A good reduced field should be semantically specific and complementary to other fields. Clusters are flagged when they show high lexical-head diversity, high span-type diversity, frequent same-span overlap with other clusters, or hub-like behavior.",
        "",
        "## Summary",
        f"- Clusters evaluated: {len(diag):,}",
        f"- Clusters needing split review: {int(diag['needs_split_review'].sum()) if not diag.empty and 'needs_split_review' in diag.columns else 0:,}",
        f"- Same-span multi-cluster smoke cases: {len(overlap):,}",
        "",
    ]
    if not diag.empty:
        show_cols = ["semantic_cluster_id", "needs_split_review", "specificity_flags", "n_source_elements", "span_type_entropy", "span_head_entropy", "n_same_span_multi_cluster_cases", "top_span_types", "top_span_heads", "top_smoke_spans"]
        show_cols = [c for c in show_cols if c in diag.columns]
        lines += ["## Top split-review candidates", "", diag[show_cols].head(20).to_markdown(index=False), ""]
    if not overlap.empty:
        lines += ["## Same-span multi-cluster examples from smoke tests", "", overlap.head(30).to_markdown(index=False), ""]
    lines += [
        "## Recommended next induction change",
        "",
        "Move from source-element clustering to source-element-sense clustering. Split broad elements by observed span type and lexical head before building equivalence clusters. Treat provision-bundle edges as composition evidence and same-span-but-different-span-type evidence as complementary, not equivalent.",
    ]
    (out / "cluster_specificity_report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery_dir", required=True)
    ap.add_argument("--smoke_dirs", default="", help="Comma-separated paths such as .../medgemma/compact,.../qwen235b/compact")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    discovery = Path(args.discovery_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clusters = load_csv(discovery / "semantic_equivalence_clusters.csv")
    mentions = load_csv(discovery / "expert_element_mentions_long.csv")
    membership = cluster_membership_metrics(clusters, mentions)

    smoke_parts = [parse_smoke_dir(p) for p in split_paths(args.smoke_dirs)]
    smoke = pd.concat([x for x in smoke_parts if not x.empty], ignore_index=True) if any(not x.empty for x in smoke_parts) else pd.DataFrame()
    overlap, smoke_metrics = smoke_overlap_metrics(smoke)
    diag = combine_and_flag(membership, smoke_metrics)

    membership.to_csv(out / "cluster_membership_specificity_metrics.csv", index=False)
    smoke.to_csv(out / "smoke_cluster_annotations_long.csv", index=False)
    overlap.to_csv(out / "smoke_same_span_multi_cluster_cases.csv", index=False)
    smoke_metrics.to_csv(out / "smoke_cluster_specificity_metrics.csv", index=False)
    diag.to_csv(out / "cluster_split_review_candidates.csv", index=False)
    write_report(out, diag, overlap)
    print(f"Wrote cluster specificity diagnostics to {out}")


if __name__ == "__main__":
    main()
