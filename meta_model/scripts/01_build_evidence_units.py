#!/usr/bin/env python
"""Build data/language evidence units for reduced consent meta-model induction.

This script converts LLM round-trip rows into phrase/source-node evidence units and
simple graph edges. It is intentionally extraction-first and schema-light: the goal
is to let source-model usage, language, co-occurrence, and preservation behavior
induce candidate meta-model units downstream.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

COL_CANDIDATES = {
    "roundtrip_id": ["roundtrip_id", "id", "source_id"],
    "sentence_id": ["sentence_id", "sent_id", "source_sentence_id"],
    "form_id": ["form_id", "form_key", "source_file"],
    "original_text": ["canonical_full_text", "full_text_original", "original_sentence", "full_text"],
    "reconstructed_text": ["reconstructed_sentence", "backward_mapping", "backward_reconstruction", "reconstruction"],
    "forward_mapping": ["annotations_serialized", "forward_mapping", "annotations_combined", "mapping"],
    "llm": ["llm", "model", "llm_name"],
    "information_model": ["information_model", "info_model", "model_family"],
    "meaning_preserved": ["meaning_preserved", "human_meaning_preserved", "label"],
}

ID_PATTERNS = [
    re.compile(r"\b(ICO[:_][A-Za-z0-9_\-.]+)\b", re.I),
    re.compile(r"\b(DUO[:_][A-Za-z0-9_\-.]+)\b", re.I),
    re.compile(r"\b(ODRL[:_.][A-Za-z0-9_\-.]+)\b", re.I),
    re.compile(r"\b(FHIR(?:\.Consent)?[A-Za-z0-9_.:-]*)\b", re.I),
    re.compile(r"\b(Consent\.[A-Za-z0-9_.:-]+)\b", re.I),
]

NODE_ID_KEYS = {"id", "node_id", "concept_id", "term_id", "code", "uri", "iri", "class_id", "element_id"}
NODE_LABEL_KEYS = {"label", "name", "term", "title", "class", "predicate", "property", "element", "concept", "node"}
TEXT_KEYS = {"text", "phrase", "span", "value", "content", "verbatim", "sentence", "description"}


def norm(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def low(x: Any) -> str:
    return norm(x).lower()


def infer_col(df: pd.DataFrame, canonical: str, required: bool = True) -> str | None:
    for c in COL_CANDIDATES[canonical]:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not infer required column for {canonical}; tried {COL_CANDIDATES[canonical]}")
    return None


def stable_id(*parts: str, n: int = 12) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return h[:n]


def load_cue_dictionary(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    cue_groups = data.get("cue_groups", data)
    return {str(k): [str(t).lower() for t in v] for k, v in cue_groups.items()}


def cue_groups_for_text(text: str, cue_dict: dict[str, list[str]]) -> list[str]:
    t = low(text)
    groups = []
    for group, terms in cue_dict.items():
        if any(term and term in t for term in terms):
            groups.append(group)
    return groups


def parse_structured_mapping(raw: str) -> Any | None:
    s = norm(raw)
    if not s:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(s)
        except Exception:
            pass
    return None


def collect_from_obj(obj: Any, path: str = "") -> list[dict[str, str]]:
    """Recursively collect node-like objects from JSON-ish mappings."""
    out: list[dict[str, str]] = []
    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        node_id = ""
        label = ""
        span = ""
        for k, v in lowered.items():
            if k in NODE_ID_KEYS and not node_id:
                node_id = norm(v)
            if k in NODE_LABEL_KEYS and not label:
                label = norm(v)
            if k in TEXT_KEYS and not span:
                span = norm(v)
        if node_id or label or span:
            out.append({
                "source_element_id": node_id,
                "source_element_label": label or span or node_id,
                "span_text": span or label or node_id,
                "extraction_method": "json_object",
                "json_path": path,
            })
        for k, v in obj.items():
            out.extend(collect_from_obj(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(collect_from_obj(v, f"{path}[{i}]"))
    return out


def collect_from_text(raw: str, information_model: str) -> list[dict[str, str]]:
    """Fallback text extraction from unstructured mappings."""
    s = norm(raw)
    out: list[dict[str, str]] = []
    seen = set()

    for pat in ID_PATTERNS:
        for m in pat.finditer(s):
            node_id = m.group(1)
            key = (node_id, node_id)
            if key not in seen:
                seen.add(key)
                out.append({
                    "source_element_id": node_id,
                    "source_element_label": node_id,
                    "span_text": node_id,
                    "extraction_method": "regex_id",
                    "json_path": "",
                })

    # Add line-level phrase evidence. This is deliberately broad: downstream
    # clustering can discard noisy singleton phrases.
    lines = [re.sub(r"^[\s\-*•\d.)]+", "", x).strip() for x in s.splitlines()]
    lines = [x for x in lines if 4 <= len(x) <= 240]
    for line in lines[:50]:
        key = ("", line.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source_element_id": "",
            "source_element_label": line,
            "span_text": line,
            "extraction_method": "mapping_line",
            "json_path": "",
        })

    if not out and s:
        out.append({
            "source_element_id": "",
            "source_element_label": information_model or "mapping_text",
            "span_text": s[:240],
            "extraction_method": "mapping_text_fallback",
            "json_path": "",
        })
    return out


def extract_mapping_units(raw: str, information_model: str) -> list[dict[str, str]]:
    parsed = parse_structured_mapping(raw)
    units = collect_from_obj(parsed) if parsed is not None else []
    units.extend(collect_from_text(raw, information_model))

    deduped = []
    seen = set()
    for u in units:
        label = norm(u.get("source_element_label"))
        node_id = norm(u.get("source_element_id"))
        span = norm(u.get("span_text"))
        if not (label or node_id or span):
            continue
        key = (node_id.lower(), label.lower(), span.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(u)
    return deduped


def build_evidence(df: pd.DataFrame, cue_dict: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {k: infer_col(df, k, required=(k != "form_id")) for k in COL_CANDIDATES}
    rows = []
    edges = []

    for idx, row in df.iterrows():
        roundtrip_id = norm(row[cols["roundtrip_id"]]) or f"row_{idx}"
        sentence_id = norm(row[cols["sentence_id"]]) or f"sentence_{idx}"
        form_id = norm(row[cols["form_id"]]) if cols.get("form_id") else ""
        llm = norm(row[cols["llm"]])
        model = norm(row[cols["information_model"]])
        y = row[cols["meaning_preserved"]]
        original = norm(row[cols["original_text"]])
        recon = norm(row[cols["reconstructed_text"]])
        mapping = norm(row[cols["forward_mapping"]])

        units = extract_mapping_units(mapping, model)
        if not units:
            units = [{"source_element_id": "", "source_element_label": model, "span_text": original[:240], "extraction_method": "original_text_fallback", "json_path": ""}]

        rt_node = f"roundtrip:{roundtrip_id}"
        sent_node = f"sentence:{sentence_id}"
        model_node = f"source_model:{model}"
        llm_node = f"llm:{llm}"
        edges.extend([
            {"source": rt_node, "target": sent_node, "edge_type": "has_sentence", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ""},
            {"source": rt_node, "target": model_node, "edge_type": "uses_information_model", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ""},
            {"source": rt_node, "target": llm_node, "edge_type": "generated_by_llm", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ""},
        ])

        for j, u in enumerate(units):
            span = norm(u.get("span_text"))
            label = norm(u.get("source_element_label"))
            node_id = norm(u.get("source_element_id"))
            unit_text = " | ".join(x for x in [span, label, node_id, model] if x)
            ev_id = f"ev_{stable_id(roundtrip_id, str(j), unit_text)}"
            source_key = node_id or f"derived:{stable_id(model, label, span)}"
            source_node = f"source_element:{model}:{source_key}"
            cues = cue_groups_for_text(" ".join([span, label, original, recon]), cue_dict)

            rows.append({
                "evidence_unit_id": ev_id,
                "roundtrip_id": roundtrip_id,
                "sentence_id": sentence_id,
                "form_id": form_id,
                "llm": llm,
                "information_model": model,
                "meaning_preserved": y,
                "source_element_id": node_id,
                "source_element_label": label,
                "source_element_key": source_key,
                "span_text": span,
                "unit_text_for_embedding": unit_text,
                "extraction_method": u.get("extraction_method", ""),
                "json_path": u.get("json_path", ""),
                "cue_groups": ";".join(cues),
                "original_text": original,
                "reconstructed_text": recon,
                "forward_mapping_text": mapping,
            })

            edges.extend([
                {"source": ev_id, "target": rt_node, "edge_type": "evidence_for_roundtrip", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ev_id},
                {"source": ev_id, "target": sent_node, "edge_type": "evidence_for_sentence", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ev_id},
                {"source": ev_id, "target": source_node, "edge_type": "maps_to_source_element", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ev_id},
                {"source": source_node, "target": model_node, "edge_type": "belongs_to_source_model", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ev_id},
            ])
            for cue in cues:
                edges.append({"source": ev_id, "target": f"cue_group:{cue}", "edge_type": "matches_cue_group", "weight": 1, "roundtrip_id": roundtrip_id, "evidence_unit_id": ev_id})

    return pd.DataFrame(rows), pd.DataFrame(edges)


def write_summaries(evidence: pd.DataFrame, out_dir: Path) -> None:
    freq = (
        evidence.groupby(["information_model", "source_element_key", "source_element_label"], dropna=False)
        .agg(n_evidence_units=("evidence_unit_id", "count"), n_sentences=("sentence_id", "nunique"), preserved_rate=("meaning_preserved", "mean"))
        .reset_index()
        .sort_values(["n_evidence_units", "n_sentences"], ascending=False)
    )
    freq.to_csv(out_dir / "source_node_frequency.csv", index=False)

    pair_counts = Counter()
    for _, g in evidence.groupby("roundtrip_id"):
        nodes = sorted(set(g["source_element_key"].dropna().astype(str)))
        for a, b in itertools.combinations(nodes, 2):
            pair_counts[(a, b)] += 1
    pd.DataFrame([
        {"source_element_a": a, "source_element_b": b, "n_roundtrips": n}
        for (a, b), n in pair_counts.items()
    ]).sort_values("n_roundtrips", ascending=False).to_csv(out_dir / "source_node_cooccurrence.csv", index=False)

    cue_rows = []
    for _, row in evidence.iterrows():
        for cue in str(row.get("cue_groups", "")).split(";"):
            if cue:
                cue_rows.append({"cue_group": cue, "meaning_preserved": row["meaning_preserved"], "evidence_unit_id": row["evidence_unit_id"]})
    if cue_rows:
        cue_df = pd.DataFrame(cue_rows)
        cue_summary = cue_df.groupby("cue_group").agg(n=("evidence_unit_id", "count"), preserved_rate=("meaning_preserved", "mean")).reset_index().sort_values("n", ascending=False)
        cue_summary.to_csv(out_dir / "cue_group_frequency.csv", index=False)
    else:
        pd.DataFrame(columns=["cue_group", "n", "preserved_rate"]).to_csv(out_dir / "cue_group_frequency.csv", index=False)

    audit = {
        "n_evidence_units": int(len(evidence)),
        "n_roundtrips": int(evidence["roundtrip_id"].nunique()) if not evidence.empty else 0,
        "n_source_element_keys": int(evidence["source_element_key"].nunique()) if not evidence.empty else 0,
        "extraction_method_counts": evidence["extraction_method"].value_counts(dropna=False).to_dict() if not evidence.empty else {},
        "information_model_counts": evidence["information_model"].value_counts(dropna=False).to_dict() if not evidence.empty else {},
    }
    (out_dir / "extraction_audit.json").write_text(json.dumps(audit, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--output_dir", default="meta_model/outputs/evidence_units")
    ap.add_argument("--cue_dictionary", default=None, help="Optional JSON with cue_groups, e.g. meaning_preservation/literature_informed_consent_cues.json")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.roundtrips_csv)
    cue_dict = load_cue_dictionary(Path(args.cue_dictionary)) if args.cue_dictionary else {}
    evidence, edges = build_evidence(df, cue_dict)

    evidence.to_csv(out_dir / "evidence_units.csv", index=False)
    edges.to_csv(out_dir / "phrase_node_graph_edges.csv", index=False)
    write_summaries(evidence, out_dir)

    print(f"Wrote {len(evidence):,} evidence units to {out_dir / 'evidence_units.csv'}")
    print(f"Wrote {len(edges):,} graph edges to {out_dir / 'phrase_node_graph_edges.csv'}")


if __name__ == "__main__":
    main()
