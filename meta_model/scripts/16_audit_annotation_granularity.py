#!/usr/bin/env python
"""Audit annotation/role granularity in Union V0, individual, and reduced V1 outputs.

This detects behavior that can inflate backward meaning preservation: many
annotations/role entries, long clause-level evidence spans, full-sentence-like
spans, or duplicated coverage of the same source tokens.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def split_arg(value: str | None) -> list[Path]:
    return [Path(x.strip()) for x in value.split(",") if x.strip()] if value else []


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(text or "")]


def parse_jsonish(text: Any) -> Any | None:
    s = norm_text(text)
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|csv|yaml)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = s.find(open_ch); end = s.rfind(close_ch)
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                pass
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({"_parse_error": line[:500]})
    return out


def span_value(annotation: dict[str, Any]) -> str:
    for key in ["span_text", "evidence_span_text", "evidence_text", "text_span", "phrase", "text"]:
        val = annotation.get(key)
        if isinstance(val, str) and val.strip():
            return norm_text(val)
    return ""


def label_value(annotation: dict[str, Any]) -> str:
    for key in ["union_element_id", "label", "source_element_id", "element_id", "role", "field"]:
        val = annotation.get(key)
        if isinstance(val, str) and val.strip():
            return norm_text(val)
    return ""


def span_token_positions(source_text: str, span: str, source_tokens: list[tuple[str, int, int]]) -> set[int]:
    if not span or not source_text:
        return set()
    idx = source_text.lower().find(span.lower())
    if idx < 0:
        pattern = r"\s+".join(re.escape(x) for x in span.split())
        m = re.search(pattern, source_text, flags=re.I)
        if not m:
            return set()
        idx, end = m.start(), m.end()
    else:
        end = idx + len(span)
    return {i for i, (_, start, stop) in enumerate(source_tokens) if start < end and stop > idx}


def summarize_items(source_text: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    src_tokens = tokens_with_offsets(source_text)
    n_src = len(src_tokens)
    spans = [span_value(a) for a in items if isinstance(a, dict) and span_value(a)]
    labels = [label_value(a) for a in items if isinstance(a, dict) and label_value(a)]
    span_token_counts = [len(tokens_with_offsets(s)) for s in spans]
    covered: set[int] = set(); total_dup = 0; full_sentence_like = 0; long_examples: list[str] = []
    for span, n_span in zip(spans, span_token_counts):
        pos = span_token_positions(source_text, span, src_tokens)
        covered |= pos
        total_dup += len(pos) if pos else n_span
        if n_span / max(1, n_src) >= 0.80 or norm_text(span).lower() == norm_text(source_text).lower():
            full_sentence_like += 1
        if n_span >= 8 and len(long_examples) < 5:
            long_examples.append(span)
    n_items = len(items)
    n_long = sum(1 for x in span_token_counts if x >= 8)
    n_clause = sum(1 for x in span_token_counts if x >= 12)
    unique_cov = len(covered)
    return {"n_annotations": n_items, "n_unique_labels": len(set(labels)), "source_token_count": n_src, "annotation_density_per_source_token": n_items / max(1, n_src), "mean_span_token_count": sum(span_token_counts) / max(1, len(span_token_counts)), "max_span_token_count": max(span_token_counts) if span_token_counts else 0, "n_long_spans_ge8_tokens": n_long, "pct_long_spans_ge8_tokens": n_long / max(1, n_items), "n_clause_spans_ge12_tokens": n_clause, "pct_clause_spans_ge12_tokens": n_clause / max(1, n_items), "n_full_sentence_like_spans": full_sentence_like, "pct_full_sentence_like_spans": full_sentence_like / max(1, n_items), "annotated_token_coverage": unique_cov / max(1, n_src), "duplicate_coverage_factor": total_dup / max(1, unique_cov), "long_span_examples_json": json.dumps(long_examples, ensure_ascii=False)}


def union_rows(model_dir: Path) -> list[dict[str, Any]]:
    rows = []; llm = model_dir.name
    for obj in read_jsonl(model_dir / "union_v0_forward_mappings.jsonl"):
        parsed = obj.get("parsed_forward") or parse_jsonish(obj.get("raw_response")) or {}
        anns = parsed.get("annotations") if isinstance(parsed, dict) else []
        anns = anns if isinstance(anns, list) else []
        source_text = norm_text(obj.get("source_text"))
        row = {"llm": llm, "condition": "union_v0_full_dictionary", "information_model": "Union_V0", "source_id": obj.get("source_id"), "source_text": source_text}
        row.update(summarize_items(source_text, [a for a in anns if isinstance(a, dict)]))
        rows.append(row)
    return rows


def individual_rows(model_dir: Path) -> list[dict[str, Any]]:
    rows = []; llm = model_dir.name
    for info_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
        info = info_dir.name
        for obj in read_jsonl(info_dir / "forward_mappings.jsonl"):
            parsed = parse_jsonish(obj.get("raw_response")) or {}
            anns = parsed.get("annotations") if isinstance(parsed, dict) else []
            anns = anns if isinstance(anns, list) else []
            source_text = norm_text(obj.get("source_text"))
            row = {"llm": llm, "condition": "individual_source_model_json", "information_model": info, "source_id": obj.get("source_id"), "source_text": source_text}
            row.update(summarize_items(source_text, [a for a in anns if isinstance(a, dict)]))
            rows.append(row)
    return rows


def v1_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for prov in parsed.get("provisions") or []:
        if not isinstance(prov, dict):
            continue
        pid = prov.get("provision_id", "")
        for role, value in prov.items():
            if role in {"provision_id", "provenance"}:
                continue
            if role == "decision" and isinstance(value, dict):
                d = dict(value); d.update({"role": role, "field": role, "provision_id": pid}); items.append(d)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        d = dict(v); d.update({"role": role, "field": role, "provision_id": pid}); items.append(d)
                    elif norm_text(v):
                        items.append({"role": role, "field": role, "evidence_span_text": norm_text(v), "provision_id": pid})
            elif norm_text(value):
                items.append({"role": role, "field": role, "evidence_span_text": norm_text(value), "provision_id": pid})
    return items


def reduced_v1_rows(model_dir: Path) -> list[dict[str, Any]]:
    rows = []
    evidence_mode = model_dir.name
    llm = model_dir.parent.name if evidence_mode in {"compact", "permissive"} else model_dir.name
    condition = f"reduced_v1_{evidence_mode}" if evidence_mode in {"compact", "permissive"} else "reduced_v1_unknown"
    for obj in read_jsonl(model_dir / "reduced_v1_forward_mappings.jsonl"):
        parsed = obj.get("parsed_forward") or parse_jsonish(obj.get("raw_response")) or {}
        source_text = norm_text(obj.get("source_text"))
        row = {"llm": llm, "condition": condition, "information_model": "Reduced_V1", "evidence_mode": evidence_mode, "source_id": obj.get("source_id"), "source_text": source_text}
        row.update(summarize_items(source_text, v1_items(parsed if isinstance(parsed, dict) else {})))
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    gcols = ["llm", "condition", "information_model"]
    metrics = ["n_annotations", "n_unique_labels", "annotation_density_per_source_token", "mean_span_token_count", "max_span_token_count", "pct_long_spans_ge8_tokens", "pct_clause_spans_ge12_tokens", "pct_full_sentence_like_spans", "annotated_token_coverage", "duplicate_coverage_factor"]
    rows = []
    for keys, g in df.groupby(gcols, dropna=False):
        row = {c: keys[i] for i, c in enumerate(gcols)}
        row["n_rows"] = len(g)
        for m in metrics:
            row[f"mean_{m}"] = pd.to_numeric(g[m], errors="coerce").mean()
            row[f"median_{m}"] = pd.to_numeric(g[m], errors="coerce").median()
        row["pct_rows_with_long_span"] = (pd.to_numeric(g["n_long_spans_ge8_tokens"], errors="coerce") > 0).mean()
        row["pct_rows_with_full_sentence_like_span"] = (pd.to_numeric(g["n_full_sentence_like_spans"], errors="coerce") > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(gcols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_model_dirs", default="")
    ap.add_argument("--individual_model_dirs", default="")
    ap.add_argument("--reduced_v1_model_dirs", default="")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for d in split_arg(args.union_model_dirs): rows.extend(union_rows(d))
    for d in split_arg(args.individual_model_dirs): rows.extend(individual_rows(d))
    for d in split_arg(args.reduced_v1_model_dirs): rows.extend(reduced_v1_rows(d))
    df = pd.DataFrame(rows)
    df.to_csv(out / "annotation_granularity_by_row.csv", index=False)
    summarize(df).to_csv(out / "annotation_granularity_summary_by_condition.csv", index=False)
    if not df.empty:
        df.sort_values(["n_full_sentence_like_spans", "max_span_token_count", "n_annotations", "duplicate_coverage_factor"], ascending=[False, False, False, False]).to_csv(out / "broad_span_annotation_audit.csv", index=False)
    print(f"Wrote annotation granularity audit to {out}")


if __name__ == "__main__":
    main()
