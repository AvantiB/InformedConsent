#!/usr/bin/env python
"""Build PI/expert-review examples with highlighted round-trip annotations.

Inputs are the standardized/scored/diagnostic round-trip CSVs produced by scripts
07/09/32. The script is intentionally tolerant of heterogeneous forward-mapping
formats used across individual models, Union V0, Manual V1, and LLM-induced V1.

Outputs:
- expert_review_examples.csv
- expert_review_examples.html
- expert_review_examples.xlsx, when openpyxl is available

The HTML file is intended to be sent to the PI/team or screenshotted for review.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SCORE_CANDIDATES = [
    "meaning_preserved_score",
    "meaning_preservation_score",
    "classifier_score",
    "predicted_probability",
    "probability",
    "score",
    "meaning_preserved_pred_proba",
    "meaning_preserved_pred",
    "mean_classifier_score",
]

TEXT_COLS = {
    "original": ["original_text", "source_text", "canonical_full_text", "full_text_original", "sentence", "sentence_text"],
    "reconstruction": ["reconstructed_text", "reconstructed_sentence", "backward_mapping", "reconstruction"],
    "mapping": ["forward_mapping", "annotations_serialized", "annotations_combined", "mapping"],
}

CONDITION_ORDER = [
    "individual_source_model_json",
    "union_v0_full_dictionary",
    "functional_v1_manual",
    "functional_v1_llm_induced",
    "functional_v1_llm_induced_consensus",
]

FIELD_KEYS = [
    "field_id", "field_name", "label", "element_id", "union_element_id", "cluster_id",
    "source_element", "source_element_id", "role", "type", "name", "id",
]
SPAN_KEYS = ["span_text", "span", "text", "value", "evidence_span_text", "verbatim", "quote"]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def safe_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def choose_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried {candidates}; available={list(df.columns)}")
    return None


def score_col(df: pd.DataFrame) -> str | None:
    return next((c for c in SCORE_CANDIDATES if c in df.columns), None)


def strip_fence(text: str) -> str:
    text = norm(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|csv|yaml)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_jsonish(text: Any) -> Any:
    s = strip_fence(norm(text))
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    for l, r in [("{", "}"), ("[", "]")]:
        a, b = s.find(l), s.rfind(r)
        if a >= 0 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                pass
    return None


def first_text(d: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        if norm(v):
            return norm(v)
    return ""


def compact_annotation_parse(text: str) -> list[dict[str, str]]:
    """Parse compact forms like: span text [Label] (decision)."""
    anns: list[dict[str, str]] = []
    if not norm(text):
        return anns
    # Non-greedy span before [label]; keep conservative to avoid swallowing all text.
    pat = re.compile(r"(?P<span>[^\[\]\n]{2,220}?)\s*\[(?P<label>[^\[\]]{1,180})\]\s*(?:\((?P<decision>[^)]{1,80})\))?", re.S)
    for m in pat.finditer(text):
        span = norm(m.group("span")).strip(" ;,.-")
        label = norm(m.group("label"))
        decision = norm(m.group("decision"))
        if span and label:
            anns.append({"span_text": span, "label": label, "decision": decision, "parse_source": "compact"})
    return anns


def annotation_from_dict(d: dict[str, Any]) -> dict[str, str] | None:
    span = first_text(d, SPAN_KEYS)
    label = first_text(d, FIELD_KEYS)
    decision = first_text(d, ["decision", "polarity", "permission", "rule_type", "sentence_decision"])
    if span and label:
        return {"span_text": span, "label": label, "decision": decision, "parse_source": "json"}
    # Sometimes Functional V1 uses fields with nested value objects.
    for k, v in d.items():
        if isinstance(v, dict):
            span2 = first_text(v, SPAN_KEYS)
            if span2:
                return {"span_text": span2, "label": str(k), "decision": decision, "parse_source": "json_nested"}
    return None


def collect_annotations(obj: Any) -> list[dict[str, str]]:
    anns: list[dict[str, str]] = []
    if obj is None:
        return anns
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                a = annotation_from_dict(item)
                if a:
                    anns.append(a)
                anns.extend(collect_annotations(item.get("annotations")))
            elif norm(item):
                anns.append({"span_text": norm(item), "label": "annotation", "decision": "", "parse_source": "json_list"})
        return anns
    if isinstance(obj, dict):
        # Standard top-level annotation lists.
        for key in ["annotations", "span_annotations", "elements", "fields", "mapped_elements"]:
            if isinstance(obj.get(key), list):
                anns.extend(collect_annotations(obj.get(key)))
        # Functional interpretation units/provisions may contain field names as keys.
        for container_key in ["interpretation_units", "provisions", "sentence_level_elements"]:
            val = obj.get(container_key)
            if isinstance(val, list):
                for unit in val:
                    if isinstance(unit, dict):
                        for k, v in unit.items():
                            if k in {"annotations", "evidence", "source"}:
                                continue
                            if isinstance(v, dict):
                                span = first_text(v, SPAN_KEYS)
                                if span:
                                    anns.append({"span_text": span, "label": k, "decision": first_text(v, ["polarity", "decision", "value"]), "parse_source": container_key})
                            elif isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict):
                                        span = first_text(item, SPAN_KEYS)
                                        if span:
                                            anns.append({"span_text": span, "label": k, "decision": first_text(item, ["polarity", "decision", "value"]), "parse_source": container_key})
                                    elif norm(item):
                                        anns.append({"span_text": norm(item), "label": k, "decision": "", "parse_source": container_key})
                            elif norm(v) and len(norm(v).split()) <= 14:
                                anns.append({"span_text": norm(v), "label": k, "decision": "", "parse_source": container_key})
        # Single annotation-shaped dict.
        a = annotation_from_dict(obj)
        if a:
            anns.append(a)
    # De-duplicate.
    seen = set()
    out = []
    for a in anns:
        key = (a.get("span_text", "").lower(), a.get("label", "").lower(), a.get("decision", "").lower())
        if key not in seen and a.get("span_text"):
            seen.add(key)
            out.append(a)
    return out


def parse_annotations(raw_mapping: Any) -> list[dict[str, str]]:
    raw = norm(raw_mapping)
    obj = parse_jsonish(raw)
    anns = collect_annotations(obj)
    if anns:
        return anns
    return compact_annotation_parse(raw)


def assign_colors(labels: list[str]) -> dict[str, str]:
    colors = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    mapping: dict[str, str] = {}
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = colors[len(mapping) % len(colors)]
    return mapping


def find_nonoverlap_spans(text: str, anns: list[dict[str, str]]) -> list[tuple[int, int, dict[str, str]]]:
    lower = text.lower()
    hits: list[tuple[int, int, dict[str, str]]] = []
    occupied: list[tuple[int, int]] = []
    for a in sorted(anns, key=lambda x: len(x.get("span_text", "")), reverse=True):
        span = norm(a.get("span_text", ""))
        if not span:
            continue
        start = lower.find(span.lower())
        if start < 0:
            # Allow punctuation-insensitive fallback for short but important phrases.
            cleaned = re.sub(r"\s+", " ", re.escape(span.lower())).replace("\\ ", r"\s+")
            m = re.search(cleaned, lower)
            if not m:
                continue
            start, end = m.start(), m.end()
        else:
            end = start + len(span)
        if any(not (end <= a0 or start >= b0) for a0, b0 in occupied):
            continue
        occupied.append((start, end))
        hits.append((start, end, a))
    return sorted(hits, key=lambda x: x[0])


def highlighted_text(text: str, anns: list[dict[str, str]]) -> str:
    text = norm(text)
    if not text:
        return ""
    labels = [norm(a.get("label")) for a in anns if norm(a.get("label"))]
    cmap = assign_colors(labels)
    hits = find_nonoverlap_spans(text, anns)
    if not hits:
        return html.escape(text)
    parts: list[str] = []
    pos = 0
    for start, end, a in hits:
        parts.append(html.escape(text[pos:start]))
        lab = norm(a.get("label")) or "annotation"
        dec = norm(a.get("decision"))
        cls = cmap.get(lab, "c1")
        title = html.escape(lab + (f" | {dec}" if dec else ""))
        parts.append(f'<mark class="ann {cls}" title="{title}">{html.escape(text[start:end])}<sup>{html.escape(short_label(lab))}</sup></mark>')
        pos = end
    parts.append(html.escape(text[pos:]))
    return "".join(parts)


def short_label(label: str, n: int = 22) -> str:
    label = re.sub(r"^[A-Za-z0-9_]+::", "", label)
    return label if len(label) <= n else label[: n - 1] + "…"


def select_examples(df: pd.DataFrame, per_group: int, max_total: int) -> pd.DataFrame:
    work = df.copy()
    sc = score_col(work)
    if sc:
        work["_score"] = pd.to_numeric(work[sc], errors="coerce")
    else:
        work["_score"] = 0.5
    if "suspected_error_count" in work.columns:
        work["_error_count"] = pd.to_numeric(work["suspected_error_count"], errors="coerce").fillna(0)
    else:
        work["_error_count"] = 0
    if "content_word_recall" in work.columns:
        work["_content_recall"] = pd.to_numeric(work["content_word_recall"], errors="coerce").fillna(1)
    else:
        work["_content_recall"] = 1

    rows = []
    group_cols = [c for c in ["condition", "information_model", "llm"] if c in work.columns]
    grouped = work.groupby(group_cols, dropna=False) if group_cols else [((), work)]
    for _, sub in grouped:
        # Include a mix: best, worst, and diagnostic-error examples.
        picks = []
        picks.append(sub.sort_values("_score", ascending=False).head(max(1, per_group // 3)))
        picks.append(sub.sort_values("_score", ascending=True).head(max(1, per_group // 3)))
        picks.append(sub.sort_values(["_error_count", "_content_recall"], ascending=[False, True]).head(max(1, per_group - sum(len(p) for p in picks))))
        merged = pd.concat(picks, ignore_index=False).drop_duplicates()
        rows.append(merged.head(per_group))
    out = pd.concat(rows, ignore_index=True) if rows else work.head(0)
    # Prioritize canonical conditions and then score/error mix.
    if "condition" in out.columns:
        out["_condition_order"] = out["condition"].astype(str).map({c: i for i, c in enumerate(CONDITION_ORDER)}).fillna(99)
    else:
        out["_condition_order"] = 0
    out = out.sort_values(["_condition_order", "llm" if "llm" in out.columns else "_score", "_score"], ascending=[True, True, False])
    return out.head(max_total).copy()


def build_review_rows(df: pd.DataFrame, per_group: int, max_total: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    orig_col = choose_col(df, TEXT_COLS["original"])
    rec_col = choose_col(df, TEXT_COLS["reconstruction"])
    map_col = choose_col(df, TEXT_COLS["mapping"], required=False)
    sc = score_col(df)
    selected = select_examples(df, per_group, max_total)

    rows = []
    html_cards = []
    for i, r in selected.iterrows():
        mapping = r.get(map_col, "") if map_col else ""
        anns = parse_annotations(mapping)
        original = norm(r.get(orig_col))
        reconstruction = norm(r.get(rec_col))
        ann_preview = "; ".join([f"{a.get('span_text')} → {a.get('label')}" for a in anns[:12]])
        common = {
            "review_id": f"REV_{len(rows)+1:04d}",
            "roundtrip_id": norm(r.get("roundtrip_id")),
            "source_id": norm(r.get("source_id")),
            "condition": norm(r.get("condition")),
            "information_model": norm(r.get("information_model")),
            "llm": norm(r.get("llm")),
            "classifier_score": safe_float(r.get(sc)) if sc else None,
            "annotation_count": safe_float(r.get("annotation_count")),
            "unique_element_count": safe_float(r.get("unique_element_count")),
            "content_word_recall": safe_float(r.get("content_word_recall")),
            "important_category_presence_recall": safe_float(r.get("important_category_presence_recall")),
            "modal_word_change_ratio": safe_float(r.get("modal_word_change_ratio")),
            "unmatched_language_rate": safe_float(r.get("unmatched_language_rate")),
            "suspected_error_flags": norm(r.get("suspected_error_flags")),
            "original_text": original,
            "reconstructed_text": reconstruction,
            "annotation_preview": ann_preview,
            "n_parsed_annotations": len(anns),
            "expert_meaning_preserved": "",
            "expert_schema_feedback": "",
            "expert_field_merge_split_notes": "",
            "expert_missing_field_notes": "",
        }
        rows.append(common)
        html_cards.append({**common, "highlighted_original": highlighted_text(original, anns), "annotations": anns})
    return pd.DataFrame(rows), html_cards


def metric_badge(name: str, val: Any) -> str:
    if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        f = float(val)
        text = f"{f:.3f}" if abs(f) <= 1 else f"{f:.1f}"
    except Exception:
        text = html.escape(str(val))
    return f'<span class="badge"><b>{html.escape(name)}:</b> {text}</span>'


def write_html(cards: list[dict[str, Any]], out: Path) -> None:
    style = """
    body{font-family:Arial,Helvetica,sans-serif;margin:28px;color:#1f2937;background:#f8fafc;}
    h1{color:#0f172a;margin-bottom:4px;} h2{color:#0f766e;margin-top:28px;}
    .subtitle{color:#475569;margin-bottom:24px;}.card{background:white;border:1px solid #dbeafe;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 2px 8px rgba(15,23,42,0.08)}
    .meta{font-size:13px;color:#475569;margin-bottom:10px}.badge{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:4px 9px;margin:3px;font-size:12px}
    .textblock{line-height:1.6;font-size:16px;background:#f8fafc;border-left:4px solid #0ea5e9;padding:12px;border-radius:8px;margin:10px 0}
    .recon{border-left-color:#14b8a6}.ann{padding:1px 3px;border-radius:4px}.ann sup{font-size:10px;margin-left:3px;color:#334155}.c1{background:#bfdbfe}.c2{background:#99f6e4}.c3{background:#ddd6fe}.c4{background:#fecaca}.c5{background:#fde68a}.c6{background:#bbf7d0}.c7{background:#fbcfe8}.c8{background:#bae6fd}.c9{background:#e9d5ff}.c10{background:#fed7aa}
    table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}td,th{border:1px solid #e2e8f0;padding:6px;text-align:left}th{background:#eff6ff}.flags{color:#b91c1c;font-weight:bold}.review{background:#fffbeb;border:1px dashed #f59e0b;border-radius:8px;padding:10px;margin-top:10px;color:#78350f}
    """
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Expert review examples</title><style>", style, "</style></head><body>"]
    parts.append("<h1>Informed Consent Meta-Model: Expert Review Examples</h1>")
    parts.append("<div class='subtitle'>Examples are stratified across schema strategies and LLMs. Highlighted spans are parsed from the forward mapping when available. Expert review boxes are intentionally left blank in the CSV/XLSX output.</div>")
    current = None
    for c in cards:
        cond = c.get("condition", "")
        if cond != current:
            current = cond
            parts.append(f"<h2>{html.escape(cond or 'Condition not available')}</h2>")
        parts.append("<div class='card'>")
        parts.append(f"<div class='meta'><b>{html.escape(c.get('review_id',''))}</b> | LLM: {html.escape(c.get('llm',''))} | Information model: {html.escape(c.get('information_model',''))} | Source ID: {html.escape(c.get('source_id',''))}</div>")
        metrics = [
            metric_badge("classifier", c.get("classifier_score")),
            metric_badge("content recall", c.get("content_word_recall")),
            metric_badge("cue-category recall", c.get("important_category_presence_recall")),
            metric_badge("modal change", c.get("modal_word_change_ratio")),
            metric_badge("annotations", c.get("annotation_count")),
            metric_badge("unique fields", c.get("unique_element_count")),
        ]
        parts.append("<div>" + " ".join([m for m in metrics if m]) + "</div>")
        if c.get("suspected_error_flags"):
            parts.append(f"<div class='flags'>Flags: {html.escape(c.get('suspected_error_flags',''))}</div>")
        parts.append("<b>Original with parsed annotations</b>")
        parts.append(f"<div class='textblock'>{c.get('highlighted_original','')}</div>")
        parts.append("<b>Backward reconstruction</b>")
        parts.append(f"<div class='textblock recon'>{html.escape(c.get('reconstructed_text',''))}</div>")
        anns = c.get("annotations", [])
        if anns:
            parts.append("<details><summary>Parsed annotations</summary><table><tr><th>Span</th><th>Label/field</th><th>Decision/polarity</th><th>Parse source</th></tr>")
            for a in anns[:80]:
                parts.append(f"<tr><td>{html.escape(a.get('span_text',''))}</td><td>{html.escape(a.get('label',''))}</td><td>{html.escape(a.get('decision',''))}</td><td>{html.escape(a.get('parse_source',''))}</td></tr>")
            parts.append("</table></details>")
        parts.append("<div class='review'><b>Expert review prompts:</b> Is the meaning preserved? Are any fields missing, overly broad, redundant, or unsafe to merge? Should Manual V1 or LLM-induced V1 be revised?</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    out.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrip_metrics_csv", required=True, help="Diagnostic row-level CSV from script 32, or scored/standardized roundtrip CSV.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--examples_per_condition_llm", type=int, default=3)
    ap.add_argument("--max_examples", type=int, default=160)
    args = ap.parse_args()

    df = pd.read_csv(args.roundtrip_metrics_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    review_df, cards = build_review_rows(df, args.examples_per_condition_llm, args.max_examples)
    review_df.to_csv(out / "expert_review_examples.csv", index=False)
    write_html(cards, out / "expert_review_examples.html")
    try:
        review_df.to_excel(out / "expert_review_examples.xlsx", index=False)
    except Exception:
        pass
    print(f"Wrote expert review examples to {out}")


if __name__ == "__main__":
    main()
