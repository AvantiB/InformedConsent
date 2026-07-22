#!/usr/bin/env python
"""Build crosswalk tables for Manual V1 and LLM-induced V1 expert review.

Manual V1 crosswalk is usually produced by script 26. This script formats it for
PI review and, when LLM-induced fold schemas/evidence cards are available, derives
an evidence-based source-model-to-LLM-field crosswalk.

The LLM-induced crosswalk is intentionally labeled as evidence-derived. It should
be reviewed by experts before being treated as a final semantic alignment.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


SOURCE_MODEL_NAMES = ["ICO", "DUO", "ODRL", "FHIR", "FHIR_Consent"]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def safe_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")


def read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def read_schema(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML schemas.")
    return yaml.safe_load(text)


def field_list(schema: dict[str, Any]) -> list[dict[str, Any]]:
    fields = schema.get("fields") or schema.get("schema_fields") or schema.get("functional_fields") or []
    if isinstance(fields, dict):
        out = []
        for fid, v in fields.items():
            d = dict(v) if isinstance(v, dict) else {"definition": v}
            d.setdefault("field_id", fid)
            out.append(d)
        return out
    if isinstance(fields, list):
        out = []
        for i, v in enumerate(fields):
            if isinstance(v, dict):
                d = dict(v)
            else:
                d = {"field_id": str(v)}
            d.setdefault("field_id", d.get("id") or d.get("name") or f"field_{i:02d}")
            out.append(d)
        return out
    return []


def listify(x: Any) -> list[Any]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        return [v.strip() for v in re.split(r"[;|,]", s) if v.strip()]
    return [x]


def normalize_manual_crosswalk(manual_crosswalk_csv: str | Path | None, out_dir: Path) -> pd.DataFrame:
    df = read_csv(manual_crosswalk_csv)
    if df is None:
        return pd.DataFrame()
    colmap = {}
    for target, candidates in {
        "source_model": ["source_model", "model", "information_model"],
        "source_element": ["source_element", "source_element_id", "element_id", "element", "source_label"],
        "source_label": ["source_label", "label", "element_label", "canonical_label", "source_element_name"],
        "manual_v1_field": ["v1_field", "functional_field", "proposed_v1_field", "field_id", "target_field"],
        "mapping_type": ["mapping_type", "relationship", "mapping_relation"],
        "context_rule": ["context_rule", "rule", "notes"],
    }.items():
        for c in candidates:
            if c in df.columns:
                colmap[target] = c
                break
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "source_model": norm(r.get(colmap.get("source_model", ""))),
            "source_element": norm(r.get(colmap.get("source_element", ""))),
            "source_label": norm(r.get(colmap.get("source_label", ""))),
            "manual_v1_field": norm(r.get(colmap.get("manual_v1_field", ""))),
            "mapping_type": norm(r.get(colmap.get("mapping_type", ""))) or "candidate_mapping",
            "context_rule": norm(r.get(colmap.get("context_rule", ""))),
            "expert_review_status": "",
            "expert_notes": "",
        })
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(out_dir / "manual_v1_source_model_crosswalk_for_review.csv", index=False)
    return out


def card_id(card: dict[str, Any]) -> str:
    for k in ["card_id", "evidence_card_id", "stability_group_id", "selected_field_id", "field_id", "id"]:
        if norm(card.get(k)):
            return norm(card.get(k))
    return ""


def flatten_source_elements(card: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    possible = []
    for key in ["source_elements", "source_model_elements", "elements", "member_elements", "crosswalk_mappings"]:
        val = card.get(key)
        if isinstance(val, list):
            possible.extend(val)
    # Some cards store element summaries as dict keyed by source model.
    for key in ["source_model_counts", "source_model_support", "source_elements_by_model"]:
        val = card.get(key)
        if isinstance(val, dict):
            for m, elems in val.items():
                for e in listify(elems):
                    if isinstance(e, dict):
                        rows.append({"source_model": norm(e.get("source_model") or m), "source_element": norm(e.get("source_element") or e.get("element_id") or e.get("label")), "source_label": norm(e.get("source_label") or e.get("label"))})
                    else:
                        rows.append({"source_model": norm(m), "source_element": norm(e), "source_label": norm(e)})
    for item in possible:
        if isinstance(item, dict):
            rows.append({
                "source_model": norm(item.get("source_model") or item.get("information_model") or item.get("model")),
                "source_element": norm(item.get("source_element") or item.get("source_element_id") or item.get("element_id") or item.get("id") or item.get("label")),
                "source_label": norm(item.get("source_label") or item.get("label") or item.get("name") or item.get("element")),
            })
        elif norm(item):
            s = norm(item)
            # Try source::element convention.
            if "::" in s:
                m, e = s.split("::", 1)
                rows.append({"source_model": m, "source_element": e, "source_label": e})
            else:
                rows.append({"source_model": "", "source_element": s, "source_label": s})
    # Fallback: parse strings from top spans or source element lists.
    seen = set()
    out = []
    for r in rows:
        key = (r["source_model"], r["source_element"], r["source_label"])
        if r["source_element"] and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load_cards(cards_root: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    if not cards_root.exists():
        return cards
    for p in cards_root.glob("fold_*/schema_induction_evidence_cards.jsonl"):
        fold = p.parent.name
        for card in read_jsonl(p):
            cid = card_id(card)
            if not cid:
                continue
            card = dict(card)
            card["fold"] = fold
            cards[f"{fold}::{cid}"] = card
            cards.setdefault(cid, card)
    return cards


def field_evidence_ids(field: dict[str, Any]) -> list[str]:
    ids = []
    for key in ["evidence_card_ids", "assigned_evidence_cards", "supporting_evidence_cards", "evidence_cards", "source_cards", "cluster_ids", "stability_group_ids"]:
        ids.extend([norm(x) for x in listify(field.get(key)) if norm(x)])
    # Also parse evidence IDs out of rationale text.
    for key in ["rationale", "evidence", "notes", "definition"]:
        text = norm(field.get(key))
        ids.extend(re.findall(r"(?:C\d{3,}|SG[_-]?\d+|field[_-]?\d+|cluster[_-]?\d+)", text, flags=re.I))
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def infer_source_elements_from_field_text(field: dict[str, Any]) -> list[dict[str, str]]:
    text = " ".join(norm(field.get(k)) for k in ["field_id", "name", "definition", "inclusion_criteria", "examples", "source_model_elements", "crosswalk"])
    rows = []
    for m in SOURCE_MODEL_NAMES:
        if re.search(rf"\b{re.escape(m)}\b", text, flags=re.I):
            rows.append({"source_model": m, "source_element": "mentioned_in_field_definition", "source_label": "mentioned_in_field_definition"})
    return rows


def build_llm_fold_crosswalk(llm_schema_root: Path, cards_root: Path, manual_crosswalk: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cards = load_cards(cards_root)
    manual_by_source = {}
    if not manual_crosswalk.empty:
        for _, r in manual_crosswalk.iterrows():
            key = (norm(r.get("source_model")).lower(), norm(r.get("source_element")).lower())
            manual_by_source.setdefault(key, set()).add(norm(r.get("manual_v1_field")))
    rows = []
    for schema_path in sorted(llm_schema_root.glob("fold_*/llm_induced_functional_v1_candidate.*")):
        if schema_path.suffix not in {".yaml", ".yml", ".json"}:
            continue
        fold = schema_path.parent.name
        schema = read_schema(schema_path)
        if not schema:
            continue
        for field in field_list(schema):
            fid = norm(field.get("field_id") or field.get("id") or field.get("name"))
            fname = norm(field.get("name") or field.get("label") or fid)
            definition = norm(field.get("definition") or field.get("description"))
            tier = norm(field.get("tier") or field.get("status") or field.get("field_type"))
            eids = field_evidence_ids(field)
            source_rows = []
            for eid in eids:
                for key in [f"{fold}::{eid}", eid]:
                    if key in cards:
                        source_rows.extend(flatten_source_elements(cards[key]))
            if not source_rows:
                source_rows = infer_source_elements_from_field_text(field)
            if not source_rows:
                rows.append({
                    "fold": fold,
                    "llm_induced_field": fid,
                    "llm_induced_field_name": fname,
                    "llm_induced_definition": definition,
                    "tier": tier,
                    "source_model": "",
                    "source_element": "",
                    "source_label": "",
                    "manual_v1_fields_linked_by_source_element": "",
                    "evidence_card_ids": "; ".join(eids),
                    "mapping_basis": "field_without_parseable_source_elements",
                    "expert_review_status": "",
                    "expert_notes": "",
                })
                continue
            seen = set()
            for sr in source_rows:
                key = (norm(sr.get("source_model")), norm(sr.get("source_element")), norm(sr.get("source_label")))
                if key in seen:
                    continue
                seen.add(key)
                manual_fields = sorted(manual_by_source.get((key[0].lower(), key[1].lower()), set()))
                rows.append({
                    "fold": fold,
                    "llm_induced_field": fid,
                    "llm_induced_field_name": fname,
                    "llm_induced_definition": definition,
                    "tier": tier,
                    "source_model": key[0],
                    "source_element": key[1],
                    "source_label": key[2],
                    "manual_v1_fields_linked_by_source_element": "; ".join(manual_fields),
                    "evidence_card_ids": "; ".join(eids),
                    "mapping_basis": "evidence_card_source_elements" if eids else "field_text_heuristic",
                    "expert_review_status": "",
                    "expert_notes": "",
                })
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(out_dir / "llm_induced_v1_source_model_crosswalk_by_fold_for_review.csv", index=False)
    return out


def build_consensus_crosswalk(consensus_mapping_csv: str | Path | None, fold_crosswalk: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    mapping = read_csv(consensus_mapping_csv)
    if mapping is None or fold_crosswalk.empty:
        return pd.DataFrame()
    cons_col = next((c for c in ["consensus_field", "consensus_field_id", "consensus_name"] if c in mapping.columns), None)
    fold_col = next((c for c in ["fold", "fold_id"] if c in mapping.columns), None)
    fold_field_col = next((c for c in ["fold_field", "fold_field_id", "llm_induced_field", "field_id"] if c in mapping.columns), None)
    if not (cons_col and fold_col and fold_field_col):
        return pd.DataFrame()
    m = mapping[[fold_col, fold_field_col, cons_col]].copy()
    m.columns = ["fold", "llm_induced_field", "llm_induced_consensus_field"]
    joined = fold_crosswalk.merge(m, on=["fold", "llm_induced_field"], how="left")
    joined.to_csv(out_dir / "llm_induced_consensus_source_model_crosswalk_for_review.csv", index=False)
    return joined


def manual_vs_llm_alignment(manual: pd.DataFrame, llm: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if manual.empty or llm.empty:
        return pd.DataFrame()
    m = manual[["source_model", "source_element", "manual_v1_field"]].dropna().drop_duplicates()
    l = llm[["source_model", "source_element", "llm_induced_field", "llm_induced_field_name", "fold"]].dropna().drop_duplicates()
    joined = m.merge(l, on=["source_model", "source_element"], how="outer")
    summary = joined.groupby(["manual_v1_field", "llm_induced_field", "llm_induced_field_name"], dropna=False).agg(
        n_source_elements=("source_element", "nunique"),
        n_source_models=("source_model", "nunique"),
        folds=("fold", lambda x: "; ".join(sorted(set(norm(v) for v in x if norm(v)))))
    ).reset_index()
    summary["expert_alignment_decision"] = ""
    summary["expert_notes"] = ""
    summary.to_csv(out_dir / "manual_v1_vs_llm_induced_field_alignment_for_review.csv", index=False)
    return summary


def write_html_summary(manual: pd.DataFrame, llm: pd.DataFrame, consensus: pd.DataFrame, align: pd.DataFrame, out: Path) -> None:
    def table(df: pd.DataFrame, n=20) -> str:
        if df.empty:
            return "<p><i>Not available.</i></p>"
        return df.head(n).to_html(index=False, escape=True)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>V1 crosswalk review</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:28px;color:#1f2937}}h1{{color:#0f172a}}h2{{color:#0f766e}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{border:1px solid #e2e8f0;padding:5px;text-align:left}}th{{background:#eff6ff}}.note{{background:#f8fafc;border-left:4px solid #0ea5e9;padding:12px;margin:14px 0}}</style></head><body>
<h1>Manual V1 and LLM-Induced V1 Crosswalks for Expert Review</h1>
<div class='note'>These tables are candidate alignments to support discussion. The LLM-induced crosswalk is evidence-derived from fold schemas and induction evidence cards; it should be expert-adjudicated before use as a final source-model alignment.</div>
<h2>Manual V1 source-model crosswalk preview</h2>{table(manual)}
<h2>LLM-induced V1 source-model crosswalk by fold preview</h2>{table(llm)}
<h2>LLM-induced consensus crosswalk preview</h2>{table(consensus)}
<h2>Manual V1 vs LLM-induced field alignment preview</h2>{table(align)}
</body></html>"""
    out.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manual_crosswalk_csv", default="meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv")
    ap.add_argument("--llm_induced_schema_root", default="meta_model/functional_v1/llm_induced")
    ap.add_argument("--evidence_cards_root", default="meta_model/functional_v1/llm_induction_cards")
    ap.add_argument("--llm_consensus_mapping_csv", default="")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manual = normalize_manual_crosswalk(args.manual_crosswalk_csv, out)
    llm = build_llm_fold_crosswalk(Path(args.llm_induced_schema_root), Path(args.evidence_cards_root), manual, out)
    consensus = build_consensus_crosswalk(args.llm_consensus_mapping_csv, llm, out)
    align = manual_vs_llm_alignment(manual, consensus if not consensus.empty else llm, out)
    write_html_summary(manual, llm, consensus, align, out / "v1_crosswalk_review_summary.html")
    print(f"Wrote V1 crosswalk review tables to {out}")


if __name__ == "__main__":
    main()
