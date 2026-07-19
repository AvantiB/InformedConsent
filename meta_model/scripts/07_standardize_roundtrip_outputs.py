#!/usr/bin/env python
"""Standardize Union V0, individual source-model, and reduced V1 round-trip outputs.

The output is a single classifier-ready CSV with one row per source sentence,
LLM, condition, and information model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

INFO_MODELS = ["DUO", "ICO", "ODRL", "FHIR_Consent"]
V1_ROLE_FIELDS = ["decision", "action", "resource", "actor", "recipient_or_grantee", "purpose", "condition", "constraint_or_exception", "temporal_scope", "privacy_identifiability", "choice_structure", "lifecycle_effect", "risk_benefit_or_results", "residual_important_content"]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def strip_fence(text: str) -> str:
    text = norm(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|csv|yaml)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_jsonish(text: str) -> Any:
    text = strip_fence(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for l, r in [("{", "}"), ("[", "]")]:
        a, b = text.find(l), text.rfind(r)
        if a >= 0 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except Exception:
                pass
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                obj["_line_no"] = i
                rows.append(obj)
            except Exception as exc:
                rows.append({"_line_no": i, "_jsonl_parse_error": repr(exc)})
    return rows


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["source_id"]): r for r in rows if r.get("source_id") is not None}


def reconstruction(row: dict[str, Any]) -> str:
    parsed = row.get("parsed_backward")
    if isinstance(parsed, dict):
        for k in ["reconstructed_sentence", "reconstruction", "sentence", "text"]:
            if norm(parsed.get(k)):
                return norm(parsed.get(k))
    raw = norm(row.get("raw_response", ""))
    parsed_raw = parse_jsonish(raw)
    if isinstance(parsed_raw, dict):
        for k in ["reconstructed_sentence", "reconstruction", "sentence", "text"]:
            if norm(parsed_raw.get(k)):
                return norm(parsed_raw.get(k))
    return raw


def annotation_counts_from_obj(obj: Any) -> tuple[int, int]:
    if isinstance(obj, dict):
        anns = obj.get("annotations")
    elif isinstance(obj, list):
        anns = obj
    else:
        anns = None
    if not isinstance(anns, list):
        return 0, 0
    labels = []
    for a in anns:
        if isinstance(a, dict):
            labels.append(norm(a.get("union_element_id") or a.get("label") or a.get("element_id") or a.get("id")))
        else:
            labels.append(norm(a))
    labels = [x for x in labels if x]
    return len(anns), len(set(labels))


def v1_role_counts(obj: Any) -> tuple[int, int]:
    if not isinstance(obj, dict):
        return 0, 0
    n = 0
    roles = set()
    for prov in obj.get("provisions") or []:
        if not isinstance(prov, dict):
            continue
        for role in V1_ROLE_FIELDS:
            val = prov.get(role)
            if role == "decision":
                if isinstance(val, dict) and (norm(val.get("value")) or norm(val.get("evidence_span_text"))):
                    n += 1; roles.add(role)
                continue
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and any(norm(x) for x in item.values()):
                        n += 1; roles.add(role)
                    elif norm(item):
                        n += 1; roles.add(role)
            elif norm(val):
                n += 1; roles.add(role)
    return n, len(roles)


def forward_counts(raw: str, parsed_forward: Any = None) -> tuple[int, int, bool]:
    obj = parsed_forward if parsed_forward else parse_jsonish(raw)
    n, u = annotation_counts_from_obj(obj)
    if n:
        return n, u, True
    vn, vu = v1_role_counts(obj)
    if vn:
        return vn, vu, True
    text = strip_fence(raw)
    if not text:
        return 0, 0, False
    try:
        rows = [[norm(c) for c in row] for row in csv.reader(text.splitlines())]
        rows = [r for r in rows if any(r)]
    except Exception:
        return 0, 0, False
    if rows and any(c.lower() in {"annotation", "duo_label", "ico_label", "odrl_label", "fhir_label"} for c in rows[0]):
        rows = rows[1:]
    labels = []
    for r in rows:
        if len(r) >= 3:
            labels.append(r[-2] if r[-1].lower() in {"permit", "deny", "permission", "prohibition"} else r[-1])
    labels = [x for x in labels if x]
    return len(rows), len(set(labels)), False


def standardize_union(model_dir: Path):
    model_key = model_dir.name
    fwd = by_id(read_jsonl(model_dir / "union_v0_forward_mappings.jsonl"))
    bwd = by_id(read_jsonl(model_dir / "union_v0_backward_reconstructions.jsonl"))
    rows, missing = [], []
    for sid in sorted(set(fwd) | set(bwd)):
        f, b = fwd.get(sid, {}), bwd.get(sid, {})
        parsed = f.get("parsed_forward") if isinstance(f.get("parsed_forward"), dict) else None
        n_ann, n_unique, parse_ok = forward_counts(norm(f.get("raw_response", "")), parsed)
        rec = reconstruction(b)
        rows.append({"roundtrip_id": f"{model_key}__Union_V0__{sid}", "source_id": sid, "source_text": norm(f.get("source_text") or b.get("source_text")), "original_text": norm(f.get("source_text") or b.get("source_text")), "reconstructed_sentence": rec, "reconstructed_text": rec, "forward_mapping": json.dumps(parsed, ensure_ascii=False) if parsed else norm(f.get("raw_response", "")), "llm": model_key, "model": f.get("model") or b.get("model") or model_key, "condition": "union_v0_full_dictionary", "information_model": "Union_V0", "info_model": "Union_V0", "annotation_count": n_ann, "unique_element_count": n_unique, "forward_parse_ok": parse_ok, "backward_parse_ok": bool(rec), "has_forward": sid in fwd, "has_backward": sid in bwd, "output_dir": str(model_dir)})
        if sid not in fwd or sid not in bwd:
            missing.append({"output_dir": str(model_dir), "source_id": sid, "has_forward": sid in fwd, "has_backward": sid in bwd})
    audit = [{"output_dir": str(model_dir), "condition": "union_v0_full_dictionary", "llm": model_key, "information_model": "Union_V0", "n_forward": len(fwd), "n_backward": len(bwd), "n_standardized": len(rows), "n_missing_pairs": len(missing)}]
    return rows, audit, missing


def standardize_reduced_v1(model_dir: Path):
    evidence_mode = model_dir.name
    model_key = model_dir.parent.name if evidence_mode in {"compact", "permissive"} else model_dir.name
    if evidence_mode not in {"compact", "permissive"}:
        evidence_mode = "unknown"
    condition = f"reduced_v1_{evidence_mode}"
    fwd = by_id(read_jsonl(model_dir / "reduced_v1_forward_mappings.jsonl"))
    bwd = by_id(read_jsonl(model_dir / "reduced_v1_backward_reconstructions.jsonl"))
    rows, missing = [], []
    for sid in sorted(set(fwd) | set(bwd)):
        f, b = fwd.get(sid, {}), bwd.get(sid, {})
        parsed = f.get("parsed_forward") if isinstance(f.get("parsed_forward"), dict) else None
        n_ann, n_unique, parse_ok = forward_counts(norm(f.get("raw_response", "")), parsed)
        rec = reconstruction(b)
        rows.append({"roundtrip_id": f"{model_key}__Reduced_V1_{evidence_mode}__{sid}", "source_id": sid, "source_text": norm(f.get("source_text") or b.get("source_text")), "original_text": norm(f.get("source_text") or b.get("source_text")), "reconstructed_sentence": rec, "reconstructed_text": rec, "forward_mapping": json.dumps(parsed, ensure_ascii=False) if parsed else norm(f.get("raw_response", "")), "llm": model_key, "model": f.get("model") or b.get("model") or model_key, "condition": condition, "information_model": "Reduced_V1", "info_model": "Reduced_V1", "evidence_mode": evidence_mode, "annotation_count": n_ann, "unique_element_count": n_unique, "forward_parse_ok": parse_ok, "backward_parse_ok": bool(rec), "has_forward": sid in fwd, "has_backward": sid in bwd, "output_dir": str(model_dir)})
        if sid not in fwd or sid not in bwd:
            missing.append({"output_dir": str(model_dir), "source_id": sid, "has_forward": sid in fwd, "has_backward": sid in bwd})
    audit = [{"output_dir": str(model_dir), "condition": condition, "llm": model_key, "information_model": "Reduced_V1", "n_forward": len(fwd), "n_backward": len(bwd), "n_standardized": len(rows), "n_missing_pairs": len(missing)}]
    return rows, audit, missing


def standardize_individual(model_dir: Path):
    model_key = model_dir.name
    rows, audit, missing = [], [], []
    for info in INFO_MODELS:
        sub = model_dir / info
        fwd = by_id(read_jsonl(sub / "forward_mappings.jsonl"))
        bwd = by_id(read_jsonl(sub / "backward_reconstructions.jsonl"))
        n_missing = 0
        for sid in sorted(set(fwd) | set(bwd)):
            f, b = fwd.get(sid, {}), bwd.get(sid, {})
            raw = norm(f.get("raw_response", ""))
            n_ann, n_unique, parse_ok = forward_counts(raw)
            rec = reconstruction(b)
            rows.append({"roundtrip_id": f"{model_key}__{info}__{sid}", "source_id": sid, "source_text": norm(f.get("source_text") or b.get("source_text")), "original_text": norm(f.get("source_text") or b.get("source_text")), "reconstructed_sentence": rec, "reconstructed_text": rec, "forward_mapping": raw, "llm": model_key, "model": f.get("model") or b.get("model") or model_key, "condition": "individual_source_model_json", "information_model": info, "info_model": info, "annotation_count": n_ann, "unique_element_count": n_unique, "forward_parse_ok": parse_ok, "backward_parse_ok": bool(rec), "has_forward": sid in fwd, "has_backward": sid in bwd, "output_dir": str(sub)})
            if sid not in fwd or sid not in bwd:
                n_missing += 1
                missing.append({"output_dir": str(sub), "source_id": sid, "has_forward": sid in fwd, "has_backward": sid in bwd})
        audit.append({"output_dir": str(sub), "condition": "individual_source_model_json", "llm": model_key, "information_model": info, "n_forward": len(fwd), "n_backward": len(bwd), "n_standardized": len(set(fwd) | set(bwd)), "n_missing_pairs": n_missing})
    return rows, audit, missing


def split_paths(x: str) -> list[Path]:
    return [Path(p.strip()) for p in x.split(",") if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--union_model_dirs", default="", help="Comma-separated Union V0 model output dirs.")
    ap.add_argument("--individual_model_dirs", default="", help="Comma-separated individual model output dirs.")
    ap.add_argument("--reduced_v1_model_dirs", default="", help="Comma-separated Reduced V1 output dirs, usually .../<model_key>/<compact|permissive>.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--require_backward", action="store_true")
    args = ap.parse_args()
    rows, audit, missing = [], [], []
    for d in split_paths(args.union_model_dirs):
        r, a, m = standardize_union(d); rows += r; audit += a; missing += m
    for d in split_paths(args.individual_model_dirs):
        r, a, m = standardize_individual(d); rows += r; audit += a; missing += m
    for d in split_paths(args.reduced_v1_model_dirs):
        r, a, m = standardize_reduced_v1(d); rows += r; audit += a; missing += m
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if args.require_backward and not df.empty:
        df = df[df["reconstructed_text"].astype(str).str.len() > 0].copy()
    df.to_csv(out / "standardized_roundtrips.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(audit).to_csv(out / "standardization_audit.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "missing_pairs.csv", index=False)
    print(f"Wrote {len(df)} rows to {out / 'standardized_roundtrips.csv'}")


if __name__ == "__main__":
    main()
