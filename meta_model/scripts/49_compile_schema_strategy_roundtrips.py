#!/usr/bin/env python
"""Compile corrected baseline and direct-schema roundtrip outputs for scoring.

This script produces one classifier-ready standardized_roundtrips.csv from:

- corrected Union V0 output directories;
- corrected individual source-model output directories;
- direct LLM schema evaluation roundtrip_rows.csv files.

It intentionally only compiles completed round-trip outputs. Scoring is still done
with 09_score_roundtrip_outputs.py so all conditions use the same classifier and
lexical diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def split_paths_or_globs(x: str) -> list[Path]:
    out: list[Path] = []
    for part in [p.strip() for p in x.split(",") if p.strip()]:
        matches = sorted(glob.glob(part, recursive=True))
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(part))
    # de-duplicate preserving order
    seen = set()
    unique = []
    for p in out:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def load_standardizer_module():
    path = Path(__file__).resolve().parent / "07_standardize_roundtrip_outputs.py"
    spec = importlib.util.spec_from_file_location("standardize_roundtrip_outputs_07", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def standardize_direct_csv(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_csv(path).fillna("")
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if df.empty:
        audit.append({"output_dir": str(path.parent), "condition": "direct_llm_schema", "llm": "", "information_model": "direct_llm_reduced_schema", "n_standardized": 0})
        return rows, audit

    for i, r in df.iterrows():
        llm = norm(r.get("llm")) or "unknown"
        gran = norm(r.get("schema_granularity")) or "unknown_granularity"
        schema_id = norm(r.get("schema_id"))
        fold_id = norm(r.get("schema_fold_id"))
        source_id = norm(r.get("source_sentence_id")) or norm(r.get("eval_row_id")) or f"direct_row_{i:06d}"
        original = norm(r.get("original_sentence") or r.get("original_text") or r.get("source_text"))
        reconstructed = norm(r.get("reconstructed_sentence") or r.get("reconstructed_text"))
        condition = f"direct_llm_{gran}"
        annotation_count = r.get("n_valid_annotations", "")
        try:
            annotation_count = int(annotation_count)
        except Exception:
            annotation_count = None
        unique_count = None
        annotations_raw = norm(r.get("annotations_serialized"))
        if annotations_raw:
            try:
                anns = json.loads(annotations_raw)
                if isinstance(anns, list):
                    labels = []
                    for a in anns:
                        if isinstance(a, dict):
                            labels.append(norm(a.get("field_name") or a.get("field_id")))
                    unique_count = len(set(x for x in labels if x))
            except Exception:
                unique_count = None
        rows.append({
            "roundtrip_id": f"{llm}__{condition}__{fold_id}__{source_id}",
            "source_id": source_id,
            "source_text": original,
            "original_text": original,
            "reconstructed_sentence": reconstructed,
            "reconstructed_text": reconstructed,
            "forward_mapping": annotations_raw,
            "llm": llm,
            "model": llm,
            "condition": condition,
            "information_model": "direct_llm_reduced_schema",
            "info_model": "direct_llm_reduced_schema",
            "schema_id": schema_id,
            "schema_fold_id": fold_id,
            "schema_granularity": gran,
            "annotation_count": annotation_count,
            "unique_element_count": unique_count,
            "forward_parse_ok": bool(r.get("forward_parse_ok", True)),
            "backward_parse_ok": bool(r.get("backward_parse_ok", bool(reconstructed))),
            "has_forward": True,
            "has_backward": True,
            "output_dir": str(path.parent),
        })
    audit.append({
        "output_dir": str(path.parent),
        "condition": "direct_llm_schema",
        "llm": norm(df.get("llm", pd.Series([""])).iloc[0]) if "llm" in df else "",
        "information_model": "direct_llm_reduced_schema",
        "schema_granularity": norm(df.get("schema_granularity", pd.Series([""])).iloc[0]) if "schema_granularity" in df else "",
        "schema_fold_id": norm(df.get("schema_fold_id", pd.Series([""])).iloc[0]) if "schema_fold_id" in df else "",
        "n_standardized": len(rows),
        "input_csv": str(path),
    })
    return rows, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--union_model_dirs", default="", help="Comma-separated dirs/globs for corrected Union V0 model dirs.")
    ap.add_argument("--individual_model_dirs", default="", help="Comma-separated dirs/globs for corrected individual model dirs.")
    ap.add_argument("--direct_roundtrip_csvs", default="", help="Comma-separated files/globs for direct schema roundtrip_rows.csv files.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--require_backward", action="store_true", help="Drop rows with empty reconstructions.")
    args = ap.parse_args()

    std = load_standardizer_module()
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for d in split_paths_or_globs(args.union_model_dirs):
        r, a, m = std.standardize_union(d)
        rows.extend(r)
        audit.extend(a)
        missing.extend(m)
    for d in split_paths_or_globs(args.individual_model_dirs):
        r, a, m = std.standardize_individual(d)
        rows.extend(r)
        audit.extend(a)
        missing.extend(m)
    for p in split_paths_or_globs(args.direct_roundtrip_csvs):
        r, a = standardize_direct_csv(p)
        rows.extend(r)
        audit.extend(a)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if args.require_backward and not df.empty:
        df = df[df["reconstructed_text"].astype(str).str.len() > 0].copy()
    df.to_csv(out / "standardized_roundtrips.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(audit).to_csv(out / "standardization_audit.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "missing_pairs.csv", index=False)
    metadata = {
        "n_rows": int(len(df)),
        "union_model_dirs": [str(p) for p in split_paths_or_globs(args.union_model_dirs)],
        "individual_model_dirs": [str(p) for p in split_paths_or_globs(args.individual_model_dirs)],
        "direct_roundtrip_csvs": [str(p) for p in split_paths_or_globs(args.direct_roundtrip_csvs)],
        "require_backward": bool(args.require_backward),
    }
    (out / "compile_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(df)} rows to {out / 'standardized_roundtrips.csv'}")


if __name__ == "__main__":
    main()
