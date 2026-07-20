#!/usr/bin/env python
"""Build a clean expert round-trip corpus from original annotation workbooks.

This script converts the original researcher handoff Excel workbooks into a
single normalized CSV that can be used as the authoritative derivation corpus
for Reduced V1 induction.

Expected workbook columns, case-insensitive:
  source_file, ID, full_text, annotations_combined, backward_mapping,
  Results/results, Notes

Outputs one row per source sentence / information model / LLM workbook row, with
annotations_json containing parsed span-level source-element mentions.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COLS = ["canonical_full_text", "full_text", "text", "sentence", "source_text"]
ID_COLS = ["ID", "id", "source_id", "sentence_id", "roundtrip_id"]
SOURCE_FILE_COLS = ["source_file", "file", "document", "doc"]
ANNOTATION_COLS = ["annotations_combined", "annotations_serialized", "annotations", "forward_mapping"]
BACKWARD_COLS = ["backward_mapping", "backward", "reconstruction", "backward_reconstruction"]
RESULT_COLS = ["Results", "results", "meaning_preserved", "human_meaning_preserved", "expert_meaning_preserved"]
NOTES_COLS = ["Notes", "notes", "comment", "comments"]

INFO_ALIASES = {
    "DUO": "DUO",
    "FHIR": "FHIR_Consent",
    "FHIR_CONSENT": "FHIR_Consent",
    "ICO": "ICO",
    "ODRL": "ODRL",
}

POSITIVE_VALUES = {"1", "true", "yes", "y", "preserved", "meaning preserved", "pass", "passed", "positive", "match", "matches"}
NEGATIVE_VALUES = {"0", "false", "no", "n", "not preserved", "failed", "fail", "negative", "mismatch", "does not match"}

CODE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]*:\d{3,}\b")
BRACKET_PAREN_RE = re.compile(r"(?P<span>[^\[]*?)\s*\[(?P<label>[^\]]+)\]\s*\((?P<decision>[^)]*)\)")
BRACKET_ONLY_RE = re.compile(r"(?P<span>[^\[]*?)\s*\[(?P<label>[^\]]+)\]")


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).replace("\r", "\n").split())


def pick_col(df: pd.DataFrame, names: list[str], required: bool = False) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried {names}; available={list(df.columns)}")
    return None


def infer_information_model(path: Path, sheet_name: str = "") -> str:
    text = f"{path.stem} {sheet_name}".upper()
    for key, val in INFO_ALIASES.items():
        if key in text:
            return val
    return "unknown"


def infer_llm(path: Path) -> str:
    text = path.stem.lower()
    if "chatgpt5" in text or "gpt5" in text:
        return "ChatGPT5"
    if "claude" in text:
        return "Claude4.5Sonnet"
    if "gemini" in text:
        return "Gemini3FlashPrev"
    return path.stem


def result_to_bool(x: Any) -> int | None:
    v = norm(x).lower()
    if v in POSITIVE_VALUES:
        return 1
    if v in NEGATIVE_VALUES:
        return 0
    try:
        f = float(v)
        if f >= 0.5:
            return 1
        if f < 0.5:
            return 0
    except Exception:
        pass
    return None


def canonical_element_id(raw_label: str, information_model: str) -> tuple[str, str]:
    """Return (union_element_id, cleaned_source_element_label)."""
    label = norm(raw_label).replace("***", "").strip()
    if not label or label.upper() == "NA":
        return "", label

    code_match = CODE_RE.search(label)
    if code_match:
        core = code_match.group(0)
    else:
        # For compact model labels like Consent.provision.data, Action_Verb, Asset_DO,
        # Permission, Constraint, Party, etc., the first token is the element name.
        first = label.split()[0]
        core = first if first else label

    return f"{information_model}::{core}" if information_model else core, label


def parse_annotation_string(raw: Any, information_model: str) -> list[dict[str, Any]]:
    text = str(raw or "").replace("\r", "\n")
    if not norm(text):
        return []

    annotations: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()

    for regex in [BRACKET_PAREN_RE, BRACKET_ONLY_RE]:
        for match in regex.finditer(text):
            loc = (match.start(), match.end())
            if loc in seen_spans:
                continue
            seen_spans.add(loc)
            span = norm(match.group("span"))
            label = norm(match.group("label"))
            decision = norm(match.groupdict().get("decision", ""))
            uid, source_label = canonical_element_id(label, information_model)
            if not uid:
                continue
            annotations.append({
                "annotation_index": len(annotations),
                "union_element_id": uid,
                "source_element_label": source_label,
                "span_text": span,
                "decision_value": decision,
                "raw_annotation_text": norm(match.group(0)),
            })

    return annotations


def read_workbook(path: Path, skip_sheets_regex: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip_re = re.compile(skip_sheets_regex, flags=re.I) if skip_sheets_regex else None
    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        if skip_re and skip_re.search(sheet_name):
            continue
        df = pd.read_excel(path, sheet_name=sheet_name).fillna("")
        if df.empty:
            continue
        text_col = pick_col(df, TEXT_COLS)
        ann_col = pick_col(df, ANNOTATION_COLS)
        res_col = pick_col(df, RESULT_COLS)
        if not text_col or not ann_col or not res_col:
            continue
        id_col = pick_col(df, ID_COLS)
        src_col = pick_col(df, SOURCE_FILE_COLS)
        back_col = pick_col(df, BACKWARD_COLS)
        notes_col = pick_col(df, NOTES_COLS)
        info_model = infer_information_model(path, sheet_name)
        llm = infer_llm(path)

        for i, row in df.iterrows():
            full_text = norm(row.get(text_col, ""))
            if not full_text:
                continue
            raw_result = row.get(res_col, "")
            meaning_preserved = result_to_bool(raw_result)
            anns = parse_annotation_string(row.get(ann_col, ""), info_model)
            source_file = norm(row.get(src_col, "")) if src_col else ""
            source_id = norm(row.get(id_col, "")) if id_col else ""
            if not source_id:
                source_id = f"{path.stem}:{sheet_name}:{i+2}"
            unique_elements = sorted({a["union_element_id"] for a in anns})
            rows.append({
                "canonical_full_text": full_text,
                "source_file": source_file,
                "source_id": source_id,
                "information_model": info_model,
                "llm": llm,
                "annotations_raw": str(row.get(ann_col, "")),
                "annotations_json": json.dumps({"annotations": anns}, ensure_ascii=False),
                "annotation_count": len(anns),
                "unique_element_count": len(unique_elements),
                "unique_elements_json": json.dumps(unique_elements, ensure_ascii=False),
                "backward_mapping": norm(row.get(back_col, "")) if back_col else "",
                "meaning_preserved": meaning_preserved,
                "meaning_preserved_raw": norm(raw_result),
                "eligible_element_analysis": bool(anns) and meaning_preserved is not None,
                "notes": norm(row.get(notes_col, "")) if notes_col else "",
                "workbook_file": path.name,
                "worksheet_name": sheet_name,
                "workbook_row_number": int(i + 2),
            })
    return rows


def write_summary(df: pd.DataFrame, output_csv: Path) -> None:
    out_dir = output_csv.parent
    summary_rows = []
    if not df.empty:
        group_cols = ["information_model", "llm"]
        for (info, llm), g in df.groupby(group_cols, dropna=False):
            summary_rows.append({
                "information_model": info,
                "llm": llm,
                "n_rows": len(g),
                "n_eligible_rows": int(g["eligible_element_analysis"].sum()),
                "n_meaning_preserved": int((g["meaning_preserved"] == 1).sum()),
                "n_not_preserved": int((g["meaning_preserved"] == 0).sum()),
                "mean_annotation_count": float(pd.to_numeric(g["annotation_count"], errors="coerce").mean()),
                "mean_unique_element_count": float(pd.to_numeric(g["unique_element_count"], errors="coerce").mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "expert_roundtrip_corpus_summary.csv", index=False)
    (out_dir / "expert_roundtrip_corpus_summary.json").write_text(
        json.dumps({
            "n_rows": int(len(df)),
            "n_eligible_rows": int(df["eligible_element_analysis"].sum()) if not df.empty else 0,
            "by_information_model_llm": summary_rows,
        }, ensure_ascii=False, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook_dir", required=True, help="Directory containing original researcher .xlsx workbooks.")
    parser.add_argument("--output_csv", required=True, help="Path for normalized expert round-trip CSV.")
    parser.add_argument("--file_glob", default="*.xlsx")
    parser.add_argument("--skip_sheets_regex", default="abstract|summary|readme|notes")
    args = parser.parse_args()

    workbook_dir = Path(args.workbook_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for path in sorted(workbook_dir.glob(args.file_glob)):
        if path.name.startswith("~$"):
            continue
        all_rows.extend(read_workbook(path, args.skip_sheets_regex))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit(f"No usable rows found in {workbook_dir} with glob={args.file_glob}")

    df.to_csv(output_csv, index=False)
    write_summary(df, output_csv)
    print(f"Wrote {len(df)} rows to {output_csv}")
    print(f"Eligible element-analysis rows: {int(df['eligible_element_analysis'].sum())}")


if __name__ == "__main__":
    main()
