#!/usr/bin/env python
"""Prepare fold-specific inputs for direct LLM-induced reduced schemas.

This script builds conservative, source-model-grounded induction packets for the
Direct LLM reduced-schema arm. It does not use Phase 1 outputs. It uses only:

- source-model dictionary rows from source_element_inventory.csv;
- training-fold consent sentences for orientation/examples;
- optional external requirements/guidance text, such as NIH repository guidance.

Sentence-level inventory rows are removed. sentence_decision remains a universal
experiment-level field and is not treated as a source-model label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

TEXT_COL_CANDIDATES = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
FORM_KEY_CANDIDATES = ["form_key", "form_id", "document_id", "source_form", "file_name"]
SOURCE_MODELS = ["DUO", "ICO", "ODRL", "FHIR_Consent"]
SENTENCE_DECISION_VALUES = ["permit", "deny", "mixed", "unclear"]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available columns={list(df.columns)}")
    return None


def stable_fold(key: str, n_folds: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_folds


def is_sentence_level_scope(value: Any) -> bool:
    return "sentence" in " ".join(str(value).lower().replace("-", "_").split())


def source_model_aliases(model: str) -> set[str]:
    return {model, model.replace("_Consent", ""), model.replace("_", "")}


def load_inventory(inventory_csv: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv).fillna("")
    required = ["source_model", "source_element_id", "source_element_label", "source_element_definition"]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    if "element_scope" not in inv.columns:
        inv["element_scope"] = "span"
    inv = inv[~inv["element_scope"].map(is_sentence_level_scope)].copy().reset_index(drop=True)
    return inv


def compact_inventory(inv: pd.DataFrame, max_rows_per_model: int | None = None) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for model in SOURCE_MODELS:
        aliases = {x.casefold() for x in source_model_aliases(model)}
        sub = inv[inv["source_model"].astype(str).str.casefold().isin(aliases)].copy()
        sub = sub.drop_duplicates(subset=["source_model", "source_element_id", "source_element_label"])
        if max_rows_per_model:
            sub = sub.head(max_rows_per_model)
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "source_model": norm(r.get("source_model")),
                "source_element_id": norm(r.get("source_element_id")),
                "source_element_label": norm(r.get("source_element_label")),
                "definition": norm(r.get("source_element_definition")),
            })
        out[model] = rows
    return out


def read_guidance(path: Path, max_chars: int) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".text"}:
        text = path.read_text(errors="replace")
    elif suffix == ".csv":
        df = pd.read_csv(path).fillna("")
        text = df.to_csv(index=False)
    elif suffix == ".json":
        text = json.dumps(json.loads(path.read_text(errors="replace")), ensure_ascii=False, indent=2)
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read PDF guidance file {path}. Install pypdf or convert it to .txt/.md first. Error: {exc}"
            )
    else:
        text = path.read_text(errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    truncated = text[:max_chars]
    return {
        "file_name": path.name,
        "source_path": str(path),
        "text": truncated,
        "truncated": str(len(text) > len(truncated)),
        "original_char_count": str(len(text)),
        "included_char_count": str(len(truncated)),
    }


def load_sentences(roundtrips_csv: Path, n_folds: int, examples_per_fold: int, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(roundtrips_csv).fillna("")
    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    form_col = pick_col(df, FORM_KEY_CANDIDATES)
    keep_cols = [form_col, text_col]
    extra_cols = [c for c in ["canonical_source_key", "sentence_id", "sentence_text_id"] if c in df.columns]
    ex = df[keep_cols + extra_cols].copy()
    ex.columns = ["form_key", "sentence_text"] + extra_cols
    ex["form_key"] = ex["form_key"].map(norm)
    ex["sentence_text"] = ex["sentence_text"].map(norm)
    ex = ex[(ex["form_key"] != "") & (ex["sentence_text"] != "")].drop_duplicates(subset=["form_key", "sentence_text"]).copy()
    forms = sorted(ex["form_key"].unique())
    fold_rows = [{"form_key": f, "fold_id": f"fold_{stable_fold(f, n_folds):02d}"} for f in forms]
    folds = pd.DataFrame(fold_rows)
    ex = ex.merge(folds, on="form_key", how="left")
    return ex, folds


def write_fold_inputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = load_inventory(Path(args.inventory_csv))
    dictionaries = compact_inventory(inv, args.max_rows_per_model)
    examples, folds = load_sentences(Path(args.roundtrips_csv), args.n_folds, args.examples_per_fold, args.random_seed)
    folds.to_csv(out_dir / "direct_llm_fold_assignments.csv", index=False)

    guidance = [read_guidance(Path(p), args.max_guidance_chars_per_file) for p in args.guidance_files]

    for fold_num in range(args.n_folds):
        fold_id = f"fold_{fold_num:02d}"
        heldout_forms = set(folds.loc[folds["fold_id"] == fold_id, "form_key"])
        train = examples[~examples["form_key"].isin(heldout_forms)].copy()
        train = train.sample(n=min(args.examples_per_fold, len(train)), random_state=args.random_seed + fold_num) if len(train) else train
        representative_sentences = []
        for _, row in train.iterrows():
            representative_sentences.append({
                "form_key": row["form_key"],
                "sentence_text": row["sentence_text"],
            })
        payload = {
            "schema_induction_arm": "direct_llm_source_model_grounded",
            "fold_id": fold_id,
            "heldout_form_keys": sorted(heldout_forms),
            "training_form_key_count": int(len(set(folds["form_key"]) - heldout_forms)),
            "heldout_form_key_count": int(len(heldout_forms)),
            "design_policy": {
                "source_model_grounding": "stay close to DUO, ICO, ODRL, and FHIR Consent source-model dictionaries",
                "sentence_decision": {"field": "sentence_decision", "allowed_values": SENTENCE_DECISION_VALUES, "not_a_dictionary_label": True},
                "schema_shape": "flat span-level dictionary with optional modifiers when justified",
                "guidance_use": "requirements coverage only; do not copy guidance headings as field names",
                "heldout_policy": "representative_sentences come only from training forms for this fold",
            },
            "source_model_dictionaries": dictionaries,
            "requirements_guidance": guidance,
            "representative_training_sentences": representative_sentences,
        }
        fold_dir = out_dir / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "direct_induction_input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    manifest = {
        "output_dir": str(out_dir),
        "n_folds": args.n_folds,
        "roundtrips_csv": args.roundtrips_csv,
        "inventory_csv": args.inventory_csv,
        "guidance_files": args.guidance_files,
        "examples_per_fold": args.examples_per_fold,
        "max_guidance_chars_per_file": args.max_guidance_chars_per_file,
        "max_rows_per_model": args.max_rows_per_model,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--guidance_files", nargs="*", default=[])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--examples_per_fold", type=int, default=60)
    ap.add_argument("--max_guidance_chars_per_file", type=int, default=12000)
    ap.add_argument("--max_rows_per_model", type=int, default=None)
    ap.add_argument("--random_seed", type=int, default=17)
    args = ap.parse_args()
    write_fold_inputs(args)


if __name__ == "__main__":
    main()
