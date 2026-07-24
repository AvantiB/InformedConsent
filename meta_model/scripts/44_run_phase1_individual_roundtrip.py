#!/usr/bin/env python
"""Phase 1 individual source-model runner with sentence decisions separated from labels.

This runner reuses the dictionary-based individual runner and applies the final
Phase 1 policy:

- source-model dictionaries exclude sentence-level inventory rows;
- sentence_decision is the only global consent-force field;
- sentence_level_elements emitted by a model are dropped before backward;
- zero-valid-span rows remain coverage failures and are excluded from schema induction.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_base_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "05_run_individual_model_roundtrip.py"
    spec = importlib.util.spec_from_file_location("individual_base_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["individual_base_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def is_sentence_level_scope(value: Any) -> bool:
    return "sentence" in " ".join(str(value).lower().replace("-", "_").split())


def apply_phase1_sentence_decision_policy(mod):
    base_load_inventory = mod.load_inventory

    def load_inventory_span_only(inventory_csv: Path) -> pd.DataFrame:
        inv = base_load_inventory(inventory_csv)
        if "element_scope" not in inv.columns:
            return inv
        return inv[~inv["element_scope"].map(is_sentence_level_scope)].copy().reset_index(drop=True)

    def extract_sentence_level_annotations(raw_forward: str, has_valid_span_annotations: bool, info_model: str, label_lookup: dict[str, Any]):
        if not has_valid_span_annotations:
            return [], {
                "n_sentence_level_annotations_backward_eligible": 0,
                "n_sentence_level_elements_dropped_by_policy": 0,
            }
        out = []
        dropped = 0
        try:
            parsed = mod.extract_json(raw_forward)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            canonical = mod.normalize_sentence_decision_value(parsed.get("sentence_decision") or parsed.get("decision"))
            if canonical:
                out.append({
                    "field": "sentence_decision",
                    "value": canonical,
                    "support": "valid_span_annotations_present",
                })
            elems = parsed.get("sentence_level_elements") or []
            dropped = len(elems) if isinstance(elems, list) else 0
        return out, {
            "n_sentence_level_annotations_backward_eligible": len(out),
            "n_sentence_level_elements_dropped_by_policy": dropped,
        }

    def build_forward_messages(dictionary_text: str, info_model: str, sentence: str):
        system = "You are an informed-consent annotation system. Return valid JSON only."
        user = f"""
Annotate the sentence using only the {info_model} dictionary rows below.

Rules:
- Return sentence_decision as one of: permit, deny, mixed, unclear.
- Create span annotations only for text supported by a dictionary row.
- Each annotation must copy source_element_id and source_element_label exactly from the same dictionary row.
- Use the smallest meaningful contiguous span when possible.
- Multiple labels may be assigned to the same or overlapping spans when supported.
- Do not create an annotation when no dictionary row fits.
- Use interpretation_units only to link annotation_ids that should be read together.

Dictionary:
{dictionary_text}

Return JSON exactly in this shape:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "source_element_id": "exact dictionary source_element_id",
      "source_element_label": "exact dictionary source_element_label",
      "overlap_group_id": "g1 or null",
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "annotation_ids": ["a1", "a2"],
      "relationship": "same_span_multiple_labels|nested_broad_narrow|complementary_roles|conflicting_or_uncertain"
    }}
  ]
}}

Sentence:
{sentence}
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    mod.load_inventory = load_inventory_span_only
    mod.extract_sentence_level_annotations = extract_sentence_level_annotations
    mod.build_forward_messages = build_forward_messages
    return mod


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mod = apply_phase1_sentence_decision_policy(load_base_runner(repo_root))
    mod.main()


if __name__ == "__main__":
    main()
