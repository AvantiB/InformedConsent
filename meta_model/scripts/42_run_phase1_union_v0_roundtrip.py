#!/usr/bin/env python
"""Phase 1 Union V0 runner with sentence decisions separated from labels.

This is the paper-facing Phase 1 Union V0 entry point. It reuses the stable
Union V0 runner but applies the final Phase 1 policy:

- sentence-level inventory rows are removed from the annotation dictionary;
- sentence_decision is the only global consent-force field;
- sentence_level_elements emitted by a model are dropped before backward;
- backward receives valid span annotations, static metadata, relationship links,
  and the universal controlled sentence_decision when span evidence exists.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_base_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "03_run_union_v0_roundtrip.py"
    spec = importlib.util.spec_from_file_location("union_v0_base_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["union_v0_base_runner"] = mod
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

    def eligible_sentence_level_annotations(parsed_forward: dict[str, Any], has_valid_span_annotations: bool, maps: dict[str, Any]):
        if not has_valid_span_annotations:
            return [], {
                "n_sentence_level_annotations_backward_eligible": 0,
                "n_sentence_level_elements_dropped_by_policy": 0,
            }
        out = []
        canonical = mod.normalize_sentence_decision_value(parsed_forward.get("sentence_decision"))
        if canonical:
            out.append({
                "field": "sentence_decision",
                "value": canonical,
                "support": "valid_span_annotations_present",
            })
        elems = parsed_forward.get("sentence_level_elements") or []
        dropped = len(elems) if isinstance(elems, list) else 0
        return out, {
            "n_sentence_level_annotations_backward_eligible": len(out),
            "n_sentence_level_elements_dropped_by_policy": dropped,
        }

    def build_forward_messages(sentence: str, dictionary_text: str):
        system = "You are an informed-consent annotation system. Return valid JSON only."
        user = f"""
Annotate the sentence using only the Union V0 dictionary rows below.

Rules:
- Return sentence_decision as one of: permit, deny, mixed, unclear.
- Create span annotations only for text supported by a dictionary row.
- Each annotation must copy union_element_id and source_element_label exactly from the same dictionary row.
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
      "union_element_id": "exact dictionary union_element_id",
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
    mod.eligible_sentence_level_annotations = eligible_sentence_level_annotations
    mod.build_forward_messages = build_forward_messages
    return mod


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mod = apply_phase1_sentence_decision_policy(load_base_runner(repo_root))
    mod.main()


if __name__ == "__main__":
    main()
