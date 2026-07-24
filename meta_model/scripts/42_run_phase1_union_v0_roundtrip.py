#!/usr/bin/env python
"""Phase 1 Union V0 runner with sentence decisions separated from labels.

This is the paper-facing Phase 1 Union V0 entry point. It reuses the stable
Union V0 runner but applies the final Phase 1 policy:

- sentence-level inventory rows are removed from the annotation dictionary;
- FHIR Consent.provision.type and ODRL Rule_TestSentence are not labels;
- sentence_decision is the only global consent-force field;
- sentence_level_elements emitted by a model are dropped before backward;
- backward still receives valid span annotations, static metadata, relationship
  links, and the universal controlled sentence_decision when span evidence exists.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
        system = (
            "You are an NLP annotator for informed-consent documents. Apply only the supplied "
            "authoritative dictionary. Return valid JSON only."
        )
        user = f"""
Task: annotate the informed-consent sentence using ONLY rows from the authoritative Union V0 dictionary below.

Important context:
- This dictionary is a naive union of multiple information models, not a reduced meta-model.
- Several elements may overlap, duplicate, specialize, or complement each other.
- The same or similar text span MAY receive more than one label.
- A larger phrase may receive a broader role, while a nested shorter phrase may receive a narrower or more specific role.
- Preserve overlaps/nesting relationships rather than forcing a single label too early.
- A phrase may be annotated with a general dictionary class even when the phrase is a named instance and the exact phrase is not in the dictionary.

Hard dictionary rules:
- Every annotation MUST copy union_element_id exactly from one dictionary row.
- Every annotation MUST copy source_element_label exactly from the same dictionary row.
- Do not invent IDs, labels, fields, or namespaces.
- Sentence-level decision fields are NOT annotation labels and are NOT included in the dictionary.
- Do not use Consent.provision.type, Rule_TestSentence, Permission, Prohibition, permit, deny, yes, no, NA, none, null, unknown, invalid, or unmatched_language as annotation labels.
- unmatched_language is only the name of the top-level audit list. It is never a dictionary label and never a valid union_element_id.
- If no dictionary row fits, put the phrase only in the top-level unmatched_language list and do not create an annotation object for that phrase.
- If you are uncertain whether an ID/label pair is valid, do not annotate that phrase.
- Do not annotate standalone “yes” or “no” as Permission or Prohibition. Phrases like “say yes or no” represent choice/decision, not permit plus prohibit.
- Phrases like “no penalty” and “no expiration date” are not sentence-level denial/prohibition; they are consequence/protection or temporal-scope expressions.

Sentence-level decision rule:
- sentence_decision is the ONLY sentence-level consent-force field.
- sentence_decision must be one of: permit, deny, mixed, unclear.
- Do not output sentence_level_elements.

Audit rules:
- You may include interpretation_units and unmatched_language for human audit.
- Audit fields will not be included directly in backward reconstruction.

Data dictionary:
{dictionary_text}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "union_element_id": "exact dictionary union_element_id",
      "source_element_label": "exact dictionary source_element_label",
      "overlap_group_id": "g1 or null",
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span",
      "rationale": "brief audit-only rationale"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "evidence_span_text": "span or phrase represented by this unit",
      "annotation_ids": ["a1", "a2"],
      "relationship": "single|same_span_multiple_labels|nested_broad_narrow|complementary_roles|conflicting_or_uncertain",
      "combined_meaning": "audit only",
      "backward_mapping_decision": "audit only",
      "rationale": "brief explanation of how the annotations should be considered together"
    }}
  ],
  "unmatched_language": [{{"span_text": "exact text span", "reason": "brief reason"}}]
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
