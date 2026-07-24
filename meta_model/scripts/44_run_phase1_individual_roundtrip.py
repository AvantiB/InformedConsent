#!/usr/bin/env python
"""Phase 1 individual source-model runner with sentence decisions separated from labels.

This runner reuses the dictionary-based individual runner and applies the final
Phase 1 policy:

- source-model dictionaries exclude sentence-level inventory rows;
- FHIR Consent.provision.type and ODRL Rule_TestSentence are not labels;
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
        system = (
            "You are an NLP annotator for informed-consent documents. Apply only the supplied "
            "source-model dictionary. Return valid JSON only."
        )
        user = f"""
Task: annotate the informed-consent sentence using ONLY rows from the authoritative {info_model} dictionary below.

Design rule:
- The forward prompt is identical across individual source-model experiments; only the dictionary changes.
- Do not use concepts from any other information model.

Hard dictionary rules:
- Every span annotation MUST copy source_element_id exactly from one dictionary row.
- Every span annotation MUST copy source_element_label exactly from the same dictionary row.
- Do not invent IDs, labels, fields, or namespaces.
- Sentence-level decision fields are NOT annotation labels and are NOT included in the dictionary.
- Do not use Consent.provision.type, Rule_TestSentence, Permission, Prohibition, permit, deny, yes, no, NA, none, null, unknown, invalid, or unmatched_language as annotation labels.
- unmatched_language is only the name of the top-level audit list. It is never a dictionary label and never a valid annotation.
- If no dictionary row fits, put the phrase only in top-level unmatched_language and do not create an annotation object for it.
- A phrase may be annotated with a general source-model class even when the phrase is a named instance and the exact phrase is not in the dictionary.
- Do not annotate standalone “yes” or “no” as Permission or Prohibition unless it directly governs a specific action. Phrases like “say yes or no” represent choice/decision, not permit plus prohibit.
- Phrases like “no penalty” and “no expiration date” are not sentence-level denial/prohibition; they are consequence/protection or temporal-scope expressions.

Sentence-level decision rule:
- sentence_decision is the ONLY sentence-level consent-force field.
- sentence_decision must be one of: permit, deny, mixed, unclear.
- Do not output sentence_level_elements.

Data dictionary:
{dictionary_text}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "source_element_id": "exact dictionary source_element_id",
      "source_element_label": "exact dictionary source_element_label",
      "overlap_group_id": "g1 or null",
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span",
      "decision_or_polarity": "controlled local value if explicitly supported, else empty",
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
    mod.extract_sentence_level_annotations = extract_sentence_level_annotations
    mod.build_forward_messages = build_forward_messages
    return mod


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mod = apply_phase1_sentence_decision_policy(load_base_runner(repo_root))
    mod.main()


if __name__ == "__main__":
    main()
