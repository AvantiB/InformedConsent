#!/usr/bin/env python
"""JSON-output wrapper for individual source-model round-trip experiments.

This wrapper preserves the original individual source-model prompt content
(task description, data dictionary, labels, and annotation rules) but overrides
only the forward output serialization to standardized JSON.

It delegates loading, resumability, sanitization, and backward reconstruction to
05_run_individual_model_roundtrip.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

INFO_MODEL_OUTPUTS = {
    "DUO": {"decision_field": "DUO.decision", "decision_values": "permit|deny"},
    "ICO": {"decision_field": "ICO.decision", "decision_values": "permit|deny"},
    "ODRL": {"decision_field": "Rule_TestSentence", "decision_values": "Permission|Prohibition"},
    "FHIR_Consent": {"decision_field": "Consent.provision.type", "decision_values": "permit|deny"},
}


def load_base_module():
    script_path = Path(__file__).with_name("05_run_individual_model_roundtrip.py")
    spec = importlib.util.spec_from_file_location("individual_roundtrip_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_info_model(prompt_text: str) -> str:
    text = prompt_text.lower()
    if "duo.decision" in text or "data use ontology" in text:
        return "DUO"
    if "ico.decision" in text or "informed consent ontology" in text:
        return "ICO"
    if "rule_testsentence" in text or "odrl" in text:
        return "ODRL"
    if "consent.provision.type" in text or "fhir consent" in text:
        return "FHIR_Consent"
    raise ValueError("Could not infer source information model from prompt text")


def json_output_override(info_model: str) -> str:
    cfg = INFO_MODEL_OUTPUTS[info_model]
    decision_field = cfg["decision_field"]
    values = cfg["decision_values"]
    return f"""
Output-format override:
- Use the original source-model prompt below for the task description, label set, data dictionary, and annotation rules.
- Ignore only the original CSV/table output-format instructions.
- Return valid JSON only; do not include markdown fences, explanations, or extra text.
- Do not include the complete original sentence in the output.
- Use exact text spans copied from the input sentence.
- Assign one best label per annotation object, using only labels allowed by the original source-model prompt.
- The sentence-level decision field is \"{decision_field}\" and its value must be one of: {values}.

Return exactly this JSON structure:
{{
  "decision_field": "{decision_field}",
  "sentence_decision": "{values}",
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact contiguous text span",
      "label": "one allowed source-model label",
      "rationale": "brief rationale or empty string"
    }}
  ],
  "unmatched_language": [
    {{
      "span_text": "exact text span",
      "reason": "brief reason"
    }}
  ]
}}
""".strip()


def make_json_forward_messages(prompt_text: str, sentence: str):
    info_model = infer_info_model(prompt_text)
    system = "You are an NLP annotator for informed-consent documents. Return valid JSON only."
    user = f"""
Use the following original source-model prompt to annotate the sentence.

Original source-model prompt:
{prompt_text}

{json_output_override(info_model)}

Sentence to annotate:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    base = load_base_module()
    base.build_forward_messages = make_json_forward_messages
    base.main()


if __name__ == "__main__":
    main()
