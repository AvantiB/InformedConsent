#!/usr/bin/env python
"""Induce direct, source-model-grounded reduced consent schemas with one LLM.

The Direct LLM arm is deliberately conservative: it starts from governed/expert
source-model dictionaries plus optional requirements guidance and training-fold
example sentences. It should stay close to DUO, ICO, ODRL, and FHIR Consent.
The more exploratory arm is the later data-driven + LLM induction pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def norm(x: Any) -> str:
    return "" if x is None else " ".join(str(x).split())


def load_model_config(path: Path, model_key: str) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    model_cfg = {**(cfg.get("defaults", {}) or {}), **((cfg.get("models", {}) or {}).get(model_key, {}))}
    if not model_cfg:
        raise KeyError(f"model_key={model_key!r} not found in {path}")
    model_cfg["model_key"] = model_key
    return model_cfg


def make_client(cfg: dict[str, Any]) -> Any:
    if str(cfg.get("provider", "")).lower() == "mayo_apigee_azure_openai":
        return None
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install with: pip install openai")
    api_key_env = cfg.get("api_key_env")
    api_key = os.getenv(str(api_key_env), "") if api_key_env else "EMPTY"
    if not api_key:
        api_key = "EMPTY"
    base_url = cfg.get("base_url")
    return OpenAI(api_key=api_key) if base_url in {"", "null", None} else OpenAI(api_key=api_key, base_url=base_url)


def call_chat(client: Any, cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    if str(cfg.get("provider", "")).lower() == "mayo_apigee_azure_openai":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from apigee_azure_client import call_apigee_chat  # type: ignore
        return call_apigee_chat(client, cfg, messages)
    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": int(cfg.get("max_tokens", 12000)),
        "timeout": float(cfg.get("timeout_seconds", 300)),
    }
    if cfg.get("temperature") is not None and not cfg.get("omit_temperature"):
        kwargs["temperature"] = cfg.get("temperature", 0)
    if cfg.get("top_p") is not None:
        kwargs["top_p"] = cfg.get("top_p")
    last = None
    for attempt in range(1, int(cfg.get("max_retries", 3)) + 1):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content or ""
        except Exception as exc:
            last = exc
            if attempt < int(cfg.get("max_retries", 3)):
                time.sleep(float(cfg.get("retry_sleep_seconds", 5)) * attempt)
    raise RuntimeError(f"LLM request failed: {last}")


def extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|yaml)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(t[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(t[start:i + 1])
                    if not isinstance(obj, dict):
                        raise ValueError("Parsed JSON is not an object")
                    return obj
    raise ValueError("Could not parse balanced JSON object")


def granularity_targets(granularity: str) -> tuple[int, int]:
    if granularity == "low":
        return 12, 18
    if granularity == "high":
        return 24, 36
    raise ValueError("granularity must be high or low")


def build_messages(payload: dict[str, Any], granularity: str) -> list[dict[str, str]]:
    target_min, target_max = granularity_targets(granularity)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    system = "You design conservative, machine-interpretable informed-consent schemas. Return valid JSON only."
    user = f"""
Task: create a {granularity}-granularity reduced informed-consent meta-model as a flat dictionary.

Use these inputs:
1. DUO, ICO, ODRL, and FHIR Consent dictionary rows.
2. Requirements/guidance text, if present.
3. Representative training-fold consent sentences.

Design priorities:
- Stay close to the source information models; they are governed/expert resources.
- Consolidate near-equivalent source elements into reduced fields.
- Keep distinct functional roles separate when merging would lose meaning.
- Use requirements/guidance only to identify coverage needs and boundary cases.
- Do not copy guideline headings as field names.
- Do not treat sentence-level decision fields as annotation labels.
- Keep sentence_decision separate from span-level fields.
- The schema must support faithful machine-interpretable representation of permissions, prohibitions, restrictions, obligations, temporal logic, conditions, identifiability, sharing, withdrawal, and consequences/protections.
- Prefer a flat span-level dictionary. Add modifiers only when they prevent unnecessary field proliferation or preserve required meaning.
- Target {target_min}-{target_max} span-level fields.
- Do not invent new fields unless a source-model-grounded field cannot represent a requirement; mark any such field as requirement_driven_extension and justify it.

Return JSON exactly in this structure:
{{
  "schema_id": "direct_llm_reduced_{granularity}_fold_specific",
  "schema_status": "direct_llm_induced_source_model_grounded_training_fold",
  "fold_id": "...",
  "granularity": "{granularity}",
  "sentence_decision": {{
    "description": "Universal sentence/provision-level consent force, not an annotation label.",
    "allowed_values": ["permit", "deny", "mixed", "unclear"]
  }},
  "fields": [
    {{
      "field_id": "DRS001",
      "field_name": "snake_case_name",
      "status": "core|extension",
      "definition": "machine-interpretable definition",
      "include": ["what belongs here"],
      "exclude": ["boundary exclusions"],
      "source_model_support": [
        {{
          "source_model": "DUO|ICO|ODRL|FHIR_Consent",
          "source_element_id": "...",
          "source_element_label": "...",
          "mapping_type": "exact|near_equivalent|broader_than_target|narrower_than_target|context_dependent"
        }}
      ],
      "requirement_basis": ["brief requirement/guidance basis if any"],
      "allowed_modifiers": ["modifier_name if applicable"],
      "machine_representation_notes": "how to encode this field in annotation"
    }}
  ],
  "modifiers": [
    {{
      "modifier_id": "MOD001",
      "modifier_name": "snake_case_modifier",
      "definition": "...",
      "allowed_values": ["..."],
      "applies_to_fields": ["field_name or all"],
      "source_or_requirement_basis": ["..."],
      "required_for_faithful_reconstruction": true
    }}
  ],
  "crosswalk": [
    {{
      "source_model": "DUO|ICO|ODRL|FHIR_Consent",
      "source_element_id": "...",
      "source_element_label": "...",
      "target_field_id": "DRS001 or null",
      "target_field_name": "... or null",
      "mapping_type": "exact|near_equivalent|broader_than_target|narrower_than_target|context_dependent|modifier_attribute|no_direct_mapping",
      "needs_expert_review": true,
      "mapping_rationale": "..."
    }}
  ],
  "requirements_coverage": [
    {{
      "requirement": "permission/prohibition/restriction/obligation/temporal/condition/identifiability/sharing/withdrawal/consequence/etc.",
      "covered_by": ["field_or_modifier_name"],
      "coverage_note": "..."
    }}
  ],
  "unsafe_merge_notes": ["..."],
  "missing_or_uncertain_functions": ["..."],
  "expert_review_flags": ["..."]
}}

Input packet:
{payload_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def induce_one(input_json: Path, output_dir: Path, granularity: str, client: Any, cfg: dict[str, Any]) -> None:
    payload = json.loads(input_json.read_text())
    fold_id = payload.get("fold_id", input_json.parent.name)
    out_dir = output_dir / str(fold_id) / granularity
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = call_chat(client, cfg, build_messages(payload, granularity))
    (out_dir / "raw_response.txt").write_text(raw)
    parsed = extract_json(raw)
    parsed.setdefault("fold_id", fold_id)
    parsed.setdefault("granularity", granularity)
    (out_dir / "schema.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    metadata = {
        "input_json": str(input_json),
        "output_dir": str(out_dir),
        "model_key": cfg.get("model_key"),
        "model": cfg.get("model"),
        "provider": cfg.get("provider"),
        "granularity": granularity,
        "fold_id": fold_id,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {out_dir / 'schema.json'}", flush=True)


def discover_inputs(input_dir: Path, folds: str) -> list[Path]:
    all_inputs = sorted(input_dir.glob("fold_*/direct_induction_input.json"))
    if folds == "all":
        return all_inputs
    wanted = {f.strip() for f in folds.split(",") if f.strip()}
    out = [p for p in all_inputs if p.parent.name in wanted]
    if not out:
        raise FileNotFoundError(f"No fold inputs matched {folds!r} under {input_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--granularity", choices=["high", "low", "both"], default="both")
    ap.add_argument("--folds", default="all", help="all or comma-separated fold IDs like fold_00,fold_01")
    args = ap.parse_args()

    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)
    granularities = ["high", "low"] if args.granularity == "both" else [args.granularity]
    inputs = discover_inputs(Path(args.input_dir), args.folds)
    for inp in inputs:
        for granularity in granularities:
            induce_one(inp, Path(args.output_dir), granularity, client, cfg)


if __name__ == "__main__":
    main()
