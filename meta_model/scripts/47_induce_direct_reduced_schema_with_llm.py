#!/usr/bin/env python
"""Induce direct, source-model-grounded reduced consent schemas with one LLM.

This direct arm is conservative and round-trip oriented. The generated schema is a
flat annotation dictionary plus optional controlled modifiers. It is intentionally
compact so GPT-5.5 does not spend the full completion budget on hidden reasoning
or verbose JSON.
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

MAX_GUIDANCE_CHARS = 5000
MAX_SENTENCES = 25
MAX_SENTENCE_CHARS = 260
MAX_DICT_DEF_CHARS = 220
MAX_DICT_ROWS_PER_MODEL = 120


def trunc(text: Any, n: int) -> str:
    s = "" if text is None else " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


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
                    obj = json.loads(t[start : i + 1])
                    if not isinstance(obj, dict):
                        raise ValueError("Parsed JSON is not an object")
                    return obj
    raise ValueError("Could not parse balanced JSON object")


def granularity_targets(granularity: str) -> tuple[int, int]:
    if granularity == "low":
        return 10, 15
    if granularity == "high":
        return 22, 30
    raise ValueError("granularity must be high or low")


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dictionaries = payload.get("source_model_dictionaries") or {}
    compact_dicts: dict[str, list[dict[str, str]]] = {}
    for model, rows in dictionaries.items():
        compact_rows = []
        for r in (rows or [])[:MAX_DICT_ROWS_PER_MODEL]:
            compact_rows.append({
                "source_model": trunc(r.get("source_model", model), 40),
                "source_element_id": trunc(r.get("source_element_id", ""), 80),
                "source_element_label": trunc(r.get("source_element_label", ""), 120),
                "definition": trunc(r.get("definition", ""), MAX_DICT_DEF_CHARS),
            })
        compact_dicts[str(model)] = compact_rows

    guidance_parts = []
    for g in payload.get("requirements_guidance") or []:
        if isinstance(g, dict):
            guidance_parts.append(trunc(g.get("text", ""), MAX_GUIDANCE_CHARS))
        else:
            guidance_parts.append(trunc(g, MAX_GUIDANCE_CHARS))
    guidance_text = trunc("\n\n".join([p for p in guidance_parts if p]), MAX_GUIDANCE_CHARS)

    sentences = []
    for s in (payload.get("representative_training_sentences") or [])[:MAX_SENTENCES]:
        if isinstance(s, dict):
            sentences.append(trunc(s.get("sentence_text", ""), MAX_SENTENCE_CHARS))
        else:
            sentences.append(trunc(s, MAX_SENTENCE_CHARS))

    return {
        "fold_id": payload.get("fold_id"),
        "policy": "Conservative source-model-grounded flat dictionary; use NIH summary only as coverage pointers.",
        "source_model_dictionaries": compact_dicts,
        "requirements_guidance_summary": guidance_text,
        "representative_training_sentences": sentences,
    }


def build_messages(payload: dict[str, Any], granularity: str) -> list[dict[str, str]]:
    target_min, target_max = granularity_targets(granularity)
    payload_text = json.dumps(compact_payload(payload), ensure_ascii=False, separators=(",", ":"))
    system = "Create compact JSON annotation schemas. Return JSON only. No prose."
    user = f"""
Create a {granularity}-granularity reduced informed-consent annotation dictionary.

Goal: conservative merge of DUO/ICO/ODRL/FHIR span-level elements for round-trip annotation.
Target {target_min}-{target_max} fields.
Keep sentence_decision separate with values permit, deny, mixed, unclear.
Flat dictionary only. Modifiers are optional controlled attributes attached to annotations, not labels.
Use NIH guidance only as coverage checks, not field names.

Output compact JSON exactly with these keys:
{{
 "schema_id":"direct_llm_reduced_{granularity}_fold_specific",
 "fold_id":"...",
 "granularity":"{granularity}",
 "sentence_decision":{{"allowed_values":["permit","deny","mixed","unclear"]}},
 "dictionary_rows":[{{
   "field_id":"DRS001",
   "field_name":"snake_case",
   "definition":"<=25 words; span-selection definition",
   "include":["<=3 short examples"],
   "exclude":["<=3 short boundaries"],
   "allowed_modifiers":["modifier_name"],
   "source_support":["MODEL:id label"]
 }}],
 "modifiers":[{{
   "modifier_name":"snake_case",
   "definition":"<=20 words",
   "allowed_values":["value1","value2"],
   "applies_to_fields":["field_name or all"]
 }}],
 "requirements_coverage":[{{"requirement":"short name","covered_by":["field_or_modifier"]}}],
 "expert_review_flags":["brief flags"]
}}

Constraints:
- No duplicate fields array; dictionary_rows is the field dictionary.
- No full source crosswalk.
- Max 8 source_support strings per field.
- Max 8 modifiers total.
- Definitions must be annotation-ready, not conceptual essays.
- Field IDs must be unique and stable.
- Do not include rationale paragraphs.

Input:{payload_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def write_prompt_files(out_dir: Path, messages: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "prompt_messages.json"
    prompt_txt_path = out_dir / "browser_prompt.txt"
    prompt_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
    prompt_txt_path.write_text("\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in messages]))
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {prompt_path}", flush=True)
    print(f"Wrote {prompt_txt_path}", flush=True)


def induce_one(input_json: Path, output_dir: Path, granularity: str, client: Any, cfg: dict[str, Any], write_prompts_only: bool) -> None:
    payload = json.loads(input_json.read_text())
    fold_id = payload.get("fold_id", input_json.parent.name)
    out_dir = output_dir / str(fold_id) / granularity
    messages = build_messages(payload, granularity)
    metadata = {
        "input_json": str(input_json),
        "output_dir": str(out_dir),
        "model_key": cfg.get("model_key"),
        "model": cfg.get("model"),
        "provider": cfg.get("provider"),
        "granularity": granularity,
        "fold_id": fold_id,
        "schema_design": "compact_roundtrip_ready_flat_dictionary_with_optional_controlled_modifiers",
        "write_prompts_only": bool(write_prompts_only),
    }
    write_prompt_files(out_dir, messages, metadata)
    if write_prompts_only:
        return
    raw = call_chat(client, cfg, messages)
    (out_dir / "raw_response.txt").write_text(raw)
    parsed = extract_json(raw)
    parsed.setdefault("fold_id", fold_id)
    parsed.setdefault("granularity", granularity)
    (out_dir / "schema.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
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
    ap.add_argument("--write_prompts_only", action="store_true", help="Write prompt files and exit without calling the LLM API.")
    args = ap.parse_args()

    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)
    granularities = ["high", "low"] if args.granularity == "both" else [args.granularity]
    inputs = discover_inputs(Path(args.input_dir), args.folds)
    for inp in inputs:
        for granularity in granularities:
            induce_one(inp, Path(args.output_dir), granularity, client, cfg, args.write_prompts_only)


if __name__ == "__main__":
    main()
