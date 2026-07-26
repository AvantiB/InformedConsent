#!/usr/bin/env python
"""Induce data-driven reduced consent schemas with one LLM.

Input packets are produced by 50_prepare_data_driven_schema_induction_inputs.py.
The model receives only training-fold evidence summaries from corrected baseline
round trips. It returns the same schema JSON shape used by the direct LLM arm so
48_run_direct_schema_roundtrip.py can evaluate it without a separate runner.
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

MAX_PROFILES = 90
MAX_EDGES = 100
MAX_EXAMPLES = 16
MAX_SENT_CHARS = 260
MAX_DEF_CHARS = 180


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
    t = (text or "").strip()
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
        return 22, 32
    raise ValueError("granularity must be high or low")


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    profiles = []
    for p in (payload.get("source_element_profiles") or [])[:MAX_PROFILES]:
        profiles.append({
            "source_element_key": trunc(p.get("source_element_key"), 120),
            "source_model": trunc(p.get("source_model"), 40),
            "source_element_label": trunc(p.get("source_element_label"), 120),
            "definition": trunc(p.get("source_element_definition"), MAX_DEF_CHARS),
            "raw_mention_count": p.get("raw_mention_count"),
            "supported_mention_count": p.get("supported_mention_count"),
            "mean_preservation_score": p.get("mean_preservation_score"),
            "example_spans": [trunc(x, 70) for x in (p.get("example_spans") or [])[:4]],
            "source_model_support": p.get("source_model_support") or [],
            "llm_support": p.get("llm_support") or [],
        })
    edges = []
    for e in (payload.get("cooccurrence_edges") or [])[:MAX_EDGES]:
        edges.append({
            "a": trunc(e.get("source_element_key_a"), 120),
            "b": trunc(e.get("source_element_key_b"), 120),
            "cooccurrence_count": e.get("cooccurrence_count"),
            "support_sentence_count": e.get("support_sentence_count"),
        })
    examples_high = []
    for e in (payload.get("high_preservation_examples") or [])[:MAX_EXAMPLES]:
        examples_high.append({
            "source_text": trunc(e.get("source_text"), MAX_SENT_CHARS),
            "reconstructed_text": trunc(e.get("reconstructed_text"), MAX_SENT_CHARS),
            "score": e.get("score"),
            "condition": e.get("condition"),
        })
    examples_low = []
    for e in (payload.get("low_preservation_examples") or [])[:MAX_EXAMPLES]:
        examples_low.append({
            "source_text": trunc(e.get("source_text"), MAX_SENT_CHARS),
            "reconstructed_text": trunc(e.get("reconstructed_text"), MAX_SENT_CHARS),
            "score": e.get("score"),
            "condition": e.get("condition"),
        })
    return {
        "fold_id": payload.get("fold_id"),
        "evidence_policy": payload.get("evidence_policy"),
        "training_evidence_counts": payload.get("training_evidence_counts"),
        "source_element_profiles": profiles,
        "cooccurrence_edges": edges,
        "high_preservation_examples": examples_high,
        "low_preservation_examples": examples_low,
    }


def build_messages(payload: dict[str, Any], granularity: str) -> list[dict[str, str]]:
    target_min, target_max = granularity_targets(granularity)
    payload_text = json.dumps(compact_payload(payload), ensure_ascii=False, separators=(",", ":"))
    system = "Create compact JSON annotation schemas from evidence summaries. Return JSON only. No prose."
    user = f"""
Create a {granularity}-granularity data-driven informed-consent meta-model.

Input: training-fold evidence from corrected baseline round trips. Evidence includes source-element profiles, preservation-supported mention counts, example spans, co-occurrence edges, and examples where structured evidence did or did not preserve meaning.
Goal: synthesize one model-agnostic schema that groups near-equivalent source-model meanings, keeps distinct meanings separate when needed, and covers the breadth of informed-consent language supported by the training evidence.
Target {target_min}-{target_max} span-level fields.
Keep sentence_decision separate with values permit, deny, mixed, unclear.
Use a flat annotation dictionary. You may create modifiers if they help preserve meaning, but choose modifier names and allowed values yourself.
Do not copy a source model wholesale. Do not create fields for purely linguistic artifacts unless they are needed as annotation fields; prefer modifiers for polarity, modality, scope, timing, identifiability, role, or condition attributes.

Output compact JSON exactly with these keys:
{{
 "schema_id":"data_driven_llm_reduced_{granularity}_fold_specific",
 "fold_id":"...",
 "granularity":"{granularity}",
 "sentence_decision":{{"allowed_values":["permit","deny","mixed","unclear"]}},
 "dictionary_rows":[{{
   "field_id":"DDS001",
   "field_name":"snake_case",
   "definition":"<=25 words; span-selection definition",
   "include":["<=3 short examples"],
   "exclude":["<=3 short boundaries"],
   "allowed_modifiers":["modifier_name"],
   "source_support":["source element keys or labels"]
 }}],
 "modifiers":[{{
   "modifier_name":"snake_case",
   "definition":"<=20 words",
   "allowed_values":["value1","value2"],
   "applies_to_fields":["field_name or all"]
 }}],
 "evidence_coverage":[{{"evidence_group":"short name","covered_by":["field_or_modifier"]}}],
 "expert_review_flags":["brief flags"]
}}

Constraints:
- dictionary_rows is the only field dictionary.
- No full source crosswalk.
- Max 8 source_support strings per field.
- Max 10 modifiers total.
- Definitions must be annotation-ready, not conceptual essays.
- Field IDs must be unique and stable.
- Do not include rationale paragraphs.

Input:{payload_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def write_prompt_files(out_dir: Path, messages: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt_messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2))
    (out_dir / "browser_prompt.txt").write_text("\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in messages]))
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {out_dir / 'prompt_messages.json'}", flush=True)
    print(f"Wrote {out_dir / 'browser_prompt.txt'}", flush=True)


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
        "schema_design": "data_driven_training_fold_evidence_flat_dictionary_with_optional_modifiers",
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
    all_inputs = sorted(input_dir.glob("fold_*/data_driven_induction_input.json"))
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
