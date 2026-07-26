#!/usr/bin/env python
"""Induce data-driven schemas from original annotation-dataset summaries.

Inputs are produced by 50_prepare_data_driven_schema_induction_inputs.py. The
packets use original non-NA annotation evidence only, not baseline round-trip
outputs, classifier scores, or reconstructions.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
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
MAX_SENTENCES = 20
MAX_SENT_CHARS = 260
MAX_DEF_CHARS = 180


def trunc(x: Any, n: int) -> str:
    s = "" if x is None else " ".join(str(x).split())
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
    return OpenAI(api_key=api_key or "EMPTY", base_url=cfg.get("base_url")) if cfg.get("base_url") else OpenAI(api_key=api_key or "EMPTY")


def call_chat(client: Any, cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    if str(cfg.get("provider", "")).lower() == "mayo_apigee_azure_openai":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from apigee_azure_client import call_apigee_chat  # type: ignore
        return call_apigee_chat(client, cfg, messages)
    kwargs = {"model": cfg["model"], "messages": messages, "max_tokens": int(cfg.get("max_tokens", 12000)), "timeout": float(cfg.get("timeout_seconds", 300))}
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
    depth = 0; in_str = False; esc = False
    for i, ch in enumerate(t[start:], start=start):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(t[start:i + 1])
                    if isinstance(obj, dict): return obj
    raise ValueError("Could not parse balanced JSON object")


def granularity_targets(g: str) -> tuple[int, int]:
    return (10, 15) if g == "low" else (22, 32)


def compact_payload(p: dict[str, Any]) -> dict[str, Any]:
    profiles = []
    for x in (p.get("source_element_profiles") or [])[:MAX_PROFILES]:
        profiles.append({
            "source_element_key": trunc(x.get("source_element_key"), 120),
            "source_model": trunc(x.get("source_model"), 40),
            "source_element_label": trunc(x.get("source_element_label"), 120),
            "definition": trunc(x.get("source_element_definition"), MAX_DEF_CHARS),
            "raw_mention_count": x.get("raw_mention_count"),
            "sentence_count": x.get("sentence_count"),
            "form_count": x.get("form_count"),
            "example_spans": [trunc(v, 70) for v in (x.get("example_spans") or [])[:5]],
            "decision_or_tag_examples": [trunc(v, 40) for v in (x.get("decision_or_tag_examples") or [])[:4]],
            "source_model_support": x.get("source_model_support") or [],
            "llm_support": x.get("llm_support") or [],
        })
    edges = [{"a": trunc(e.get("source_element_key_a"), 120), "b": trunc(e.get("source_element_key_b"), 120), "cooccurrence_count": e.get("cooccurrence_count"), "form_count": e.get("form_count")} for e in (p.get("cooccurrence_edges") or [])[:MAX_EDGES]]
    sents = [trunc(s.get("sentence_text") if isinstance(s, dict) else s, MAX_SENT_CHARS) for s in (p.get("representative_training_sentences") or [])[:MAX_SENTENCES]]
    return {"fold_id": p.get("fold_id"), "evidence_policy": p.get("evidence_policy"), "training_evidence_counts": p.get("training_evidence_counts"), "source_element_profiles": profiles, "cooccurrence_edges": edges, "representative_training_sentences": sents}


def build_messages(payload: dict[str, Any], granularity: str) -> list[dict[str, str]]:
    lo, hi = granularity_targets(granularity)
    payload_text = json.dumps(compact_payload(payload), ensure_ascii=False, separators=(",", ":"))
    system = "Create compact JSON annotation schemas from original annotation evidence. Return JSON only."
    user = f"""
Create a {granularity}-granularity data-driven informed-consent meta-model.

Input: training-fold evidence from the original annotation dataset after excluding NA-only annotation rows. Evidence includes source-element profiles, mention counts, example spans, co-occurrence edges, and representative training sentences.
Goal: synthesize one model-agnostic schema that groups near-equivalent source-model meanings, keeps distinct meanings separate when needed, and covers the breadth of informed-consent language represented in the original annotations.
Target {lo}-{hi} span-level fields.
Keep sentence_decision separate with values permit, deny, mixed, unclear.
Use a flat annotation dictionary. You may create modifiers if they help preserve meaning, but choose modifier names and allowed values yourself.
Prefer modifiers for polarity, modality, scope, timing, identifiability, role, or condition attributes.

Output JSON exactly with these keys:
{{"schema_id":"data_driven_llm_reduced_{granularity}_fold_specific","fold_id":"...","granularity":"{granularity}","sentence_decision":{{"allowed_values":["permit","deny","mixed","unclear"]}},"dictionary_rows":[{{"field_id":"DDS001","field_name":"snake_case","definition":"<=25 words; span-selection definition","include":["<=3 short examples"],"exclude":["<=3 short boundaries"],"allowed_modifiers":["modifier_name"],"source_support":["source element keys or labels"]}}],"modifiers":[{{"modifier_name":"snake_case","definition":"<=20 words","allowed_values":["value1","value2"],"applies_to_fields":["field_name or all"]}}],"evidence_coverage":[{{"evidence_group":"short name","covered_by":["field_or_modifier"]}}],"expert_review_flags":["brief flags"]}}

Constraints: dictionary_rows is the only field dictionary; no full source crosswalk; max 8 source_support strings per field; max 10 modifiers; annotation-ready definitions; unique stable field IDs; no rationale paragraphs.

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
    metadata = {"input_json": str(input_json), "output_dir": str(out_dir), "model_key": cfg.get("model_key"), "model": cfg.get("model"), "provider": cfg.get("provider"), "granularity": granularity, "fold_id": fold_id, "schema_design": "data_driven_original_annotation_dataset_flat_dictionary", "baseline_roundtrip_outputs_used": False, "classifier_scores_used": False, "write_prompts_only": bool(write_prompts_only)}
    write_prompt_files(out_dir, messages, metadata)
    if write_prompts_only: return
    raw = call_chat(client, cfg, messages)
    (out_dir / "raw_response.txt").write_text(raw)
    parsed = extract_json(raw)
    parsed.setdefault("fold_id", fold_id)
    parsed.setdefault("granularity", granularity)
    (out_dir / "schema.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    print(f"Wrote {out_dir / 'schema.json'}", flush=True)


def discover_inputs(input_dir: Path, folds: str) -> list[Path]:
    all_inputs = sorted(input_dir.glob("fold_*/data_driven_induction_input.json"))
    if folds == "all": return all_inputs
    wanted = {f.strip() for f in folds.split(",") if f.strip()}
    out = [p for p in all_inputs if p.parent.name in wanted]
    if not out: raise FileNotFoundError(f"No fold inputs matched {folds!r} under {input_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--granularity", choices=["high", "low", "both"], default="both")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--write_prompts_only", action="store_true")
    args = ap.parse_args()
    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)
    granularities = ["high", "low"] if args.granularity == "both" else [args.granularity]
    for inp in discover_inputs(Path(args.input_dir), args.folds):
        for granularity in granularities:
            induce_one(inp, Path(args.output_dir), granularity, client, cfg, args.write_prompts_only)


if __name__ == "__main__":
    main()
