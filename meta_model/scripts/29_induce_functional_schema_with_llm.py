#!/usr/bin/env python
"""Induce a Reduced Functional V1 schema from evidence cards using one fixed LLM.

This is the LLM-induced schema arm. It should be run with a single strong
induction model (for example Mayo GPT-5.5 Thinking) while downstream round-trip
evaluation can use multiple LLMs. The manual V1 schema is intentionally not
provided to the induction prompt.

Stages:
1. induce   -> propose a compact functional schema from evidence cards.
2. critique -> audit overlap, unsafe merges, missing roles, and weak names.
3. revise   -> revise into a final fold-specific induced schema.
4. validate -> produce machine-readable validation/audit reports.
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

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_model_config(path: Path, model_key: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model_config_yaml does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"model_config_yaml is a directory, expected YAML file: {path}")
    cfg = yaml.safe_load(path.read_text())
    model_cfg = {**(cfg.get("defaults", {}) or {}), **((cfg.get("models", {}) or {}).get(model_key, {}))}
    if not model_cfg:
        raise KeyError(f"model_key={model_key!r} not found in {path}. Available keys={sorted((cfg.get('models') or {}).keys())}")
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
        "max_tokens": int(cfg.get("max_tokens", 8192)),
        "timeout": float(cfg.get("timeout_seconds", 240)),
    }
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg.get("temperature", 0)
    last = None
    for attempt in range(1, int(cfg.get("max_retries", 3)) + 1):
        try:
            return (client.chat.completions.create(**kwargs).choices[0].message.content or "")
        except Exception as exc:
            last = exc
            if attempt < int(cfg.get("max_retries", 3)):
                time.sleep(float(cfg.get("retry_sleep_seconds", 5)) * attempt)
    raise RuntimeError(f"LLM request failed: {last}")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|yaml)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start=start):
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
                    obj = json.loads(text[start:i + 1])
                    if not isinstance(obj, dict):
                        raise ValueError("Parsed JSON is not an object")
                    return obj
    raise ValueError("Could not parse balanced JSON object")


def compact_cards(cards: list[dict[str, Any]], max_cards: int, max_spans: int) -> list[dict[str, Any]]:
    out = []
    for c in cards[:max_cards]:
        d = dict(c)
        d["top_spans"] = d.get("top_spans", [])[:max_spans]
        d["top_source_elements"] = d.get("top_source_elements", [])[:12]
        d["crosswalk_hints_from_source_elements"] = d.get("crosswalk_hints_from_source_elements", [])[:8]
        d["near_equivalence_or_related_edges"] = d.get("near_equivalence_or_related_edges", [])[:6]
        d["complementary_cooccurrence_edges"] = d.get("complementary_cooccurrence_edges", [])[:6]
        d["example_sentences"] = d.get("example_sentences", [])[:2]
        out.append(d)
    return out


def induce_messages(cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    payload = json.dumps(compact_cards(cards, args.max_cards, args.max_spans_per_card), ensure_ascii=False, indent=2)
    system = "You induce compact functional schemas from evidence. Return valid JSON only."
    user = f"""
Task: induce a compact functional informed-consent schema from the evidence cards below.

Important constraints:
- Do NOT use a pre-existing manual schema. Infer roles only from the evidence cards.
- The schema must support forward annotation and backward reconstruction of consent sentences.
- Prefer complementary functional roles over lexical/topic clusters.
- Separate sentence/provision-level decision from span-level semantics.
- Co-occurrence is complementarity evidence, not merge evidence.
- Split roles that mix actor/resource/action/purpose/time/repository/privacy.
- Target roughly {args.target_min_fields}-{args.target_max_fields} span-level fields.
- Each field must have a clear name, definition, include examples, exclude examples, and boundary notes.
- Use core/extension status.

Return JSON with exactly this structure:
{{
  "schema_id": "llm_induced_functional_v1_candidate",
  "schema_status": "llm_induced_from_training_fold_evidence",
  "sentence_decision": {{"allowed_values": ["permit", "deny", "obligation", "mixed", "unclear"]}},
  "fields": [
    {{
      "name": "snake_case_field_name",
      "status": "core|extension",
      "definition": "...",
      "include": ["example span/function"],
      "exclude": ["boundary exclusion"],
      "assigned_evidence_cards": ["candidate_field_id or stability_group_id"],
      "source_model_support_summary": "...",
      "common_complementary_fields": ["field_name"],
      "boundary_notes": "..."
    }}
  ],
  "unsafe_merge_notes": ["..."],
  "missing_or_uncertain_functions": ["..."]
}}

Evidence cards:
{payload}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def critique_messages(schema: dict[str, Any], cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    payload = json.dumps(compact_cards(cards, args.max_cards, max(4, args.max_spans_per_card // 2)), ensure_ascii=False, indent=2)
    system = "You are a critical reviewer of functional annotation schemas. Return valid JSON only."
    user = f"""
Task: critique the induced informed-consent schema for overlap, unsafe merges, missing functions, poor names, and insufficient boundaries.

Focus on whether the fields are mostly non-overlapping and suitable for round-trip meaning preservation.
Pay special attention to:
- decision cue vs sentence decision
- participant vs authorized actor
- institution/custodian vs repository/registry
- action vs resource vs purpose
- temporal phrase vs temporal target/attachment
- condition vs restriction/prohibition
- privacy/identifiability vs resource
- consequence/protection vs general purpose

Return JSON:
{{
  "overall_assessment": "...",
  "recommended_merges": [{{"fields": ["..."], "reason": "..."}}],
  "recommended_splits": [{{"field": "...", "proposed_fields": ["..."], "reason": "..."}}],
  "renaming_suggestions": [{{"old_name": "...", "new_name": "...", "reason": "..."}}],
  "missing_fields": [{{"name": "...", "reason": "..."}}],
  "unsafe_or_ambiguous_boundaries": [{{"fields": ["..."], "issue": "...", "fix": "..."}}],
  "cards_poorly_covered": [{{"candidate_field_id": "...", "issue": "..."}}]
}}

Schema to critique:
{schema_json}

Evidence cards, abbreviated:
{payload}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def revise_messages(schema: dict[str, Any], critique: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    system = "You revise functional schemas using critique. Return valid JSON only."
    user = f"""
Task: revise the induced informed-consent schema using the critique.

Requirements:
- Keep the schema compact and functional.
- Target roughly {args.target_min_fields}-{args.target_max_fields} span-level fields.
- Keep sentence_decision separate from span-level fields.
- Use snake_case names.
- Every field needs definition, include, exclude, and boundary notes.
- Do not include residual_important_content or provenance in the fields list; they are metadata fields.

Return JSON:
{{
  "meta_model_id": "llm_induced_functional_v1_candidate",
  "status": "llm_induced_training_fold_candidate",
  "annotation_policy": {{
    "sentence_decision": "provision-level only; do not use permit/deny as span-level labels",
    "atomic_spans": "prefer smallest phrase expressing one semantic function",
    "multi_label": "allowed only when same span truly expresses multiple functions",
    "residual": "use residual_important_content for meaning-critical text outside schema"
  }},
  "sentence_decision": {{"scope": "sentence_or_provision_level", "allowed_values": ["permit", "deny", "obligation", "mixed", "unclear"]}},
  "fields": [
    {{
      "name": "snake_case_field_name",
      "status": "core|extension",
      "definition": "...",
      "include": ["..."],
      "exclude": ["..."],
      "data_seed": "brief evidence-card rationale",
      "boundary_notes": "..."
    }}
  ],
  "residual_important_content": {{"description": "Meaning-critical span not captured by current fields."}},
  "provenance": {{"required": true, "note": "Preserve source sentence, form, span, field, model, fold."}}
}}

Initial schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Critique:
{json.dumps(critique, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    fields = schema.get("fields") or []
    names = [norm(f.get("name")) for f in fields if isinstance(f, dict)]
    dupes = sorted([x for x in set(names) if names.count(x) > 1])
    missing = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        for k in ["name", "definition", "include", "exclude"]:
            if not f.get(k):
                missing.append({"field": norm(f.get("name")), "missing": k})
    bad_names = [n for n in names if not re.match(r"^[a-z][a-z0-9_]*$", n or "")]
    return {
        "n_fields": len(names),
        "n_core": sum(1 for f in fields if isinstance(f, dict) and norm(f.get("status")) == "core"),
        "n_extension": sum(1 for f in fields if isinstance(f, dict) and norm(f.get("status")) == "extension"),
        "duplicate_names": dupes,
        "bad_snake_case_names": bad_names,
        "missing_required_content": missing,
        "passes_basic_validation": not dupes and not bad_names and not missing and len(names) > 0,
    }


def write_schema(schema: dict[str, Any], out_yaml: Path) -> None:
    out_yaml.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True))
    out_yaml.with_suffix(".json").write_text(json.dumps(schema, indent=2, ensure_ascii=False))


def call_stage(stage_name: str, client: Any, cfg: dict[str, Any], messages: list[dict[str, str]], raw_path: Path) -> dict[str, Any]:
    approx_chars = sum(len(m.get("content", "")) for m in messages)
    log(f"Starting LLM stage={stage_name}; prompt_chars={approx_chars}; raw_output={raw_path}")
    t0 = time.time()
    raw = call_chat(client, cfg, messages)
    raw_path.write_text(raw)
    log(f"Finished LLM stage={stage_name}; elapsed_sec={time.time() - t0:.1f}; response_chars={len(raw)}")
    return extract_json(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence_cards_jsonl", required=True)
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True, help="Use one fixed strong induction model, e.g. mayo_gpt55.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["induce", "critique", "revise", "validate", "all"], default="all")
    ap.add_argument("--max_cards", type=int, default=80)
    ap.add_argument("--max_spans_per_card", type=int, default=12)
    ap.add_argument("--target_min_fields", type=int, default=16)
    ap.add_argument("--target_max_fields", type=int, default=28)
    ap.add_argument("--limit_cards", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cards_path = Path(args.evidence_cards_jsonl)
    cfg_path = Path(args.model_config_yaml)
    log(f"Loading evidence cards: {cards_path}")
    cards = load_jsonl(cards_path, args.limit_cards)
    log(f"Loaded cards={len(cards)}; max_cards_sent={args.max_cards}; max_spans_per_card={args.max_spans_per_card}")

    cfg = load_model_config(cfg_path, args.model_key)
    log(
        "Loaded model config: "
        f"model_key={args.model_key}; provider={cfg.get('provider')}; "
        f"model={cfg.get('model')}; deployment={cfg.get('deployment')}; "
        f"timeout={cfg.get('timeout_seconds')}; retries={cfg.get('max_retries')}"
    )
    if str(cfg.get("provider", "")).lower() == "mayo_apigee_azure_openai":
        if cfg.get("oauth_client_id_env") and cfg.get("oauth_client_secret_env"):
            id_set = bool(os.getenv(str(cfg.get("oauth_client_id_env"))))
            secret_set = bool(os.getenv(str(cfg.get("oauth_client_secret_env"))))
            log(f"Mayo OAuth mode detected; client_id_env_set={id_set}; client_secret_env_set={secret_set}")
        else:
            token_env = cfg.get("api_key_env") or "APIGEE_TOKEN"
            token_file_env = cfg.get("api_key_file_env") or "APIGEE_TOKEN_FILE"
            log(
                "Mayo static-token mode detected; "
                f"token_env_set={bool(os.getenv(str(token_env)))}; "
                f"token_file_env_set={bool(os.getenv(str(token_file_env)))}"
            )

    client = make_client(cfg)

    initial_path = out / "llm_induced_schema.initial.json"
    critique_path = out / "llm_induced_schema.critique.json"
    final_path = out / "llm_induced_functional_v1_candidate.yaml"

    if args.stage in {"induce", "all"}:
        initial = call_stage("induce", client, cfg, induce_messages(cards, args), out / "llm_induction_raw_response.txt")
        initial_path.write_text(json.dumps(initial, indent=2, ensure_ascii=False))
        log(f"Wrote initial schema JSON: {initial_path}")
    else:
        log(f"Loading existing initial schema JSON: {initial_path}")
        initial = json.loads(initial_path.read_text())

    if args.stage in {"critique", "all"}:
        critique = call_stage("critique", client, cfg, critique_messages(initial, cards, args), out / "llm_critique_raw_response.txt")
        critique_path.write_text(json.dumps(critique, indent=2, ensure_ascii=False))
        log(f"Wrote critique JSON: {critique_path}")
    else:
        log(f"Loading existing critique JSON: {critique_path}")
        critique = json.loads(critique_path.read_text())

    if args.stage in {"revise", "all"}:
        final_schema = call_stage("revise", client, cfg, revise_messages(initial, critique, args), out / "llm_revision_raw_response.txt")
        write_schema(final_schema, final_path)
        log(f"Wrote final schema YAML/JSON: {final_path}")
    elif final_path.exists():
        log(f"Loading existing final schema YAML: {final_path}")
        final_schema = yaml.safe_load(final_path.read_text())
    else:
        final_schema = initial

    if args.stage in {"validate", "all"}:
        report = validate_schema(final_schema)
        report_path = out / "llm_induced_schema_validation.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        log(f"Wrote validation report: {report_path}")
    log(f"Wrote LLM-induced schema artifacts to {out}")


if __name__ == "__main__":
    main()
