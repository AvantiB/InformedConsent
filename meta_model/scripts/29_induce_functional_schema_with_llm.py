#!/usr/bin/env python
"""Induce a data-driven reduced consent schema from evidence cards.

This is the LLM-assisted schema induction arm. It reuses the organized staged
workflow:

1. induce   -> propose a compact functional schema from evidence cards.
2. critique -> audit overlap, unsafe merges, missing roles, and weak names.
3. revise   -> revise into a final fold-specific induced schema.
4. validate -> produce machine-readable validation/audit reports.

The only evidence sent to the LLM should be evidence cards produced by script 28
from the data-driven pipeline in script 23. No manual schema, baseline round-trip
reconstruction, or classifier score is provided to the induction LLM.
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
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

SENTENCE_DECISION_SET = {"permit", "deny", "mixed", "unclear"}


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


def fold_id_from_cards(cards: list[dict[str, Any]]) -> str:
    vals = [norm(c.get("fold_id")) for c in cards if norm(c.get("fold_id"))]
    return vals[0] if vals else "fold_unknown"


def default_targets(granularity: str, lo: int | None, hi: int | None) -> tuple[int, int]:
    if lo is not None and hi is not None:
        return lo, hi
    return (lo or 10, hi or 15) if granularity == "low" else (lo or 22, hi or 32)


def compact_cards(cards: list[dict[str, Any]], max_cards: int, max_spans: int) -> list[dict[str, Any]]:
    keys = [
        "fold_id", "seed_cluster_id", "sense_ids", "source_elements_included",
        "source_models_represented", "llms_represented", "n_mentions", "n_forms", "n_sentences",
        "top_spans", "suggested_terms", "source_element_labels", "source_element_sense_nodes",
        "equivalence_support_edges", "complementary_or_proximity_neighbors", "unsafe_merge_warnings",
        "polarity_patterns", "decision_value_patterns", "quality_flags", "example_sentences",
    ]
    out = []
    for c in cards[:max_cards]:
        d = {k: c.get(k) for k in keys if k in c}
        d["top_spans"] = (d.get("top_spans") or [])[:max_spans]
        d["source_element_sense_nodes"] = (d.get("source_element_sense_nodes") or [])[:8]
        d["equivalence_support_edges"] = (d.get("equivalence_support_edges") or [])[:6]
        d["complementary_or_proximity_neighbors"] = (d.get("complementary_or_proximity_neighbors") or [])[:6]
        d["unsafe_merge_warnings"] = (d.get("unsafe_merge_warnings") or [])[:6]
        d["example_sentences"] = (d.get("example_sentences") or [])[:2]
        out.append(d)
    return out


def schema_shape(schema_id: str, fold_id: str, granularity: str) -> str:
    return f'''{{
  "schema_id": "{schema_id}",
  "fold_id": "{fold_id}",
  "granularity": "{granularity}",
  "sentence_decision": {{"allowed_values": ["permit", "deny", "mixed", "unclear"]}},
  "dictionary_rows": [{{
    "field_id": "DDS001",
    "field_name": "snake_case_field_name",
    "definition": "annotation-ready span definition",
    "include": ["short examples"],
    "exclude": ["boundary exclusions"],
    "allowed_modifiers": ["modifier_name"],
    "assigned_evidence_cards": ["seed_cluster_id"],
    "source_support": ["brief source/sense support"],
    "boundary_notes": "brief boundary note"
  }}],
  "modifiers": [{{
    "modifier_name": "snake_case_modifier",
    "definition": "cross-cutting attribute definition",
    "allowed_values": ["value1", "value2"],
    "applies_to_fields": ["field_name or all"]
  }}],
  "evidence_coverage": [{{"evidence_group": "seed_cluster_id or role group", "covered_by": ["field_or_modifier"]}}],
  "unsafe_merge_notes": ["brief notes"],
  "missing_or_uncertain_functions": ["brief notes"]
}}'''


def induce_messages(cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    lo, hi = default_targets(args.granularity, args.target_min_fields, args.target_max_fields)
    fold_id = fold_id_from_cards(cards)
    payload = json.dumps(compact_cards(cards, args.max_cards, args.max_spans_per_card), ensure_ascii=False, indent=2)
    user = f"""
Induce a {args.granularity}-granularity functional informed-consent schema from data-driven evidence cards.

Evidence meaning:
- cards come from original annotation evidence where human meaning_preserved = 1;
- seed clusters were formed only from near-equivalence edges;
- complementary/proximity edges are relationship evidence, not merge evidence;
- unsafe-merge warnings mark boundaries that should not be collapsed.

Rules:
- Do not use a manual schema or external ontology as the schema.
- Keep sentence_decision separate from span-level fields: permit, deny, mixed, unclear.
- Target roughly {lo}-{hi} fields.
- Create functional fields suitable for forward annotation and backward reconstruction.
- Merge cards only when functional equivalence is supported; split mixed actor/resource/action/purpose/time/repository/privacy/condition roles.
- Use modifiers only for cross-cutting attributes.

Return JSON exactly in this shape:
{schema_shape(f"data_driven_llm_{args.granularity}_fold_specific", fold_id, args.granularity)}

Evidence cards:
{payload}
""".strip()
    return [{"role": "system", "content": "You induce compact functional annotation schemas from evidence cards. Return valid JSON only."}, {"role": "user", "content": user}]


def critique_messages(schema: dict[str, Any], cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    payload = json.dumps(compact_cards(cards, args.max_cards, max(4, args.max_spans_per_card // 2)), ensure_ascii=False, indent=2)
    user = f"""
Critique the induced schema for overlap, unsafe merges, missing roles, weak names, missing evidence-card coverage, and any use of complementary/proximity edges as merge evidence.

Return JSON with keys: overall_assessment, recommended_merges, recommended_splits, renaming_suggestions, missing_fields, unsafe_or_ambiguous_boundaries, evidence_cards_poorly_covered, complementary_edges_misused_as_merges.

Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Evidence cards:
{payload}
""".strip()
    return [{"role": "system", "content": "You critically audit data-driven functional annotation schemas. Return valid JSON only."}, {"role": "user", "content": user}]


def revise_messages(schema: dict[str, Any], critique: dict[str, Any], cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    lo, hi = default_targets(args.granularity, args.target_min_fields, args.target_max_fields)
    fold_id = fold_id_from_cards(cards)
    payload = json.dumps(compact_cards(cards, args.max_cards, max(4, args.max_spans_per_card // 2)), ensure_ascii=False, indent=2)
    user = f"""
Revise the schema using the critique and evidence cards.

Requirements:
- Target roughly {lo}-{hi} fields.
- Keep sentence_decision separate: permit, deny, mixed, unclear.
- dictionary_rows is the only span-level field dictionary.
- Do not merge cards merely because they co-occur.
- Every field needs field_id, field_name, definition, include, exclude, assigned_evidence_cards, source_support, and boundary_notes.

Return final JSON exactly in this shape:
{schema_shape(f"data_driven_llm_{args.granularity}_fold_specific", fold_id, args.granularity)}

Initial schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Critique:
{json.dumps(critique, ensure_ascii=False, indent=2)}

Evidence cards:
{payload}
""".strip()
    return [{"role": "system", "content": "You revise data-driven annotation schemas using critique. Return valid JSON only."}, {"role": "user", "content": user}]


def validate_schema(schema: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    rows = schema.get("dictionary_rows") or []
    field_ids = [norm(f.get("field_id")) for f in rows if isinstance(f, dict)]
    names = [norm(f.get("field_name")) for f in rows if isinstance(f, dict)]
    dup_ids = sorted([x for x in set(field_ids) if x and field_ids.count(x) > 1])
    dup_names = sorted([x for x in set(names) if x and names.count(x) > 1])
    missing = []
    covered_cards = set()
    for f in rows:
        if not isinstance(f, dict):
            continue
        fname = norm(f.get("field_name"))
        for k in ["field_id", "field_name", "definition", "include", "exclude", "assigned_evidence_cards", "source_support"]:
            if not f.get(k):
                missing.append({"field": fname, "missing": k})
        for cid in f.get("assigned_evidence_cards") or []:
            if norm(cid):
                covered_cards.add(norm(cid))
    card_ids = {norm(c.get("seed_cluster_id")) for c in cards if norm(c.get("seed_cluster_id"))}
    bad_names = [n for n in names if not re.match(r"^[a-z][a-z0-9_]*$", n or "")]
    sent_vals = schema.get("sentence_decision", {}).get("allowed_values", []) if isinstance(schema.get("sentence_decision"), dict) else []
    sent_set = {norm(x) for x in sent_vals}
    return {
        "n_fields": len(names),
        "duplicate_field_ids": dup_ids,
        "duplicate_field_names": dup_names,
        "bad_snake_case_field_names": bad_names,
        "missing_required_content": missing,
        "n_evidence_cards": len(card_ids),
        "n_evidence_cards_assigned_to_fields": len(card_ids & covered_cards),
        "unassigned_evidence_cards": sorted(card_ids - covered_cards),
        "sentence_decision_allowed_values": sorted(sent_set),
        "sentence_decision_valid": sent_set == SENTENCE_DECISION_SET,
        "passes_basic_validation": not dup_ids and not dup_names and not bad_names and not missing and bool(names) and sent_set == SENTENCE_DECISION_SET,
    }


def call_stage(name: str, client: Any, cfg: dict[str, Any], messages: list[dict[str, str]], raw_path: Path) -> dict[str, Any]:
    log(f"Starting stage={name}; prompt_chars={sum(len(m.get('content', '')) for m in messages)}")
    raw = call_chat(client, cfg, messages)
    raw_path.write_text(raw)
    log(f"Finished stage={name}; response_chars={len(raw)}")
    return extract_json(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence_cards_jsonl", required=True)
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["induce", "critique", "revise", "validate", "all"], default="all")
    ap.add_argument("--granularity", choices=["low", "high"], default="high")
    ap.add_argument("--max_cards", type=int, default=90)
    ap.add_argument("--max_spans_per_card", type=int, default=18)
    ap.add_argument("--target_min_fields", type=int, default=None)
    ap.add_argument("--target_max_fields", type=int, default=None)
    ap.add_argument("--limit_cards", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cards = load_jsonl(Path(args.evidence_cards_jsonl), args.limit_cards)
    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)

    initial_path = out / "schema.initial.json"
    critique_path = out / "schema.critique.json"
    final_json = out / "schema.json"
    final_yaml = out / "schema.yaml"

    if args.stage in {"induce", "all"}:
        initial = call_stage("induce", client, cfg, induce_messages(cards, args), out / "stage1_induce_raw_response.txt")
        initial_path.write_text(json.dumps(initial, indent=2, ensure_ascii=False))
    else:
        initial = json.loads(initial_path.read_text())

    if args.stage in {"critique", "all"}:
        critique = call_stage("critique", client, cfg, critique_messages(initial, cards, args), out / "stage2_critique_raw_response.txt")
        critique_path.write_text(json.dumps(critique, indent=2, ensure_ascii=False))
    else:
        critique = json.loads(critique_path.read_text())

    if args.stage in {"revise", "all"}:
        final = call_stage("revise", client, cfg, revise_messages(initial, critique, cards, args), out / "stage3_revise_raw_response.txt")
        final_json.write_text(json.dumps(final, indent=2, ensure_ascii=False))
        final_yaml.write_text(yaml.safe_dump(final, sort_keys=False, allow_unicode=True))
    elif final_json.exists():
        final = json.loads(final_json.read_text())
    elif final_yaml.exists():
        final = yaml.safe_load(final_yaml.read_text())
    else:
        final = initial

    if args.stage in {"validate", "all"}:
        report = validate_schema(final, cards)
        (out / "schema.validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    log(f"Wrote schema induction artifacts to {out}")


if __name__ == "__main__":
    main()
