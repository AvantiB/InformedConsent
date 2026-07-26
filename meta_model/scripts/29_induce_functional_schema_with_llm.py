#!/usr/bin/env python
"""Induce a data-driven reduced consent schema from evidence cards.

This is the LLM-assisted schema induction arm. It reuses the organized staged
workflow:

1. induce   -> propose a compact functional schema from evidence cards.
2. critique -> audit overlap, unsafe merges, missing roles, and weak names.
3. revise   -> revise into a final fold-specific induced schema.
4. validate -> produce machine-readable validation/audit reports.

The only evidence sent to the LLM should be evidence cards produced by script 28
from the data-driven pipeline in script 23. No manual schema, baseline
round-trip reconstruction, or classifier score is provided to the induction LLM.
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

SENTENCE_DECISIONS = ["permit", "deny", "mixed", "unclear"]


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
                    obj = json.loads(t[start:i + 1])
                    if not isinstance(obj, dict):
                        raise ValueError("Parsed JSON is not an object")
                    return obj
    raise ValueError("Could not parse balanced JSON object")


def compact_cards(cards: list[dict[str, Any]], max_cards: int, max_spans: int) -> list[dict[str, Any]]:
    out = []
    for c in cards[:max_cards]:
        d = {
            "fold_id": c.get("fold_id"),
            "seed_cluster_id": c.get("seed_cluster_id"),
            "sense_ids": c.get("sense_ids", [])[:8],
            "source_elements_included": c.get("source_elements_included", [])[:16],
            "source_models_represented": c.get("source_models_represented", []),
            "llms_represented": c.get("llms_represented", []),
            "n_mentions": c.get("n_mentions"),
            "n_forms": c.get("n_forms"),
            "n_sentences": c.get("n_sentences"),
            "top_spans": c.get("top_spans", [])[:max_spans],
            "suggested_terms": c.get("suggested_terms", [])[:12],
            "source_element_labels": c.get("source_element_labels", [])[:12],
            "source_element_sense_nodes": c.get("source_element_sense_nodes", [])[:8],
            "equivalence_support_edges": c.get("equivalence_support_edges", [])[:6],
            "complementary_or_proximity_neighbors": c.get("complementary_or_proximity_neighbors", [])[:6],
            "unsafe_merge_warnings": c.get("unsafe_merge_warnings", [])[:6],
            "polarity_patterns": c.get("polarity_patterns", {}),
            "decision_value_patterns": c.get("decision_value_patterns", {}),
            "quality_flags": c.get("quality_flags", []),
            "example_sentences": c.get("example_sentences", [])[:2],
        }
        out.append(d)
    return out


def default_targets(granularity: str, lo: int | None, hi: int | None) -> tuple[int, int]:
    if lo is not None and hi is not None:
        return lo, hi
    if granularity == "low":
        return lo or 10, hi or 15
    if granularity == "high":
        return lo or 22, hi or 32
    return lo or 16, hi or 28


def fold_id_from_cards(cards: list[dict[str, Any]]) -> str:
    vals = [norm(c.get("fold_id")) for c in cards if norm(c.get("fold_id"))]
    return vals[0] if vals else "fold_unknown"


def induce_messages(cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    target_min, target_max = default_targets(args.granularity, args.target_min_fields, args.target_max_fields)
    payload = json.dumps(compact_cards(cards, args.max_cards, args.max_spans_per_card), ensure_ascii=False, indent=2)
    system = "You induce compact functional annotation schemas from evidence cards. Return valid JSON only."
    user = f"""
Task: induce a {args.granularity}-granularity functional informed-consent schema from the data-driven evidence cards below.

Evidence-card meaning:
- seed clusters are derived only from valid original annotation evidence where human meaning preservation = 1.
- only near-equivalence edges were used to form seed clusters.
- complementary/proximity edges are relationship/bundle evidence, not merge evidence.
- unsafe-merge warnings identify boundaries that should not be collapsed.

Constraints:
- Do NOT use a manual schema or external ontology as the schema.
- Create functional span-level fields suitable for forward annotation and backward reconstruction.
- Keep sentence_decision separate from span-level fields with allowed values: permit, deny, mixed, unclear.
- Target roughly {target_min}-{target_max} span-level fields.
- Merge evidence cards only when their functional roles and unsafe-merge warnings support it.
- Split cards when evidence mixes actor/resource/action/purpose/time/repository/privacy/condition roles.
- Use modifiers only for cross-cutting attributes, not as primary schema fields.
- Every field must have clear boundaries, include examples, exclude examples, and assigned evidence cards.

Return JSON exactly with this structure:
{{
  "schema_id": "data_driven_llm_{args.granularity}_fold_specific",
  "fold_id": "{fold_id_from_cards(cards)}",
  "granularity": "{args.granularity}",
  "sentence_decision": {{"allowed_values": ["permit", "deny", "mixed", "unclear"]}},
  "dictionary_rows": [
    {{
      "field_id": "DDS001",
      "field_name": "snake_case_field_name",
      "definition": "annotation-ready span definition",
      "include": ["short examples"],
      "exclude": ["boundary exclusions"],
      "allowed_modifiers": ["modifier_name"],
      "assigned_evidence_cards": ["seed_cluster_id"],
      "source_support": ["brief source/sense support"]
    }}
  ],
  "modifiers": [
    {{
      "modifier_name": "snake_case_modifier",
      "definition": "cross-cutting attribute definition",
      "allowed_values": ["value1", "value2"],
      "applies_to_fields": ["field_name or all"]
    }}
  ],
  "evidence_coverage": [{{"evidence_group": "seed_cluster_id or role group", "covered_by": ["field_or_modifier"]}}],
  "unsafe_merge_notes": ["brief notes"],
  "missing_or_uncertain_functions": ["brief notes"]
}}

Evidence cards:
{payload}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def critique_messages(schema: dict[str, Any], cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    payload = json.dumps(compact_cards(cards, args.max_cards, max(4, args.max_spans_per_card // 2)), ensure_ascii=False, indent=2)
    system = "You are a critical reviewer of data-driven functional annotation schemas. Return valid JSON only."
    user = f"""
Task: critique the induced schema using the evidence cards.

Audit specifically for:
- overlap between fields;
- unsafe merges contradicted by unsafe_merge_warnings;
- missing roles/functions present in evidence cards;
- weak or lexical field names;
- cards not covered by any field;
- complementary evidence incorrectly treated as equivalence;
- sentence decision cues incorrectly turned into span-level fields.

Return JSON:
{{
  "overall_assessment": "...",
  "recommended_merges": [{{"fields": ["..."], "reason": "..."}}],
  "recommended_splits": [{{"field": "...", "proposed_fields": ["..."], "reason": "..."}}],
  "renaming_suggestions": [{{"old_name": "...", "new_name": "...", "reason": "..."}}],
  "missing_fields": [{{"name": "...", "reason": "..."}}],
  "unsafe_or_ambiguous_boundaries": [{{"fields": ["..."], "issue": "...", "fix": "..."}}],
  "evidence_cards_poorly_covered": [{{"seed_cluster_id": "...", "issue": "..."}}],
  "complementary_edges_misused_as_merges": [{{"field": "...", "issue": "..."}}]
}}

Schema to critique:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Evidence cards, abbreviated:
{payload}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def revise_messages(schema: dict[str, Any], critique: dict[str, Any], cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    target_min, target_max = default_targets(args.granularity, args.target_min_fields, args.target_max_fields)
    payload = json.dumps(compact_cards(cards, args.max_cards, max(4, args.max_spans_per_card // 2)), ensure_ascii=False, indent=2)
    system = "You revise data-driven annotation schemas using critique. Return valid JSON only."
    user = f"""
Task: revise the induced informed-consent schema using the critique and evidence cards.

Requirements:
- Target roughly {target_min}-{target_max} span-level fields.
- Keep sentence_decision separate: permit, deny, mixed, unclear.
- Use only dictionary_rows as the span-level field dictionary.
- Use snake_case field and modifier names.
- Every dictionary row needs field_id, field_name, definition, include, exclude, assigned_evidence_cards, and source_support.
- Do not merge evidence cards merely because they co-occur; merge only functional near-equivalence.
- Keep complementary/proximity information as boundary notes, source_support, or evidence_coverage, not as merge justification.

Return JSON exactly with this structure:
{{
  "schema_id": "data_driven_llm_{args.granularity}_fold_specific",
  "fold_id": "{fold_id_from_cards(cards)}",
  "granularity": "{args.granularity}",
  "schema_status": "revised_after_critique",
  "annotation_policy": {{
    "sentence_decision": "provision-level only; do not use permit/deny as span-level labels",
    "atomic_spans": "prefer the smallest phrase expressing one semantic function",
    "multi_label": "allowed only when the same span truly expresses multiple functions"
  }},
  "sentence_decision": {{"allowed_values": ["permit", "deny", "mixed", "unclear"]}},
  "dictionary_rows": [
    {{
      "field_id": "DDS001",
      "field_name": "snake_case_field_name",
      "definition": "annotation-ready span definition",
      "include": ["short examples"],
      "exclude": ["boundary exclusions"],
      "allowed_modifiers": ["modifier_name"],
      "assigned_evidence_cards": ["seed_cluster_id"],
      "source_support": ["brief source/sense support"],
      "boundary_notes": "brief boundary note"
    }}
  ],
  "modifiers": [
    {{
      "modifier_name": "snake_case_modifier",
      "definition": "cross-cutting attribute definition",
      "allowed_values": ["value1", "value2"],
      "applies_to_fields": ["field_name or all"]
    }}
  ],
  "evidence_coverage": [{{"evidence_group": "seed_cluster_id or role group", "covered_by": ["field_or_modifier"]}}],
  "unsafe_merge_notes": ["brief notes"],
  "missing_or_uncertain_functions": ["brief notes"]
}}

Initial schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Critique:
{json.dumps(critique, ensure_ascii=False, indent=2)}

Evidence cards, abbreviated:
{payload}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_schema(schema: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    rows = schema.get("dictionary_rows") or schema.get("fields") or []
    field_ids = [norm(f.get("field_id")) for f in rows if isinstance(f, dict)]
    names = [norm(f.get("field_name") or f.get("name")) for f in rows if isinstance(f, dict)]
    dup_ids = sorted([x for x in set(field_ids) if x and field_ids.count(x) > 1])
    dup_names = sorted([x for x in set(names) if x and names.count(x) > 1])
    missing = []
    covered_cards = set()
    for f in rows:
        if not isinstance(f, dict):
            continue
        fname = norm(f.get("field_name") or f.get("name"))
        for k in ["field_name", "definition", "include", "exclude"]:
            if k == "field_name":
                ok = bool(fname)
            else:
                ok = bool(f.get(k))
            if not ok:
                missing.append({"field": fname, "missing": k})
        for cid in f.get("assigned_evidence_cards") or []:
            if norm(cid):
                covered_cards.add(norm(cid))
    card_ids = {norm(c.get("seed_cluster_id")) for c in cards if norm(c.get("seed_cluster_id"))}
    bad_names = [n for n in names if not re.match(r"^[a-z][a-z0-9_]*$", n or "")]
    sent_vals = schema.get("sentence_decision", {}).get("allowed_values", []) if isinstance(schema.get("sentence_decision"), dict) else []
    sent_vals_norm = sorted(norm(x) for x in sent_vals)
    return {
        "n_fields": len(names),
        "duplicate_field_ids": dup_ids,
        "duplicate_field_names": dup_names,
        "bad_snake_case_field_names": bad_names,
        "missing_required_content": missing,
        "n_evidence_cards": len(card_ids),
        "n_evidence_cards_assigned_to_fields": len(card_ids & covered_cards),
        "unassigned_evidence_cards": sorted(card_ids - covered_cards),
        "sentence_decision_allowed_values": sent_vals_norm,
        "sentence_decision_valid": sent_vals_norm == SENTENCE_DECISIONS,
        "passes_basic_validation": not dup_ids and not dup_names and not bad_names and not missing and bool(names) and sent_vals_norm == SENTENCE_DECISIONS,
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def call_stage(stage_name: str, client: Any, cfg: dict[str, Any], messages: list[dict[str, str]], raw_path: Path) -> dict[str, Any]:
    approx_chars = sum(len(m.get("content", "")) for m in messages)
    log(f"Starting LLM stage={stage_name}; prompt_chars={approx_chars}; raw_output={raw_path}")
    raw = call_chat(client, cfg, messages)
    raw_path.write_text(raw)
    log(f"Finished LLM stage={stage_name}; response_chars={len(raw)}")
    return extract_json(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence_cards_jsonl", required=True)
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True, help="Use one fixed strong induction model.")
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
    log(f"Loaded evidence cards={len(cards)} from {args.evidence_cards_jsonl}")
    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)

    initial_path = out / "schema.initial.json"
    critique_path = out / "schema.critique.json"
    final_json = out / "schema.json"
    final_yaml = out / "schema.yaml"

    if args.stage in {"induce", "all"}:
        initial = call_stage("induce", client, cfg, induce_messages(cards, args), out / "stage1_induce_raw_response.txt")
        write_json(initial_path, initial)
    else:
        initial = json.loads(initial_path.read_text())

    if args.stage in {"critique", "all"}:
        critique = call_stage("critique", client, cfg, critique_messages(initial, cards, args), out / "stage2_critique_raw_response.txt")
        write_json(critique_path, critique)
    else:
        critique = json.loads(critique_path.read_text())

    if args.stage in {"revise", "all"}:
        final = call_stage("revise", client, cfg, revise_messages(initial, critique, cards, args), out / "stage3_revise_raw_response.txt")
        write_json(final_json, final)
        final_yaml.write_text(yaml.safe_dump(final, sort_keys=False, allow_unicode=True))
    elif final_json.exists():
        final = json.loads(final_json.read_text())
    elif final_yaml.exists():
        final = yaml.safe_load(final_yaml.read_text())
    else:
        final = initial

    if args.stage in {"validate", "all"}:
        report = validate_schema(final, cards)
        write_json(out / "schema.validation.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    log(f"Wrote schema induction artifacts to {out}")


if __name__ == "__main__":
    main()
