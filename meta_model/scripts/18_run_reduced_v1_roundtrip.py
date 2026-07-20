#!/usr/bin/env python
"""Run reduced V1 meta-model forward/backward round-trip experiments.

This runner is schema-dynamic: it reads field names from the supplied V1 YAML.
That allows both audited V1 schemas and provisional empirical cluster schemas
(e.g., semantic_cluster_C001) to be evaluated with the same round-trip protocol.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

TEXT_COL_CANDIDATES = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "source_id", "id"]
MASK = "[ORIGINAL_SENTENCE_REMOVED]"
NON_LIST_FIELDS = {"decision", "provenance"}
AUDIT_FIELDS = {"residual_important_content", "provenance"}


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")
    return None


def load_rows(roundtrips_csv: Path, limit: int | None, no_dedupe_sentences: bool) -> pd.DataFrame:
    df = pd.read_csv(roundtrips_csv)
    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    id_col = pick_col(df, ID_COL_CANDIDATES, required=False)
    out = df.copy()
    out["_source_text"] = out[text_col].map(norm_text)
    out["_source_id"] = out[id_col].astype(str) if id_col else out["_source_text"].map(stable_id)
    out = out[out["_source_text"].astype(bool)].copy()
    if not no_dedupe_sentences:
        out = out.drop_duplicates(subset=["_source_text"]).copy()
        out["_source_id"] = out["_source_text"].map(stable_id)
    out = out.reset_index(drop=True)
    if limit is not None:
        out = out.head(limit).copy()
    return out[["_source_id", "_source_text"]]


def load_model_config(path: Path, model_key: str) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    model_cfg = {**(cfg.get("defaults", {}) or {}), **((cfg.get("models", {}) or {}).get(model_key, {}))}
    if not model_cfg:
        raise KeyError(f"model_key={model_key!r} not found in {path}")
    model_cfg["model_key"] = model_key
    return model_cfg


def make_client(model_cfg: dict[str, Any]) -> Any:
    if str(model_cfg.get("provider", "")).lower() == "mayo_apigee_azure_openai":
        return None
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install with: pip install openai")
    api_key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(str(api_key_env), "") if api_key_env else ""
    if not api_key:
        api_key = "EMPTY"
    base_url = model_cfg.get("base_url")
    return OpenAI(api_key=api_key) if base_url in {"", "null", None} else OpenAI(api_key=api_key, base_url=base_url)


def call_chat(client: Any, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    if str(model_cfg.get("provider", "")).lower() == "mayo_apigee_azure_openai":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from apigee_azure_client import call_apigee_chat  # type: ignore
        return call_apigee_chat(client, model_cfg, messages)
    kwargs = {"model": model_cfg["model"], "messages": messages, "max_tokens": int(model_cfg.get("max_tokens", 4096)), "timeout": float(model_cfg.get("timeout_seconds", 120))}
    if model_cfg.get("temperature") is not None:
        kwargs["temperature"] = model_cfg.get("temperature", 0)
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_exc}")


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
    depth, in_str, esc = 0, False, False
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


def load_schema(schema_path: Path) -> tuple[dict[str, Any], str, list[str]]:
    data = yaml.safe_load(schema_path.read_text())
    fields = data.get("fields", []) or []
    field_names = [str(f.get("name")) for f in fields if f.get("name") and f.get("name") not in {"decision", "provenance"}]
    lines = [f"Meta-model: {data.get('meta_model_id', 'reduced_consent_metamodel_v1')}", f"Status: {data.get('status', '')}", f"Design goal: {data.get('design_goal', '')}", "", "Fields:"]
    for f in fields:
        lines.append(f"- {f.get('name')} [{f.get('status', '')}]: {f.get('description', '')}")
        ev = f.get("selection_evidence") or {}
        if ev:
            lines.append(f"  Selection evidence: {json.dumps(ev, ensure_ascii=False)[:800]}")
        support = f.get("source_element_support") or []
        if support:
            lines.append(f"  Source support examples: {', '.join(str(x) for x in support[:10])}")
        spans = f.get("positive_span_examples") or []
        if spans:
            lines.append(f"  Positive span examples: {'; '.join(str(x) for x in spans[:8])}")
    lines += ["", "Provision structure:", json.dumps(data.get("provision_structure", {}), ensure_ascii=False, indent=2)]
    return data, "\n".join(lines), field_names


def evidence_rules(evidence_mode: str, max_evidence_tokens: int) -> str:
    if evidence_mode == "compact":
        return f"""Evidence-span rules for compact mode:
- Use short evidence phrases only, preferably <= {max_evidence_tokens} tokens.
- Do not copy the full sentence into any field.
- Avoid full clauses unless the clause is the minimal expression of a condition or exception.
- Prefer normalized_value plus short evidence_span_text.
- Put unmodeled but important meaning in residual_important_content, but each residual phrase must be short."""
    if evidence_mode == "permissive":
        return """Evidence-span rules for permissive mode:
- Use the same reduced V1 schema.
- Evidence_span_text may be a longer phrase or clause when needed to preserve meaning.
- Do not copy the full sentence verbatim into any single field.
- Prefer normalized_value plus evidence_span_text; longer evidence is allowed only when it carries condition, exception, temporal, privacy, or governance meaning."""
    raise ValueError("--evidence_mode must be compact or permissive")


def provision_template(field_names: list[str]) -> str:
    lines = [
        '{',
        '  "provision_id": "p1",',
        '  "decision": {"value": "permit|deny|obligation|mixed|unclear", "evidence_span_text": "..."},'
    ]
    for name in field_names:
        if name in {"residual_important_content", "provenance"}:
            continue
        lines.append(f'  "{name}": [{{"normalized_value": "...", "evidence_span_text": "...", "confidence": "high|medium|low"}}],')
    lines += [
        '  "residual_important_content": [{"evidence_span_text": "...", "reason": "why this content matters"}],',
        '  "provenance": {"source_field_support": ["field names used"], "rationale": "brief rationale"}',
        '}'
    ]
    return "\n".join(lines)


def build_forward_messages(sentence: str, schema: str, field_names: list[str], evidence_mode: str, max_evidence_tokens: int) -> list[dict[str, str]]:
    system = "You are an NLP annotator for informed-consent documents. Apply the supplied reduced functional meta-model consistently. Return valid JSON only."
    field_list = ", ".join(field_names)
    user = f"""
Task: map the informed-consent sentence to the supplied reduced V1 functional consent meta-model.

Important context:
- This schema may be an audited V1 or a provisional empirical cluster schema.
- Use only the provided field names. Do not create new top-level provision fields.
- For provisional semantic_cluster_C### fields, use the schema descriptions/source examples to decide which cluster best captures each meaning unit.
- A sentence may contain one or more provisions.
- Use normalized values wherever possible, with evidence spans for audit.

Allowed provision fields: decision, {field_list}, residual_important_content, provenance

{evidence_rules(evidence_mode, max_evidence_tokens)}

Reduced V1 schema:
{schema}

Return JSON with exactly this top-level structure:
{{
  "sentence_decision": "permit|deny|obligation|mixed|unclear",
  "provisions": [
{provision_template(field_names)}
  ],
  "unmatched_language": [{{"span_text": "...", "reason": "brief reason"}}],
  "schema_coverage_notes": "brief note"
}}

Use empty lists for fields that are not present. Use multiple provision objects only when the sentence contains distinct consent provisions.

Sentence:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mask_original_sentence_in_text(text: Any, source_text: str) -> Any:
    if not isinstance(text, str) or not source_text:
        return text
    out = text
    for variant in {source_text, norm_text(source_text)}:
        if variant:
            out = re.sub(re.escape(variant), MASK, out, flags=re.I)
    return out


def mask_original_sentence_in_obj(obj: Any, source_text: str) -> Any:
    if isinstance(obj, str):
        return mask_original_sentence_in_text(obj, source_text)
    if isinstance(obj, list):
        return [mask_original_sentence_in_obj(x, source_text) for x in obj]
    if isinstance(obj, dict):
        return {k: mask_original_sentence_in_obj(v, source_text) for k, v in obj.items()}
    return obj


def build_backward_packet(parsed_forward: dict[str, Any], source_text: str, evidence_mode: str) -> dict[str, Any]:
    masked = mask_original_sentence_in_obj(copy.deepcopy(parsed_forward), source_text)
    return {"sentence_decision": masked.get("sentence_decision", ""), "provisions": masked.get("provisions", []), "unmatched_language": masked.get("unmatched_language", []), "schema_coverage_notes": masked.get("schema_coverage_notes", ""), "evidence_mode": evidence_mode, "sanitization_note": "Original full sentence and raw forward response are not included; exact full-sentence echoes are masked."}


def build_backward_messages(backward_packet: dict[str, Any], schema: str) -> list[dict[str, str]]:
    system = "You reconstruct informed-consent sentence meaning from a reduced functional schema. You do not see the original sentence. Return valid JSON only."
    mapping_text = json.dumps(backward_packet, ensure_ascii=False, indent=2)
    user = f"""
Task: reconstruct one concise natural-language consent sentence that preserves the meaning of the reduced V1 mapping.

Critical leakage rule:
- The original sentence is intentionally not provided.
- Use only the reduced V1 mapping below.
- Do not add details not supported by the mapping.

Reconstruction rules:
- Preserve decision/rule type, entities/participants, governed data/specimens/resources, actions, purpose, conditions, exceptions/restrictions, privacy/identifiability, temporal scope, choice, withdrawal/lifecycle effects, and result/risk meaning when present.
- Include residual_important_content and unmatched_language when needed for meaning preservation.
- Do not reconstruct by listing field names.
- Write a natural consent sentence.

Reduced V1 schema:
{schema}

Sanitized reduced V1 mapping for reconstruction:
{mapping_text}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_done(path: Path, key_field: str = "source_id") -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if key_field in obj:
                    done.add(str(obj[key_field]))
            except Exception:
                continue
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[str(obj["source_id"])] = obj
    return out


def count_role_entries(parsed: dict[str, Any], field_names: list[str]) -> tuple[int, int]:
    n = 0
    roles = set()
    fields = ["decision"] + field_names + ["residual_important_content"]
    for prov in parsed.get("provisions") or []:
        if not isinstance(prov, dict):
            continue
        for role in fields:
            val = prov.get(role)
            if role == "decision":
                if isinstance(val, dict) and (norm_text(val.get("value")) or norm_text(val.get("evidence_span_text"))):
                    n += 1
                    roles.add(role)
                continue
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and any(norm_text(v) for v in item.values()):
                        n += 1
                        roles.add(role)
                    elif norm_text(item):
                        n += 1
                        roles.add(role)
            elif norm_text(val):
                n += 1
                roles.add(role)
    return n, len(roles)


def write_roundtrip_csv(forward_path: Path, backward_path: Path, out_csv: Path, evidence_mode: str, field_names: list[str]) -> None:
    forward = load_jsonl_by_id(forward_path)
    backward = load_jsonl_by_id(backward_path)
    rows = []
    for sid, fwd in forward.items():
        parsed = fwd.get("parsed_forward") or {}
        bwd = backward.get(sid, {})
        n_entries, n_roles = count_role_entries(parsed if isinstance(parsed, dict) else {}, field_names)
        rows.append({"source_id": sid, "source_text": fwd.get("source_text", ""), "evidence_mode": evidence_mode, "schema_fields_json": json.dumps(field_names, ensure_ascii=False), "sentence_decision": parsed.get("sentence_decision", "") if isinstance(parsed, dict) else "", "n_role_entries": n_entries, "n_unique_roles": n_roles, "forward_parse_ok": fwd.get("parse_ok", False), "backward_parse_ok": bwd.get("parse_ok", False), "reconstructed_sentence": (bwd.get("parsed_backward") or {}).get("reconstructed_sentence", ""), "v1_mapping_json": json.dumps(parsed, ensure_ascii=False), "backward_packet_json": json.dumps(bwd.get("backward_packet", {}), ensure_ascii=False), "forward_raw": fwd.get("raw_response", ""), "backward_raw": bwd.get("raw_response", "")})
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_forward(rows: pd.DataFrame, client: Any, model_cfg: dict[str, Any], schema: str, field_names: list[str], evidence_mode: str, max_evidence_tokens: int, out_dir: Path) -> None:
    forward_path = out_dir / "reduced_v1_forward_mappings.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    done = read_done(forward_path)
    for i, row in rows.iterrows():
        sid = str(row["_source_id"])
        if sid in done:
            continue
        sent = str(row["_source_text"])
        try:
            raw = call_chat(client, model_cfg, build_forward_messages(sent, schema, field_names, evidence_mode, max_evidence_tokens))
            parsed = extract_json(raw)
            append_jsonl(forward_path, {"source_id": sid, "source_text": sent, "model_key": model_cfg["model_key"], "model": model_cfg.get("model", model_cfg["model_key"]), "condition": f"reduced_v1_{evidence_mode}", "evidence_mode": evidence_mode, "stage": "forward", "parse_ok": True, "parsed_forward": parsed, "raw_response": raw})
            done.add(sid)
            n_entries, n_roles = count_role_entries(parsed, field_names)
            print(f"[V1 {evidence_mode} forward] {i + 1}/{len(rows)} ok {sid} role_entries={n_entries} roles={n_roles}")
        except Exception as exc:
            append_jsonl(failures_path, {"stage": "forward", "source_id": sid, "source_text": sent, "error": repr(exc), "evidence_mode": evidence_mode})
            print(f"[V1 {evidence_mode} forward] {i + 1}/{len(rows)} FAILED {sid}: {exc}")


def run_backward(rows: pd.DataFrame, client: Any, model_cfg: dict[str, Any], schema: str, evidence_mode: str, out_dir: Path) -> None:
    forward_path = out_dir / "reduced_v1_forward_mappings.jsonl"
    backward_path = out_dir / "reduced_v1_backward_reconstructions.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    forward = load_jsonl_by_id(forward_path)
    done = read_done(backward_path)
    for i, row in rows.iterrows():
        sid = str(row["_source_id"])
        if sid in done:
            continue
        fwd = forward.get(sid)
        if not fwd:
            continue
        source_text = fwd.get("source_text", "")
        try:
            parsed_forward = fwd.get("parsed_forward") or extract_json(fwd.get("raw_response", ""))
            packet = build_backward_packet(parsed_forward, source_text, evidence_mode)
            raw = call_chat(client, model_cfg, build_backward_messages(packet, schema))
            parsed_back = extract_json(raw)
            append_jsonl(backward_path, {"source_id": sid, "source_text": source_text, "model_key": model_cfg["model_key"], "model": model_cfg.get("model", model_cfg["model_key"]), "condition": f"reduced_v1_{evidence_mode}", "evidence_mode": evidence_mode, "stage": "backward", "parse_ok": True, "parsed_backward": parsed_back, "backward_packet": packet, "raw_response": raw})
            done.add(sid)
            print(f"[V1 {evidence_mode} backward] {i + 1}/{len(rows)} ok {sid}")
        except Exception as exc:
            append_jsonl(failures_path, {"stage": "backward", "source_id": sid, "error": repr(exc), "evidence_mode": evidence_mode})
            print(f"[V1 {evidence_mode} backward] {i + 1}/{len(rows)} FAILED {sid}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--metamodel_yaml", required=True)
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--evidence_mode", choices=["compact", "permissive"], default="compact")
    ap.add_argument("--max_evidence_tokens", type=int, default=7)
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()
    rows = load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    model_cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(model_cfg)
    _, schema, field_names = load_schema(Path(args.metamodel_yaml))
    out_dir = Path(args.output_dir) / args.model_key / args.evidence_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"forward", "both"}:
        run_forward(rows, client, model_cfg, schema, field_names, args.evidence_mode, args.max_evidence_tokens, out_dir)
    if args.stage in {"backward", "both"}:
        run_backward(rows, client, model_cfg, schema, args.evidence_mode, out_dir)
    write_roundtrip_csv(out_dir / "reduced_v1_forward_mappings.jsonl", out_dir / "reduced_v1_backward_reconstructions.jsonl", out_dir / "reduced_v1_roundtrip_outputs.csv", args.evidence_mode, field_names)
    print(f"Wrote V1 outputs under {out_dir}")


if __name__ == "__main__":
    main()
