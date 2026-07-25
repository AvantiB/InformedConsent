#!/usr/bin/env python
"""Run forward/backward round-trip evaluation for direct LLM-induced schemas.

This runner evaluates a generated reduced schema JSON as an annotation dictionary.
It keeps the corrected annotation-only protocol:

- forward mapping sees the sentence and the schema dictionary;
- backward reconstruction sees only valid annotation spans, static field metadata,
  controlled modifiers, sanitized relationship links, and sentence_decision;
- backward never sees original sentence text, raw forward response, unmatched text,
  rationale, evidence text from relationship units, or combined meanings.

Typical use:

python meta_model/scripts/48_run_direct_schema_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --schema_json meta_model/direct_llm_reduced_schema/outputs/fold_00/high/schema.json \
  --fold_assignments_csv meta_model/direct_llm_reduced_schema/inputs/direct_llm_fold_assignments.csv \
  --fold_id fold_00 \
  --eval_split heldout \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key medgemma \
  --output_dir meta_model/direct_llm_reduced_schema/roundtrip_eval \
  --stage both
"""
from __future__ import annotations

import argparse
import csv
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
FORM_KEY_CANDIDATES = ["form_key", "form_id", "document_id", "source_form", "file_name"]
ID_COL_CANDIDATES = ["sentence_text_id", "sentence_id", "roundtrip_id", "canonical_source_key"]
SENTENCE_DECISIONS = {"permit", "deny", "mixed", "unclear"}
RELATIONSHIP_TYPES = {
    "same_span_multiple_labels",
    "same_span_multiple_fields",
    "nested_broad_narrow",
    "complementary_roles",
    "complementary_fields",
    "single",
    "conflicting_or_uncertain",
}
BANNED_BACKWARD_KEYS = {
    "unmatched_language",
    "evidence_span_text",
    "combined_meaning",
    "backward_mapping_decision",
    "rationale",
    "raw_response",
    "original_sentence",
    "original_text",
    "sentence_text",
}


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available columns={list(df.columns)}")
    return None


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
        "max_tokens": int(cfg.get("max_tokens", 4096)),
        "timeout": float(cfg.get("timeout_seconds", 120)),
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


def load_schema(schema_json: Path) -> dict[str, Any]:
    schema = json.loads(schema_json.read_text())
    rows = schema.get("dictionary_rows") or schema.get("fields") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Schema {schema_json} has no dictionary_rows/fields list")
    seen: set[str] = set()
    for r in rows:
        fid = norm(r.get("field_id"))
        fname = norm(r.get("field_name"))
        if not fid or not fname:
            raise ValueError(f"Every dictionary row must contain field_id and field_name. Bad row={r}")
        if fid in seen:
            raise ValueError(f"Duplicate field_id in schema: {fid}")
        seen.add(fid)
    schema["dictionary_rows"] = rows
    schema.setdefault("modifiers", [])
    schema.setdefault("sentence_decision", {"allowed_values": sorted(SENTENCE_DECISIONS)})
    return schema


def schema_maps(schema: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    fields = {norm(r.get("field_id")): r for r in schema.get("dictionary_rows", [])}
    modifiers = {norm(m.get("modifier_name")): m for m in schema.get("modifiers", []) if norm(m.get("modifier_name"))}
    return fields, modifiers


def field_modifier_allowed(field: dict[str, Any], modifier: str) -> bool:
    allowed = field.get("allowed_modifiers") or []
    if isinstance(allowed, str):
        allowed = [allowed]
    allowed_norm = {norm(x) for x in allowed if norm(x)}
    return not allowed_norm or modifier in allowed_norm


def dictionary_prompt_text(schema: dict[str, Any]) -> str:
    lines = []
    for r in schema["dictionary_rows"]:
        includes = "; ".join(norm(x) for x in (r.get("include") or []) if norm(x))
        excludes = "; ".join(norm(x) for x in (r.get("exclude") or []) if norm(x))
        mods = ", ".join(norm(x) for x in (r.get("allowed_modifiers") or []) if norm(x))
        line = (
            f"{norm(r.get('field_id'))} | {norm(r.get('field_name'))} | "
            f"definition: {norm(r.get('definition'))}"
        )
        if includes:
            line += f" | include: {includes}"
        if excludes:
            line += f" | exclude: {excludes}"
        if mods:
            line += f" | allowed_modifiers: {mods}"
        lines.append(line)
    return "\n".join(lines)


def modifier_prompt_text(schema: dict[str, Any]) -> str:
    lines = []
    for m in schema.get("modifiers", []):
        name = norm(m.get("modifier_name"))
        if not name:
            continue
        vals = ", ".join(norm(v) for v in (m.get("allowed_values") or []) if norm(v))
        applies = ", ".join(norm(v) for v in (m.get("applies_to_fields") or []) if norm(v))
        line = f"{name} | definition: {norm(m.get('definition'))}"
        if vals:
            line += f" | allowed_values: {vals}"
        if applies:
            line += f" | applies_to_fields: {applies}"
        lines.append(line)
    return "\n".join(lines) if lines else "No modifiers defined."


def build_forward_messages(schema: dict[str, Any], sentence: str) -> list[dict[str, str]]:
    dictionary = dictionary_prompt_text(schema)
    modifiers = modifier_prompt_text(schema)
    system = "You annotate informed-consent sentences using a fixed schema dictionary. Return valid JSON only."
    user = f"""
Annotate the sentence using only the reduced schema dictionary rows below.

Rules:
- Return sentence_decision as one of: permit, deny, mixed, unclear.
- Create span annotations only for text supported by a dictionary row.
- Each annotation must copy field_id and field_name exactly from the same dictionary row.
- Use the smallest meaningful contiguous span when possible.
- Multiple fields may be assigned to the same or overlapping spans when supported.
- Attach modifiers only when they are supported by the modifier dictionary and helpful for meaning preservation.
- Do not invent field IDs, field names, modifier names, or modifier values.
- Do not create an annotation when no dictionary row fits.
- Use interpretation_units only to link annotation_ids that should be read together.
- Do not output rationale, unmatched text, residual text, or the original sentence outside the requested JSON.

Dictionary rows:
{dictionary}

Modifier dictionary:
{modifiers}

Sentence:
{sentence}

Return JSON exactly with this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "field_id": "DRS001",
      "field_name": "field_name",
      "modifiers": [{{"modifier_name": "modifier_name", "value": "allowed_value"}}]
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "relationship_type": "same_span_multiple_fields|nested_broad_narrow|complementary_roles|complementary_fields|single|conflicting_or_uncertain",
      "annotation_ids": ["a1", "a2"]
    }}
  ]
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def sanitize_modifiers(raw_mods: Any, field: dict[str, Any], modifier_defs: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    if not isinstance(raw_mods, list):
        return out
    field_name = norm(field.get("field_name"))
    for m in raw_mods:
        if not isinstance(m, dict):
            continue
        name = norm(m.get("modifier_name"))
        value = norm(m.get("value"))
        if not name or name not in modifier_defs or not field_modifier_allowed(field, name):
            continue
        mdef = modifier_defs[name]
        applies = {norm(x) for x in (mdef.get("applies_to_fields") or []) if norm(x)}
        if applies and "all" not in applies and field_name not in applies:
            continue
        allowed_values = {norm(x) for x in (mdef.get("allowed_values") or []) if norm(x)}
        if allowed_values and value not in allowed_values:
            continue
        out.append({"modifier_name": name, "value": value})
    return out


def sanitize_forward(parsed: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    field_defs, modifier_defs = schema_maps(schema)
    decision = norm(parsed.get("sentence_decision")).lower()
    if decision not in SENTENCE_DECISIONS:
        decision = "unclear"
    valid_annotations = []
    invalid_annotations = []
    seen_ids: set[str] = set()
    for i, ann in enumerate(parsed.get("annotations") or []):
        if not isinstance(ann, dict):
            continue
        fid = norm(ann.get("field_id"))
        span = norm(ann.get("span_text"))
        if fid not in field_defs or not span:
            invalid_annotations.append({"reason": "unknown_field_or_empty_span", "field_id": fid, "span_text": span})
            continue
        field = field_defs[fid]
        aid = norm(ann.get("annotation_id")) or f"a{len(valid_annotations) + 1}"
        if aid in seen_ids:
            aid = f"a{len(valid_annotations) + 1}"
        seen_ids.add(aid)
        valid_annotations.append({
            "annotation_id": aid,
            "span_text": span,
            "field_id": fid,
            "field_name": norm(field.get("field_name")),
            "field_definition": norm(field.get("definition")),
            "modifiers": sanitize_modifiers(ann.get("modifiers"), field, modifier_defs),
        })
    valid_ids = {a["annotation_id"] for a in valid_annotations}
    relationship_links = []
    for j, unit in enumerate(parsed.get("interpretation_units") or []):
        if not isinstance(unit, dict):
            continue
        ids = [norm(x) for x in (unit.get("annotation_ids") or []) if norm(x) in valid_ids]
        ids = list(dict.fromkeys(ids))
        if len(ids) < 2:
            continue
        rel = norm(unit.get("relationship_type") or unit.get("relationship") or "complementary_roles")
        if rel not in RELATIONSHIP_TYPES:
            rel = "complementary_roles"
        relationship_links.append({
            "relationship_id": norm(unit.get("unit_id")) or f"r{j + 1}",
            "relationship_type": rel,
            "annotation_ids": ids,
        })
    clean = {
        "sentence_decision": decision,
        "annotations": valid_annotations,
        "relationship_links": relationship_links,
    }
    audit = {
        "n_valid_annotations": len(valid_annotations),
        "n_invalid_annotations": len(invalid_annotations),
        "invalid_annotations": invalid_annotations,
        "n_relationship_links": len(relationship_links),
    }
    return clean, audit


def backward_packet(clean_forward: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    modifier_defs = {
        norm(m.get("modifier_name")): {
            "modifier_name": norm(m.get("modifier_name")),
            "definition": norm(m.get("definition")),
            "allowed_values": [norm(v) for v in (m.get("allowed_values") or []) if norm(v)],
        }
        for m in schema.get("modifiers", [])
        if norm(m.get("modifier_name"))
    }
    used_modifiers = sorted({m["modifier_name"] for a in clean_forward.get("annotations", []) for m in a.get("modifiers", [])})
    return {
        "sentence_decision": clean_forward.get("sentence_decision", "unclear"),
        "annotations": clean_forward.get("annotations", []),
        "relationship_links": clean_forward.get("relationship_links", []),
        "modifier_definitions": [modifier_defs[m] for m in used_modifiers if m in modifier_defs],
    }


def build_backward_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    mapping_text = json.dumps(packet, ensure_ascii=False, indent=2)
    system = "You reconstruct consent language from annotation-only mappings. Return valid JSON only."
    user = f"""
Task: reconstruct one concise natural-language consent sentence using only the annotation-only mapping below.

Instructions:
- Use only information explicitly present in the annotation-only mapping.
- Use field_name and field_definition to interpret annotation fields.
- Use modifiers only as controlled attributes attached to listed annotation spans.
- Use relationship_links only as structural cues for how listed annotations relate to each other.
- Relationship links do not add source wording beyond annotation spans and static field/modifier metadata.
- Use sentence_decision only as a controlled consent-force cue attached to the listed span evidence.
- You may add minimal grammar/function words needed to make the reconstruction readable, but do not add unsupported content.
- If the annotation evidence is empty or insufficient, return an empty reconstructed_sentence and explain that annotation evidence was insufficient.

Relationship link types:
- same_span_multiple_labels: the listed annotations describe the same evidence span using multiple labels.
- same_span_multiple_fields: the listed annotations describe the same evidence span using multiple fields.
- nested_broad_narrow: the listed annotations describe overlapping or nested spans where one is broader and another is narrower.
- complementary_roles: the listed annotations describe different parts of one local meaning unit.
- complementary_fields: the listed annotations describe different fields that should be considered together.
- single: the source forward output marked this as a one-annotation unit.
- conflicting_or_uncertain: the relationship among the listed annotations is uncertain or potentially conflicting.

Annotation-only mapping:
{mapping_text}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_eval_rows(roundtrips_csv: Path, fold_assignments_csv: Path | None, fold_id: str | None, eval_split: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(roundtrips_csv).fillna("")
    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    form_col = pick_col(df, FORM_KEY_CANDIDATES)
    id_col = pick_col(df, ID_COL_CANDIDATES, required=False)
    cols = [form_col, text_col] + ([id_col] if id_col else [])
    extra = [c for c in ["canonical_source_key", "sentence_id", "sentence_text_id"] if c in df.columns and c not in cols]
    ex = df[cols + extra].copy()
    new_cols = ["form_key", "sentence_text"] + (["source_sentence_id"] if id_col else []) + extra
    ex.columns = new_cols
    ex["form_key"] = ex["form_key"].map(norm)
    ex["sentence_text"] = ex["sentence_text"].map(norm)
    ex = ex[(ex["form_key"] != "") & (ex["sentence_text"] != "")].drop_duplicates(subset=["form_key", "sentence_text"]).copy()
    if "source_sentence_id" not in ex.columns:
        ex["source_sentence_id"] = [f"sent_{i:06d}" for i in range(len(ex))]

    if eval_split != "all":
        if not fold_assignments_csv or not fold_id:
            raise ValueError("--fold_assignments_csv and --fold_id are required when --eval_split is heldout or training")
        folds = pd.read_csv(fold_assignments_csv).fillna("")
        if "form_key" not in folds.columns or "fold_id" not in folds.columns:
            raise ValueError("fold_assignments_csv must contain form_key and fold_id columns")
        heldout = set(folds.loc[folds["fold_id"] == fold_id, "form_key"].map(norm))
        if eval_split == "heldout":
            ex = ex[ex["form_key"].isin(heldout)].copy()
        elif eval_split == "training":
            ex = ex[~ex["form_key"].isin(heldout)].copy()
        else:
            raise ValueError(f"Unknown eval_split={eval_split}")

    ex = ex.reset_index(drop=True)
    if limit is not None:
        ex = ex.head(limit).copy()
    ex["eval_row_id"] = [f"row_{i:06d}" for i in range(len(ex))]
    return ex


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def run_forward(rows: pd.DataFrame, schema: dict[str, Any], client: Any, cfg: dict[str, Any], out_dir: Path) -> Path:
    out_path = out_dir / "forward_outputs.jsonl"
    for _, row in rows.iterrows():
        messages = build_forward_messages(schema, row["sentence_text"])
        raw = ""
        parsed: dict[str, Any] = {}
        parse_ok = False
        error = ""
        try:
            raw = call_chat(client, cfg, messages)
            parsed = extract_json(raw)
            parse_ok = True
        except Exception as exc:
            error = str(exc)
        clean, audit = sanitize_forward(parsed if parse_ok else {}, schema)
        append_jsonl(out_path, {
            "eval_row_id": row["eval_row_id"],
            "form_key": row["form_key"],
            "source_sentence_id": row.get("source_sentence_id", ""),
            "sentence_text": row["sentence_text"],
            "forward_parse_ok": parse_ok,
            "forward_error": error,
            "raw_forward_response": raw,
            "parsed_forward": parsed,
            "clean_forward": clean,
            "annotation_audit": audit,
        })
        print(f"forward {row['eval_row_id']} parse_ok={parse_ok} valid={audit['n_valid_annotations']}", flush=True)
    return out_path


def run_backward(schema: dict[str, Any], client: Any, cfg: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    fwd_path = out_dir / "forward_outputs.jsonl"
    bwd_path = out_dir / "backward_outputs.jsonl"
    csv_path = out_dir / "roundtrip_rows.csv"
    rows = read_jsonl(fwd_path)
    csv_rows = []
    for rec in rows:
        clean = rec.get("clean_forward") or {}
        packet = backward_packet(clean, schema)
        raw = ""
        parsed: dict[str, Any] = {}
        parse_ok = False
        error = ""
        if not packet.get("annotations"):
            parsed = {
                "reconstructed_sentence": "",
                "reconstruction_notes": "annotation evidence was insufficient",
            }
            parse_ok = True
        else:
            try:
                raw = call_chat(client, cfg, build_backward_messages(packet))
                parsed = extract_json(raw)
                parse_ok = True
            except Exception as exc:
                error = str(exc)
        reconstructed = norm(parsed.get("reconstructed_sentence")) if parse_ok else ""
        notes = norm(parsed.get("reconstruction_notes")) if parse_ok else ""
        append_jsonl(bwd_path, {
            "eval_row_id": rec["eval_row_id"],
            "form_key": rec["form_key"],
            "source_sentence_id": rec.get("source_sentence_id", ""),
            "backward_parse_ok": parse_ok,
            "backward_error": error,
            "backward_packet": packet,
            "raw_backward_response": raw,
            "parsed_backward": parsed,
            "reconstructed_sentence": reconstructed,
            "reconstruction_notes": notes,
        })
        csv_rows.append({
            "eval_row_id": rec["eval_row_id"],
            "form_key": rec["form_key"],
            "source_sentence_id": rec.get("source_sentence_id", ""),
            "original_sentence": rec["sentence_text"],
            "reconstructed_sentence": reconstructed,
            "schema_id": schema.get("schema_id", ""),
            "schema_fold_id": schema.get("fold_id", ""),
            "schema_granularity": schema.get("granularity", ""),
            "llm": cfg.get("model_key", ""),
            "information_model": "direct_llm_reduced_schema",
            "forward_parse_ok": rec.get("forward_parse_ok", False),
            "backward_parse_ok": parse_ok,
            "n_valid_annotations": (rec.get("annotation_audit") or {}).get("n_valid_annotations", 0),
            "n_relationship_links": (rec.get("annotation_audit") or {}).get("n_relationship_links", 0),
            "sentence_decision": (rec.get("clean_forward") or {}).get("sentence_decision", "unclear"),
            "annotations_serialized": json.dumps((rec.get("clean_forward") or {}).get("annotations", []), ensure_ascii=False),
        })
        print(f"backward {rec['eval_row_id']} parse_ok={parse_ok} reconstructed_len={len(reconstructed)}", flush=True)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)
    return bwd_path, csv_path


def write_summary(out_dir: Path, schema: dict[str, Any], rows: pd.DataFrame, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    fwd = read_jsonl(out_dir / "forward_outputs.jsonl")
    bwd = read_jsonl(out_dir / "backward_outputs.jsonl")
    summary = {
        "schema_json": args.schema_json,
        "schema_id": schema.get("schema_id"),
        "schema_fold_id": schema.get("fold_id"),
        "schema_granularity": schema.get("granularity"),
        "model_key": cfg.get("model_key"),
        "eval_split": args.eval_split,
        "fold_id": args.fold_id,
        "n_eval_rows": int(len(rows)),
        "n_forward_rows": len(fwd),
        "n_backward_rows": len(bwd),
        "n_fields": len(schema.get("dictionary_rows", [])),
        "n_modifiers": len(schema.get("modifiers", [])),
        "forward_parse_ok": sum(1 for r in fwd if r.get("forward_parse_ok")),
        "backward_parse_ok": sum(1 for r in bwd if r.get("backward_parse_ok")),
        "zero_valid_annotation_rows": sum(1 for r in fwd if (r.get("annotation_audit") or {}).get("n_valid_annotations", 0) == 0),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--fold_assignments_csv", default=None)
    ap.add_argument("--fold_id", default=None)
    ap.add_argument("--eval_split", choices=["heldout", "training", "all"], default="heldout")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    schema = load_schema(Path(args.schema_json))
    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)
    rows = load_eval_rows(
        Path(args.roundtrips_csv),
        Path(args.fold_assignments_csv) if args.fold_assignments_csv else None,
        args.fold_id,
        args.eval_split,
        args.limit,
    )
    gran = norm(schema.get("granularity")) or "schema"
    fold = norm(schema.get("fold_id")) or (args.fold_id or "fold")
    out_dir = Path(args.output_dir) / str(cfg.get("model_key")) / fold / gran
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schema_used.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.to_csv(out_dir / "eval_rows.csv", index=False)

    if args.stage in {"forward", "both"}:
        run_forward(rows, schema, client, cfg, out_dir)
    if args.stage in {"backward", "both"}:
        run_backward(schema, client, cfg, out_dir)
    write_summary(out_dir, schema, rows, cfg, args)
    print(f"Done. Outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
