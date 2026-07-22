#!/usr/bin/env python
"""Run individual source-model prompt round-trip experiments.

Replication condition: new LLMs + original individual source-model prompts.

Forward uses the original source-model forward prompt text and can be reused from
previous runs. Backward reconstruction is now strict and universal across all
experiments: it receives only valid span-level annotations plus sentence-level
annotations only when at least one valid span annotation exists. It does not
receive unmatched/residual language, interpretation units, rationales, raw
forward text, previous reconstructions, or the original sentence.

Rows with no backward-eligible annotations are not sent to the LLM; their
reconstruction is intentionally blank so residual-only mappings cannot receive
inflated meaning-preservation scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. Install with: pip install openai") from exc

TEXT_COL_CANDIDATES = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "id"]
INFO_MODELS = ["DUO", "ICO", "ODRL", "FHIR_Consent"]
PROMPT_PATTERNS = {"DUO": [r"duo"], "ICO": [r"ico"], "ODRL": [r"odrl"], "FHIR_Consent": [r"fhir", r"r03_fhir"]}
SPAN_KEYS = ["span_text", "evidence_span_text", "evidence_text", "text_span", "phrase", "text", "span", "value", "verbatim"]
LABEL_KEYS = ["field_name", "field_id", "label", "element", "element_id", "source_element_id", "union_element_id", "node", "term", "class", "category", "path", "role", "type", "id"]
DECISION_KEYS = ["decision", "polarity", "consent_force", "permission", "rule_type", "value"]
DECISION_VALUES = {"permit", "deny", "denied", "prohibit", "prohibition", "permission", "obligation", "mixed", "unclear", "allow", "allowed"}
MASK = "[ORIGINAL_SENTENCE_REMOVED]"
STRICT_POLICY = "strict_annotation_only_no_unmatched_language_no_interpretation_units_no_rationales"
NO_ANNOTATION_NOTE = "No valid backward-eligible annotations were available; reconstruction intentionally left blank."


def norm_text(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
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


def make_client(model_cfg: dict[str, Any]) -> OpenAI:
    api_key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if not api_key:
        api_key = "EMPTY"
    base_url = model_cfg.get("base_url")
    if base_url in {"", "null", None}:
        return OpenAI(api_key=api_key)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_chat(client: OpenAI, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
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


def find_prompt_file(prompt_dir: Path, info_model: str) -> Path:
    patterns = PROMPT_PATTERNS[info_model]
    files = [p for p in prompt_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    matches = [p for p in files if any(re.search(pattern, p.name.lower()) for pattern in patterns)]
    if not matches:
        raise FileNotFoundError(f"Could not find prompt file for {info_model} in {prompt_dir}")
    return sorted(matches, key=lambda p: ("forward" not in p.name.lower(), len(p.name), p.name.lower()))[0]


def find_backward_prompt_file(backward_dir: Path | None, info_model: str) -> Path | None:
    # Retained for backwards-compatible CLI metadata only. The current backward
    # evaluator always uses the universal strict prompt below.
    if backward_dir is None or not backward_dir.exists():
        return None
    patterns = PROMPT_PATTERNS[info_model]
    files = [p for p in backward_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    matches = [p for p in files if any(re.search(pattern, p.name.lower()) for pattern in patterns)]
    if not matches:
        return None
    return sorted(matches, key=lambda p: ("back" not in p.name.lower(), len(p.name), p.name.lower()))[0]


def build_forward_messages(prompt_text: str, sentence: str) -> list[dict[str, str]]:
    system = "You are an NLP annotator for informed-consent documents. Return only the requested output format."
    user = f"""
Use the following original source-model prompt to annotate the sentence.

Original source-model prompt:
{prompt_text}

Sentence to annotate:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|csv)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json(text: str) -> Any:
    stripped = strip_code_fence(text)
    try:
        return json.loads(stripped)
    except Exception:
        candidates = []
        start_obj, end_obj = stripped.find("{"), stripped.rfind("}")
        start_arr, end_arr = stripped.find("["), stripped.rfind("]")
        if start_obj >= 0 and end_obj > start_obj:
            candidates.append(stripped[start_obj : end_obj + 1])
        if start_arr >= 0 and end_arr > start_arr:
            candidates.append(stripped[start_arr : end_arr + 1])
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                pass
    raise ValueError("Could not parse JSON from output")


def parse_csv_like(text: str) -> list[list[str]]:
    stripped = strip_code_fence(text)
    if not stripped or "\n" not in stripped:
        return []
    try:
        rows = []
        for row in csv.reader(io.StringIO(stripped)):
            cells = [norm_text(cell) for cell in row]
            if any(cells):
                rows.append(cells)
        return rows
    except Exception:
        return []


def find_span_bounds(sentence: str, span: Any) -> tuple[int | None, int | None]:
    if not isinstance(span, str) or not span.strip():
        return None, None
    span_norm = norm_text(span)
    idx = sentence.lower().find(span_norm.lower())
    if idx >= 0:
        return idx, idx + len(span_norm)
    pattern = r"\s+".join(re.escape(part) for part in span_norm.split())
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None, None


def is_full_sentence_like(span: Any, source_text: str, threshold: float = 0.85) -> bool:
    s1 = re.sub(r"\W+", " ", norm_text(span).lower()).strip()
    s2 = re.sub(r"\W+", " ", norm_text(source_text).lower()).strip()
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    return len(s1.split()) / max(1, len(s2.split())) >= threshold and (s1 in s2 or s2 in s1)


def first_value(d: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = d.get(key)
        if isinstance(value, (str, int, float)) and norm_text(value):
            return norm_text(value)
    return ""


def get_span_value(d: dict[str, Any]) -> str:
    return first_value(d, SPAN_KEYS)


def get_label_value(d: dict[str, Any], span: str = "") -> str:
    for key in LABEL_KEYS:
        value = norm_text(d.get(key))
        if value and value != span and key not in SPAN_KEYS:
            return value
    return ""


def get_decision_value(d: dict[str, Any]) -> str:
    for key in DECISION_KEYS:
        value = norm_text(d.get(key))
        if value and value.lower() in DECISION_VALUES:
            return value
    return ""


def annotation_from_dict(d: dict[str, Any]) -> dict[str, str] | None:
    span = get_span_value(d)
    label = get_label_value(d, span)
    if span and label:
        return {
            "annotation_id": first_value(d, ["annotation_id", "id"]),
            "span_text": span,
            "label": label,
            "decision_or_polarity": get_decision_value(d),
            "span_relation": first_value(d, ["span_relation", "relationship"]),
            "overlap_group_id": first_value(d, ["overlap_group_id", "group_id"]),
            "parse_source": "json_annotation",
        }
    return None


def collect_json_annotations(obj: Any) -> list[dict[str, str]]:
    annotations: list[dict[str, str]] = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                a = annotation_from_dict(item)
                if a:
                    annotations.append(a)
                if isinstance(item.get("annotations"), list):
                    annotations.extend(collect_json_annotations(item.get("annotations")))
        return annotations
    if isinstance(obj, dict):
        for key in ["annotations", "span_annotations", "mapped_elements", "elements", "results"]:
            if isinstance(obj.get(key), list):
                annotations.extend(collect_json_annotations(obj.get(key)))
        a = annotation_from_dict(obj)
        if a:
            annotations.append(a)
    return annotations


def is_probable_label_or_decision(value: str) -> bool:
    v = norm_text(value).lower()
    if not v:
        return True
    if v in {"full_text", "sentence", "text"} | DECISION_VALUES:
        return True
    if re.fullmatch(r"[A-Za-z_:-]{1,12}", value) and " " not in value:
        return True
    return False


def choose_row_span(cells: list[str], source_text: str) -> tuple[str, int | None, int | None]:
    candidates = []
    for cell in cells:
        if not cell or cell == MASK or is_probable_label_or_decision(cell):
            continue
        start, end = find_span_bounds(source_text, cell)
        if start is None or end is None:
            continue
        if is_full_sentence_like(cell, source_text):
            continue
        candidates.append((cell, start, end))
    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2] - x[1]))
        return candidates[0]
    return "", None, None


def compact_annotations(text: str) -> list[dict[str, str]]:
    out = []
    pattern = re.compile(r"(?P<span>[^\[\]\n]{2,240}?)\s*\[(?P<label>[^\[\]]{1,220})\]\s*(?:\((?P<decision>[^)]{1,120})\))?", re.S)
    for m in pattern.finditer(text):
        span = norm_text(m.group("span")).strip(" ;,.-")
        label = norm_text(m.group("label"))
        decision = norm_text(m.group("decision"))
        if span and label:
            out.append({"annotation_id": "", "span_text": span, "label": label, "decision_or_polarity": decision, "span_relation": "", "overlap_group_id": "", "parse_source": "compact_bracket"})
    return out


def csv_annotations(text: str, source_text: str) -> list[dict[str, str]]:
    out = []
    rows = parse_csv_like(text)
    for row in rows:
        span, _, _ = choose_row_span(row, source_text)
        if not span:
            continue
        labels = [c for c in row if c and c != span and c != MASK and not is_full_sentence_like(c, source_text)]
        labels = [c for c in labels if not c.lower() in {"span", "text", "sentence", "full_text"}]
        label = next((c for c in labels if c.lower() not in DECISION_VALUES), "")
        decision = next((c for c in labels if c.lower() in DECISION_VALUES), "")
        if label:
            out.append({"annotation_id": "", "span_text": span, "label": label, "decision_or_polarity": decision, "span_relation": "", "overlap_group_id": "", "parse_source": "csv_like"})
    return out


def parse_span_annotations(raw_forward: str, source_text: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    annotations: list[dict[str, str]] = []
    parse_mode = ""
    try:
        parsed = extract_json(raw_forward)
        annotations = collect_json_annotations(parsed)
        parse_mode = "json_like"
    except Exception:
        parsed = None
    if not annotations:
        annotations = compact_annotations(raw_forward)
        parse_mode = "compact_bracket" if annotations else parse_mode
    if not annotations:
        annotations = csv_annotations(raw_forward, source_text)
        parse_mode = "csv_like" if annotations else parse_mode
    seen = set()
    kept = []
    dropped_full_sentence = 0
    for i, ann in enumerate(annotations, start=1):
        span = norm_text(ann.get("span_text"))
        label = norm_text(ann.get("label"))
        if not span or not label:
            continue
        if is_full_sentence_like(span, source_text):
            dropped_full_sentence += 1
            continue
        key = (span.lower(), label.lower(), norm_text(ann.get("decision_or_polarity")).lower())
        if key in seen:
            continue
        seen.add(key)
        ann = dict(ann)
        ann["annotation_id"] = norm_text(ann.get("annotation_id")) or f"a{len(kept) + 1}"
        kept.append(ann)
    audit = {"annotation_parse_mode": parse_mode or "none", "n_annotations_parsed": len(annotations), "n_annotations_backward_eligible_strict": len(kept), "n_full_sentence_spans_dropped": dropped_full_sentence}
    return kept, audit


def extract_sentence_level_annotations(raw_forward: str, has_valid_span_annotations: bool) -> list[dict[str, str]]:
    if not has_valid_span_annotations:
        return []
    out: list[dict[str, str]] = []
    try:
        parsed = extract_json(raw_forward)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ["sentence_decision", "decision", "rule_type", "permission", "prohibition"]:
            val = norm_text(parsed.get(key))
            if val:
                out.append({"field": key, "value": val, "support_policy": "included_only_because_valid_span_annotations_exist"})
        elems = parsed.get("sentence_level_elements") or []
        if isinstance(elems, list):
            for item in elems:
                if isinstance(item, dict):
                    d = {k: norm_text(v) for k, v in item.items() if norm_text(v)}
                    if d:
                        d["support_policy"] = "included_only_because_valid_span_annotations_exist"
                        out.append(d)
    return out


def build_sanitized_forward_material(raw_forward: str, source_text: str) -> dict[str, Any]:
    """Build strict annotation-only backward input from an existing forward output."""
    annotations, audit = parse_span_annotations(raw_forward, source_text)
    ordered = []
    for ann in annotations:
        start, end = find_span_bounds(source_text, ann.get("span_text", ""))
        ordered.append({
            "annotation_id": ann.get("annotation_id", ""),
            "span_text": ann.get("span_text", ""),
            "label": ann.get("label", ""),
            "decision_or_polarity": ann.get("decision_or_polarity", ""),
            "span_relation": ann.get("span_relation", ""),
            "overlap_group_id": ann.get("overlap_group_id", ""),
            "span_start": start,
            "span_end": end,
        })
    ordered.sort(key=lambda x: (10**9 if x.get("span_start") is None else int(x.get("span_start")), str(x.get("annotation_id", ""))))
    for i, item in enumerate(ordered, start=1):
        item["sentence_order_index"] = i
    return {
        "backward_input_policy": STRICT_POLICY,
        "ordered_reconstruction_items": ordered,
        "sentence_level_annotations": extract_sentence_level_annotations(raw_forward, bool(ordered)),
        "annotation_audit": {**audit, "excluded_from_backward": ["unmatched_language", "interpretation_units", "combined_meaning", "rationale", "raw_forward_response", "original_sentence"]},
    }


def build_backward_messages(info_model: str, sanitized_material: dict[str, Any], backward_prompt_text: str | None = None) -> list[dict[str, str]]:
    _ = backward_prompt_text  # intentionally ignored: universal backward prompt for all experiments
    system = "You reconstruct informed-consent sentence meaning from annotation evidence only. You do not see the original sentence. Return valid JSON only."
    material_text = json.dumps(sanitized_material, ensure_ascii=False, indent=2)
    user = f"""
Task: reconstruct one concise natural-language consent sentence using ONLY the annotation-only mapping below.

Universal strict backward policy:
- The original sentence is intentionally not provided.
- Use only ordered_reconstruction_items and sentence_level_annotations from the mapping.
- Do not use, infer, or request unmatched/residual language, interpretation units, rationales, combined meanings, raw forward responses, or the original sentence.
- If ordered_reconstruction_items are empty or insufficient, return an empty reconstructed_sentence and explain that the annotations are insufficient.
- Preserve only the meaning supported by the annotation spans and labels.
- Do not add actors, actions, resources, purposes, conditions, restrictions, or temporal details that are not present in the annotation-only mapping.

Information model / label source: {info_model}

Annotation-only mapping for reconstruction:
{material_text}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(str(obj.get("source_id")))
            except Exception:
                pass
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[str(obj["source_id"])] = obj
    return out


def parse_backward_response(raw: str) -> dict[str, str]:
    try:
        obj = extract_json(raw)
        if isinstance(obj, dict):
            return {"reconstructed_sentence": norm_text(obj.get("reconstructed_sentence")), "reconstruction_notes": norm_text(obj.get("reconstruction_notes"))}
    except Exception:
        pass
    return {"reconstructed_sentence": norm_text(raw), "reconstruction_notes": "non_json_backward_response"}


def write_csv(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    fwd = load_jsonl_by_id(forward_path)
    bwd = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, f in fwd.items():
        b = bwd.get(source_id, {})
        packet = b.get("sanitized_forward_material", {})
        items = packet.get("ordered_reconstruction_items") or []
        parsed_back = b.get("parsed_backward") or parse_backward_response(b.get("raw_response", ""))
        rows.append({
            "source_id": source_id,
            "source_text": f.get("source_text", ""),
            "condition": f"individual_{f.get('info_model', '')}_strict",
            "information_model": f.get("info_model", ""),
            "forward_raw": f.get("raw_response", ""),
            "backward_raw": b.get("raw_response", ""),
            "backward_input_sanitized": b.get("backward_input_sanitized", False),
            "backward_input_policy": packet.get("backward_input_policy", STRICT_POLICY),
            "sanitized_forward_material_json": json.dumps(packet, ensure_ascii=False),
            "annotation_count": len(items),
            "unique_element_count": len({norm_text(x.get("label")) for x in items if isinstance(x, dict) and norm_text(x.get("label"))}),
            "n_annotations_backward_eligible": len(items),
            "n_full_sentence_spans_dropped": (packet.get("annotation_audit") or {}).get("n_full_sentence_spans_dropped", ""),
            "annotation_parse_mode": (packet.get("annotation_audit") or {}).get("annotation_parse_mode", ""),
            "forward_parse_ok": bool(items) or bool(f.get("raw_response", "")),
            "backward_parse_ok": b.get("parse_ok", False),
            "reconstructed_sentence": parsed_back.get("reconstructed_sentence", ""),
            "reconstruction_notes": parsed_back.get("reconstruction_notes", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_info_model(rows: pd.DataFrame, client: OpenAI, model_cfg: dict[str, Any], info_model: str, prompt_text: str, backward_prompt_text: str | None, out_dir: Path, stage: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    forward_path = out_dir / "forward_mappings.jsonl"
    backward_path = out_dir / "backward_reconstructions.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"

    if stage in {"forward", "both"}:
        done = read_done(forward_path)
        for i, row in rows.iterrows():
            source_id = str(row["_source_id"])
            if source_id in done:
                continue
            sentence = row["_source_text"]
            try:
                raw = call_chat(client, model_cfg, build_forward_messages(prompt_text, sentence))
                append_jsonl(forward_path, {"source_id": source_id, "source_text": sentence, "model_key": model_cfg["model_key"], "model": model_cfg["model"], "info_model": info_model, "stage": "forward", "raw_response": raw})
                done.add(source_id)
                print(f"[{info_model} forward] {i + 1}/{len(rows)} ok {source_id}")
            except Exception as exc:
                append_jsonl(failures_path, {"source_id": source_id, "info_model": info_model, "stage": "forward", "error": repr(exc)})
                print(f"[{info_model} forward] FAILED {source_id}: {exc}", file=sys.stderr)

    if stage in {"backward", "both"}:
        fwd_by_id = load_jsonl_by_id(forward_path)
        done = read_done(backward_path)
        for i, (source_id, fwd) in enumerate(fwd_by_id.items()):
            if source_id in done:
                continue
            try:
                source_text = fwd.get("source_text", "")
                sanitized_material = build_sanitized_forward_material(fwd.get("raw_response", ""), source_text)
                if not sanitized_material.get("ordered_reconstruction_items"):
                    parsed_back = {"reconstructed_sentence": "", "reconstruction_notes": NO_ANNOTATION_NOTE}
                    raw = json.dumps(parsed_back, ensure_ascii=False)
                else:
                    raw = call_chat(client, model_cfg, build_backward_messages(info_model, sanitized_material, backward_prompt_text))
                    parsed_back = parse_backward_response(raw)
                append_jsonl(backward_path, {"source_id": source_id, "source_text": source_text, "model_key": model_cfg["model_key"], "model": model_cfg["model"], "info_model": info_model, "stage": "backward", "backward_input_sanitized": True, "backward_input_policy": STRICT_POLICY, "sanitized_forward_material": sanitized_material, "parsed_backward": parsed_back, "parse_ok": True, "raw_response": raw})
                done.add(source_id)
                print(f"[{info_model} backward] {i + 1}/{len(fwd_by_id)} ok {source_id} eligible_annotations={len(sanitized_material.get('ordered_reconstruction_items') or [])}")
            except Exception as exc:
                append_jsonl(failures_path, {"source_id": source_id, "info_model": info_model, "stage": "backward", "error": repr(exc)})
                print(f"[{info_model} backward] FAILED {source_id}: {exc}", file=sys.stderr)

    write_csv(forward_path, backward_path, out_dir / "roundtrip_outputs.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--prompt_dir", required=True)
    ap.add_argument("--backward_prompt_dir", default=None, help="Deprecated/ignored for evaluation; strict universal backward prompt is always used.")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--info_models", default="all", help="Comma-separated list or all")
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()

    info_models = INFO_MODELS if args.info_models == "all" else [x.strip() for x in args.info_models.split(",") if x.strip()]
    unknown = [m for m in info_models if m not in INFO_MODELS]
    if unknown:
        raise ValueError(f"Unknown info_models: {unknown}. Allowed: {INFO_MODELS}")

    rows = load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    model_cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(model_cfg)
    prompt_dir = Path(args.prompt_dir)
    backward_dir = Path(args.backward_prompt_dir) if args.backward_prompt_dir else None

    base_out = Path(args.output_dir) / args.model_key
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "run_metadata.json").write_text(json.dumps({
        "model_key": args.model_key,
        "model": model_cfg.get("model"),
        "n_input_rows": int(len(rows)),
        "info_models": info_models,
        "roundtrips_csv": args.roundtrips_csv,
        "prompt_dir": args.prompt_dir,
        "backward_prompt_dir": args.backward_prompt_dir,
        "stage": args.stage,
        "backward_input": STRICT_POLICY,
        "backward_prompt": "universal_strict_annotation_only",
    }, indent=2))

    for info_model in info_models:
        prompt_path = find_prompt_file(prompt_dir, info_model)
        backward_path = find_backward_prompt_file(backward_dir, info_model)
        prompt_text = prompt_path.read_text(errors="replace")
        backward_text = backward_path.read_text(errors="replace") if backward_path else None
        out_dir = base_out / info_model
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt_files.json").write_text(json.dumps({"forward_prompt_file": str(prompt_path), "backward_prompt_file_deprecated_not_used": str(backward_path) if backward_path else None, "uses_universal_strict_backward_prompt": True, "backward_input_policy": STRICT_POLICY}, indent=2))
        run_info_model(rows, client, model_cfg, info_model, prompt_text, backward_text, out_dir, args.stage)

    print(f"Wrote individual-model outputs under {base_out}")


if __name__ == "__main__":
    main()
