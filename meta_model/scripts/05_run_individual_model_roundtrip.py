#!/usr/bin/env python
"""Run individual source-model prompt round-trip experiments.

Replication condition: new LLMs + original individual source-model prompts.

The forward step uses the original source-model forward prompt text. The backward
step uses a matching backward prompt if supplied; otherwise it uses a generic
reconstruction prompt. Backward mapping never receives the original sentence; it
receives only sanitized forward output with exact original-sentence echoes masked.
If the forward output is JSON-like, annotation-like lists are ordered by the
position of their span text in the original sentence before reconstruction.

Outputs are append-only JSONL files and are safe to resume.
"""
from __future__ import annotations

import argparse
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
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. Install with: pip install openai") from exc

TEXT_COL_CANDIDATES = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "id"]
INFO_MODELS = ["DUO", "ICO", "ODRL", "FHIR_Consent"]
PROMPT_PATTERNS = {"DUO": [r"duo"], "ICO": [r"ico"], "ODRL": [r"odrl"], "FHIR_Consent": [r"fhir", r"r03_fhir"]}
SPAN_KEYS = ["span_text", "evidence_span_text", "evidence_text", "text_span", "phrase", "text"]
MASK = "[ORIGINAL_SENTENCE_REMOVED]"


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
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
    kwargs = {
        "model": model_cfg["model"],
        "messages": messages,
        "max_tokens": int(model_cfg.get("max_tokens", 4096)),
        "timeout": float(model_cfg.get("timeout_seconds", 120)),
    }
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
    matches = []
    for p in files:
        name = p.name.lower()
        if any(re.search(pattern, name) for pattern in patterns):
            matches.append(p)
    if not matches:
        raise FileNotFoundError(f"Could not find prompt file for {info_model} in {prompt_dir}")
    matches = sorted(matches, key=lambda p: ("forward" not in p.name.lower(), len(p.name), p.name.lower()))
    return matches[0]


def find_backward_prompt_file(backward_dir: Path | None, info_model: str) -> Path | None:
    if backward_dir is None or not backward_dir.exists():
        return None
    patterns = PROMPT_PATTERNS[info_model]
    files = [p for p in backward_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    matches = []
    for p in files:
        name = p.name.lower()
        if any(re.search(pattern, name) for pattern in patterns):
            matches.append(p)
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: ("back" not in p.name.lower(), len(p.name), p.name.lower()))
    return matches[0]


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


def extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except Exception:
        start_obj = stripped.find("{")
        end_obj = stripped.rfind("}")
        start_arr = stripped.find("[")
        end_arr = stripped.rfind("]")
        candidates = []
        if start_obj >= 0 and end_obj > start_obj:
            candidates.append(stripped[start_obj : end_obj + 1])
        if start_arr >= 0 and end_arr > start_arr:
            candidates.append(stripped[start_arr : end_arr + 1])
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                pass
    raise ValueError("Could not parse JSON from forward output")


def mask_original_sentence_in_text(text: Any, source_text: str) -> Any:
    if not isinstance(text, str) or not source_text:
        return text
    out = text
    for variant in {source_text, norm_text(source_text)}:
        if variant:
            out = re.sub(re.escape(variant), MASK, out, flags=re.IGNORECASE)
    return out


def find_span_bounds(sentence: str, span: Any) -> tuple[int | None, int | None]:
    if not isinstance(span, str) or not span.strip():
        return None, None
    span_norm = " ".join(span.split())
    idx = sentence.lower().find(span_norm.lower())
    if idx >= 0:
        return idx, idx + len(span_norm)
    pattern = r"\s+".join(re.escape(part) for part in span_norm.split())
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None, None


def get_span_value(d: dict[str, Any]) -> str:
    for key in SPAN_KEYS:
        value = d.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def order_annotation_like_list(items: list[Any], source_text: str) -> list[Any]:
    if not all(isinstance(x, dict) for x in items):
        return items
    if not any(get_span_value(x) for x in items if isinstance(x, dict)):
        return items
    ordered = []
    for item in items:
        item2 = dict(item)
        span = get_span_value(item2)
        start, end = find_span_bounds(source_text, span)
        item2["span_start"] = start
        item2["span_end"] = end
        ordered.append(item2)

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        start = item.get("span_start")
        end = item.get("span_end")
        if start is None:
            start = 10**9
        if end is None:
            end = start
        return int(start), -int(end - start)

    ordered = sorted(ordered, key=sort_key)
    for i, item in enumerate(ordered, start=1):
        item["sentence_order_index"] = i
    return ordered


def sanitize_json_like_obj(obj: Any, source_text: str) -> Any:
    if isinstance(obj, str):
        return mask_original_sentence_in_text(obj, source_text)
    if isinstance(obj, list):
        cleaned = [sanitize_json_like_obj(x, source_text) for x in obj]
        return order_annotation_like_list(cleaned, source_text)
    if isinstance(obj, dict):
        return {k: sanitize_json_like_obj(v, source_text) for k, v in obj.items()}
    return obj


def build_sanitized_forward_material(raw_forward: str, source_text: str) -> dict[str, Any]:
    """Build the only forward-derived object allowed in the backward prompt."""
    try:
        parsed = extract_json(raw_forward)
        sanitized = sanitize_json_like_obj(parsed, source_text)
        return {
            "forward_output_format": "json_like",
            "sanitized_forward_output": sanitized,
            "sanitization_note": (
                "Original full sentence was removed if echoed. Annotation-like lists were sorted by span position when possible."
            ),
        }
    except Exception:
        sanitized_text = mask_original_sentence_in_text(raw_forward, source_text)
        return {
            "forward_output_format": "raw_text_sanitized",
            "sanitized_forward_output": sanitized_text,
            "sanitization_note": "Original full sentence was removed if echoed. Raw text could not be parsed as JSON for ordering.",
        }


def build_backward_messages(info_model: str, sanitized_material: dict[str, Any], backward_prompt_text: str | None) -> list[dict[str, str]]:
    system = (
        "You reconstruct informed-consent sentence meaning from a sanitized forward mapping. "
        "You do not see the original sentence. Return only the requested reconstruction."
    )
    material_text = json.dumps(sanitized_material, ensure_ascii=False, indent=2)
    if backward_prompt_text:
        instructions = f"Original backward prompt/instructions:\n{backward_prompt_text}"
    else:
        instructions = f"""
Generic backward reconstruction instructions for {info_model}:
- Reconstruct one concise natural-language consent sentence that preserves the meaning of the sanitized forward mapping.
- Do not add details that are not present in the mapping.
- Do not ask for or assume access to the original sentence.
- If annotation-like items include sentence_order_index, use that order as the main reconstruction order.
- Preserve permission/denial, action, object, actor/recipient, purpose, condition, restriction, and temporal meaning when present.
""".strip()
    user = f"""
{instructions}

Critical leakage rule:
- The original sentence is intentionally not provided.
- Use only the sanitized forward mapping below.

Sanitized forward mapping:
{material_text}
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


def write_csv(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    fwd = load_jsonl_by_id(forward_path)
    bwd = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, f in fwd.items():
        b = bwd.get(source_id, {})
        rows.append({
            "source_id": source_id,
            "source_text": f.get("source_text", ""),
            "forward_raw": f.get("raw_response", ""),
            "backward_raw": b.get("raw_response", ""),
            "backward_input_sanitized": b.get("backward_input_sanitized", False),
            "sanitized_forward_material_json": json.dumps(b.get("sanitized_forward_material", {}), ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_info_model(
    rows: pd.DataFrame,
    client: OpenAI,
    model_cfg: dict[str, Any],
    info_model: str,
    prompt_text: str,
    backward_prompt_text: str | None,
    out_dir: Path,
    stage: str,
) -> None:
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
                append_jsonl(forward_path, {
                    "source_id": source_id,
                    "source_text": sentence,
                    "model_key": model_cfg["model_key"],
                    "model": model_cfg["model"],
                    "info_model": info_model,
                    "stage": "forward",
                    "raw_response": raw,
                })
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
                raw = call_chat(client, model_cfg, build_backward_messages(info_model, sanitized_material, backward_prompt_text))
                append_jsonl(backward_path, {
                    "source_id": source_id,
                    "source_text": source_text,
                    "model_key": model_cfg["model_key"],
                    "model": model_cfg["model"],
                    "info_model": info_model,
                    "stage": "backward",
                    "backward_input_sanitized": True,
                    "sanitized_forward_material": sanitized_material,
                    "raw_response": raw,
                })
                done.add(source_id)
                print(f"[{info_model} backward] {i + 1}/{len(fwd_by_id)} ok {source_id}")
            except Exception as exc:
                append_jsonl(failures_path, {"source_id": source_id, "info_model": info_model, "stage": "backward", "error": repr(exc)})
                print(f"[{info_model} backward] FAILED {source_id}: {exc}", file=sys.stderr)

    write_csv(forward_path, backward_path, out_dir / "roundtrip_outputs.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--prompt_dir", required=True)
    ap.add_argument("--backward_prompt_dir", default=None)
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
        "backward_input": "sanitized_forward_output_no_original_sentence_exact_sentence_echoes_masked_annotation_like_lists_ordered_when_parseable",
    }, indent=2))

    for info_model in info_models:
        prompt_path = find_prompt_file(prompt_dir, info_model)
        backward_path = find_backward_prompt_file(backward_dir, info_model)
        prompt_text = prompt_path.read_text(errors="replace")
        backward_text = backward_path.read_text(errors="replace") if backward_path else None
        out_dir = base_out / info_model
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt_files.json").write_text(json.dumps({
            "forward_prompt_file": str(prompt_path),
            "backward_prompt_file": str(backward_path) if backward_path else None,
            "uses_generic_backward_prompt": backward_path is None,
            "backward_input_sanitized": True,
        }, indent=2))
        run_info_model(rows, client, model_cfg, info_model, prompt_text, backward_text, out_dir, args.stage)

    print(f"Wrote individual-model outputs under {base_out}")


if __name__ == "__main__":
    main()
