#!/usr/bin/env python
"""Run individual source-model prompt round-trip experiments.

Replication condition: new LLMs + original individual source-model prompts.

The forward step uses the original source-model forward prompt text. The
backward step uses a matching backward prompt if supplied; otherwise it uses a
generic reconstruction prompt that does not see the original sentence.

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
    model_cfg = {**(cfg.get("defaults", {}) or {}), **(cfg.get("models", {}) or {}).get(model_key, {})}
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
    kwargs: dict[str, Any] = {
        "model": model_cfg["model"],
        "messages": messages,
        "max_tokens": int(model_cfg.get("max_tokens", 4096)),
        "timeout": float(model_cfg.get("timeout_seconds", 120)),
    }
    if model_cfg.get("temperature") is not None:
        kwargs["temperature"] = model_cfg.get("temperature", 0)
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    last_exc: Exception | None = None
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
    files = [p for p in prompt_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    pats = PROMPT_PATTERNS[info_model]
    matches = []
    for p in files:
        name = p.name.lower()
        if any(re.search(pat, name) for pat in pats) and "forward" in name:
            matches.append(p)
    if not matches:
        for p in files:
            name = p.name.lower()
            if any(re.search(pat, name) for pat in pats):
                matches.append(p)
    if not matches:
        raise FileNotFoundError(f"No prompt file found for {info_model} in {prompt_dir}")
    return sorted(matches, key=lambda x: len(x.name))[0]


def find_backward_prompt_file(prompt_dir: Path | None, info_model: str) -> Path | None:
    if prompt_dir is None or not prompt_dir.exists():
        return None
    files = [p for p in prompt_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    pats = PROMPT_PATTERNS[info_model]
    matches = []
    for p in files:
        name = p.name.lower()
        if any(re.search(pat, name) for pat in pats) and ("backward" in name or "reconstruct" in name):
            matches.append(p)
    return sorted(matches, key=lambda x: len(x.name))[0] if matches else None


def build_forward_messages(prompt_text: str, sentence: str) -> list[dict[str, str]]:
    system = "You are an NLP annotator for informed-consent documents. Follow the provided data dictionary prompt exactly."
    user = f"""{prompt_text.strip()}

Now apply the above prompt to this sentence.

Sentence:
{sentence}

Return only the annotation output requested by the prompt."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_backward_messages(info_model: str, forward_output: str, backward_prompt_text: str | None) -> list[dict[str, str]]:
    system = "You reconstruct informed-consent sentence meaning from structured annotations. Do not see the original sentence."
    if backward_prompt_text:
        user = f"""{backward_prompt_text.strip()}

Use this forward mapping as input. Do not use the original sentence.

Forward mapping:
{forward_output}

Return the reconstructed sentence only, unless the prompt requires a specific output format."""
    else:
        user = f"""Task: reconstruct a concise natural-language informed-consent sentence from the {info_model} forward mapping below.

Rules:
- Do not add details that are not present in the mapping.
- Preserve permission/denial, action, data/specimen object, actor/recipient, purpose, condition, restriction, and temporal meaning when present.
- Do not see or infer from the original sentence.
- Return valid JSON only with this structure:
{{"reconstructed_sentence": "...", "reconstruction_notes": "brief note or empty string"}}

Forward mapping:
{forward_output}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


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
                done.add(str(obj["source_id"]))
            except Exception:
                continue
    return done


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
    forward = load_jsonl_by_id(forward_path)
    backward = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, fwd in forward.items():
        bwd = backward.get(source_id, {})
        rows.append({
            "source_id": source_id,
            "source_text": fwd.get("source_text", ""),
            "info_model": fwd.get("info_model", ""),
            "forward_raw": fwd.get("raw_response", ""),
            "backward_raw": bwd.get("raw_response", ""),
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
                raw = call_chat(client, model_cfg, build_backward_messages(info_model, fwd.get("raw_response", ""), backward_prompt_text))
                append_jsonl(backward_path, {
                    "source_id": source_id,
                    "source_text": fwd.get("source_text", ""),
                    "model_key": model_cfg["model_key"],
                    "model": model_cfg["model"],
                    "info_model": info_model,
                    "stage": "backward",
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
    }, indent=2))

    for info_model in info_models:
        prompt_path = find_prompt_file(prompt_dir, info_model)
        backward_path = find_backward_prompt_file(backward_dir, info_model)
        prompt_text = prompt_path.read_text(errors="replace")
        backward_text = backward_path.read_text(errors="replace") if backward_path else None
        out_dir = base_out / info_model
        (out_dir / "prompt_files.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt_files.json").write_text(json.dumps({
            "forward_prompt_file": str(prompt_path),
            "backward_prompt_file": str(backward_path) if backward_path else None,
            "uses_generic_backward_prompt": backward_path is None,
        }, indent=2))
        run_info_model(rows, client, model_cfg, info_model, prompt_text, backward_text, out_dir, args.stage)

    print(f"Wrote individual-model outputs under {base_out}")


if __name__ == "__main__":
    main()
