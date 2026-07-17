#!/usr/bin/env python
"""Run Union V0 full-dictionary forward/backward round-trip experiments.

This runner is designed for one model at a time. Open-source models can be
served with vLLM's OpenAI-compatible API, while closed-source models can use the
same OpenAI client interface. Outputs are append-only JSONL files so interrupted
runs can be resumed safely.

The forward step intentionally has two layers:
1. raw span annotations, where multiple labels may be assigned to the same,
   similar, overlapping, or nested spans; and
2. interpretation_units, where the LLM explains how those overlapping labels
   should be considered together for backward reconstruction.

The backward step uses interpretation_units as the primary reconstruction input
and raw annotations as supporting evidence. This preserves the fact that the
naive Union V0 inventory is redundant while still forcing an explicit decision
about the final meaning to reconstruct.
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: openai. Install with: pip install openai") from exc

TEXT_COL_CANDIDATES = [
    "canonical_full_text",
    "full_text_original",
    "original_sentence",
    "full_text",
    "sentence",
    "text",
]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "id"]


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")
    return None


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_rows(roundtrips_csv: Path, limit: int | None, no_dedupe_sentences: bool) -> pd.DataFrame:
    df = pd.read_csv(roundtrips_csv)
    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    id_col = pick_col(df, ID_COL_CANDIDATES, required=False)
    out = df.copy()
    out["_source_text"] = out[text_col].map(norm_text)
    if id_col:
        out["_source_id"] = out[id_col].astype(str)
    else:
        out["_source_id"] = out["_source_text"].map(stable_id)
    out = out[out["_source_text"].astype(bool)].copy()

    if not no_dedupe_sentences:
        out = out.drop_duplicates(subset=["_source_text"]).copy()
        out["_source_id"] = out["_source_text"].map(stable_id)

    out = out.reset_index(drop=True)
    if limit is not None:
        out = out.head(limit).copy()
    return out[["_source_id", "_source_text"]]


def load_inventory(inventory_csv: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv).fillna("")
    required = [
        "union_element_id",
        "source_model",
        "source_element_id",
        "source_element_label",
        "source_element_definition",
    ]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    if "element_scope" not in inv.columns:
        inv["element_scope"] = "span"
    return inv


def build_dictionary_text(inv: pd.DataFrame) -> str:
    lines = []
    for _, row in inv.iterrows():
        scope = row.get("element_scope", "span") or "span"
        definition = norm_text(row["source_element_definition"])
        label = norm_text(row["source_element_label"])
        if definition:
            desc = f"{label}: {definition}"
        else:
            desc = label
        lines.append(f"- {row['union_element_id']} [{row['source_model']}; {scope}] {desc}")
    return "\n".join(lines)


def load_model_config(path: Path, model_key: str) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {}) or {}
    models = cfg.get("models", {}) or {}
    if model_key not in models:
        raise KeyError(f"model_key={model_key!r} not found in {path}. Available: {list(models)}")
    model_cfg = {**defaults, **models[model_key]}
    model_cfg["model_key"] = model_key
    return model_cfg


def make_client(model_cfg: dict[str, Any]) -> OpenAI:
    api_key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if not api_key:
        # vLLM accepts any non-empty key by default; OpenAI requires a real key.
        api_key = "EMPTY"
    base_url = model_cfg.get("base_url")
    if base_url in {"", "null", None}:
        return OpenAI(api_key=api_key)
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_chat(client: OpenAI, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    timeout = float(model_cfg.get("timeout_seconds", 120))
    kwargs: dict[str, Any] = {
        "model": model_cfg["model"],
        "messages": messages,
        "max_tokens": int(model_cfg.get("max_tokens", 2200)),
        "timeout": timeout,
    }
    if model_cfg.get("temperature") is not None:
        kwargs["temperature"] = model_cfg.get("temperature", 0)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - depends on external server
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_exc}")


def build_forward_messages(sentence: str, dictionary_text: str) -> list[dict[str, str]]:
    system = (
        "You are an NLP annotator for informed-consent documents. "
        "Your task is to consistently apply the combined source-model data dictionary "
        "to the input sentence. Return valid JSON only. Do not include markdown or explanations outside JSON."
    )
    user = f"""
Task: annotate the informed-consent sentence using ONLY element IDs from the data dictionary below.

Important context:
- This dictionary is a naive union of multiple information models, not a reduced meta-model.
- Several elements may overlap, duplicate, specialize, or complement each other.
- Therefore, the same or similar text span MAY receive more than one label.
- A larger phrase may receive a broader role, while a nested shorter phrase may receive a narrower or more specific role.
- Preserve those overlaps/nesting relationships rather than forcing a single label too early.

Annotation rules:
- Find the smallest meaningful contiguous text span for each concept when possible.
- Assign one best union_element_id per annotation object.
- If the same span maps clearly to multiple source-model elements, output multiple annotation objects with the same span_text and a shared overlap_group_id.
- If a broad phrase and a nested narrower phrase both carry meaning, output both annotations and link them with a shared overlap_group_id.
- Do not invent or rephrase spans.
- Do not annotate language that does not clearly map to a dictionary element.
- Sentence-level elements may be used only in sentence_level_elements, not as span annotations.
- If important language is not captured by the dictionary, include it in unmatched_language.
- sentence_decision must be one of: permit, deny, mixed, unclear.

Interpretation rules for backward mapping:
- After producing raw annotations, create interpretation_units.
- Each interpretation_unit should explain how related annotations should be considered together for reconstruction.
- Use interpretation_units to decide whether overlapping labels are equivalent, complementary, broad/narrow, conflicting, or uncertain.
- Do not merely collapse overlapping labels as redundant. Preserve specificity when a nested or narrower annotation adds meaning.
- If two labels mean essentially the same thing for the sentence, select the interpretation that best preserves meaning and record the redundancy in rationale.
- If one label is broad and another is narrower, preserve both in a combined meaning when both are needed.

Data dictionary:
{dictionary_text}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "sentence_level_elements": [
    {{"union_element_id": "...", "value": "..."}}
  ],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "union_element_id": "...",
      "overlap_group_id": "g1 or null",
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span",
      "rationale": "brief rationale"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "evidence_span_text": "span or phrase represented by this unit",
      "annotation_ids": ["a1", "a2"],
      "relationship": "single|same_span_multiple_labels|nested_broad_narrow|complementary_roles|conflicting_or_uncertain",
      "combined_meaning": "final meaning to preserve for backward reconstruction",
      "backward_mapping_decision": "use_as_core_meaning|use_as_modifier|preserve_broad_and_specific|choose_more_specific|choose_broader|flag_uncertain",
      "rationale": "brief explanation of how the annotations should be considered together"
    }}
  ],
  "unmatched_language": [
    {{"span_text": "exact text span", "reason": "brief reason"}}
  ]
}}

Sentence:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_backward_messages(forward_obj: dict[str, Any], dictionary_text: str) -> list[dict[str, str]]:
    system = (
        "You reconstruct informed-consent sentence meaning from structured annotations. "
        "Do not see the original sentence. Return valid JSON only."
    )
    mapping_text = json.dumps(forward_obj, ensure_ascii=False, indent=2)
    user = f"""
Task: reconstruct a concise natural-language consent sentence that preserves the meaning of the structured mapping.

Use the mapping as follows:
- interpretation_units are the primary source for reconstruction.
- annotations are supporting evidence and should be used to understand exact source-model elements and span relationships.
- Do not reconstruct by simply listing every annotation label.
- If multiple labels refer to the same span and interpretation_units mark them as equivalent, express the shared meaning once.
- If a broader span and a nested narrower span both add meaning, preserve the combined broad+narrow meaning.
- If annotations are complementary, include all complementary meaning needed for preservation.
- If interpretation_units flag uncertainty or conflict, reconstruct the most cautious meaning and mention the uncertainty only if needed.
- Do not add details that are not in the mapping.
- Preserve permission/denial, action, object, actor/recipient, purpose, condition, restriction, and temporal meaning when present.
- Use unmatched_language only when it is needed to preserve meaning.

Data dictionary:
{dictionary_text}

Structured mapping:
{mapping_text}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_done_keys(path: Path, key_field: str = "source_id") -> set[str]:
    done: set[str] = set()
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


def write_roundtrip_csv(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    forward = load_jsonl_by_id(forward_path)
    backward = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, fwd in forward.items():
        parsed_forward = fwd.get("parsed_forward") or {}
        annotations = parsed_forward.get("annotations") or []
        interpretation_units = parsed_forward.get("interpretation_units") or []
        bwd = backward.get(source_id, {})
        rows.append({
            "source_id": source_id,
            "source_text": fwd.get("source_text", ""),
            "sentence_decision": parsed_forward.get("sentence_decision", ""),
            "n_annotations": len(annotations) if isinstance(annotations, list) else "",
            "n_interpretation_units": len(interpretation_units) if isinstance(interpretation_units, list) else "",
            "forward_parse_ok": fwd.get("parse_ok", False),
            "backward_parse_ok": bwd.get("parse_ok", False),
            "reconstructed_sentence": (bwd.get("parsed_backward") or {}).get("reconstructed_sentence", ""),
            "annotations_json": json.dumps(annotations, ensure_ascii=False),
            "interpretation_units_json": json.dumps(interpretation_units, ensure_ascii=False),
            "forward_raw": fwd.get("raw_response", ""),
            "backward_raw": bwd.get("raw_response", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_forward(rows: pd.DataFrame, client: OpenAI, model_cfg: dict[str, Any], dictionary_text: str, out_dir: Path) -> None:
    forward_path = out_dir / "union_v0_forward_mappings.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    done = read_done_keys(forward_path)
    for i, row in rows.iterrows():
        source_id = str(row["_source_id"])
        if source_id in done:
            continue
        sentence = row["_source_text"]
        try:
            raw = call_chat(client, model_cfg, build_forward_messages(sentence, dictionary_text))
            parsed = extract_json(raw)
            rec = {
                "source_id": source_id,
                "source_text": sentence,
                "model_key": model_cfg["model_key"],
                "model": model_cfg["model"],
                "stage": "forward",
                "parse_ok": True,
                "parsed_forward": parsed,
                "raw_response": raw,
            }
            append_jsonl(forward_path, rec)
            done.add(source_id)
            print(f"[forward] {i + 1}/{len(rows)} ok {source_id}")
        except Exception as exc:
            append_jsonl(failures_path, {"source_id": source_id, "stage": "forward", "error": repr(exc)})
            print(f"[forward] {i + 1}/{len(rows)} FAILED {source_id}: {exc}", file=sys.stderr)


def run_backward(client: OpenAI, model_cfg: dict[str, Any], dictionary_text: str, out_dir: Path) -> None:
    forward_path = out_dir / "union_v0_forward_mappings.jsonl"
    backward_path = out_dir / "union_v0_backward_reconstructions.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    forward = load_jsonl_by_id(forward_path)
    done = read_done_keys(backward_path)
    items = list(forward.items())
    for i, (source_id, fwd) in enumerate(items):
        if source_id in done:
            continue
        try:
            forward_obj = fwd.get("parsed_forward") or {}
            raw = call_chat(client, model_cfg, build_backward_messages(forward_obj, dictionary_text))
            parsed = extract_json(raw)
            rec = {
                "source_id": source_id,
                "source_text": fwd.get("source_text", ""),
                "model_key": model_cfg["model_key"],
                "model": model_cfg["model"],
                "stage": "backward",
                "parse_ok": True,
                "parsed_backward": parsed,
                "raw_response": raw,
            }
            append_jsonl(backward_path, rec)
            done.add(source_id)
            print(f"[backward] {i + 1}/{len(items)} ok {source_id}")
        except Exception as exc:
            append_jsonl(failures_path, {"source_id": source_id, "stage": "backward", "error": repr(exc)})
            print(f"[backward] {i + 1}/{len(items)} FAILED {source_id}: {exc}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    inv = load_inventory(Path(args.inventory_csv))
    dictionary_text = build_dictionary_text(inv)
    model_cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(model_cfg)

    run_meta = {
        "model_key": args.model_key,
        "model": model_cfg.get("model"),
        "stage": args.stage,
        "n_input_rows": int(len(rows)),
        "n_union_elements": int(len(inv)),
        "inventory_csv": args.inventory_csv,
        "roundtrips_csv": args.roundtrips_csv,
        "overlap_aware_forward_mapping": True,
        "backward_uses_interpretation_units": True,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    if args.stage in {"forward", "both"}:
        run_forward(rows, client, model_cfg, dictionary_text, output_dir)
    if args.stage in {"backward", "both"}:
        run_backward(client, model_cfg, dictionary_text, output_dir)

    write_roundtrip_csv(
        output_dir / "union_v0_forward_mappings.jsonl",
        output_dir / "union_v0_backward_reconstructions.jsonl",
        output_dir / "union_v0_roundtrip_outputs.csv",
    )
    print(f"Wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
