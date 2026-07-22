#!/usr/bin/env python
"""Run forward/backward round-trip assessment using a functional schema.

Forward mapping may include rich audit material for human review. Backward
evaluation is strict and universal: it receives only an annotation-only mapping
built from valid span annotations, annotation-attached modifiers when present,
and eligible sentence-level annotations. Rows with no backward-eligible
annotations are not sent to the LLM; their reconstruction is intentionally blank.
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
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

TEXT_COLS = ["canonical_full_text", "full_text_original", "original_sentence", "source_text", "full_text", "sentence", "text"]
ID_COLS = ["sentence_id", "source_sentence_id", "roundtrip_id", "source_id", "id"]
STRICT_POLICY = "strict_annotation_only"
NO_ANNOTATION_NOTE = "Annotation evidence was empty or insufficient."

UNIVERSAL_BACKWARD_SYSTEM = (
    "You reconstruct informed-consent sentence meaning from an annotation-only mapping. "
    "Return valid JSON only."
)
UNIVERSAL_BACKWARD_USER_TEMPLATE = """
Task: reconstruct one concise natural-language consent sentence using only the annotation-only mapping below.

Instructions:
- Use only information explicitly present in the annotation-only mapping.
- Preserve the order indicated by sentence_order_index when available.
- If the annotation evidence is empty or insufficient, return an empty reconstructed_sentence and explain that annotation evidence was insufficient.

Annotation-only mapping:
{mapping_text}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()


def norm(x: Any) -> str:
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


def pick(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Missing required column from {candidates}; available={list(df.columns)}")
    return None


def load_rows(path: Path, limit: int | None, no_dedupe: bool) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    tc = pick(df, TEXT_COLS, required=True)
    ic = pick(df, ID_COLS, required=False)
    out = df.copy()
    out["_source_text"] = out[tc].map(norm)
    out["_source_id"] = out[ic].astype(str) if ic else out["_source_text"].map(stable_id)
    out = out[out["_source_text"].astype(bool)].copy()
    if not no_dedupe:
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
    kwargs = {"model": cfg["model"], "messages": messages, "max_tokens": int(cfg.get("max_tokens", 4096)), "timeout": float(cfg.get("timeout_seconds", 120))}
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg.get("temperature", 0)
    last = None
    for attempt in range(1, int(cfg.get("max_retries", 3)) + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover
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


def as_list(x: Any) -> list[str]:
    if isinstance(x, list):
        return [norm(v) for v in x if norm(v)]
    s = norm(x)
    return [s] if s else []


def load_field_dictionary(schema_path: Path) -> tuple[str, list[str]]:
    data = yaml.safe_load(schema_path.read_text())
    fields = data.get("fields", []) or []
    names: list[str] = []
    lines = ["Functional field dictionary:", "Use only these field names for span-level annotations.", "sentence_decision is provision-level only and must not be used as a span label.", ""]
    for f in fields:
        name = norm(f.get("name"))
        if not name or name in {"residual_important_content", "provenance"}:
            continue
        names.append(name)
        lines.append(f"- {name}")
        if norm(f.get("definition")):
            lines.append(f"  definition: {norm(f.get('definition'))}")
        include = as_list(f.get("include"))
        if include:
            lines.append(f"  include examples: {', '.join(include[:10])}")
        exclude = as_list(f.get("exclude"))
        if exclude:
            lines.append(f"  exclude/avoid: {', '.join(exclude[:8])}")
        lines.append("")
    if not names:
        raise ValueError(f"No functional fields found in {schema_path}")
    return "\n".join(lines), names


def evidence_rules(mode: str, max_tokens: int) -> str:
    if mode == "compact":
        return f"""Evidence-span rules:
- Prefer short atomic evidence spans, ideally <= {max_tokens} tokens.
- Do not copy the full sentence into one annotation.
- Split a phrase when it contains multiple functions, e.g. action + resource + repository.
- Use residual_important_content only for forward audit; it will not be included in the annotation-only backward mapping."""
    return """Evidence-span rules:
- Evidence spans may be longer when needed to preserve condition, exception, temporal, privacy, or governance meaning.
- Do not copy the full sentence verbatim into a single annotation.
- Split complementary functions whenever possible.
- Use residual_important_content only for forward audit; it will not be included in the annotation-only backward mapping."""


def forward_messages(sentence: str, dictionary: str, field_names: list[str], mode: str, max_tokens: int) -> list[dict[str, str]]:
    allowed = ", ".join(field_names + ["residual_important_content"])
    system = "You are an NLP annotator for informed-consent documents. Apply the supplied functional schema. Return valid JSON only."
    user = f"""
Task: annotate the informed-consent sentence using ONLY the functional fields below.

Core principles:
- sentence_decision is a sentence/provision-level label only: permit, deny, obligation, mixed, or unclear.
- Do not annotate individual spans as permit, deny, mixed, or unclear.
- Prefer atomic, mostly non-overlapping spans. If a phrase contains multiple functions, split it.
- Multi-labeling the same span is allowed only when one exact span truly expresses multiple functions.
- Modifiers such as negation, permission, prohibition, obligation, condition, and uncertainty should be attached to the relevant annotation when possible, not represented as standalone schema fields unless the schema explicitly provides such a field.
- Use residual_important_content only for audit; it will not be used for backward reconstruction.

Allowed span-level fields:
{allowed}

{evidence_rules(mode, max_tokens)}

{dictionary}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|obligation|mixed|unclear",
  "sentence_level_elements": [
    {{
      "element_type": "decision|scope_note|other",
      "value": "permit|deny|obligation|mixed|unclear|other",
      "supported_annotation_ids": ["a1"],
      "rationale": "brief rationale"
    }}
  ],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "field_name": "one allowed field name",
      "modifiers": [{{"modifier_type": "consent_force|linguistic_polarity|condition|temporality|uncertainty|other", "value": "canonical value"}}],
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span",
      "overlap_group_id": "g1 or null",
      "rationale": "brief rationale"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "evidence_span_text": "span or phrase represented by this unit",
      "annotation_ids": ["a1"],
      "relationship": "single|same_span_multiple_fields|nested_broad_narrow|complementary_fields|conflicting_or_uncertain",
      "combined_meaning": "audit only",
      "rationale": "brief explanation"
    }}
  ],
  "unmatched_language": [{{"span_text": "exact text span", "reason": "brief reason"}}]
}}

Sentence:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def span_bounds(sentence: str, span: Any) -> tuple[int | None, int | None]:
    span = norm(span)
    if not span:
        return None, None
    idx = sentence.lower().find(span.lower())
    if idx >= 0:
        return idx, idx + len(span)
    m = re.search(r"\s+".join(re.escape(p) for p in span.split()), sentence, flags=re.I)
    return (m.start(), m.end()) if m else (None, None)


def is_full_sentence_like(span: Any, source: str, threshold: float = 0.85) -> bool:
    s1 = re.sub(r"\W+", " ", norm(span).lower()).strip()
    s2 = re.sub(r"\W+", " ", norm(source).lower()).strip()
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    return len(s1.split()) / max(1, len(s2.split())) >= threshold and (s1 in s2 or s2 in s1)


def ordered_annotations(parsed: dict[str, Any], source: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    dropped_full = 0
    for ann in parsed.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        span = norm(ann.get("span_text"))
        field = norm(ann.get("field_name") or ann.get("field_id") or ann.get("label"))
        if not span or not field:
            continue
        if field in {"residual_important_content", "provenance"}:
            continue
        if is_full_sentence_like(span, source):
            dropped_full += 1
            continue
        start, end = span_bounds(source, span)
        rows.append({
            "annotation_id": norm(ann.get("annotation_id")),
            "span_text": span,
            "label": field,
            "modifiers": ann.get("modifiers", []) if isinstance(ann.get("modifiers", []), list) else [],
            "span_relation": norm(ann.get("span_relation")),
            "overlap_group_id": norm(ann.get("overlap_group_id")),
            "span_start": start,
            "span_end": end,
        })
    rows = sorted(rows, key=lambda x: (10**9 if x.get("span_start") is None else int(x.get("span_start")), 0 if x.get("span_end") is None else -int(x.get("span_end") - (x.get("span_start") or 0)), str(x.get("annotation_id", ""))))
    for i, x in enumerate(rows, start=1):
        x["sentence_order_index"] = i
    return rows, {"n_full_sentence_spans_dropped": dropped_full, "n_annotations_backward_eligible_strict": len(rows)}


def eligible_sentence_level_annotations(parsed: dict[str, Any], has_valid_span_annotations: bool) -> list[dict[str, Any]]:
    if not has_valid_span_annotations:
        return []
    out = []
    if norm(parsed.get("sentence_decision")):
        out.append({"field": "sentence_decision", "value": norm(parsed.get("sentence_decision")), "support": "valid_span_annotations_present"})
    elems = parsed.get("sentence_level_elements") or []
    if isinstance(elems, list):
        for item in elems:
            if isinstance(item, dict):
                d = {k: v for k, v in item.items() if norm(v)}
                if d:
                    d["support"] = "valid_span_annotations_present"
                    out.append(d)
    return out


def backward_packet(parsed: dict[str, Any], source: str, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered, audit = ordered_annotations(parsed, source)
    packet = {
        "backward_input_policy": STRICT_POLICY,
        "ordered_reconstruction_items": ordered,
        "sentence_level_annotations": eligible_sentence_level_annotations(parsed, bool(ordered)),
    }
    audit = {**audit, "evidence_mode": mode, "n_sentence_level_annotations_backward_eligible": len(packet["sentence_level_annotations"])}
    return packet, audit


def backward_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    mapping = json.dumps(packet, ensure_ascii=False, indent=2)
    user = UNIVERSAL_BACKWARD_USER_TEMPLATE.format(mapping_text=mapping)
    return [{"role": "system", "content": UNIVERSAL_BACKWARD_SYSTEM}, {"role": "user", "content": user}]


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            try:
                out.add(str(json.loads(line).get("source_id")))
            except Exception:
                pass
    return out


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[str(obj["source_id"])] = obj
    return out


def count_fields(parsed: Any) -> tuple[int, int]:
    labels = []
    if isinstance(parsed, dict):
        for ann in parsed.get("annotations") or []:
            if isinstance(ann, dict):
                field = norm(ann.get("field_name") or ann.get("field_id") or ann.get("cluster_id") or ann.get("label"))
                if field:
                    labels.append(field)
    return len(labels), len(set(labels))


def write_csv(forward_path: Path, backward_path: Path, out_csv: Path, mode: str) -> None:
    fwd, bwd = by_id(forward_path), by_id(backward_path)
    rows = []
    for source_id, f in fwd.items():
        parsed = f.get("parsed_forward") or {}
        b = bwd.get(source_id, {})
        n, u = count_fields(parsed)
        packet = b.get("backward_packet", {})
        audit = b.get("backward_annotation_audit", {})
        strict_items = packet.get("ordered_reconstruction_items") or []
        rows.append({
            "source_id": source_id,
            "source_text": f.get("source_text", ""),
            "condition": f"functional_v1_{mode}_strict",
            "evidence_mode": mode,
            "sentence_decision": parsed.get("sentence_decision", "") if isinstance(parsed, dict) else "",
            "n_role_entries": n,
            "n_unique_roles": u,
            "annotation_count": len(strict_items),
            "unique_element_count": len({norm(x.get("label")) for x in strict_items if isinstance(x, dict) and norm(x.get("label"))}),
            "n_full_sentence_spans_dropped": audit.get("n_full_sentence_spans_dropped", ""),
            "forward_parse_ok": f.get("parse_ok", False),
            "backward_parse_ok": b.get("parse_ok", False),
            "reconstructed_sentence": (b.get("parsed_backward") or {}).get("reconstructed_sentence", ""),
            "reconstruction_notes": (b.get("parsed_backward") or {}).get("reconstruction_notes", ""),
            "backward_input_policy": packet.get("backward_input_policy", STRICT_POLICY),
            "v1_mapping_json": json.dumps(parsed, ensure_ascii=False),
            "backward_packet_json": json.dumps(packet, ensure_ascii=False),
            "backward_annotation_audit_json": json.dumps(audit, ensure_ascii=False),
            "forward_raw": f.get("raw_response", ""),
            "backward_raw": b.get("raw_response", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_forward(rows: pd.DataFrame, client: Any, cfg: dict[str, Any], dictionary: str, field_names: list[str], mode: str, max_tokens: int, out_dir: Path) -> None:
    path, failures = out_dir / "functional_v1_forward_mappings.jsonl", out_dir / "failed_requests.jsonl"
    done = read_done(path)
    for i, row in rows.iterrows():
        source_id, sent = str(row["_source_id"]), str(row["_source_text"])
        if source_id in done:
            continue
        try:
            raw = call_chat(client, cfg, forward_messages(sent, dictionary, field_names, mode, max_tokens))
            parsed = extract_json(raw)
            append_jsonl(path, {"source_id": source_id, "source_text": sent, "model_key": cfg["model_key"], "model": cfg.get("model", cfg["model_key"]), "condition": f"functional_v1_{mode}", "evidence_mode": mode, "stage": "forward", "parse_ok": True, "parsed_forward": parsed, "raw_response": raw})
            n, u = count_fields(parsed)
            print(f"[Functional V1 {mode} forward] {i+1}/{len(rows)} ok {source_id} annotations={n} fields={u}")
        except Exception as exc:
            append_jsonl(failures, {"stage": "forward", "source_id": source_id, "source_text": sent, "error": repr(exc), "evidence_mode": mode})
            print(f"[Functional V1 {mode} forward] {i+1}/{len(rows)} FAILED {source_id}: {exc}")


def run_backward(rows: pd.DataFrame, client: Any, cfg: dict[str, Any], dictionary: str, mode: str, out_dir: Path) -> None:
    _ = dictionary
    fwd_path = out_dir / "functional_v1_forward_mappings.jsonl"
    bwd_path = out_dir / "functional_v1_backward_reconstructions.jsonl"
    failures = out_dir / "failed_requests.jsonl"
    fwd, done = by_id(fwd_path), read_done(bwd_path)
    for i, row in rows.iterrows():
        source_id = str(row["_source_id"])
        if source_id in done or source_id not in fwd:
            continue
        try:
            f = fwd[source_id]
            source_text = f.get("source_text", "")
            parsed = f.get("parsed_forward") or extract_json(f.get("raw_response", ""))
            packet, audit = backward_packet(parsed, source_text, mode)
            if not packet.get("ordered_reconstruction_items"):
                parsed_back = {"reconstructed_sentence": "", "reconstruction_notes": NO_ANNOTATION_NOTE}
                raw = json.dumps(parsed_back, ensure_ascii=False)
            else:
                raw = call_chat(client, cfg, backward_messages(packet))
                parsed_back = extract_json(raw)
            append_jsonl(bwd_path, {"source_id": source_id, "source_text": source_text, "model_key": cfg["model_key"], "model": cfg.get("model", cfg["model_key"]), "condition": f"functional_v1_{mode}_strict", "evidence_mode": mode, "stage": "backward", "parse_ok": True, "backward_input_policy": STRICT_POLICY, "parsed_backward": parsed_back, "backward_packet": packet, "backward_annotation_audit": audit, "raw_response": raw})
            print(f"[Functional V1 {mode} backward] {i+1}/{len(rows)} ok {source_id} eligible_annotations={len(packet.get('ordered_reconstruction_items') or [])}")
        except Exception as exc:
            append_jsonl(failures, {"stage": "backward", "source_id": source_id, "error": repr(exc), "evidence_mode": mode})
            print(f"[Functional V1 {mode} backward] {i+1}/{len(rows)} FAILED {source_id}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--metamodel_yaml", default="meta_model/schemas/reduced_functional_v1_candidate.yaml")
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
    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)
    dictionary, field_names = load_field_dictionary(Path(args.metamodel_yaml))
    out_dir = Path(args.output_dir) / args.model_key / args.evidence_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps({"model_key": args.model_key, "model": cfg.get("model"), "evidence_mode": args.evidence_mode, "stage": args.stage, "roundtrips_csv": args.roundtrips_csv, "metamodel_yaml": args.metamodel_yaml, "backward_input": STRICT_POLICY, "backward_prompt": "minimal_universal_annotation_only"}, indent=2))

    if args.stage in {"forward", "both"}:
        run_forward(rows, client, cfg, dictionary, field_names, args.evidence_mode, args.max_evidence_tokens, out_dir)
    if args.stage in {"backward", "both"}:
        run_backward(rows, client, cfg, dictionary, args.evidence_mode, out_dir)
    write_csv(out_dir / "functional_v1_forward_mappings.jsonl", out_dir / "functional_v1_backward_reconstructions.jsonl", out_dir / "functional_v1_roundtrip_outputs.csv", args.evidence_mode)
    print(f"Wrote Functional V1 outputs under {out_dir}")


if __name__ == "__main__":
    main()
