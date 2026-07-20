#!/usr/bin/env python
"""Run reduced-schema smoke tests with a V0-style cluster prompt.

This runner is intentionally neutral: it treats discovered semantic clusters as
data-dictionary IDs, not named roles, and keeps the forward output close to the
Union V0 structure: sentence_decision, sentence_level_elements, annotations,
interpretation_units, and unmatched_language.
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, json, os, re, sys, time
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

TEXT_COLS = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
ID_COLS = ["sentence_id", "source_sentence_id", "roundtrip_id", "source_id", "id"]
MASK = "[ORIGINAL_SENTENCE_REMOVED]"


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def sid(text: str) -> str:
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
    df = pd.read_csv(path)
    tc, ic = pick(df, TEXT_COLS), pick(df, ID_COLS, required=False)
    out = df.copy()
    out["_source_text"] = out[tc].map(norm)
    out["_source_id"] = out[ic].astype(str) if ic else out["_source_text"].map(sid)
    out = out[out["_source_text"].astype(bool)].copy()
    if not no_dedupe:
        out = out.drop_duplicates(subset=["_source_text"]).copy()
        out["_source_id"] = out["_source_text"].map(sid)
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
    depth = 0; in_str = False; esc = False
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
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [norm(a) for a in v if norm(a)]
    except Exception:
        pass
    return [s]


def load_cluster_dictionary(schema_path: Path) -> str:
    data = yaml.safe_load(schema_path.read_text())
    fields = data.get("fields", []) or []
    lines = [
        "Cluster dictionary:",
        "Use only these cluster IDs for span-level annotations.",
        "The IDs are provisional dictionary entries. Use member/source elements and evidence-span examples to infer coverage.",
        "",
    ]
    n = 0
    for f in fields:
        name = norm(f.get("name", ""))
        if not name or name in {"decision", "provenance", "residual_important_content"}:
            continue
        n += 1
        ev = f.get("selection_evidence") or {}
        terms = ", ".join(as_list(ev.get("name_suggestion_terms", []))) if isinstance(ev, dict) else ""
        support = as_list(f.get("source_element_support", []))[:12]
        spans = as_list(f.get("positive_span_examples", []))[:12]
        lines.append(f"- {name}")
        if terms:
            lines.append(f"  data-derived terms: {terms}")
        if support:
            lines.append(f"  member/source element examples: {', '.join(support)}")
        if spans:
            lines.append(f"  positive evidence-span examples: {'; '.join(spans)}")
        if isinstance(ev, dict):
            bits = [f"{k}={ev.get(k)}" for k in ["selection_status", "n_positive_source_sentences_max", "n_positive_information_models_max", "n_positive_llms_max"] if norm(ev.get(k))]
            if bits:
                lines.append(f"  empirical support: {', '.join(bits)}")
        lines.append("")
    if n == 0:
        raise ValueError(f"No cluster fields found in {schema_path}")
    return "\n".join(lines)


def evidence_rules(mode: str, max_tokens: int) -> str:
    if mode == "compact":
        return f"""Evidence-span rules:
- Use short evidence phrases, preferably <= {max_tokens} tokens.
- Do not copy the full sentence into any field.
- Prefer the smallest phrase that supports the annotation.
- Put meaning-critical content that does not fit any cluster in unmatched_language."""
    return """Evidence-span rules:
- Evidence spans may be longer when needed to preserve condition, exception, temporal, privacy, repository, or governance meaning.
- Do not copy the full sentence verbatim into a single annotation.
- Put meaning-critical content that does not fit any cluster in unmatched_language."""


def forward_messages(sentence: str, dictionary: str, mode: str, max_tokens: int) -> list[dict[str, str]]:
    system = "You are an NLP annotator for informed-consent documents. Apply the provided cluster dictionary to the input sentence. Return valid JSON only."
    user = f"""
Task: annotate the informed-consent sentence using ONLY cluster IDs from the cluster dictionary below.

Important context:
- Cluster IDs are not role names. Use the member/source element examples and positive evidence-span examples to infer what each cluster covers.
- Several clusters may overlap, duplicate, specialize, or complement each other.
- The same or similar text span MAY receive more than one cluster label when that preserves meaning.
- A larger phrase may receive a broader cluster label, while a nested shorter phrase may receive a narrower or complementary cluster label.

Decision rules:
- sentence_decision is a sentence/provision-level label only and must be one of: permit, deny, mixed, unclear.
- Do not annotate individual spans as permit, deny, mixed, or unclear.
- Decision cue spans such as "agree", "allow", "may", "will not", or "can withdraw" should go in sentence_level_elements as cues supporting the sentence/provision-level decision.
- A cue span may also receive a cluster annotation only if a listed cluster specifically captures that cue's semantic content.

Annotation rules:
- Find the smallest meaningful contiguous text span for each concept when possible.
- Assign one best cluster_id per annotation object.
- Copy cluster_id EXACTLY from the cluster dictionary.
- Do not create new cluster IDs.
- If no cluster fits a meaning-critical phrase, put it in unmatched_language.
- If the same span maps clearly to multiple clusters, output multiple annotation objects with the same span_text and a shared overlap_group_id.
- If a broad phrase and a nested narrower phrase both carry meaning, output both annotations and link them with a shared overlap_group_id.

Interpretation rules for backward mapping:
- After producing raw annotations, create interpretation_units.
- Each interpretation_unit should explain how related annotations should be considered together for reconstruction.
- Do not merely collapse overlapping labels as redundant. Preserve specificity when a nested or narrower annotation adds meaning.
- If annotations are complementary, include all complementary meaning needed for preservation.

{evidence_rules(mode, max_tokens)}

{dictionary}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "sentence_level_elements": [
    {{
      "element_type": "decision_cue",
      "value": "permit|deny|mixed|unclear",
      "cue_span_text": "exact text span",
      "cue_type": "agreement|permission|denial|restriction|withdrawal|other",
      "rationale": "brief rationale"
    }}
  ],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "cluster_id": "semantic_cluster_C001",
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
      "relationship": "single|same_span_multiple_clusters|nested_broad_narrow|complementary_clusters|conflicting_or_uncertain",
      "combined_meaning": "final meaning to preserve for backward reconstruction",
      "backward_mapping_decision": "use_as_core_meaning|use_as_modifier|preserve_broad_and_specific|choose_more_specific|choose_broader|flag_uncertain",
      "rationale": "brief explanation of how the annotations should be considered together"
    }}
  ],
  "unmatched_language": [{{"span_text": "exact text span", "reason": "brief reason"}}]
}}

Sentence:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mask_text(text: Any, source: str) -> Any:
    if not isinstance(text, str) or not source:
        return text
    out = text
    for variant in {source, norm(source)}:
        if variant:
            out = re.sub(re.escape(variant), MASK, out, flags=re.I)
    return out


def mask_obj(obj: Any, source: str) -> Any:
    if isinstance(obj, str):
        return mask_text(obj, source)
    if isinstance(obj, list):
        return [mask_obj(x, source) for x in obj]
    if isinstance(obj, dict):
        return {k: mask_obj(v, source) for k, v in obj.items()}
    return obj


def span_bounds(sentence: str, span: Any) -> tuple[int | None, int | None]:
    span = norm(span)
    if not span:
        return None, None
    idx = sentence.lower().find(span.lower())
    if idx >= 0:
        return idx, idx + len(span)
    m = re.search(r"\s+".join(re.escape(p) for p in span.split()), sentence, flags=re.I)
    return (m.start(), m.end()) if m else (None, None)


def ordered_annotations(parsed: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for ann in parsed.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        x = copy.deepcopy(ann)
        start, end = span_bounds(source, x.get("span_text", ""))
        x["span_start"], x["span_end"] = start, end
        rows.append(x)

    def key(x: dict[str, Any]) -> tuple[int, int, str]:
        start, end = x.get("span_start"), x.get("span_end")
        if start is None:
            start = 10**9
        if end is None:
            end = start
        return (int(start), -int(end - start), str(x.get("annotation_id", "")))

    rows = sorted(rows, key=key)
    for i, x in enumerate(rows, start=1):
        x["sentence_order_index"] = i
    return rows


def backward_packet(parsed: dict[str, Any], source: str, mode: str) -> dict[str, Any]:
    masked = mask_obj(copy.deepcopy(parsed), source)
    packet = {
        "sentence_decision": masked.get("sentence_decision", ""),
        "sentence_level_elements": masked.get("sentence_level_elements", []),
        "ordered_reconstruction_items": ordered_annotations(masked, source),
        "interpretation_units": masked.get("interpretation_units", []),
        "unmatched_language": masked.get("unmatched_language", []),
        "evidence_mode": mode,
        "sanitization_note": "Original full sentence and raw forward response are not included. ordered_reconstruction_items are sorted by span position.",
    }
    return mask_obj(packet, source)


def backward_messages(packet: dict[str, Any], dictionary: str) -> list[dict[str, str]]:
    system = "You reconstruct informed-consent sentence meaning from structured annotations. You do not see the original sentence. Return valid JSON only."
    mapping = json.dumps(packet, ensure_ascii=False, indent=2)
    user = f"""
Task: reconstruct one concise natural-language consent sentence that preserves the meaning of the structured mapping.

Critical leakage rule:
- The original sentence is intentionally not provided.
- Use only the ordered spans, cluster IDs, sentence-level decision cues, interpretation units, and unmatched-language fragments in the structured mapping.
- Do not add details that are not in the mapping.

Use the mapping as follows:
- sentence_decision is the sentence/provision-level decision, not a span-level annotation.
- sentence_level_elements contain cues that support the sentence/provision-level decision.
- ordered_reconstruction_items are cluster-grounded annotations sorted in original span order.
- interpretation_units are the primary source for deciding how overlapping or nested clusters should be combined.
- Do not reconstruct by simply listing every cluster ID.

{dictionary}

Sanitized structured mapping for reconstruction:
{mapping}

Return JSON with exactly this structure:
{{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
                obj = json.loads(line)
                out.add(str(obj.get("source_id")))
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


def count_clusters(parsed: Any) -> tuple[int, int]:
    labels = []
    if isinstance(parsed, dict):
        for ann in parsed.get("annotations") or []:
            if isinstance(ann, dict):
                cid = norm(ann.get("cluster_id") or ann.get("union_element_id") or ann.get("label"))
                if cid:
                    labels.append(cid)
    return len(labels), len(set(labels))


def write_csv(forward_path: Path, backward_path: Path, out_csv: Path, mode: str) -> None:
    fwd, bwd = by_id(forward_path), by_id(backward_path)
    rows = []
    for source_id, f in fwd.items():
        parsed = f.get("parsed_forward") or {}
        b = bwd.get(source_id, {})
        n, u = count_clusters(parsed)
        rows.append({
            "source_id": source_id,
            "source_text": f.get("source_text", ""),
            "evidence_mode": mode,
            "sentence_decision": parsed.get("sentence_decision", "") if isinstance(parsed, dict) else "",
            "n_role_entries": n,
            "n_unique_roles": u,
            "n_cluster_annotations": n,
            "n_unique_clusters": u,
            "forward_parse_ok": f.get("parse_ok", False),
            "backward_parse_ok": b.get("parse_ok", False),
            "reconstructed_sentence": (b.get("parsed_backward") or {}).get("reconstructed_sentence", ""),
            "v1_mapping_json": json.dumps(parsed, ensure_ascii=False),
            "backward_packet_json": json.dumps(b.get("backward_packet", {}), ensure_ascii=False),
            "forward_raw": f.get("raw_response", ""),
            "backward_raw": b.get("raw_response", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_forward(rows: pd.DataFrame, client: Any, cfg: dict[str, Any], dictionary: str, mode: str, max_tokens: int, out_dir: Path) -> None:
    path, failures = out_dir / "reduced_v1_forward_mappings.jsonl", out_dir / "failed_requests.jsonl"
    done = read_done(path)
    for i, row in rows.iterrows():
        source_id, sent = str(row["_source_id"]), str(row["_source_text"])
        if source_id in done:
            continue
        try:
            raw = call_chat(client, cfg, forward_messages(sent, dictionary, mode, max_tokens))
            parsed = extract_json(raw)
            append_jsonl(path, {"source_id": source_id, "source_text": sent, "model_key": cfg["model_key"], "model": cfg.get("model", cfg["model_key"]), "condition": f"reduced_v1_{mode}", "evidence_mode": mode, "stage": "forward", "parse_ok": True, "parsed_forward": parsed, "raw_response": raw})
            n, u = count_clusters(parsed)
            print(f"[V1 {mode} forward] {i+1}/{len(rows)} ok {source_id} annotations={n} clusters={u}")
        except Exception as exc:
            append_jsonl(failures, {"stage": "forward", "source_id": source_id, "source_text": sent, "error": repr(exc), "evidence_mode": mode})
            print(f"[V1 {mode} forward] {i+1}/{len(rows)} FAILED {source_id}: {exc}")


def run_backward(rows: pd.DataFrame, client: Any, cfg: dict[str, Any], dictionary: str, mode: str, out_dir: Path) -> None:
    fwd_path = out_dir / "reduced_v1_forward_mappings.jsonl"
    bwd_path = out_dir / "reduced_v1_backward_reconstructions.jsonl"
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
            packet = backward_packet(parsed, source_text, mode)
            raw = call_chat(client, cfg, backward_messages(packet, dictionary))
            parsed_back = extract_json(raw)
            append_jsonl(bwd_path, {"source_id": source_id, "source_text": source_text, "model_key": cfg["model_key"], "model": cfg.get("model", cfg["model_key"]), "condition": f"reduced_v1_{mode}", "evidence_mode": mode, "stage": "backward", "parse_ok": True, "parsed_backward": parsed_back, "backward_packet": packet, "raw_response": raw})
            print(f"[V1 {mode} backward] {i+1}/{len(rows)} ok {source_id}")
        except Exception as exc:
            append_jsonl(failures, {"stage": "backward", "source_id": source_id, "error": repr(exc), "evidence_mode": mode})
            print(f"[V1 {mode} backward] {i+1}/{len(rows)} FAILED {source_id}: {exc}")


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
    cfg = load_model_config(Path(args.model_config_yaml), args.model_key)
    client = make_client(cfg)
    dictionary = load_cluster_dictionary(Path(args.metamodel_yaml))
    out_dir = Path(args.output_dir) / args.model_key / args.evidence_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"forward", "both"}:
        run_forward(rows, client, cfg, dictionary, args.evidence_mode, args.max_evidence_tokens, out_dir)
    if args.stage in {"backward", "both"}:
        run_backward(rows, client, cfg, dictionary, args.evidence_mode, out_dir)
    write_csv(out_dir / "reduced_v1_forward_mappings.jsonl", out_dir / "reduced_v1_backward_reconstructions.jsonl", out_dir / "reduced_v1_roundtrip_outputs.csv", args.evidence_mode)
    print(f"Wrote V1 outputs under {out_dir}")


if __name__ == "__main__":
    main()
