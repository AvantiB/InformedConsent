#!/usr/bin/env python
"""Run individual source-model prompt round-trip experiments.

Forward uses each original source-model prompt as the information-model specific
schema, but adds a strict Phase 1 output contract around it: annotation labels
should be copied verbatim from the source-model dictionary, reserved non-label
strings are not valid annotation IDs, and sentence-level values must be
controlled consent-force labels only.

Backward evaluation uses the same universal structured protocol as Union V0:
valid span annotations enriched with static label metadata, sanitized
relationship links, and controlled sentence-level decision labels. Rows with no
backward-eligible annotations are not sent to the LLM; their reconstruction is
intentionally blank.
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
SPAN_KEYS = ["span_text", "evidence_span_text", "evidence_text", "text_span", "phrase", "text", "span", "verbatim"]
LABEL_KEYS = [
    "source_element_id",
    "source_element_label",
    "field_name",
    "field_id",
    "label",
    "element",
    "element_id",
    "union_element_id",
    "node",
    "term",
    "class",
    "category",
    "path",
    "role",
    "type",
    "id",
]
DECISION_KEYS = ["sentence_decision", "decision", "polarity", "consent_force", "permission", "rule_type", "value"]
RESERVED_NON_LABEL_IDS = {"", "unmatched_language", "unmatched", "no_match", "no match", "none", "null", "unknown", "invalid", "n/a", "na"}
STRICT_POLICY = "annotation_dictionary_relationships"
NO_ANNOTATION_NOTE = "Annotation evidence was empty or insufficient."
RELATIONSHIP_TYPES = {
    "same_span_multiple_labels",
    "same_span_multiple_fields",
    "nested_broad_narrow",
    "complementary_roles",
    "complementary_fields",
    "single",
    "conflicting_or_uncertain",
    "unknown",
}
SENTENCE_LEVEL_DECISION_LABELS = {
    "Rule_TestSentence",
    "Consent.provision.type",
    "Consent.decision",
    "DUO.decision",
    "ICO.decision",
}

UNIVERSAL_BACKWARD_SYSTEM = (
    "You reconstruct informed-consent sentence meaning from an annotation-only mapping. "
    "Return valid JSON only."
)
UNIVERSAL_BACKWARD_USER_TEMPLATE = """
Task: reconstruct one concise natural-language consent sentence using only the annotation-only mapping below.

Instructions:
- Use only information explicitly present in the annotation-only mapping.
- Use label_name and label_definition to interpret annotation labels.
- Use relationship_links only as structural cues for how listed annotations relate to each other.
- Relationship links do not add source wording beyond the annotation spans and static label metadata.
- Use sentence_level_annotations only as controlled consent-force cues attached to the listed span evidence.
- Preserve the order indicated by sentence_order_index when available.
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


def norm_text(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def norm_key(x: Any) -> str:
    raw = norm_text(x).casefold()
    parts = re.split(r"[^a-z0-9]+", raw)
    out = []
    for part in parts:
        if not part:
            continue
        out.append(str(int(part)) if part.isdigit() else part)
    return "|".join(out)


def is_reserved_non_label_id(x: Any) -> bool:
    return norm_text(x).casefold() in RESERVED_NON_LABEL_IDS


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


def load_label_lookup(inventory_csv: Path | None) -> dict[str, Any]:
    lookup: dict[str, Any] = {"by_info_key": {}, "by_info_label": {}, "metadata_by_union_id": {}}
    if inventory_csv is None or not inventory_csv.exists():
        return lookup
    inv = pd.read_csv(inventory_csv).fillna("")
    required = {"source_model", "source_element_id", "source_element_label", "source_element_definition", "union_element_id"}
    if not required.issubset(set(inv.columns)):
        return lookup
    for _, row in inv.iterrows():
        source_model = norm_text(row["source_model"])
        union_id = norm_text(row["union_element_id"])
        source_id = norm_text(row["source_element_id"])
        label = norm_text(row["source_element_label"])
        definition = norm_text(row["source_element_definition"])
        meta = {
            "union_label_id": union_id,
            "source_model": source_model,
            "source_element_id": source_id,
            "label_name": label,
            "label_definition": definition,
        }
        lookup["metadata_by_union_id"][union_id] = meta
        aliases = {source_model, source_model.replace("_Consent", ""), source_model.replace("_", "")}
        candidates = {union_id, source_id, label}
        for alias in aliases:
            alias_key = alias.casefold()
            for cand in candidates:
                lookup["by_info_key"][(alias_key, norm_key(cand))] = meta
            lookup["by_info_label"].setdefault((alias_key, norm_key(label)), []).append(meta)
    return lookup


def info_model_aliases(info_model: str) -> list[str]:
    return [info_model, info_model.replace("_Consent", ""), info_model.replace("_", "")]


def resolve_label_metadata(label: str, info_model: str, lookup: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    label = norm_text(label)
    if not label:
        return {}, "invalid", "empty_label"
    if is_reserved_non_label_id(label):
        return {}, "routed_to_unmatched", "reserved_non_label_id"
    keys = [(alias.casefold(), norm_key(label)) for alias in info_model_aliases(info_model)]
    for key in keys:
        meta = lookup.get("by_info_key", {}).get(key)
        if meta:
            return meta, "valid", "exact_inventory_match"
    # Last safe rescue: exact label text uniquely identifies one row within this source model.
    candidates: list[dict[str, Any]] = []
    for alias in info_model_aliases(info_model):
        candidates.extend(lookup.get("by_info_label", {}).get((alias.casefold(), norm_key(label)), []))
    unique = {m.get("union_label_id", ""): m for m in candidates if m.get("union_label_id")}
    if len(unique) == 1:
        return next(iter(unique.values())), "repaired", "unique_source_element_label_match"
    if len(unique) > 1:
        return {}, "invalid", "ambiguous_label_match"
    return {}, "invalid", "no_inventory_match"


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
    if backward_dir is None or not backward_dir.exists():
        return None
    patterns = PROMPT_PATTERNS[info_model]
    files = [p for p in backward_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    matches = [p for p in files if any(re.search(pattern, p.name.lower()) for pattern in patterns)]
    if not matches:
        return None
    return sorted(matches, key=lambda p: ("back" not in p.name.lower(), len(p.name), p.name.lower()))[0]


def build_forward_messages(prompt_text: str, sentence: str) -> list[dict[str, str]]:
    system = "You are an NLP annotator for informed-consent documents. Apply only the supplied source-model dictionary. Return valid JSON only."
    user = f"""
Use the original source-model prompt below as the information-model schema, but follow the strict Phase 1 output contract.

Strict Phase 1 output contract:
- Return JSON only.
- Every span annotation must copy a source-model element ID or label exactly from the source-model prompt/dictionary.
- When available, include both source_element_id and source_element_label copied verbatim from the same source-model row.
- Do not invent IDs, labels, fields, or namespaces.
- Never use reserved non-label strings as annotation IDs or labels: unmatched_language, unmatched, no_match, none, null, unknown, invalid, n/a.
- unmatched_language is only the name of the top-level audit list. It is never a valid annotation label.
- If no source-model dictionary row fits, place the phrase only in top-level unmatched_language and do not create an annotation object for it.
- A phrase may be annotated with a general source-model class even when the phrase is a named instance and the exact phrase is not in the dictionary.
- Do not annotate standalone “yes” or “no” as Permission or Prohibition unless it directly governs a specific action. Phrases like “say yes or no” represent choice/decision, not permit plus prohibit.
- Phrases like “no penalty” and “no expiration date” are not sentence-level denial/prohibition; they are consequence/protection or temporal-scope expressions.
- sentence_decision must be one of: permit, deny, mixed, unclear.
- sentence_level_elements.value must be a controlled decision value only, e.g., permit, deny, mixed, unclear, Permission, Prohibition, Duty. Do not write explanatory summaries in sentence_level_elements.value.

Return JSON with this structure when possible:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "sentence_level_elements": [{{"source_element_id": "exact source ID", "source_element_label": "exact source label", "value": "controlled decision value only"}}],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "source_element_id": "exact source-model element ID if available",
      "source_element_label": "exact source-model element label if available",
      "label": "same as source_element_label if no separate field is required",
      "overlap_group_id": "g1 or null",
      "span_relation": "single|same_span|broader_span|narrower_nested_span|partially_overlapping_span",
      "decision_or_polarity": "controlled local value if explicitly supported, else empty",
      "rationale": "brief audit-only rationale"
    }}
  ],
  "interpretation_units": [
    {{
      "unit_id": "u1",
      "evidence_span_text": "span or phrase represented by this unit",
      "annotation_ids": ["a1", "a2"],
      "relationship": "single|same_span_multiple_labels|nested_broad_narrow|complementary_roles|conflicting_or_uncertain",
      "combined_meaning": "audit only",
      "backward_mapping_decision": "audit only",
      "rationale": "brief explanation of how the annotations should be considered together"
    }}
  ],
  "unmatched_language": [{{"span_text": "exact text span", "reason": "brief reason"}}]
}}

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
    # Prefer explicit source ID, then explicit source label, then legacy label-like fields.
    for key in LABEL_KEYS:
        value = norm_text(d.get(key))
        if value and value != span and key not in SPAN_KEYS:
            return value
    return ""


def get_label_name_value(d: dict[str, Any]) -> str:
    return first_value(d, ["source_element_label", "label_name", "label", "field_name", "element", "term", "class", "category", "role", "type"])


def get_decision_value(d: dict[str, Any]) -> str:
    for key in DECISION_KEYS:
        value = normalize_sentence_decision_value(d.get(key))
        if value:
            return value
    return ""


def normalize_sentence_decision_value(value: Any) -> str:
    raw = norm_text(value)
    low = raw.casefold()
    if not low:
        return ""
    if len(raw) > 40 or len(raw.split()) > 4:
        return ""
    permit = {"permit", "permission", "permitted", "allow", "allowed", "yes", "consent", "authorized", "authorization"}
    deny = {"deny", "denial", "prohibition", "prohibit", "prohibited", "forbid", "forbidden", "no", "refuse", "refusal"}
    obligation = {"duty", "obligation", "obligated", "mandatory", "required", "must"}
    if low in permit:
        return "permit"
    if low in deny:
        return "deny"
    if low in obligation:
        return "obligation"
    if low in {"mixed", "both"}:
        return "mixed"
    if low in {"unclear", "unknown", "not clear", "ambiguous"}:
        return "unclear"
    return ""


def annotation_from_dict(d: dict[str, Any]) -> dict[str, str] | None:
    span = get_span_value(d)
    label = get_label_value(d, span)
    if span and label:
        return {
            "annotation_id": first_value(d, ["annotation_id", "id"]),
            "span_text": span,
            "label": label,
            "returned_source_element_label": get_label_name_value(d),
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
    if v in {"full_text", "sentence", "text"} or normalize_sentence_decision_value(v):
        return True
    if re.fullmatch(r"[A-Za-z_:-]{1,20}", value) and " " not in value:
        return True
    return False


def choose_row_span(cells: list[str], source_text: str) -> tuple[str, int | None, int | None]:
    candidates = []
    for cell in cells:
        if not cell or is_probable_label_or_decision(cell):
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
        decision = normalize_sentence_decision_value(m.group("decision"))
        if span and label:
            out.append({"annotation_id": "", "span_text": span, "label": label, "returned_source_element_label": label, "decision_or_polarity": decision, "span_relation": "", "overlap_group_id": "", "parse_source": "compact_bracket"})
    return out


def csv_annotations(text: str, source_text: str) -> list[dict[str, str]]:
    out = []
    rows = parse_csv_like(text)
    for row in rows:
        span, _, _ = choose_row_span(row, source_text)
        if not span:
            continue
        labels = [c for c in row if c and c != span and not is_full_sentence_like(c, source_text)]
        labels = [c for c in labels if c.lower() not in {"span", "text", "sentence", "full_text"}]
        label = next((c for c in labels if not normalize_sentence_decision_value(c)), "")
        decision = next((normalize_sentence_decision_value(c) for c in labels if normalize_sentence_decision_value(c)), "")
        if label:
            out.append({"annotation_id": "", "span_text": span, "label": label, "returned_source_element_label": label, "decision_or_polarity": decision, "span_relation": "", "overlap_group_id": "", "parse_source": "csv_like"})
    return out


def ensure_unmatched_language_list(parsed: Any) -> list[dict[str, str]]:
    if not isinstance(parsed, dict):
        return []
    existing = parsed.get("unmatched_language")
    if not isinstance(existing, list):
        return []
    out = []
    for item in existing:
        if isinstance(item, dict):
            span = norm_text(item.get("span_text"))
            reason = norm_text(item.get("reason"))
            if span or reason:
                out.append({"span_text": span, "reason": reason})
        elif norm_text(item):
            out.append({"span_text": norm_text(item), "reason": "raw_unmatched_language_string"})
    return out


def parse_span_annotations(raw_forward: str, source_text: str, info_model: str, label_lookup: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    parsed_obj: Any = None
    annotations: list[dict[str, str]] = []
    parse_mode = ""
    try:
        parsed_obj = extract_json(raw_forward)
        annotations = collect_json_annotations(parsed_obj)
        parse_mode = "json_like"
    except Exception:
        pass
    if not annotations:
        annotations = compact_annotations(raw_forward)
        parse_mode = "compact_bracket" if annotations else parse_mode
    if not annotations:
        annotations = csv_annotations(raw_forward, source_text)
        parse_mode = "csv_like" if annotations else parse_mode

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    routed: list[dict[str, Any]] = []
    seen = set()
    dropped_full_sentence = 0
    for ann in annotations:
        span = norm_text(ann.get("span_text"))
        label = norm_text(ann.get("label"))
        if not span or not label:
            continue
        if is_full_sentence_like(span, source_text):
            dropped_full_sentence += 1
            continue
        meta, status, reason = resolve_label_metadata(label, info_model, label_lookup)
        clean = dict(ann)
        clean["annotation_id"] = norm_text(clean.get("annotation_id")) or f"a{len(valid) + len(invalid) + len(routed) + 1}"
        clean["id_validation_status"] = status
        clean["id_validation_reason"] = reason
        if status == "routed_to_unmatched":
            clean["invalid_label"] = label
            routed.append(clean)
            continue
        if status not in {"valid", "repaired"}:
            clean["invalid_label"] = label
            invalid.append(clean)
            continue
        key = (span.lower(), meta.get("union_label_id", label).lower(), norm_text(clean.get("decision_or_polarity")).lower())
        if key in seen:
            continue
        seen.add(key)
        clean.update({
            "label_id": meta.get("source_element_id", label),
            "label": meta.get("source_element_id", label),
            "union_label_id": meta.get("union_label_id", ""),
            "label_name": meta.get("label_name", label),
            "label_definition": meta.get("label_definition", ""),
            "source_model": meta.get("source_model", info_model),
            "source_element_id": meta.get("source_element_id", label),
        })
        valid.append(clean)

    unmatched = ensure_unmatched_language_list(parsed_obj)
    for ann in routed:
        span = norm_text(ann.get("span_text"))
        if span:
            unmatched.append({"span_text": span, "reason": "LLM placed reserved non-label value inside annotations; routed to unmatched_language audit."})
    audit = {
        "annotation_parse_mode": parse_mode or "none",
        "n_annotations_parsed": len(annotations),
        "n_annotations_valid": len(valid),
        "n_annotations_invalid": len(invalid),
        "n_annotations_routed_to_unmatched": len(routed),
        "n_annotations_backward_eligible_strict": len(valid),
        "n_full_sentence_spans_dropped": dropped_full_sentence,
    }
    return valid, audit, invalid, routed, unmatched


def canonical_sentence_level_element(item: dict[str, Any], info_model: str, lookup: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    label = first_value(item, ["source_element_id", "source_element_label", "union_element_id", "field", "field_name", "label", "id"])
    value = first_value(item, ["value", "decision", "rule_type", "permission", "prohibition"])
    canonical_value = normalize_sentence_decision_value(value)
    if not canonical_value:
        return None, "non_controlled_or_explanatory_value"
    if not label:
        return {"field": "sentence_level_decision", "value": canonical_value, "support": "valid_span_annotations_present"}, "included_without_label"
    meta, status, reason = resolve_label_metadata(label, info_model, lookup)
    if status not in {"valid", "repaired"}:
        return None, f"invalid_sentence_level_label:{reason}"
    label_name = meta.get("label_name", label)
    if label_name not in SENTENCE_LEVEL_DECISION_LABELS and meta.get("source_element_id") not in SENTENCE_LEVEL_DECISION_LABELS:
        return None, "not_approved_sentence_decision_field"
    return {
        "field": "sentence_level_decision",
        "label_id": meta.get("source_element_id", label),
        "union_label_id": meta.get("union_label_id", ""),
        "label_name": label_name,
        "label_definition": meta.get("label_definition", ""),
        "source_model": meta.get("source_model", info_model),
        "source_element_id": meta.get("source_element_id", label),
        "value": canonical_value,
        "support": "valid_span_annotations_present",
        "id_resolution_status": status,
        "id_resolution_reason": reason,
    }, "included"


def extract_sentence_level_annotations(raw_forward: str, has_valid_span_annotations: bool, info_model: str, label_lookup: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not has_valid_span_annotations:
        return [], {"n_sentence_level_annotations_backward_eligible": 0, "n_sentence_level_elements_dropped_by_policy": 0}
    out: list[dict[str, Any]] = []
    dropped = 0
    try:
        parsed = extract_json(raw_forward)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        canonical = normalize_sentence_decision_value(parsed.get("sentence_decision") or parsed.get("decision"))
        if canonical:
            out.append({"field": "sentence_decision", "value": canonical, "support": "valid_span_annotations_present"})
        for key in ["rule_type", "permission", "prohibition"]:
            canonical = normalize_sentence_decision_value(parsed.get(key))
            if canonical:
                out.append({"field": key, "value": canonical, "support": "valid_span_annotations_present"})
        elems = parsed.get("sentence_level_elements") or []
        if isinstance(elems, list):
            for item in elems:
                if not isinstance(item, dict):
                    dropped += 1
                    continue
                canonical_elem, _reason = canonical_sentence_level_element(item, info_model, label_lookup)
                if canonical_elem is None:
                    dropped += 1
                else:
                    out.append(canonical_elem)
    return out, {"n_sentence_level_annotations_backward_eligible": len(out), "n_sentence_level_elements_dropped_by_policy": dropped}


def normalize_relationship_type(x: Any) -> str:
    rel = norm_text(x).lower().replace(" ", "_").replace("-", "_")
    if not rel:
        return "unknown"
    aliases = {
        "same_span_multiple_label": "same_span_multiple_labels",
        "same_span_multiple_labels": "same_span_multiple_labels",
        "same_span_multiple_fields": "same_span_multiple_fields",
        "same_span_multiple_field": "same_span_multiple_fields",
        "nested_broader_narrower": "nested_broad_narrow",
        "nested_broad_narrow": "nested_broad_narrow",
        "broad_narrow": "nested_broad_narrow",
        "complementary_role": "complementary_roles",
        "complementary_roles": "complementary_roles",
        "complementary_fields": "complementary_fields",
        "single": "single",
        "conflicting": "conflicting_or_uncertain",
        "uncertain": "conflicting_or_uncertain",
        "conflicting_or_uncertain": "conflicting_or_uncertain",
    }
    return aliases.get(rel, rel if rel in RELATIONSHIP_TYPES else "unknown")


def extract_relationship_links(raw_forward: str, valid_annotation_ids: set[str]) -> list[dict[str, Any]]:
    try:
        parsed = extract_json(raw_forward)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    units = parsed.get("interpretation_units") or []
    if not isinstance(units, list):
        return []
    links = []
    for idx, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            continue
        ids = unit.get("annotation_ids") or []
        if not isinstance(ids, list):
            ids = []
        ann_ids = [norm_text(x) for x in ids if norm_text(x) in valid_annotation_ids]
        if len(ann_ids) < 2:
            continue
        links.append({"relationship_id": norm_text(unit.get("unit_id")) or f"rel{idx}", "relationship_type": normalize_relationship_type(unit.get("relationship")), "annotation_ids": ann_ids})
    return links


def build_sanitized_forward_material(raw_forward: str, source_text: str, info_model: str, label_lookup: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    annotations, audit, invalid_annotations, routed_unmatched, unmatched = parse_span_annotations(raw_forward, source_text, info_model, label_lookup)
    ordered = []
    for ann in annotations:
        start, end = find_span_bounds(source_text, ann.get("span_text", ""))
        ordered.append({
            "annotation_id": ann.get("annotation_id", ""),
            "span_text": ann.get("span_text", ""),
            "label_id": ann.get("label_id", ann.get("label", "")),
            "label": ann.get("label", ""),
            "union_label_id": ann.get("union_label_id", ""),
            "label_name": ann.get("label_name", ann.get("label", "")),
            "label_definition": ann.get("label_definition", ""),
            "source_model": ann.get("source_model", info_model),
            "source_element_id": ann.get("source_element_id", ann.get("label", "")),
            "id_resolution_status": ann.get("id_validation_status", ""),
            "id_resolution_reason": ann.get("id_validation_reason", ""),
            "decision_or_polarity": ann.get("decision_or_polarity", ""),
            "span_relation": ann.get("span_relation", ""),
            "overlap_group_id": ann.get("overlap_group_id", ""),
            "span_start": start,
            "span_end": end,
        })
    ordered.sort(key=lambda x: (10**9 if x.get("span_start") is None else int(x.get("span_start")), str(x.get("annotation_id", ""))))
    for i, item in enumerate(ordered, start=1):
        item["sentence_order_index"] = i
    valid_annotation_ids = {norm_text(x.get("annotation_id")) for x in ordered if norm_text(x.get("annotation_id"))}
    sent_level, sent_audit = extract_sentence_level_annotations(raw_forward, bool(ordered), info_model, label_lookup)
    packet = {
        "backward_input_policy": STRICT_POLICY,
        "ordered_reconstruction_items": ordered,
        "relationship_links": extract_relationship_links(raw_forward, valid_annotation_ids),
        "sentence_level_annotations": sent_level,
    }
    audit = {
        **audit,
        **sent_audit,
        "n_relationship_links_backward_eligible": len(packet["relationship_links"]),
        "n_invalid_annotations": len(invalid_annotations),
        "n_routed_unmatched_annotations": len(routed_unmatched),
    }
    audit_material = {
        "invalid_annotations": invalid_annotations,
        "routed_unmatched_annotations": routed_unmatched,
        "unmatched_language": unmatched,
    }
    return packet, {**audit, **audit_material}


def build_backward_messages(sanitized_material: dict[str, Any]) -> list[dict[str, str]]:
    material_text = json.dumps(sanitized_material, ensure_ascii=False, indent=2)
    user = UNIVERSAL_BACKWARD_USER_TEMPLATE.format(mapping_text=material_text)
    return [{"role": "system", "content": UNIVERSAL_BACKWARD_SYSTEM}, {"role": "user", "content": user}]


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


def write_label_audit(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    fwd = load_jsonl_by_id(forward_path)
    bwd = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, f in fwd.items():
        b = bwd.get(source_id, {})
        audit = b.get("backward_annotation_audit", {})
        for group_name in ["invalid_annotations", "routed_unmatched_annotations"]:
            for ann in audit.get(group_name, []) if isinstance(audit.get(group_name), list) else []:
                if not isinstance(ann, dict):
                    continue
                rows.append({
                    "source_id": source_id,
                    "source_text": f.get("source_text", ""),
                    "info_model": f.get("info_model", ""),
                    "group": group_name,
                    "annotation_id": norm_text(ann.get("annotation_id")),
                    "span_text": norm_text(ann.get("span_text")),
                    "label": norm_text(ann.get("label")),
                    "status": norm_text(ann.get("id_validation_status")),
                    "reason": norm_text(ann.get("id_validation_reason")),
                    "rationale_audit_only": norm_text(ann.get("rationale")),
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def write_csv(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    fwd = load_jsonl_by_id(forward_path)
    bwd = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, f in fwd.items():
        b = bwd.get(source_id, {})
        packet = b.get("sanitized_forward_material", {})
        audit = b.get("backward_annotation_audit", {})
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
            "backward_annotation_audit_json": json.dumps(audit, ensure_ascii=False),
            "invalid_annotations_json": json.dumps(audit.get("invalid_annotations", []), ensure_ascii=False),
            "routed_unmatched_annotations_json": json.dumps(audit.get("routed_unmatched_annotations", []), ensure_ascii=False),
            "unmatched_language_json": json.dumps(audit.get("unmatched_language", []), ensure_ascii=False),
            "annotation_count": len(items),
            "unique_element_count": len({norm_text(x.get("label_id") or x.get("label")) for x in items if isinstance(x, dict) and norm_text(x.get("label_id") or x.get("label"))}),
            "n_annotations_backward_eligible": len(items),
            "n_annotations_valid": audit.get("n_annotations_valid", ""),
            "n_annotations_invalid": audit.get("n_annotations_invalid", ""),
            "n_annotations_routed_to_unmatched": audit.get("n_annotations_routed_to_unmatched", ""),
            "n_relationship_links_backward_eligible": audit.get("n_relationship_links_backward_eligible", ""),
            "n_sentence_level_annotations_backward_eligible": audit.get("n_sentence_level_annotations_backward_eligible", ""),
            "n_sentence_level_elements_dropped_by_policy": audit.get("n_sentence_level_elements_dropped_by_policy", ""),
            "n_full_sentence_spans_dropped": audit.get("n_full_sentence_spans_dropped", ""),
            "annotation_parse_mode": audit.get("annotation_parse_mode", ""),
            "forward_parse_ok": bool(items) or bool(f.get("raw_response", "")),
            "backward_parse_ok": b.get("parse_ok", False),
            "reconstructed_sentence": parsed_back.get("reconstructed_sentence", ""),
            "reconstruction_notes": parsed_back.get("reconstruction_notes", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def run_info_model(rows: pd.DataFrame, client: OpenAI, model_cfg: dict[str, Any], info_model: str, prompt_text: str, backward_prompt_text: str | None, out_dir: Path, stage: str, label_lookup: dict[str, Any] | None = None) -> None:
    _ = backward_prompt_text
    label_lookup = label_lookup or {}
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
        if not fwd_by_id:
            raise FileNotFoundError(f"No forward mappings found for {info_model}. Expected non-empty file: {forward_path}")
        done = read_done(backward_path)
        for i, (source_id, fwd) in enumerate(fwd_by_id.items()):
            if source_id in done:
                continue
            try:
                source_text = fwd.get("source_text", "")
                sanitized_material, audit = build_sanitized_forward_material(fwd.get("raw_response", ""), source_text, info_model, label_lookup)
                if not sanitized_material.get("ordered_reconstruction_items"):
                    parsed_back = {"reconstructed_sentence": "", "reconstruction_notes": NO_ANNOTATION_NOTE}
                    raw = json.dumps(parsed_back, ensure_ascii=False)
                else:
                    raw = call_chat(client, model_cfg, build_backward_messages(sanitized_material))
                    parsed_back = parse_backward_response(raw)
                append_jsonl(backward_path, {"source_id": source_id, "source_text": source_text, "model_key": model_cfg["model_key"], "model": model_cfg["model"], "info_model": info_model, "stage": "backward", "backward_input_sanitized": True, "backward_input_policy": STRICT_POLICY, "sanitized_forward_material": sanitized_material, "backward_annotation_audit": audit, "parsed_backward": parsed_back, "parse_ok": True, "raw_response": raw})
                done.add(source_id)
                print(f"[{info_model} backward] {i + 1}/{len(fwd_by_id)} ok {source_id} eligible_annotations={len(sanitized_material.get('ordered_reconstruction_items') or [])} links={len(sanitized_material.get('relationship_links') or [])}")
            except Exception as exc:
                append_jsonl(failures_path, {"source_id": source_id, "info_model": info_model, "stage": "backward", "error": repr(exc)})
                print(f"[{info_model} backward] FAILED {source_id}: {exc}", file=sys.stderr)

    write_csv(forward_path, backward_path, out_dir / "roundtrip_outputs.csv")
    write_label_audit(forward_path, backward_path, out_dir / "label_validation_audit.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--prompt_dir", required=True)
    ap.add_argument("--backward_prompt_dir", default=None, help="Deprecated/ignored for evaluation; strict universal backward prompt is always used.")
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv", help="Optional Union V0/source inventory for static label metadata.")
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
    label_lookup = load_label_lookup(Path(args.inventory_csv) if args.inventory_csv else None)

    base_out = Path(args.output_dir) / args.model_key
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "run_metadata.json").write_text(json.dumps({
        "model_key": args.model_key,
        "model": model_cfg.get("model"),
        "n_input_rows": int(len(rows)),
        "info_models": info_models,
        "roundtrips_csv": args.roundtrips_csv,
        "prompt_dir": args.prompt_dir,
        "inventory_csv": args.inventory_csv,
        "backward_prompt_dir_deprecated_not_used": args.backward_prompt_dir,
        "stage": args.stage,
        "prompt_design": "source_model_forward_requires_verbatim_id_label_and_controlled_sentence_decisions",
        "id_validation": "source_model_inventory_label_validation_with_reserved_non_label_routing",
        "sentence_level_backward_policy": "controlled_decision_values_only_no_explanatory_summaries",
        "backward_input": STRICT_POLICY,
        "backward_prompt": "universal_annotation_dictionary_relationships",
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
            "backward_prompt_file_deprecated_not_used": str(backward_path) if backward_path else None,
            "uses_universal_structured_backward_prompt": True,
            "backward_input_policy": STRICT_POLICY,
            "strict_forward_contract_applied": True,
        }, indent=2))
        run_info_model(rows, client, model_cfg, info_model, prompt_text, backward_text, out_dir, args.stage, label_lookup)

    print(f"Wrote individual-model outputs under {base_out}")


if __name__ == "__main__":
    main()
