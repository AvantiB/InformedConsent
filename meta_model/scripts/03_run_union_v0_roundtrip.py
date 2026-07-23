#!/usr/bin/env python
"""Run Union V0 full-dictionary forward/backward round-trip experiments.

Forward mapping remains overlap-aware and may include rich audit material for
human review. The forward output is validated against the authoritative Union V0
inventory table: every span annotation must use a valid union_element_id and,
for fresh runs, should also return the exact source_element_label copied from
the same inventory row. If the LLM mistakenly places unmatched_language or other
reserved non-label strings inside annotations, the validator routes those spans
back to unmatched-language audit instead of counting them as dictionary-ID
failures. Backward evaluation uses one universal structured protocol: valid span
annotations enriched with static dictionary metadata, sanitized relationship
links, and eligible sentence-level annotations. Rows with no backward-eligible
annotations are not sent to the LLM; their reconstruction is intentionally blank.
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
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. Install with: pip install openai") from exc

TEXT_COL_CANDIDATES = ["canonical_full_text", "full_text_original", "original_sentence", "full_text", "sentence", "text"]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "id"]
STRICT_POLICY = "annotation_dictionary_relationships"
NO_ANNOTATION_NOTE = "Annotation evidence was empty or insufficient."
RESERVED_NON_LABEL_IDS = {"", "unmatched_language", "unmatched", "no_match", "no match", "none", "null", "unknown", "invalid", "n/a", "na"}
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


def norm_label(x: Any) -> str:
    return norm_text(x).casefold()


def is_reserved_non_label_id(x: Any) -> bool:
    return norm_text(x).casefold() in RESERVED_NON_LABEL_IDS


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")
    return None


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


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


def load_inventory(inventory_csv: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv).fillna("")
    required = ["union_element_id", "source_model", "source_element_id", "source_element_label", "source_element_definition"]
    missing = [c for c in required if c not in inv.columns]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")
    if "element_scope" not in inv.columns:
        inv["element_scope"] = "span"
    return inv


def build_dictionary_text(inv: pd.DataFrame) -> str:
    lines = [
        "Authoritative Union V0 dictionary. Each row has:",
        "union_element_id | source_model | source_element_label | source_element_definition",
        "Copy both union_element_id and source_element_label verbatim into every annotation.",
        "",
    ]
    for _, row in inv.iterrows():
        scope = row.get("element_scope", "span") or "span"
        label = norm_text(row["source_element_label"])
        definition = norm_text(row["source_element_definition"])
        lines.append(
            f"- union_element_id={row['union_element_id']} | source_model={row['source_model']} | "
            f"source_element_label={label} | element_scope={scope} | source_element_definition={definition}"
        )
    return "\n".join(lines)


def normalized_id_key(x: Any) -> str:
    raw = norm_text(x).lower()
    parts = re.split(r"[^a-z0-9]+", raw)
    out = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            out.append(str(int(part)))
        else:
            out.append(part)
    return "|".join(out)


def add_norm_index(index: dict[str, set[str]], key: Any, union_id: str) -> None:
    norm_key = normalized_id_key(key)
    if norm_key:
        index.setdefault(norm_key, set()).add(union_id)


def metadata_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "label_id": norm_text(row["union_element_id"]),
        "source_model": norm_text(row["source_model"]),
        "source_element_id": norm_text(row["source_element_id"]),
        "label_name": norm_text(row["source_element_label"]),
        "label_definition": norm_text(row["source_element_definition"]),
        "element_scope": norm_text(row.get("element_scope", "span")),
    }


def build_inventory_maps(inv: pd.DataFrame) -> dict[str, Any]:
    valid_ids = set(inv["union_element_id"].astype(str))
    by_pair: dict[tuple[str, str], str] = {}
    by_source_id: dict[str, list[str]] = {}
    normalized_index: dict[str, set[str]] = {}
    by_label: dict[str, list[str]] = {}
    by_source_label: dict[tuple[str, str], list[str]] = {}
    metadata_by_union_id: dict[str, dict[str, Any]] = {}
    for _, row in inv.iterrows():
        source_model = str(row["source_model"])
        source_element_id = str(row["source_element_id"])
        source_label = str(row["source_element_label"])
        union_element_id = str(row["union_element_id"])
        metadata_by_union_id[union_element_id] = metadata_from_row(row)
        by_label.setdefault(norm_label(source_label), []).append(union_element_id)
        by_source_label.setdefault((source_model.casefold(), norm_label(source_label)), []).append(union_element_id)
        aliases = {source_model, source_model.replace("_Consent", ""), source_model.replace("_", "")}
        for alias in aliases:
            by_pair[(alias, source_element_id)] = union_element_id
            add_norm_index(normalized_index, f"{alias}::{source_element_id}", union_element_id)
            add_norm_index(normalized_index, f"{alias}:{source_element_id}", union_element_id)
            if ":" in source_element_id:
                prefix, suffix = source_element_id.split(":", 1)
                if alias == prefix:
                    by_pair[(alias, suffix)] = union_element_id
                    add_norm_index(normalized_index, f"{alias}::{suffix}", union_element_id)
        by_source_id.setdefault(source_element_id, []).append(union_element_id)
        add_norm_index(normalized_index, union_element_id, union_element_id)
        add_norm_index(normalized_index, source_element_id, union_element_id)
        add_norm_index(normalized_index, source_label, union_element_id)
    return {
        "valid_ids": valid_ids,
        "by_pair": by_pair,
        "by_source_id": by_source_id,
        "normalized_index": normalized_index,
        "by_label": by_label,
        "by_source_label": by_source_label,
        "metadata_by_union_id": metadata_by_union_id,
    }


def returned_source_label(ann: dict[str, Any]) -> str:
    for key in ["source_element_label", "label_name", "element_label"]:
        value = norm_text(ann.get(key))
        if value:
            return value
    return ""


def label_matches_union_id(label: str, union_id: str, maps: dict[str, Any]) -> bool:
    if not label:
        return True
    meta = maps["metadata_by_union_id"].get(union_id, {})
    return norm_label(label) == norm_label(meta.get("label_name", ""))


def parse_source_model_from_id(uid: str) -> str:
    if "::" in uid:
        return uid.split("::", 1)[0]
    if ":" in uid:
        return uid.split(":", 1)[0]
    return ""


def unique_label_match(label: str, source_model_hint: str, maps: dict[str, Any]) -> str | None:
    if not label:
        return None
    if source_model_hint:
        matches = maps["by_source_label"].get((source_model_hint.casefold(), norm_label(label)), [])
        if len(matches) == 1:
            return matches[0]
    matches = maps["by_label"].get(norm_label(label), [])
    if len(matches) == 1:
        return matches[0]
    return None


def repair_union_id(uid: Any, label: str, maps: dict[str, Any]) -> tuple[str, str, str]:
    if not isinstance(uid, str):
        return str(uid), "invalid", "not_string"
    uid = uid.strip()
    if not uid:
        return uid, "invalid", "empty"
    if is_reserved_non_label_id(uid):
        return uid, "routed_to_unmatched", "reserved_non_label_id"
    if uid in maps["valid_ids"]:
        if label and not label_matches_union_id(label, uid, maps):
            return uid, "invalid", "label_mismatch_for_exact_union_id"
        return uid, "valid", "exact_union_id_label_match" if label else "exact_union_id_no_label_returned"

    source_model_hint = parse_source_model_from_id(uid)
    matches = maps["by_source_id"].get(uid, [])
    if len(matches) == 1 and label_matches_union_id(label, matches[0], maps):
        return matches[0], "repaired", "exact_source_element_id_label_match" if label else "exact_source_element_id_no_label_returned"

    if "::" not in uid and ":" in uid:
        source_model, rest = uid.split(":", 1)
        candidates = [(source_model, rest), (source_model, f"{source_model}:{rest}"), (source_model.replace("FHIR", "FHIR_Consent"), rest)]
        for key in candidates:
            if key in maps["by_pair"] and label_matches_union_id(label, maps["by_pair"][key], maps):
                return maps["by_pair"][key], "repaired", "source_model_pair_label_match" if label else "source_model_pair_no_label_returned"

    # Only allow normalized-id repair when the returned label confirms the same inventory row.
    norm_matches = maps["normalized_index"].get(normalized_id_key(uid), set())
    if len(norm_matches) == 1:
        candidate = next(iter(norm_matches))
        if label and label_matches_union_id(label, candidate, maps):
            return candidate, "repaired", "normalized_id_and_label_match"
        if not label:
            return uid, "invalid", "normalized_id_match_requires_label"
    if len(norm_matches) > 1:
        return uid, "invalid", "ambiguous_normalized_id_match"

    # Last safe rescue: malformed ID plus exact returned source_element_label uniquely identifies one inventory row.
    label_candidate = unique_label_match(label, source_model_hint, maps)
    if label_candidate:
        return label_candidate, "repaired", "unique_source_element_label_match"

    return uid, "invalid", "no_inventory_match"


def ensure_unmatched_language_list(obj: dict[str, Any]) -> list[dict[str, str]]:
    existing = obj.get("unmatched_language")
    if not isinstance(existing, list):
        existing = []
    out: list[dict[str, str]] = []
    for item in existing:
        if isinstance(item, dict):
            span = norm_text(item.get("span_text"))
            reason = norm_text(item.get("reason"))
            if span or reason:
                out.append({"span_text": span, "reason": reason})
        elif norm_text(item):
            out.append({"span_text": norm_text(item), "reason": "raw_unmatched_language_string"})
    return out


def add_routed_unmatched(obj: dict[str, Any], ann: dict[str, Any], reason: str) -> None:
    unmatched = ensure_unmatched_language_list(obj)
    span = norm_text(ann.get("span_text"))
    if span:
        key = (span.casefold(), reason.casefold())
        seen = {(norm_text(x.get("span_text")).casefold(), norm_text(x.get("reason")).casefold()) for x in unmatched if isinstance(x, dict)}
        if key not in seen:
            unmatched.append({"span_text": span, "reason": reason})
    obj["unmatched_language"] = unmatched


def validate_forward_obj(forward_obj: dict[str, Any], maps: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = copy.deepcopy(forward_obj)
    annotations = obj.get("annotations") or []
    if not isinstance(annotations, list):
        annotations = []
    obj["unmatched_language"] = ensure_unmatched_language_list(obj)
    valid_annotations: list[dict[str, Any]] = []
    invalid_annotations: list[dict[str, Any]] = []
    routed_unmatched_annotations: list[dict[str, Any]] = []
    n_valid = n_repaired = n_invalid = n_routed_unmatched = 0
    for ann in annotations:
        if not isinstance(ann, dict):
            n_invalid += 1
            invalid_annotations.append({"raw_annotation": ann, "id_validation_status": "invalid_non_object"})
            continue
        original_uid = ann.get("union_element_id", "")
        returned_label = returned_source_label(ann)
        repaired_uid, status, reason = repair_union_id(original_uid, returned_label, maps)
        ann = copy.deepcopy(ann)
        ann["returned_source_element_label"] = returned_label
        ann["id_validation_status"] = status
        ann["id_validation_reason"] = reason
        if status == "routed_to_unmatched":
            n_routed_unmatched += 1
            routed = copy.deepcopy(ann)
            routed["invalid_union_element_id"] = original_uid
            routed_unmatched_annotations.append(routed)
            add_routed_unmatched(obj, ann, "LLM placed reserved non-label ID inside annotations; routed to unmatched_language audit.")
            continue
        if status in {"valid", "repaired"}:
            meta = maps["metadata_by_union_id"].get(repaired_uid, {})
            ann["source_element_label"] = meta.get("label_name", returned_label)
            ann["source_element_definition"] = meta.get("label_definition", "")
            ann["source_model"] = meta.get("source_model", "")
            ann["source_element_id"] = meta.get("source_element_id", "")
        if status == "valid":
            n_valid += 1
            valid_annotations.append(ann)
        elif status == "repaired":
            n_repaired += 1
            ann["original_union_element_id"] = original_uid
            ann["union_element_id"] = repaired_uid
            valid_annotations.append(ann)
        else:
            n_invalid += 1
            bad = copy.deepcopy(ann)
            bad["invalid_union_element_id"] = original_uid
            invalid_annotations.append(bad)
    obj["annotations"] = valid_annotations
    obj["invalid_annotations"] = invalid_annotations
    obj["routed_unmatched_annotations"] = routed_unmatched_annotations
    validation = {
        "n_annotations_raw": len(annotations),
        "n_annotations_valid": n_valid,
        "n_annotations_repaired": n_repaired,
        "n_annotations_invalid": n_invalid,
        "n_annotations_routed_to_unmatched": n_routed_unmatched,
        "n_annotations_backward_eligible": len(valid_annotations),
        "n_interpretation_units": len(obj.get("interpretation_units") or []) if isinstance(obj.get("interpretation_units"), list) else 0,
        "has_invalid_ids": n_invalid > 0,
        "has_routed_unmatched_annotations": n_routed_unmatched > 0,
    }
    obj["validation_summary"] = validation
    return obj, validation


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


def annotation_item(ann: dict[str, Any], source_text: str, maps: dict[str, Any]) -> dict[str, Any] | None:
    span = norm_text(ann.get("span_text", ""))
    uid = norm_text(ann.get("union_element_id", ""))
    if not span or not uid or is_full_sentence_like(span, source_text):
        return None
    start, end = find_span_bounds(source_text, span)
    meta = maps["metadata_by_union_id"].get(uid, {})
    item = {
        "annotation_id": norm_text(ann.get("annotation_id")),
        "span_text": span,
        "label_id": uid,
        "label": uid,
        "label_name": meta.get("label_name", norm_text(ann.get("source_element_label"))),
        "label_definition": meta.get("label_definition", norm_text(ann.get("source_element_definition"))),
        "source_model": meta.get("source_model", norm_text(ann.get("source_model"))),
        "source_element_id": meta.get("source_element_id", norm_text(ann.get("source_element_id"))),
        "element_scope": meta.get("element_scope", ""),
        "id_resolution_status": norm_text(ann.get("id_validation_status")),
        "id_resolution_reason": norm_text(ann.get("id_validation_reason")),
        "span_relation": norm_text(ann.get("span_relation")),
        "overlap_group_id": norm_text(ann.get("overlap_group_id")),
        "span_start": start,
        "span_end": end,
    }
    if norm_text(ann.get("original_union_element_id")):
        item["original_label_id"] = norm_text(ann.get("original_union_element_id"))
    if norm_text(ann.get("returned_source_element_label")):
        item["returned_source_element_label"] = norm_text(ann.get("returned_source_element_label"))
    return item


def ordered_annotations_for_backward(parsed_forward: dict[str, Any], source_text: str, maps: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotations = parsed_forward.get("annotations") or []
    ordered: list[dict[str, Any]] = []
    dropped_full_sentence = 0
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if is_full_sentence_like(ann.get("span_text", ""), source_text):
            dropped_full_sentence += 1
            continue
        item = annotation_item(ann, source_text, maps)
        if item is not None:
            ordered.append(item)
    ordered = sorted(ordered, key=lambda item: (10**9 if item.get("span_start") is None else int(item.get("span_start")), -int((item.get("span_end") or item.get("span_start") or 0) - (item.get("span_start") or 0)), str(item.get("annotation_id", ""))))
    for i, ann in enumerate(ordered, start=1):
        ann["sentence_order_index"] = i
    return ordered, {"n_full_sentence_spans_dropped": dropped_full_sentence, "n_annotations_backward_eligible_strict": len(ordered)}


def eligible_sentence_level_annotations(parsed_forward: dict[str, Any], has_valid_span_annotations: bool) -> list[dict[str, Any]]:
    if not has_valid_span_annotations:
        return []
    out = []
    if norm_text(parsed_forward.get("sentence_decision")):
        out.append({"field": "sentence_decision", "value": norm_text(parsed_forward.get("sentence_decision")), "support": "valid_span_annotations_present"})
    elems = parsed_forward.get("sentence_level_elements") or []
    if isinstance(elems, list):
        for item in elems:
            if isinstance(item, dict):
                val = {k: norm_text(v) for k, v in item.items() if norm_text(v)}
                if val:
                    val["support"] = "valid_span_annotations_present"
                    out.append(val)
    return out


def normalize_relationship_type(x: Any) -> str:
    rel = norm_text(x).lower().replace(" ", "_").replace("-", "_")
    if not rel:
        return "unknown"
    aliases = {
        "same_span_multiple_label": "same_span_multiple_labels",
        "same_span_multiple_fields": "same_span_multiple_fields",
        "nested_broad_narrow": "nested_broad_narrow",
        "nested_broader_narrower": "nested_broad_narrow",
        "broad_narrow": "nested_broad_narrow",
        "complementary_role": "complementary_roles",
        "complementary_roles": "complementary_roles",
        "complementary_fields": "complementary_fields",
        "single": "single",
        "conflicting_or_uncertain": "conflicting_or_uncertain",
        "conflicting": "conflicting_or_uncertain",
        "uncertain": "conflicting_or_uncertain",
    }
    return aliases.get(rel, rel if rel in RELATIONSHIP_TYPES else "unknown")


def sanitized_relationship_links(parsed_forward: dict[str, Any], valid_annotation_ids: set[str]) -> list[dict[str, Any]]:
    links = []
    units = parsed_forward.get("interpretation_units") or []
    if not isinstance(units, list):
        return links
    for idx, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            continue
        ann_ids_raw = unit.get("annotation_ids") or []
        if not isinstance(ann_ids_raw, list):
            ann_ids_raw = []
        ann_ids = [norm_text(x) for x in ann_ids_raw if norm_text(x) in valid_annotation_ids]
        if len(ann_ids) < 2:
            continue
        links.append({
            "relationship_id": norm_text(unit.get("unit_id")) or f"rel{idx}",
            "relationship_type": normalize_relationship_type(unit.get("relationship")),
            "annotation_ids": ann_ids,
        })
    return links


def build_backward_packet(parsed_forward: dict[str, Any], source_text: str, maps: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered, audit = ordered_annotations_for_backward(parsed_forward, source_text, maps)
    valid_annotation_ids = {norm_text(x.get("annotation_id")) for x in ordered if norm_text(x.get("annotation_id"))}
    packet = {
        "backward_input_policy": STRICT_POLICY,
        "ordered_reconstruction_items": ordered,
        "relationship_links": sanitized_relationship_links(parsed_forward, valid_annotation_ids),
        "sentence_level_annotations": eligible_sentence_level_annotations(parsed_forward, bool(ordered)),
    }
    audit = {
        **audit,
        "n_annotations_forward_valid": len(parsed_forward.get("annotations") or []) if isinstance(parsed_forward.get("annotations"), list) else 0,
        "n_relationship_links_backward_eligible": len(packet["relationship_links"]),
        "n_sentence_level_annotations_backward_eligible": len(packet["sentence_level_annotations"]),
    }
    return packet, audit


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


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        raise
    raise ValueError("Parsed JSON is not an object")


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


def build_forward_messages(sentence: str, dictionary_text: str) -> list[dict[str, str]]:
    system = "You are an NLP annotator for informed-consent documents. Apply only the supplied authoritative dictionary. Return valid JSON only."
    user = f"""
Task: annotate the informed-consent sentence using ONLY rows from the authoritative Union V0 dictionary below.

Important context:
- This dictionary is a naive union of multiple information models, not a reduced meta-model.
- Several elements may overlap, duplicate, specialize, or complement each other.
- The same or similar text span MAY receive more than one label.
- A larger phrase may receive a broader role, while a nested shorter phrase may receive a narrower or more specific role.
- Preserve overlaps/nesting relationships rather than forcing a single label too early.
- A phrase may be annotated with a general dictionary class even when the phrase is a named instance and the exact phrase is not in the dictionary. For example, a named database, repository, dataset, or biobank can be labeled with the best valid dictionary concept for that type when present.

Hard dictionary rules:
- Every annotation MUST copy union_element_id exactly from one dictionary row.
- Every annotation MUST copy source_element_label exactly from the same dictionary row.
- Do not invent IDs, labels, fields, or namespaces.
- Never use these reserved non-label strings as union_element_id: unmatched_language, unmatched, no_match, none, null, unknown, invalid, n/a.
- unmatched_language is only the name of the top-level audit list. It is never a dictionary label and never a valid union_element_id.
- If no dictionary row fits, put the phrase only in the top-level unmatched_language list and do not create an annotation object for that phrase.
- If you are uncertain whether an ID/label pair is valid, do not annotate that phrase.

Annotation rules:
- Find the smallest meaningful contiguous text span for each concept when possible.
- Assign one best union_element_id and its exact source_element_label per annotation object.
- If the same span maps clearly to multiple source-model elements, output multiple annotation objects with the same span_text and a shared overlap_group_id.
- If a broad phrase and a nested narrower phrase both carry meaning, output both annotations and link them with a shared overlap_group_id.
- Sentence-level elements may be used only in sentence_level_elements, not as span annotations.
- sentence_decision must be one of: permit, deny, mixed, unclear.

Audit rules:
- You may include interpretation_units and unmatched_language for human audit.
- These audit fields will not be included directly in the backward mapping.

Data dictionary:
{dictionary_text}

Return JSON with exactly this structure:
{{
  "sentence_decision": "permit|deny|mixed|unclear",
  "sentence_level_elements": [{{"union_element_id": "...", "source_element_label": "exact dictionary label", "value": "..."}}],
  "annotations": [
    {{
      "annotation_id": "a1",
      "span_text": "exact text span",
      "union_element_id": "exact dictionary union_element_id",
      "source_element_label": "exact dictionary source_element_label",
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
      "combined_meaning": "audit only",
      "backward_mapping_decision": "audit only",
      "rationale": "brief explanation of how the annotations should be considered together"
    }}
  ],
  "unmatched_language": [{{"span_text": "exact text span", "reason": "brief reason"}}]
}}

Sentence:
{sentence}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_backward_messages(backward_packet: dict[str, Any]) -> list[dict[str, str]]:
    mapping_text = json.dumps(backward_packet, ensure_ascii=False, indent=2)
    user = UNIVERSAL_BACKWARD_USER_TEMPLATE.format(mapping_text=mapping_text)
    return [{"role": "system", "content": UNIVERSAL_BACKWARD_SYSTEM}, {"role": "user", "content": user}]


def read_done_keys(path: Path, key_field: str = "source_id") -> set[str]:
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


def write_roundtrip_csv(forward_path: Path, backward_path: Path, out_csv: Path) -> None:
    forward = load_jsonl_by_id(forward_path)
    backward = load_jsonl_by_id(backward_path)
    rows = []
    for source_id, fwd in forward.items():
        parsed = fwd.get("parsed_forward") or {}
        validation = fwd.get("validation_summary") or parsed.get("validation_summary") or {}
        annotations = parsed.get("annotations") or []
        invalid_annotations = parsed.get("invalid_annotations") or []
        routed_unmatched_annotations = parsed.get("routed_unmatched_annotations") or []
        units = parsed.get("interpretation_units") or []
        bwd = backward.get(source_id, {})
        packet = bwd.get("backward_packet", {})
        audit = bwd.get("backward_annotation_audit", {})
        strict_items = packet.get("ordered_reconstruction_items") or []
        rows.append({
            "source_id": source_id,
            "source_text": fwd.get("source_text", ""),
            "condition": "union_v0_strict",
            "sentence_decision": parsed.get("sentence_decision", ""),
            "n_annotations_raw": validation.get("n_annotations_raw", ""),
            "n_annotations_valid": validation.get("n_annotations_valid", ""),
            "n_annotations_repaired": validation.get("n_annotations_repaired", ""),
            "n_annotations_invalid": validation.get("n_annotations_invalid", ""),
            "n_annotations_routed_to_unmatched": validation.get("n_annotations_routed_to_unmatched", ""),
            "n_annotations_backward_eligible": len(strict_items),
            "n_relationship_links_backward_eligible": audit.get("n_relationship_links_backward_eligible", ""),
            "n_full_sentence_spans_dropped": audit.get("n_full_sentence_spans_dropped", ""),
            "n_interpretation_units": len(units) if isinstance(units, list) else "",
            "forward_parse_ok": fwd.get("parse_ok", False),
            "backward_parse_ok": bwd.get("parse_ok", False),
            "reconstructed_sentence": (bwd.get("parsed_backward") or {}).get("reconstructed_sentence", ""),
            "reconstruction_notes": (bwd.get("parsed_backward") or {}).get("reconstruction_notes", ""),
            "annotation_count": len(strict_items),
            "unique_element_count": len({norm_text(x.get("label_id") or x.get("label")) for x in strict_items if isinstance(x, dict) and norm_text(x.get("label_id") or x.get("label"))}),
            "backward_input_policy": packet.get("backward_input_policy", STRICT_POLICY),
            "annotations_json": json.dumps(annotations, ensure_ascii=False),
            "invalid_annotations_json": json.dumps(invalid_annotations, ensure_ascii=False),
            "routed_unmatched_annotations_json": json.dumps(routed_unmatched_annotations, ensure_ascii=False),
            "unmatched_language_json": json.dumps(parsed.get("unmatched_language") or [], ensure_ascii=False),
            "interpretation_units_json": json.dumps(units, ensure_ascii=False),
            "backward_packet_json": json.dumps(packet, ensure_ascii=False),
            "backward_annotation_audit_json": json.dumps(audit, ensure_ascii=False),
            "forward_raw": fwd.get("raw_response", ""),
            "backward_raw": bwd.get("raw_response", ""),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def write_invalid_id_audit(forward_path: Path, audit_csv: Path) -> None:
    rows = []
    forward = load_jsonl_by_id(forward_path)
    for source_id, fwd in forward.items():
        parsed = fwd.get("parsed_forward") or {}
        groups = [
            (parsed.get("annotations") or [], "valid_or_repaired"),
            (parsed.get("invalid_annotations") or [], "invalid"),
            (parsed.get("routed_unmatched_annotations") or [], "routed_to_unmatched"),
        ]
        for group, status_group in groups:
            if not isinstance(group, list):
                continue
            for ann in group:
                if not isinstance(ann, dict):
                    continue
                status = norm_text(ann.get("id_validation_status"))
                if status_group == "valid_or_repaired" and status != "repaired":
                    continue
                rows.append({
                    "source_id": source_id,
                    "source_text": fwd.get("source_text", ""),
                    "annotation_id": norm_text(ann.get("annotation_id")),
                    "span_text": norm_text(ann.get("span_text")),
                    "status": status,
                    "reason": norm_text(ann.get("id_validation_reason")),
                    "original_union_element_id": norm_text(ann.get("original_union_element_id") or ann.get("invalid_union_element_id") or ann.get("union_element_id")),
                    "resolved_union_element_id": norm_text(ann.get("union_element_id")) if status == "repaired" else "",
                    "returned_source_element_label": norm_text(ann.get("returned_source_element_label")),
                    "resolved_source_element_label": norm_text(ann.get("source_element_label")),
                    "rationale_audit_only": norm_text(ann.get("rationale")),
                })
    pd.DataFrame(rows).to_csv(audit_csv, index=False)


def run_forward(rows: pd.DataFrame, client: OpenAI, model_cfg: dict[str, Any], dictionary_text: str, maps: dict[str, Any], out_dir: Path) -> None:
    forward_path = out_dir / "union_v0_forward_mappings.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    done = read_done_keys(forward_path)
    for i, row in rows.iterrows():
        source_id = str(row["_source_id"])
        if source_id in done:
            continue
        try:
            raw = call_chat(client, model_cfg, build_forward_messages(row["_source_text"], dictionary_text))
            parsed_raw = extract_json(raw)
            parsed, validation = validate_forward_obj(parsed_raw, maps)
            append_jsonl(forward_path, {"source_id": source_id, "source_text": row["_source_text"], "model_key": model_cfg["model_key"], "model": model_cfg["model"], "stage": "forward", "parse_ok": True, "validation_summary": validation, "parsed_forward": parsed, "raw_response": raw})
            done.add(source_id)
            print(f"[forward] {i + 1}/{len(rows)} ok {source_id} valid={validation['n_annotations_valid']} repaired={validation['n_annotations_repaired']} invalid={validation['n_annotations_invalid']} routed_unmatched={validation['n_annotations_routed_to_unmatched']}")
        except Exception as exc:
            append_jsonl(failures_path, {"source_id": source_id, "stage": "forward", "error": repr(exc)})
            print(f"[forward] {i + 1}/{len(rows)} FAILED {source_id}: {exc}", file=sys.stderr)


def run_backward(client: OpenAI, model_cfg: dict[str, Any], dictionary_text: str, maps: dict[str, Any], out_dir: Path) -> None:
    _ = dictionary_text
    forward_path = out_dir / "union_v0_forward_mappings.jsonl"
    backward_path = out_dir / "union_v0_backward_reconstructions.jsonl"
    failures_path = out_dir / "failed_requests.jsonl"
    forward = load_jsonl_by_id(forward_path)
    if not forward:
        raise FileNotFoundError(f"No forward mappings found. Expected non-empty file: {forward_path}")
    done = read_done_keys(backward_path)
    for i, (source_id, fwd) in enumerate(forward.items()):
        if source_id in done:
            continue
        try:
            parsed_forward = fwd.get("parsed_forward") or {}
            source_text = fwd.get("source_text", "")
            packet, audit = build_backward_packet(parsed_forward, source_text, maps)
            if not packet.get("ordered_reconstruction_items"):
                parsed = {"reconstructed_sentence": "", "reconstruction_notes": NO_ANNOTATION_NOTE}
                raw = json.dumps(parsed, ensure_ascii=False)
            else:
                raw = call_chat(client, model_cfg, build_backward_messages(packet))
                parsed = extract_json(raw)
            append_jsonl(backward_path, {"source_id": source_id, "source_text": source_text, "model_key": model_cfg["model_key"], "model": model_cfg["model"], "stage": "backward", "parse_ok": True, "backward_input_sanitized": True, "backward_input_policy": STRICT_POLICY, "backward_packet": packet, "backward_annotation_audit": audit, "parsed_backward": parsed, "raw_response": raw})
            done.add(source_id)
            print(f"[backward] {i + 1}/{len(forward)} ok {source_id} eligible_annotations={len(packet.get('ordered_reconstruction_items') or [])} links={len(packet.get('relationship_links') or [])}")
        except Exception as exc:
            append_jsonl(failures_path, {"source_id": source_id, "stage": "backward", "error": repr(exc)})
            print(f"[backward] {i + 1}/{len(forward)} FAILED {source_id}: {exc}", file=sys.stderr)


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
    maps = build_inventory_maps(inv)
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
        "prompt_design": "overlap_aware_forward_requires_verbatim_id_and_label_and_routes_unmatched_audit",
        "id_validation": "exact_id_plus_label_validation_with_reserved_non_label_routing",
        "backward_input": STRICT_POLICY,
        "backward_prompt": "universal_annotation_dictionary_relationships",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    if args.stage in {"forward", "both"}:
        run_forward(rows, client, model_cfg, dictionary_text, maps, output_dir)
    if args.stage in {"backward", "both"}:
        run_backward(client, model_cfg, dictionary_text, maps, output_dir)

    write_roundtrip_csv(output_dir / "union_v0_forward_mappings.jsonl", output_dir / "union_v0_backward_reconstructions.jsonl", output_dir / "union_v0_roundtrip_outputs.csv")
    write_invalid_id_audit(output_dir / "union_v0_forward_mappings.jsonl", output_dir / "invalid_id_audit.csv")
    print(f"Wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
