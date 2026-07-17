#!/usr/bin/env python
"""Build Union V0 source-element inventory from original forward prompts.

The Union V0 inventory is the unreduced set of source-model elements from the
original full-dictionary prompts for ICO, DUO, FHIR Consent, and ODRL. The
script is intentionally parser-first and schema-light: it preserves the source
model labels/definitions while adding minimal metadata needed for downstream
retrieval and evidence-graph construction.

The parser is conservative. It starts a new element only when a line begins with
a recognizable element identifier, appends wrapped definition lines to the
current element, and stops before annotation/output instructions. This avoids
turning wrapped definitions or prompt instructions into fake elements.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

STOP_MARKERS = [
    "annotation rules",
    "output format",
    "use header",
    "quote fields",
    "do not include explanations",
    "each row represents",
    "input:",
    "now annotate",
    "this column must appear",
    "header name:",
    "allowed values:",
    "choose the value",
    "the value must be consistent",
]

HEADER_PATTERNS = [
    r"^element\s+id\s*(?:\||\s+)\s*definition$",
    r"^element\s+id$",
    r"^definition$",
    r"^use\s+only\s+these\s+labels:?\s*$",
]

SOURCE_ALIASES = {
    "ico": "ICO",
    "duo": "DUO",
    "fhir": "FHIR_Consent",
    "fhir_consent": "FHIR_Consent",
    "odrl": "ODRL",
}

ODRL_IDS = {
    "Rule_TestSentence",
    "Permission",
    "Duty",
    "Prohibition",
    "Constraint",
    "Party",
    "Asset_DO",
    "Action_Verb",
}

SENTENCE_LEVEL_IDS = {
    ("ODRL", "Rule_TestSentence"),
    ("FHIR_Consent", "Consent.provision.type"),
}

OUTPUT_COL_IDS = {
    "ICO.decision",
    "DUO.decision",
    "Consent.provision.type",
    "Rule_TestSentence",
}


def infer_source_model(path: Path) -> str:
    stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    for key, value in SOURCE_ALIASES.items():
        if key in stem:
            return value
    return path.stem


def normalize_line_endings(text: str) -> str:
    """Normalize prompt-specific separators while preserving line structure."""
    text = str(text)
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\xa0", " ")
    )


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def is_header_line(line: str) -> bool:
    low = line.lower().strip()
    return any(re.match(pat, low) for pat in HEADER_PATTERNS)


def is_stop_line(line: str) -> bool:
    low = line.lower().strip()
    return any(marker in low for marker in STOP_MARKERS)


def looks_like_element_start(line: str, source_model: str) -> bool:
    """Return True when a line begins a source-model element/card."""
    line = normalize_space(line).lstrip("-•* ").strip()
    if not line or is_header_line(line):
        return False

    # Created (ICO:0000429) residual clinical biospecimen ...
    if re.match(r"^Created\s*\([A-Za-z][A-Za-z0-9_.-]*:[A-Za-z0-9_.-]+\)\s+\S+", line):
        return True

    # Ontology CURIEs: ICO:..., IAO:..., DUO:..., NCIT:..., OBI:...
    if re.match(r"^[A-Za-z][A-Za-z0-9_.-]*:[A-Za-z0-9_.-]+\s+\S+", line):
        return True

    # FHIR Consent fields: Consent, Consent.category, Consent.provision.actor, ...
    if source_model == "FHIR_Consent" and re.match(r"^Consent(?:\.[A-Za-z0-9_-]+)*\b", line):
        return True

    # DUO acronym table rows: GRU | General research use ...
    if source_model == "DUO" and re.match(r"^[A-Z][A-Z0-9]{1,10}\s*\|\s*\S+", line):
        return True

    # ODRL prompt labels.
    first = line.split(maxsplit=1)[0]
    if source_model == "ODRL" and first in ODRL_IDS:
        return True

    return False


def extract_dictionary_block(text: str, source_model: str) -> tuple[list[str], list[str]]:
    """Return complete source-element strings and skipped/orphan lines.

    Prompt dictionaries often wrap long definitions across multiple physical
    lines. This function appends non-element-start lines to the current element
    rather than parsing each physical line as a new element.
    """
    text = normalize_line_endings(text)
    raw_lines = [normalize_space(x) for x in text.split("\n")]

    start = 0
    for i, line in enumerate(raw_lines):
        low = line.lower()
        if "use only" in low and ("label" in low or "element" in low):
            start = i + 1
            break

    element_lines: list[str] = []
    skipped_lines: list[str] = []
    current: str | None = None

    for line in raw_lines[start:]:
        if not line or is_header_line(line):
            continue
        if is_stop_line(line):
            break

        if looks_like_element_start(line, source_model):
            if current:
                element_lines.append(current)
            current = line
        else:
            if current:
                current = normalize_space(current + " " + line)
            else:
                skipped_lines.append(line)

    if current:
        element_lines.append(current)

    return element_lines, skipped_lines


def split_label_definition_from_rest(rest: str, element_id: str, source_model: str) -> tuple[str, str]:
    rest = normalize_space(rest)
    if not rest:
        return element_id, ""

    # FHIR and ODRL prompts use the element identifier itself as the label.
    # This must happen before dash splitting because FHIR definitions contain
    # phrases like "Action to take – permit or deny".
    if source_model in {"FHIR_Consent", "ODRL"}:
        return element_id, rest

    # Prefer quoted definitions when available; this handles most ICO rows.
    quote_match = re.search(r"[\"“](.+)[\"”]", rest)
    if quote_match:
        definition = normalize_space(quote_match.group(1))
        label = normalize_space(rest[: quote_match.start()])
        return (label or element_id), definition

    # DUO rows use "Label – definition".
    dash_match = re.search(r"\s+[–—-]\s+", rest)
    if dash_match:
        label = normalize_space(rest[: dash_match.start()])
        definition = normalize_space(rest[dash_match.end() :])
        return (label or element_id), definition

    # For ontology rows without quotes, split before a common definition starter.
    m = re.search(r"\s+(A|An|The|a|an|the)\s+", rest)
    if m and m.start() > 2:
        label = normalize_space(rest[: m.start()])
        definition = normalize_space(rest[m.start() :])
        return (label or element_id), definition

    return rest, ""


def split_element_line(line: str, source_model: str) -> dict[str, str]:
    raw = normalize_space(line)
    cleaned = normalize_space(raw.replace("***", " "))

    element_id = ""
    rest = ""

    created = re.match(r"^Created\s*\(([^)]+)\)\s+(.+)$", cleaned)
    pipe = re.match(r"^([A-Z][A-Z0-9]{1,10})\s*\|\s*(.+)$", cleaned)
    curie = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*:[A-Za-z0-9_.-]+)\s+(.+)$", cleaned)
    fhir = re.match(r"^(Consent(?:\.[A-Za-z0-9_-]+)*)\s+(.+)$", cleaned)
    odrl_ids_regex = "|".join(re.escape(x) for x in sorted(ODRL_IDS, key=len, reverse=True))
    odrl = re.match(rf"^({odrl_ids_regex})\s+(.+)$", cleaned)

    if created:
        element_id = created.group(1).strip()
        rest = created.group(2).strip()
    elif source_model == "DUO" and pipe:
        element_id = pipe.group(1).strip()
        rest = pipe.group(2).strip()
    elif curie:
        element_id = curie.group(1).strip()
        rest = curie.group(2).strip()
    elif source_model == "FHIR_Consent" and fhir:
        element_id = fhir.group(1).strip()
        rest = fhir.group(2).strip()
    elif source_model == "ODRL" and odrl:
        element_id = odrl.group(1).strip()
        rest = odrl.group(2).strip()
    else:
        parts = cleaned.split(maxsplit=1)
        element_id = parts[0] if parts else cleaned
        rest = parts[1] if len(parts) > 1 else ""

    label, definition = split_label_definition_from_rest(rest, element_id, source_model)
    element_scope = "sentence_level" if (source_model, element_id) in SENTENCE_LEVEL_IDS else "span"
    searchable_text = normalize_space(" ".join([element_id, label, definition, raw]))

    return {
        "source_element_id": element_id,
        "source_element_label": label,
        "source_element_definition": definition,
        "element_scope": element_scope,
        "is_sentence_level": str(element_scope == "sentence_level"),
        "source_prompt_text": raw,
        "searchable_text": searchable_text,
    }


def build_inventory(prompt_paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for path in prompt_paths:
        source_model = infer_source_model(path)
        text = path.read_text(errors="ignore")
        element_lines, skipped_lines = extract_dictionary_block(text, source_model)

        for rank, line in enumerate(element_lines, start=1):
            parsed = split_element_line(line, source_model)
            parsed.update({
                "source_model": source_model,
                "source_prompt_file": str(path),
                "source_order": rank,
                "union_element_id": f"{source_model}::{parsed['source_element_id']}",
            })
            rows.append(parsed)

        audit_rows.append({
            "source_prompt_file": str(path),
            "source_model": source_model,
            "n_parsed_elements": len(element_lines),
            "n_skipped_orphan_lines_before_first_element": len(skipped_lines),
            "skipped_orphan_lines_preview": " || ".join(skipped_lines[:5]),
        })

    df = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if df.empty:
        return df, audit

    cols = [
        "union_element_id",
        "source_model",
        "source_element_id",
        "source_element_label",
        "source_element_definition",
        "element_scope",
        "is_sentence_level",
        "source_prompt_text",
        "searchable_text",
        "source_prompt_file",
        "source_order",
    ]
    df = df[cols].drop_duplicates(subset=["union_element_id", "source_prompt_text"])
    return df, audit


def write_cards(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            card = {
                "union_element_id": row["union_element_id"],
                "source_model": row["source_model"],
                "source_element_id": row["source_element_id"],
                "label": row["source_element_label"],
                "definition": row["source_element_definition"],
                "element_scope": row["element_scope"],
                "card_text": (
                    f"[{row['source_model']}] {row['source_element_id']} | "
                    f"{row['source_element_label']} | {row['source_element_definition']}"
                ),
            }
            f.write(json.dumps(card, ensure_ascii=False) + "\n")


def approximate_token_count(text: str) -> int:
    # Lightweight approximation for prompt-size planning.
    return max(1, int(len(normalize_line_endings(text).split()) * 1.3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt_dir",
        required=True,
        help="Directory containing original source-model forward prompts/data dictionaries.",
    )
    ap.add_argument("--glob", default="*.txt", help="File glob for prompt files. Default: *.txt")
    ap.add_argument("--output_dir", default="meta_model/v0_union")
    args = ap.parse_args()

    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = sorted(prompt_dir.glob(args.glob))
    if not prompt_paths:
        raise FileNotFoundError(f"No prompt files matched {prompt_dir / args.glob}")

    inventory, parse_audit = build_inventory(prompt_paths)
    if inventory.empty:
        raise ValueError("No source elements were parsed. Check prompt formatting or parser markers.")

    inventory.to_csv(output_dir / "source_element_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_cards(inventory, output_dir / "element_cards.jsonl")
    parse_audit.to_csv(output_dir / "parse_audit.csv", index=False)

    prompt_size_rows = []
    for path in prompt_paths:
        text = path.read_text(errors="ignore")
        prompt_size_rows.append({
            "source_prompt_file": str(path),
            "source_model": infer_source_model(path),
            "approx_tokens_full_prompt": approximate_token_count(text),
            "n_parsed_elements": int((inventory["source_prompt_file"] == str(path)).sum()),
        })
    prompt_sizes = pd.DataFrame(prompt_size_rows)
    union_prompt_text = "\n".join(inventory["source_prompt_text"].astype(str).tolist())
    prompt_sizes.loc[len(prompt_sizes)] = {
        "source_prompt_file": "UNION_V0_ELEMENTS_ONLY",
        "source_model": "UNION_V0",
        "approx_tokens_full_prompt": approximate_token_count(union_prompt_text),
        "n_parsed_elements": len(inventory),
    }
    prompt_sizes.to_csv(output_dir / "source_model_prompt_sizes.csv", index=False)

    print(f"Wrote {len(inventory):,} Union V0 elements to {output_dir / 'source_element_inventory.csv'}")
    print(f"Wrote element cards to {output_dir / 'element_cards.jsonl'}")
    print(f"Wrote parse audit to {output_dir / 'parse_audit.csv'}")
    print(f"Wrote prompt-size summary to {output_dir / 'source_model_prompt_sizes.csv'}")


if __name__ == "__main__":
    main()
