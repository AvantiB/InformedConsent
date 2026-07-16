#!/usr/bin/env python
"""Build Union V0 source-element inventory from original full-dictionary prompts.

This script treats each original information-model prompt as an input data
dictionary. It extracts the allowed labels/elements and writes a combined Union
V0 inventory. The inventory is intentionally unreduced and can be used either as
a full-dictionary baseline or as the source for retrieval-augmented candidate
selection.
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
]

SOURCE_ALIASES = {
    "ico": "ICO",
    "duo": "DUO",
    "fhir": "FHIR_Consent",
    "fhir_consent": "FHIR_Consent",
    "odrl": "ODRL",
}


def infer_source_model(path: Path) -> str:
    stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    for key, value in SOURCE_ALIASES.items():
        if key in stem:
            return value
    return path.stem


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\u2028", " ").replace("\u2029", " ")).strip()


def extract_dictionary_block(text: str) -> list[str]:
    """Return likely element lines from a prompt.

    The original prompts generally contain a phrase such as "Use ONLY these
    labels:" followed by a table/list of element IDs and definitions. This parser
    is intentionally permissive because prompt formatting differs across models.
    """
    lines = [normalize_space(x) for x in text.splitlines()]
    start = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "use only" in low and ("label" in low or "element" in low):
            start = i + 1
            break
    out: list[str] = []
    for line in lines[start:]:
        if not line:
            continue
        low = line.lower()
        if any(marker in low for marker in STOP_MARKERS):
            break
        if low in {"element id definition", "element id", "definition"}:
            continue
        out.append(line)
    return out


def split_element_line(line: str) -> dict:
    raw = normalize_space(line)
    cleaned = raw.replace("***", " ")
    cleaned = normalize_space(cleaned)

    element_id = ""
    label = ""
    definition = ""

    created = re.match(r"^Created\s*\(([^)]+)\)\s+(.+)$", cleaned)
    curie = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*:[A-Za-z0-9_.-]+)\s+(.+)$", cleaned)
    dotted = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*(?:\.[A-Za-z0-9_.-]+)+)\s+(.+)$", cleaned)

    if created:
        element_id = created.group(1).strip()
        rest = created.group(2).strip()
    elif curie:
        element_id = curie.group(1).strip()
        rest = curie.group(2).strip()
    elif dotted:
        element_id = dotted.group(1).strip()
        rest = dotted.group(2).strip()
    else:
        parts = cleaned.split(maxsplit=1)
        element_id = parts[0] if parts else cleaned
        rest = parts[1] if len(parts) > 1 else ""

    # Prefer quoted definitions when available.
    quote_match = re.search(r"[\"“](.+?)[\"”]", rest)
    if quote_match:
        definition = quote_match.group(1).strip()
        label = normalize_space(rest[: quote_match.start()])
    else:
        # Heuristic split at common definition starts.
        m = re.search(r"\s+(A|An|The|a|an|the)\s+", rest)
        if m and m.start() > 2:
            label = normalize_space(rest[: m.start()])
            definition = normalize_space(rest[m.start() :])
        else:
            label = rest
            definition = ""

    if not label:
        label = element_id
    searchable_text = normalize_space(" ".join([element_id, label, definition, raw]))
    return {
        "source_element_id": element_id,
        "source_element_label": label,
        "source_element_definition": definition,
        "source_prompt_text": raw,
        "searchable_text": searchable_text,
    }


def build_inventory(prompt_paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in prompt_paths:
        source_model = infer_source_model(path)
        text = path.read_text(errors="ignore")
        element_lines = extract_dictionary_block(text)
        for rank, line in enumerate(element_lines, start=1):
            parsed = split_element_line(line)
            parsed.update({
                "source_model": source_model,
                "source_prompt_file": str(path),
                "source_order": rank,
                "union_element_id": f"{source_model}::{parsed['source_element_id']}",
            })
            rows.append(parsed)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = [
        "union_element_id",
        "source_model",
        "source_element_id",
        "source_element_label",
        "source_element_definition",
        "source_prompt_text",
        "searchable_text",
        "source_prompt_file",
        "source_order",
    ]
    return df[cols].drop_duplicates(subset=["union_element_id", "source_prompt_text"])


def write_cards(df: pd.DataFrame, path: Path) -> None:
    with path.open("w") as f:
        for _, row in df.iterrows():
            card = {
                "union_element_id": row["union_element_id"],
                "source_model": row["source_model"],
                "source_element_id": row["source_element_id"],
                "label": row["source_element_label"],
                "definition": row["source_element_definition"],
                "card_text": f"[{row['source_model']}] {row['source_element_id']} | {row['source_element_label']} | {row['source_element_definition']}",
            }
            f.write(json.dumps(card, ensure_ascii=False) + "\n")


def approximate_token_count(text: str) -> int:
    # Lightweight approximation for prompt-size planning.
    return max(1, int(len(text.split()) * 1.3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_dir", required=True, help="Directory containing original source-model forward prompts/data dictionaries.")
    ap.add_argument("--glob", default="*.txt", help="File glob for prompt files. Default: *.txt")
    ap.add_argument("--output_dir", default="meta_model/v0_union")
    args = ap.parse_args()

    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = sorted(prompt_dir.glob(args.glob))
    if not prompt_paths:
        raise FileNotFoundError(f"No prompt files matched {prompt_dir / args.glob}")

    inventory = build_inventory(prompt_paths)
    if inventory.empty:
        raise ValueError("No source elements were parsed. Check prompt formatting or parser markers.")

    inventory.to_csv(output_dir / "source_element_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_cards(inventory, output_dir / "element_cards.jsonl")

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
    print(f"Wrote prompt-size summary to {output_dir / 'source_model_prompt_sizes.csv'}")


if __name__ == "__main__":
    main()
