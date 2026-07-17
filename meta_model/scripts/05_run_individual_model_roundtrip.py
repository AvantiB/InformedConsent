#!/usr/bin/env python
"""Run individual source-model prompt round-trip experiments.

This script is intended for the replication condition:
new LLMs + original individual source-model prompts.

It reuses the same OpenAI-compatible model config used by the Union V0 runner.
The forward step uses the original source-model forward prompt text as the data
annotation prompt. The backward step uses either a matching backward prompt, if
provided, or a generic reconstruction prompt that does not see the original
sentence.

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

TEXT_COL_CANDIDATES = [
    "canonical_full_text",
    "full_text_original",
    "original_sentence",
    "full_text",
    "sentence",
    "text",
]
ID_COL_CANDIDATES = ["sentence_id", "source_sentence_id", "roundtrip_id", "id"]
INFO_MODELS = ["DUO", "ICO", "ODRL", "FHIR_Consent"]

PROMPT_PATTERNS = {
    "DUO": [r"duo"],
    "ICO": [r"ico"],
    "ODRL": [r"odrl"],
    "FHIR_Consent": [r"fhir", r"r03_fhir"],
}


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
        raise ValueError(f"Could not find any of columns {candidates}.