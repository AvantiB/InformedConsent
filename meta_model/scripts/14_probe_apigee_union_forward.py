#!/usr/bin/env python
"""Probe the exact Union V0 forward prompt through Mayo Apigee/Azure.

Use this when a simple Apigee token test works but the Union V0 runner fails.
It calls the same Apigee helper and the same Union V0 forward prompt for one row,
then prints response length, a preview, and JSON parse diagnostics without writing
round-trip outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from apigee_azure_client import build_endpoint, build_payload, call_apigee_chat


def load_union_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "03_run_union_v0_roundtrip.py"
    spec = importlib.util.spec_from_file_location("union_v0_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["union_v0_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--source_id", default=None)
    ap.add_argument("--row_index", type=int, default=0)
    ap.add_argument("--preview_chars", type=int, default=3000)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mod = load_union_runner(repo_root)

    rows = mod.load_rows(Path(args.roundtrips_csv), limit=None, no_dedupe_sentences=False)
    if args.source_id:
        rows = rows[rows["_source_id"].astype(str).eq(str(args.source_id))].copy()
        if rows.empty:
            raise SystemExit(f"source_id not found: {args.source_id}")
        row = rows.iloc[0]
    else:
        if args.row_index < 0 or args.row_index >= len(rows):
            raise SystemExit(f"row_index out of range: {args.row_index}; n_rows={len(rows)}")
        row = rows.iloc[args.row_index]

    inv = mod.load_inventory(Path(args.inventory_csv))
    dictionary_text = mod.build_dictionary_text(inv)
    model_cfg = mod.load_model_config(Path(args.model_config_yaml), args.model_key)
    messages = mod.build_forward_messages(str(row["_source_text"]), dictionary_text)

    payload = build_payload(model_cfg, messages)
    print("source_id:", row["_source_id"])
    print("source_text:", row["_source_text"])
    print("endpoint:", build_endpoint(model_cfg))
    print("payload keys:", sorted(payload.keys()))
    print("max_completion_tokens:", payload.get("max_completion_tokens"))
    print("temperature in payload:", "temperature" in payload)
    print("top_p in payload:", "top_p" in payload)
    print("reasoning_effort in payload:", payload.get("reasoning_effort"))
    print("n_messages:", len(messages))
    print("system chars:", len(messages[0].get("content", "")))
    print("user chars:", len(messages[1].get("content", "")))

    raw = call_apigee_chat(None, model_cfg, messages)
    print("raw_length:", len(raw))
    print("raw_preview_begin")
    print(raw[: args.preview_chars])
    print("raw_preview_end")

    try:
        parsed = mod.extract_json(raw)
        print("json_parse_ok: true")
        print("top_level_keys:", sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
    except Exception as exc:
        print("json_parse_ok: false")
        print("json_parse_error:", repr(exc))
        print("first_200_chars_repr:", repr(raw[:200]))


if __name__ == "__main__":
    main()
