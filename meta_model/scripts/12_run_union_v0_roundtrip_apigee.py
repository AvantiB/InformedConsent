#!/usr/bin/env python
"""Run Union V0 round trips through the Mayo Apigee Azure OpenAI endpoint.

This wrapper reuses 03_run_union_v0_roundtrip.py and only swaps the chat client.
Use it for model config entries with provider: mayo_apigee_azure_openai.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from apigee_azure_client import call_apigee_chat


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
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mod = load_union_runner(repo_root)
    mod.call_chat = call_apigee_chat

    output_dir = Path(args.output_dir) / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = mod.load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    inv = mod.load_inventory(Path(args.inventory_csv))
    dictionary_text = mod.build_dictionary_text(inv)
    maps = mod.build_inventory_maps(inv)
    model_cfg = mod.load_model_config(Path(args.model_config_yaml), args.model_key)

    provider = str(model_cfg.get("provider", ""))
    if provider not in {"mayo_apigee_azure_openai", "apigee_azure_openai"}:
        print(f"[WARN] model_key={args.model_key} provider={provider!r}; still using Apigee wrapper.")

    run_meta = {
        "model_key": args.model_key,
        "model": model_cfg.get("model"),
        "deployment": model_cfg.get("deployment") or model_cfg.get("engine") or model_cfg.get("model"),
        "provider": provider,
        "stage": args.stage,
        "n_input_rows": int(len(rows)),
        "n_union_elements": int(len(inv)),
        "inventory_csv": args.inventory_csv,
        "roundtrips_csv": args.roundtrips_csv,
        "prompt_design": "overlap_aware_forward_requires_verbatim_id_label_and_controlled_sentence_decisions",
        "id_validation": "exact_id_plus_label_validation_with_reserved_non_label_routing",
        "sentence_level_backward_policy": "controlled_decision_values_only_no_explanatory_summaries",
        "backward_input": mod.STRICT_POLICY,
        "backward_prompt": "universal_annotation_dictionary_relationships",
        "chat_transport": "mayo_apigee_azure_openai",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    client = None
    if args.stage in {"forward", "both"}:
        mod.run_forward(rows, client, model_cfg, dictionary_text, maps, output_dir)
    if args.stage in {"backward", "both"}:
        mod.run_backward(client, model_cfg, dictionary_text, maps, output_dir)

    mod.write_roundtrip_csv(output_dir / "union_v0_forward_mappings.jsonl", output_dir / "union_v0_backward_reconstructions.jsonl", output_dir / "union_v0_roundtrip_outputs.csv")
    mod.write_invalid_id_audit(output_dir / "union_v0_forward_mappings.jsonl", output_dir / "invalid_id_audit.csv")
    print(f"Wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
