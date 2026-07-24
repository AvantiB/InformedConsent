#!/usr/bin/env python
"""Run individual source-model round trips through Mayo Apigee Azure OpenAI.

This wrapper reuses 05_run_individual_model_roundtrip.py and only swaps the chat
client. Use it for model config entries with provider: mayo_apigee_azure_openai.
Primary Phase 1 individual runs use a universal forward prompt and source-model
specific dictionaries derived from the inventory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from apigee_azure_client import call_apigee_chat


def load_individual_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "05_run_individual_model_roundtrip.py"
    spec = importlib.util.spec_from_file_location("individual_roundtrip_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["individual_roundtrip_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrips_csv", required=True)
    ap.add_argument("--prompt_dir", default=None, help="Deprecated/reference only. Primary Phase 1 uses universal dictionary-based prompts.")
    ap.add_argument("--backward_prompt_dir", default=None)
    ap.add_argument("--inventory_csv", default="meta_model/v0_union/source_element_inventory.csv", help="Union/source inventory used as authoritative per-model dictionary.")
    ap.add_argument("--model_config_yaml", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--info_models", default="all", help="Comma-separated list or all")
    ap.add_argument("--stage", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_dedupe_sentences", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mod = load_individual_runner(repo_root)
    mod.call_chat = call_apigee_chat

    info_models = mod.INFO_MODELS if args.info_models == "all" else [x.strip() for x in args.info_models.split(",") if x.strip()]
    unknown = [m for m in info_models if m not in mod.INFO_MODELS]
    if unknown:
        raise ValueError(f"Unknown info_models: {unknown}. Allowed: {mod.INFO_MODELS}")

    rows = mod.load_rows(Path(args.roundtrips_csv), args.limit, args.no_dedupe_sentences)
    model_cfg = mod.load_model_config(Path(args.model_config_yaml), args.model_key)
    provider = str(model_cfg.get("provider", ""))
    if provider not in {"mayo_apigee_azure_openai", "apigee_azure_openai"}:
        print(f"[WARN] model_key={args.model_key} provider={provider!r}; still using Apigee wrapper.")

    inventory_csv = Path(args.inventory_csv)
    inv = mod.load_inventory(inventory_csv)
    label_lookup = mod.load_label_lookup(inventory_csv)
    backward_dir = Path(args.backward_prompt_dir) if args.backward_prompt_dir else None
    base_out = Path(args.output_dir) / args.model_key
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "run_metadata.json").write_text(json.dumps({
        "model_key": args.model_key,
        "model": model_cfg.get("model"),
        "deployment": model_cfg.get("deployment") or model_cfg.get("engine") or model_cfg.get("model"),
        "provider": provider,
        "n_input_rows": int(len(rows)),
        "info_models": info_models,
        "roundtrips_csv": args.roundtrips_csv,
        "prompt_dir_deprecated_reference_only": args.prompt_dir,
        "inventory_csv": args.inventory_csv,
        "backward_prompt_dir_deprecated_not_used": args.backward_prompt_dir,
        "stage": args.stage,
        "prompt_design": "universal_forward_prompt_with_source_model_dictionary_only",
        "id_validation": "source_model_inventory_label_validation_with_reserved_non_label_routing",
        "sentence_level_backward_policy": "controlled_decision_values_only_no_explanatory_summaries",
        "zero_annotation_policy": "no_backward_llm_call_and_exclude_from_schema_induction",
        "backward_input": mod.STRICT_POLICY,
        "backward_prompt": "universal_annotation_dictionary_relationships",
        "chat_transport": "mayo_apigee_azure_openai",
    }, indent=2))

    client = None
    for info_model in info_models:
        source_inv = mod.inventory_for_info_model(inv, info_model)
        dictionary_text = mod.build_source_dictionary_text(source_inv, info_model)
        backward_path = mod.find_backward_prompt_file(backward_dir, info_model)
        backward_text = backward_path.read_text(errors="replace") if backward_path else None
        out_dir = base_out / info_model
        out_dir.mkdir(parents=True, exist_ok=True)
        legacy_prompt = None
        if args.prompt_dir:
            try:
                legacy_prompt = str(mod.find_prompt_file(Path(args.prompt_dir), info_model))
            except Exception:
                legacy_prompt = None
        (out_dir / "prompt_files.json").write_text(json.dumps({
            "legacy_forward_prompt_file_deprecated_not_used": legacy_prompt,
            "backward_prompt_file_deprecated_not_used": str(backward_path) if backward_path else None,
            "uses_universal_dictionary_forward_prompt": True,
            "uses_universal_structured_backward_prompt": True,
            "dictionary_source_model": info_model,
            "dictionary_rows": int(len(source_inv)),
            "backward_input_policy": mod.STRICT_POLICY,
            "strict_forward_contract_applied": True,
        }, indent=2))
        mod.run_info_model(rows, client, model_cfg, info_model, dictionary_text, backward_text, out_dir, args.stage, label_lookup)

    print(f"Wrote individual-model outputs under {base_out}")


if __name__ == "__main__":
    main()
