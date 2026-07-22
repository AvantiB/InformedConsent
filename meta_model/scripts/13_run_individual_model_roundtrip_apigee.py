#!/usr/bin/env python
"""Run individual source-model round trips through Mayo Apigee Azure OpenAI.

This wrapper reuses 05_run_individual_model_roundtrip.py and only swaps the chat
client. Use it for model config entries with provider: mayo_apigee_azure_openai.
Backward reconstruction inherits the universal strict annotation-only policy from
05_run_individual_model_roundtrip.py.
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
    ap.add_argument("--prompt_dir", required=True)
    ap.add_argument("--backward_prompt_dir", default=None)
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

    prompt_dir = Path(args.prompt_dir)
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
        "prompt_dir": args.prompt_dir,
        "backward_prompt_dir_deprecated_not_used": args.backward_prompt_dir,
        "stage": args.stage,
        "backward_input": mod.STRICT_POLICY,
        "backward_prompt": "universal_strict_annotation_only",
        "chat_transport": "mayo_apigee_azure_openai",
    }, indent=2))

    client = None
    for info_model in info_models:
        prompt_path = mod.find_prompt_file(prompt_dir, info_model)
        backward_path = mod.find_backward_prompt_file(backward_dir, info_model)
        prompt_text = prompt_path.read_text(errors="replace")
        backward_text = backward_path.read_text(errors="replace") if backward_path else None
        out_dir = base_out / info_model
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt_files.json").write_text(json.dumps({
            "forward_prompt_file": str(prompt_path),
            "backward_prompt_file_deprecated_not_used": str(backward_path) if backward_path else None,
            "uses_universal_strict_backward_prompt": True,
            "backward_input_policy": mod.STRICT_POLICY,
        }, indent=2))
        mod.run_info_model(rows, client, model_cfg, info_model, prompt_text, backward_text, out_dir, args.stage)

    print(f"Wrote individual-model outputs under {base_out}")


if __name__ == "__main__":
    main()
