#!/usr/bin/env python
"""Phase 1 individual source-model Mayo Apigee runner with sentence decisions separated from labels."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from apigee_azure_client import call_apigee_chat


def load_phase1_runner(repo_root: Path):
    script_path = repo_root / "meta_model" / "scripts" / "44_run_phase1_individual_roundtrip.py"
    spec = importlib.util.spec_from_file_location("phase1_individual_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase1_individual_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    phase1 = load_phase1_runner(repo_root)
    base = phase1.apply_phase1_sentence_decision_policy(phase1.load_base_runner(repo_root))
    base.call_chat = call_apigee_chat
    base.main()


if __name__ == "__main__":
    main()
