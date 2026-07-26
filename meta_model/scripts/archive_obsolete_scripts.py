#!/usr/bin/env python
"""Move superseded meta-model scripts into meta_model/scripts/archive/.

Run from the repository root. By default this script prints what would be moved.
Use --apply to perform the moves with git mv when available.

This is intentionally conservative: it archives older exploratory/manual V1,
PI-review-package, and pre-corrected duplicate runner scripts, while leaving
current Phase 1, direct LLM, data-driven LLM, scoring, and compiler scripts in
place.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ARCHIVE_FILES = [
    # Superseded API/probe/repair variants from pre-Phase-1 cleanup.
    "meta_model/scripts/10_repair_union_v0_parse_failures.py",
    "meta_model/scripts/11_backfill_union_v0_compact_forward.py",
    "meta_model/scripts/12_run_union_v0_roundtrip_apigee.py",
    "meta_model/scripts/13_run_individual_model_roundtrip_apigee.py",
    "meta_model/scripts/14_probe_apigee_union_forward.py",
    # Exploratory analysis scripts no longer used in the current held-out evaluation path.
    "meta_model/scripts/15_analyze_roundtrip_scored_outputs.py",
    "meta_model/scripts/16_audit_annotation_granularity.py",
    # Older held-out/manual/functional V1 pipeline.
    "meta_model/scripts/23_refined_metamodel_cv_pipeline.py",
    "meta_model/scripts/24_refined_cv_postprocess.py",
    "meta_model/scripts/25_make_heldout_roundtrips.py",
    "meta_model/scripts/26_build_functional_v1_crosswalk.py",
    "meta_model/scripts/27_run_functional_v1_roundtrip.py",
    "meta_model/scripts/28_build_llm_schema_induction_cards.py",
    "meta_model/scripts/29_induce_functional_schema_with_llm.py",
    "meta_model/scripts/30_relabel_functional_v1_outputs.py",
    "meta_model/scripts/31_compile_schema_condition_comparison.py",
    # Old expert-review/PI package builders tied to pre-corrected/manual V1 outputs.
    "meta_model/scripts/34_build_expert_review_examples.py",
    "meta_model/scripts/35_build_v1_crosswalk_review_tables.py",
    "meta_model/scripts/36_prepare_pi_expert_review_package.py",
    "meta_model/scripts/37_rebuild_pi_expert_review_package_v2.py",
    "meta_model/scripts/38_build_expert_review_workbook.py",
    "meta_model/scripts/39_prepare_pi_expert_review_package_v3.py",
]


def git_mv(src: Path, dst: Path) -> None:
    try:
        subprocess.run(["git", "mv", str(src), str(dst)], check=True)
    except Exception:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Actually move files. Without this, only print the plan.")
    args = ap.parse_args()

    root = Path.cwd()
    archive_dir = root / "meta_model/scripts/archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    missing = []
    for rel in ARCHIVE_FILES:
        src = root / rel
        if src.exists():
            dst = archive_dir / src.name
            existing.append((src, dst))
        else:
            missing.append(rel)

    print("Files to archive:")
    for src, dst in existing:
        print(f"  {src} -> {dst}")
    if missing:
        print("\nFiles not found; skipped:")
        for rel in missing:
            print(f"  {rel}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to move files.")
        return

    for src, dst in existing:
        if dst.exists():
            raise FileExistsError(f"Archive target already exists: {dst}")
        git_mv(src, dst)
    print(f"\nArchived {len(existing)} files into {archive_dir}")


if __name__ == "__main__":
    main()
