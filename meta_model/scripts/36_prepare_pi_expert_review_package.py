#!/usr/bin/env python
"""Prepare a PI-facing expert review package after all round-trip experiments finish.

This orchestrates the final analysis/presentation outputs:
1. score standardized round trips, when needed
2. compute holistic diagnostic metrics
3. compile schema-condition comparisons
4. generate result visualizations
5. build highlighted annotation examples
6. build Manual V1 and LLM-induced V1 crosswalk review tables
7. write README and email-summary draft
8. optionally create a zip archive

The script assumes the round-trip outputs have already been generated. It does not
run any LLMs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "meta_model" / "scripts"


def exists(p: str | Path | None) -> bool:
    return bool(p) and Path(p).exists()


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"SKIP: missing {path}")
        return None
    return pd.read_csv(path)


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def summarize_csv(path: Path, max_rows: int = 8) -> str:
    df = read_csv(path)
    if df is None or df.empty:
        return "Not available."
    return df.head(max_rows).to_markdown(index=False)


def condition_snapshot(diag_csv: Path) -> pd.DataFrame:
    df = read_csv(diag_csv)
    if df is None or df.empty or "condition" not in df.columns:
        return pd.DataFrame()
    score_col = next((c for c in ["meaning_preserved_score", "predicted_probability", "probability", "score", "meaning_preserved_pred"] if c in df.columns), None)
    agg = {}
    if score_col:
        agg[score_col] = "mean"
    for c in ["content_word_recall", "important_category_presence_recall", "modal_word_change_ratio", "annotation_count", "unique_element_count", "forward_parse_ok", "backward_parse_ok", "suspected_error_count"]:
        if c in df.columns:
            agg[c] = "mean"
    if not agg:
        return pd.DataFrame()
    out = df.groupby("condition", dropna=False).agg(agg).reset_index()
    rename = {
        score_col: "mean_classifier_score" if score_col else score_col,
        "content_word_recall": "mean_content_recall",
        "important_category_presence_recall": "mean_cue_category_recall",
        "modal_word_change_ratio": "mean_modal_change",
        "annotation_count": "mean_annotations",
        "unique_element_count": "mean_unique_fields",
        "forward_parse_ok": "forward_parse_rate",
        "backward_parse_ok": "backward_parse_rate",
        "suspected_error_count": "mean_error_flags",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k})


def write_package_readme(package_dir: Path, out_root: Path, snapshot: pd.DataFrame) -> None:
    plots = sorted((package_dir / "plots").glob("*.png")) if (package_dir / "plots").exists() else []
    lines = [
        "# PI Expert Review Package: Functional Informed Consent Meta-Model",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "This package supports PI/domain-expert review of the informed-consent meta-model experiments. It combines process evidence, source-model crosswalks, highlighted annotation examples, quantitative evaluation results, and qualitative error-review samples.",
        "",
        "## Recommended review order",
        "",
        "1. Open `summary_for_pi.md` for the narrative summary.",
        "2. Review `plots/` for process and results figures.",
        "3. Open `expert_review_examples/expert_review_examples.html` for highlighted annotation/reconstruction examples.",
        "4. Review `crosswalks/v1_crosswalk_review_summary.html` and the CSV tables for Manual V1 and LLM-induced V1 source-model mappings.",
        "5. Use `diagnostics/qualitative_relationship_error_review_sample.csv` to adjudicate representative errors.",
        "",
        "## Key folders",
        "",
        "```text",
        "plots/                    Paper/deck-ready figures",
        "comparison/               Schema-condition summary CSVs",
        "diagnostics/              Row-level and aggregate diagnostic metrics",
        "expert_review_examples/    Highlighted annotation examples for review",
        "crosswalks/               Manual V1 and LLM-induced V1 crosswalk tables",
        "classifier/               Classifier training summary and feature outputs, when available",
        "```",
        "",
        "## Schema conditions",
        "",
        "```text",
        "individual_source_model_json        Individual ICO/DUO/ODRL/FHIR round trips",
        "union_v0_full_dictionary            Naive union of source-model elements",
        "functional_v1_manual                Data-seeded manually organized functional schema",
        "functional_v1_llm_induced           Fold-specific LLM-induced functional schemas",
        "functional_v1_llm_induced_consensus Optional descriptive post-CV consensus schema, if run",
        "```",
        "",
        "## Snapshot of available final metrics",
        "",
    ]
    if not snapshot.empty:
        lines.append(snapshot.to_markdown(index=False))
    else:
        lines.append("Metric snapshot was not available. Check whether diagnostics finished successfully.")
    lines += [
        "",
        "## Available plots",
        "",
    ]
    if plots:
        for p in plots:
            lines.append(f"- `plots/{p.name}`")
    else:
        lines.append("No plots were copied. Check the visualization script output.")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Use fold-specific LLM-induced schemas for the primary held-out CV result.",
        "- Treat the LLM-induced consensus schema as a post-CV artifact for expert review or descriptive validation.",
        "- Qualitative relationship-error flags are triage signals, not ground-truth labels.",
        "- Permit/deny captures sentence-level polarity, but negation/prohibition should also be reviewed through cue-preservation and qualitative error flags.",
    ]
    (package_dir / "README_PI_REVIEW_PACKAGE.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary_for_pi(package_dir: Path, snapshot: pd.DataFrame) -> None:
    text = [
        "# Summary for PI Review",
        "",
        "## What was evaluated",
        "",
        "We compared four modeling strategies for informed-consent round-trip meaning preservation:",
        "",
        "1. Individual information models: ICO, DUO, ODRL, and FHIR Consent.",
        "2. Union V0: a naive combined dictionary of source-model elements.",
        "3. Manual Functional V1: a data-seeded, analyst-organized reduced functional schema.",
        "4. LLM-induced Functional V1: fold-specific schemas induced from evidence cards by a fixed strong induction model, then tested on held-out forms.",
        "",
        "## What experts are being asked to review",
        "",
        "- Whether Manual V1 fields are appropriately defined, too broad, too narrow, redundant, or missing important consent functions.",
        "- Whether LLM-induced V1 fields identify useful semantic boundaries or introduce confusing/unsafe merges.",
        "- Whether the source-model crosswalks correctly map ICO/DUO/ODRL/FHIR elements into Manual V1 and LLM-induced V1 fields.",
        "- Whether highlighted examples preserve the meaning of the original consent sentence after reconstruction.",
        "",
        "## Available quantitative evidence",
        "",
        "The package includes classifier-based meaning preservation, content-word retention, cue/category retention, modal-word change ratio, annotation burden, parse success, unmatched-language rate, and qualitative error flags.",
        "",
    ]
    if not snapshot.empty:
        text += ["### Metric snapshot", "", snapshot.to_markdown(index=False), ""]
    text += [
        "## Suggested expert adjudication focus",
        "",
        "1. Start with `expert_review_examples/expert_review_examples.html` and inspect examples where Manual V1 and LLM-induced V1 differ.",
        "2. Use `crosswalks/manual_v1_source_model_crosswalk_for_review.csv` and `crosswalks/llm_induced_v1_source_model_crosswalk_by_fold_for_review.csv` to review field alignment.",
        "3. Use `plots/07_functional_role_x_source_model_heatmap.png` and the crosswalk tables to discuss source-model overlap and complementarity.",
        "4. Use `plots/01_condition_x_llm_classifier_score_heatmap.png`, `plots/03_holistic_metrics_by_condition.png`, and `plots/04_annotation_burden_by_condition.png` to evaluate performance vs usability tradeoffs.",
        "5. Use `diagnostics/qualitative_relationship_error_review_sample.csv` to identify schema failures that may not be fully captured by the classifier score.",
        "",
        "## Preliminary interpretation template",
        "",
        "- If Manual V1 performs similarly to Union V0 with fewer annotations, this supports a compact expert-organized functional schema.",
        "- If LLM-induced V1 performs similarly to Manual V1, this supports LLM-assisted schema discovery as a useful development strategy.",
        "- If one reduced schema has weaker classifier score but much lower annotation burden, interpret it as a preservation-usability tradeoff rather than a simple failure.",
        "- Any polarity/negation flips, actor-resource-action swaps, or temporal attachment errors should be prioritized for expert adjudication.",
    ]
    (package_dir / "summary_for_pi.md").write_text("\n".join(text), encoding="utf-8")


def write_email_draft(package_dir: Path) -> None:
    text = """Subject: Informed consent meta-model review package and annotation examples

Dear [PI/team],

I have prepared a review package summarizing the informed-consent meta-model development and evaluation results. The package includes:

1. Quantitative comparisons across individual information models, Union V0, Manual Functional V1, and LLM-induced Functional V1.
2. Meaning-preservation classifier results and diagnostic metrics, including content-word retention, cue/category preservation, modal-word change ratio, annotation burden, parse success, unmatched-language rate, and qualitative error flags.
3. Highlighted annotation examples showing original sentences, parsed forward annotations, and backward reconstructions across modeling strategies.
4. Crosswalk tables mapping source-model elements from ICO, DUO, ODRL, and FHIR Consent into Manual V1 and LLM-induced V1 fields.
5. Visualizations of the schema-reduction process, source-model overlap, seed/evidence-card recurrence, classifier insights, and final evaluation results.

For review, I would suggest starting with `summary_for_pi.md`, then opening `expert_review_examples/expert_review_examples.html` for highlighted examples and `crosswalks/v1_crosswalk_review_summary.html` for the source-model crosswalks.

The main decisions we would appreciate expert input on are:
- whether Manual V1 fields should be kept, merged, split, renamed, or expanded;
- whether LLM-induced V1 identifies useful additional functional boundaries;
- whether any source-model mappings appear incorrect or context-dependent;
- whether reconstruction errors suggest missing fields or unclear field definitions.

Best,
[Your Name]
"""
    (package_dir / "email_draft_to_pi.md").write_text(text, encoding="utf-8")


def copy_result_folders(out_root: Path, package_dir: Path) -> None:
    for folder in ["comparison", "diagnostics", "plots"]:
        src = out_root / folder
        dst = package_dir / folder
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    classifier_dir = REPO_ROOT / "meta_model" / "outputs" / "final_classifier"
    if classifier_dir.exists():
        dst = package_dir / "classifier"
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ["final_classifier_training_summary.json", "training_features.csv", "classifier_feature_mean_differences.csv", "classifier_feature_importance_top25.csv"]:
            copy_if_exists(classifier_dir / fname, dst)


def archive_package(package_dir: Path) -> Path:
    zip_base = package_dir.with_suffix("")
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=package_dir)
    return Path(zip_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", default="meta_model/functional_v1_experiments")
    ap.add_argument("--package_dir", default="", help="Defaults to <out_root>/pi_expert_review_package")
    ap.add_argument("--standardized_csv", default="", help="Defaults to <out_root>/scoring_inputs/standardized_roundtrips.csv")
    ap.add_argument("--classifier_bundle", default="meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib")
    ap.add_argument("--manual_crosswalk_csv", default="meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv")
    ap.add_argument("--llm_induced_schema_root", default="meta_model/functional_v1/llm_induced")
    ap.add_argument("--evidence_cards_root", default="meta_model/functional_v1/llm_induction_cards")
    ap.add_argument("--llm_consensus_fields_csv", default="")
    ap.add_argument("--llm_consensus_support_csv", default="")
    ap.add_argument("--llm_fold_mapping_csv", default="")
    ap.add_argument("--examples_per_condition_llm", type=int, default=3)
    ap.add_argument("--max_examples", type=int, default=160)
    ap.add_argument("--skip_score", action="store_true")
    ap.add_argument("--skip_diagnostics", action="store_true")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    package_dir = Path(args.package_dir) if args.package_dir else out_root / "pi_expert_review_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    standardized_csv = Path(args.standardized_csv) if args.standardized_csv else out_root / "scoring_inputs" / "standardized_roundtrips.csv"
    scored_csv = out_root / "scored_roundtrips" / "scored_roundtrips.csv"
    diag_csv = out_root / "diagnostics" / "roundtrip_diagnostic_metrics.csv"

    if not args.skip_score and standardized_csv.exists() and not scored_csv.exists():
        run([
            sys.executable, str(SCRIPT_DIR / "09_score_roundtrip_outputs.py"),
            "--standardized_csv", str(standardized_csv),
            "--classifier_bundle", str(args.classifier_bundle),
            "--output_dir", str(out_root / "scored_roundtrips"),
        ], args.dry_run)
    elif not standardized_csv.exists() and not scored_csv.exists():
        print(f"WARNING: neither standardized CSV nor scored CSV was found. Expected {standardized_csv} or {scored_csv}")

    if not args.skip_diagnostics and scored_csv.exists():
        run([
            sys.executable, str(SCRIPT_DIR / "32_compute_roundtrip_diagnostic_metrics.py"),
            "--roundtrips_csv", str(scored_csv),
            "--classifier_bundle", str(args.classifier_bundle),
            "--output_dir", str(out_root / "diagnostics"),
            "--review_sample_per_condition", "25",
        ], args.dry_run)

    if diag_csv.exists():
        run([
            sys.executable, str(SCRIPT_DIR / "31_compile_schema_condition_comparison.py"),
            "--scored_csv", str(diag_csv),
            "--output_dir", str(out_root / "comparison"),
        ], args.dry_run)

    if not args.skip_plots:
        cmd = [
            sys.executable, str(SCRIPT_DIR / "33_make_meta_model_result_visualizations.py"),
            "--out_root", str(out_root),
            "--output_dir", str(out_root / "plots"),
        ]
        if args.llm_consensus_fields_csv:
            cmd += ["--llm_consensus_fields_csv", args.llm_consensus_fields_csv]
        if args.llm_consensus_support_csv:
            cmd += ["--llm_consensus_support_csv", args.llm_consensus_support_csv]
        if args.llm_fold_mapping_csv:
            cmd += ["--llm_fold_mapping_csv", args.llm_fold_mapping_csv]
        run(cmd, args.dry_run)

    examples_dir = package_dir / "expert_review_examples"
    if diag_csv.exists():
        run([
            sys.executable, str(SCRIPT_DIR / "34_build_expert_review_examples.py"),
            "--roundtrip_metrics_csv", str(diag_csv),
            "--output_dir", str(examples_dir),
            "--examples_per_condition_llm", str(args.examples_per_condition_llm),
            "--max_examples", str(args.max_examples),
        ], args.dry_run)

    crosswalk_dir = package_dir / "crosswalks"
    cmd = [
        sys.executable, str(SCRIPT_DIR / "35_build_v1_crosswalk_review_tables.py"),
        "--manual_crosswalk_csv", str(args.manual_crosswalk_csv),
        "--llm_induced_schema_root", str(args.llm_induced_schema_root),
        "--evidence_cards_root", str(args.evidence_cards_root),
        "--output_dir", str(crosswalk_dir),
    ]
    if args.llm_fold_mapping_csv:
        cmd += ["--llm_consensus_mapping_csv", str(args.llm_fold_mapping_csv)]
    run(cmd, args.dry_run)

    if not args.dry_run:
        copy_result_folders(out_root, package_dir)
        snapshot = condition_snapshot(diag_csv)
        write_summary_for_pi(package_dir, snapshot)
        write_package_readme(package_dir, out_root, snapshot)
        write_email_draft(package_dir)
        if args.zip:
            z = archive_package(package_dir)
            print(f"Wrote zip archive: {z}")
        print(f"\nPI expert review package ready: {package_dir}")


if __name__ == "__main__":
    main()
