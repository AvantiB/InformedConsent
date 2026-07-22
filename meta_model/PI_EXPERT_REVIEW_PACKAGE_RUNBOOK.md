# PI Expert Review Package Runbook

This runbook creates the complete PI-facing package after all LLM round-trip runs finish.

The package includes:

```text
1. Meaning-preservation classifier results and visualizations
2. Holistic diagnostic metrics
3. Individual source-model, Union V0, Manual V1, and LLM-induced V1 comparisons
4. Schema-reduction / seed-cluster / evidence-card visualizations
5. Manual V1 and LLM-induced V1 crosswalk review tables
6. Highlighted annotation examples for expert adjudication
7. Qualitative relationship-error review samples
8. A short PI summary and draft email
```

## 1. Compile the new scripts

```bash
git pull origin main

python -m py_compile meta_model/scripts/31_compile_schema_condition_comparison.py
python -m py_compile meta_model/scripts/32_compute_roundtrip_diagnostic_metrics.py
python -m py_compile meta_model/scripts/33_make_meta_model_result_visualizations.py
python -m py_compile meta_model/scripts/34_build_expert_review_examples.py
python -m py_compile meta_model/scripts/35_build_v1_crosswalk_review_tables.py
python -m py_compile meta_model/scripts/36_prepare_pi_expert_review_package.py
```

## 2. Confirm the standardization output exists

The package builder expects all completed round-trip outputs to already be standardized into:

```text
$OUT_ROOT/scoring_inputs/standardized_roundtrips.csv
```

Set:

```bash
export OUT_ROOT=meta_model/functional_v1_experiments
```

Check:

```bash
ls -lh "$OUT_ROOT/scoring_inputs/standardized_roundtrips.csv"
```

If it is missing, first run Step H from `MANUAL_VS_LLM_INDUCED_V1_EXPERIMENT_RUNBOOK.md`.

## 3. Build the complete PI package

Basic run:

```bash
python meta_model/scripts/36_prepare_pi_expert_review_package.py \
  --out_root "$OUT_ROOT" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --manual_crosswalk_csv meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced \
  --evidence_cards_root meta_model/functional_v1/llm_induction_cards \
  --zip
```

This will run:

```text
09_score_roundtrip_outputs.py, if scored outputs are missing
32_compute_roundtrip_diagnostic_metrics.py
31_compile_schema_condition_comparison.py
33_make_meta_model_result_visualizations.py
34_build_expert_review_examples.py
35_build_v1_crosswalk_review_tables.py
```

and then create:

```text
$OUT_ROOT/pi_expert_review_package/
$OUT_ROOT/pi_expert_review_package.zip
```

## 4. Include LLM-induced consensus mapping files, if available

If the consensus mapping files have been saved locally or in the repository, include them:

```bash
python meta_model/scripts/36_prepare_pi_expert_review_package.py \
  --out_root "$OUT_ROOT" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --manual_crosswalk_csv meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced \
  --evidence_cards_root meta_model/functional_v1/llm_induction_cards \
  --llm_consensus_fields_csv meta_model/functional_v1/llm_consensus/llm_induced_consensus_fields.csv \
  --llm_consensus_support_csv meta_model/functional_v1/llm_consensus/llm_induced_consensus_support_summary.csv \
  --llm_fold_mapping_csv meta_model/functional_v1/llm_consensus/llm_induced_fold_field_to_consensus_mapping.csv \
  --zip
```

If those consensus files are outside the repository, replace the paths with their local paths.

## 5. Main package outputs

```text
pi_expert_review_package/
  README_PI_REVIEW_PACKAGE.md
  summary_for_pi.md
  email_draft_to_pi.md

  plots/
    01_condition_x_llm_classifier_score_heatmap.png
    02_classifier_score_by_condition.png
    03_holistic_metrics_by_condition.png
    04_annotation_burden_by_condition.png
    05_qualitative_error_flag_heatmap.png
    06_cue_category_retention_heatmap.png
    07_functional_role_x_source_model_heatmap.png
    08_role_source_model_overlap_distribution.png
    09_selected_seed_cluster_counts_by_fold.png
    10_top_seed_clusters_by_support.png
    11_cross_fold_seed_recurrence_distribution.png
    12_llm_consensus_field_recurrence.png, if consensus inputs are provided
    13_llm_fold_field_to_consensus_heatmap.png, if consensus inputs are provided
    14_llm_consensus_fields_by_tier.png, if consensus inputs are provided
    15_classifier_feature_mean_differences.png
    16_classifier_feature_importance.png

  comparison/
    schema_condition_summary.csv
    schema_condition_by_llm.csv
    schema_condition_by_information_model.csv
    schema_condition_overall.csv
    paired_condition_scores_wide.csv, when possible
    paired_condition_score_differences.csv, when possible

  diagnostics/
    roundtrip_diagnostic_metrics.csv
    condition_diagnostic_summary.csv
    condition_llm_diagnostic_summary.csv
    condition_information_model_diagnostic_summary.csv
    cue_group_retention_summary_by_condition.csv
    cue_group_retention_summary_by_condition_llm.csv
    qualitative_relationship_error_review_sample.csv
    evaluation_dictionary_used.json

  expert_review_examples/
    expert_review_examples.html
    expert_review_examples.csv
    expert_review_examples.xlsx, when openpyxl is available

  crosswalks/
    manual_v1_source_model_crosswalk_for_review.csv
    llm_induced_v1_source_model_crosswalk_by_fold_for_review.csv
    llm_induced_consensus_source_model_crosswalk_for_review.csv, if consensus mapping is provided
    manual_v1_vs_llm_induced_field_alignment_for_review.csv
    v1_crosswalk_review_summary.html
```

## 6. What to send to the PI/team today

Send the zip plus a brief note. The most useful files for immediate expert review are:

```text
README_PI_REVIEW_PACKAGE.md
summary_for_pi.md
expert_review_examples/expert_review_examples.html
crosswalks/v1_crosswalk_review_summary.html
crosswalks/manual_v1_source_model_crosswalk_for_review.csv
crosswalks/llm_induced_v1_source_model_crosswalk_by_fold_for_review.csv
comparison/schema_condition_overall.csv
plots/01_condition_x_llm_classifier_score_heatmap.png
plots/03_holistic_metrics_by_condition.png
plots/04_annotation_burden_by_condition.png
plots/07_functional_role_x_source_model_heatmap.png
plots/15_classifier_feature_mean_differences.png
```

## 7. Expert review questions to include

```text
1. Are Manual V1 fields clinically/ethically meaningful and non-overlapping?
2. Are any Manual V1 fields too broad, too narrow, redundant, or missing?
3. Do LLM-induced V1 fields identify useful distinctions not present in Manual V1?
4. Are any LLM-induced fields unsafe merges that should be split?
5. Are source-model elements correctly mapped into Manual V1 and LLM-induced V1?
6. In highlighted examples, is meaning preserved after reconstruction?
7. Do failures suggest missing fields, unclear definitions, or relationship/polarity issues?
```

## 8. Interpretation guardrails

- Use fold-specific LLM-induced schemas for primary held-out CV performance.
- Treat the consensus LLM-induced schema as a post-CV artifact for expert review or descriptive validation.
- Qualitative error flags are triage signals, not ground truth.
- Permit/deny captures sentence-level polarity, but local negation/prohibition still needs cue-preservation and expert review.
- Proximity-weighted evidence is a useful future refinement, but this version uses recurrence, overlap, co-occurrence, cross-model support, and round-trip preservation evidence.
