# Meta-Model Visualization Runbook

This runbook generates the result plots for the paper/presentation deck after the model runs are complete.

The plots combine four evidence layers:

```text
1. Data-driven reduction evidence
   - source-model crosswalk
   - seed cluster / selected field recurrence
   - LLM-induced fold-to-consensus mapping, when available

2. Meaning-preservation classifier insights
   - classifier score
   - classifier feature importances
   - feature differences between preserved vs not-preserved examples

3. Round-trip evaluation results
   - individual information models x LLMs
   - Union V0 x LLMs
   - Manual Functional V1 x LLMs
   - LLM-induced Functional V1 x LLMs

4. Holistic diagnostic metrics
   - content-word preservation
   - cue/category preservation
   - modal-word change ratio
   - unmatched-language rate
   - annotation burden
   - qualitative relationship-error review flags
```

## 1. Standardize, score, and diagnose

After all round-trip runs are finished, build the standardized CSV as described in `MANUAL_VS_LLM_INDUCED_V1_EXPERIMENT_RUNBOOK.md`.

Then score with the final classifier:

```bash
export OUT_ROOT=meta_model/functional_v1_experiments

python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv "$OUT_ROOT/scoring_inputs/standardized_roundtrips.csv" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir "$OUT_ROOT/scored_roundtrips"
```

Compute holistic diagnostics:

```bash
python meta_model/scripts/32_compute_roundtrip_diagnostic_metrics.py \
  --roundtrips_csv "$OUT_ROOT/scored_roundtrips/scored_roundtrips.csv" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir "$OUT_ROOT/diagnostics" \
  --review_sample_per_condition 25
```

Compile paper-facing summaries from the enriched diagnostic CSV:

```bash
python meta_model/scripts/31_compile_schema_condition_comparison.py \
  --scored_csv "$OUT_ROOT/diagnostics/roundtrip_diagnostic_metrics.csv" \
  --output_dir "$OUT_ROOT/comparison"
```

## 2. Generate the plots

Basic run:

```bash
python meta_model/scripts/33_make_meta_model_result_visualizations.py \
  --out_root "$OUT_ROOT" \
  --output_dir "$OUT_ROOT/plots"
```

This creates plots for whatever data files exist and skips missing optional inputs.

## 3. Include optional LLM consensus mapping outputs

If the LLM-induced consensus files are available, pass them explicitly:

```bash
python meta_model/scripts/33_make_meta_model_result_visualizations.py \
  --out_root "$OUT_ROOT" \
  --output_dir "$OUT_ROOT/plots" \
  --llm_consensus_fields_csv meta_model/functional_v1/llm_consensus/llm_induced_consensus_fields.csv \
  --llm_consensus_support_csv meta_model/functional_v1/llm_consensus/llm_induced_consensus_support_summary.csv \
  --llm_fold_mapping_csv meta_model/functional_v1/llm_consensus/llm_induced_fold_field_to_consensus_mapping.csv
```

If these files are outside the repository, replace the paths with their actual local locations.

## 4. Generated plot files

The visualization script writes a manifest:

```text
$OUT_ROOT/plots/plot_manifest.csv
```

Common outputs include:

```text
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
12_llm_consensus_field_recurrence.png
13_llm_fold_field_to_consensus_heatmap.png
14_llm_consensus_fields_by_tier.png
15_classifier_feature_mean_differences.png
16_classifier_feature_importance.png
```

## 5. How to use in the deck

Recommended plot-to-slide mapping:

```text
Information-model overlap slide:
  07_functional_role_x_source_model_heatmap.png
  08_role_source_model_overlap_distribution.png

Cluster/evidence-card seed slide:
  09_selected_seed_cluster_counts_by_fold.png
  10_top_seed_clusters_by_support.png
  11_cross_fold_seed_recurrence_distribution.png

LLM-induced consensus slide:
  12_llm_consensus_field_recurrence.png
  13_llm_fold_field_to_consensus_heatmap.png
  14_llm_consensus_fields_by_tier.png

Meaning-preservation classifier slide:
  15_classifier_feature_mean_differences.png
  16_classifier_feature_importance.png

Final results slide:
  01_condition_x_llm_classifier_score_heatmap.png
  02_classifier_score_by_condition.png
  03_holistic_metrics_by_condition.png
  04_annotation_burden_by_condition.png
  06_cue_category_retention_heatmap.png

Qualitative error slide:
  05_qualitative_error_flag_heatmap.png
  $OUT_ROOT/diagnostics/qualitative_relationship_error_review_sample.csv
```

## 6. Interpretation guardrails

Use fold-specific LLM-induced schemas for primary held-out CV performance. Treat the consensus LLM-induced schema as the post-CV final artifact for expert review or descriptive validation.

The qualitative error flags are heuristic triage signals, not ground-truth labels. Use the review sample CSV for manual inspection before making strong claims about specific error categories.
