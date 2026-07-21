# Manual Functional V1 vs LLM-Induced Functional V1 Experiment Runbook

This runbook is the complete experimental pipeline for comparing four schema conditions:

```text
1. Individual information models: DUO, ICO, ODRL, FHIR Consent
2. Union V0: naive combined source-model dictionary
3. Manual Functional V1: data-seeded manually organized reduced schema
4. LLM-Induced Functional V1: schema induced from evidence cards by one fixed strong LLM
```

## Non-negotiable prompt-control rule

For all forward/backward round-trip experiments, keep the round-trip prompt template constant within each broad experiment family. The only experimental variable should be the schema/dictionary supplied to the prompt.

```text
Individual source model condition: existing individual source-model prompt family.
Union V0 condition: existing Union V0 prompt family.
Manual Functional V1 and LLM-Induced Functional V1: same script, same prompt, same evidence_mode, same downstream LLMs; only the YAML schema changes.
```

The LLM induction prompt is separate because schema induction is a different task.

## Model-key convention

Use the exact keys present in `meta_model/configs/union_v0_models.local.yaml`.

For Mayo GPT-5.5, the template key is:

```text
mayo_gpt55
```

Use `--model_key mayo_gpt55`, not `gpt55`, unless your local YAML explicitly defines a separate `gpt55` alias.

The Mayo Apigee wrapper can read credentials in two modes:

```text
OAuth auto-refresh:
  configure oauth_client_id_env and oauth_client_secret_env in the local YAML.

Static/refreshed bearer token:
  configure APIGEE_TOKEN_FILE or APIGEE_TOKEN.
  The wrapper reads the token file/env during requests, but token creation/refresh is external unless OAuth is configured.
```

## One hosted model at a time

For local vLLM models, run only the model that is currently hosted. Do not run the all-model loops unless all endpoints are actually available. Use the single-model command blocks below and set:

```bash
export RUN_MODEL=medgemma      # or qwen235b, llama4, mayo_gpt55, etc.
```

For Mayo GPT-5.5, no local vLLM hosting is needed, but the Apigee token/OAuth configuration must be valid.

## A. Preparation

```bash
git pull origin main

python -m py_compile meta_model/scripts/07_standardize_roundtrip_outputs.py
python -m py_compile meta_model/scripts/08_train_final_meaning_classifier.py
python -m py_compile meta_model/scripts/09_score_roundtrip_outputs.py
python -m py_compile meta_model/scripts/23_refined_metamodel_cv_pipeline.py
python -m py_compile meta_model/scripts/24_refined_cv_postprocess.py
python -m py_compile meta_model/scripts/25_make_heldout_roundtrips.py
python -m py_compile meta_model/scripts/26_build_functional_v1_crosswalk.py
python -m py_compile meta_model/scripts/27_run_functional_v1_roundtrip.py
python -m py_compile meta_model/scripts/28_build_llm_schema_induction_cards.py
python -m py_compile meta_model/scripts/29_induce_functional_schema_with_llm.py
python -m py_compile meta_model/scripts/30_relabel_functional_v1_outputs.py
python -m py_compile meta_model/scripts/31_compile_schema_condition_comparison.py
```

Set common variables:

```bash
export ROUNDTRIPS_CSV=/dgx1data/aii/tao/m338824/R03-InformedConsent/roundtrips.csv
export MODEL_CONFIG=meta_model/configs/union_v0_models.local.yaml
export OUT_ROOT=meta_model/functional_v1_experiments
```

## B. Build/confirm the shared form-level CV evidence

Create or refresh form-level folds:

```bash
python meta_model/scripts/23_refined_metamodel_cv_pipeline.py make-folds \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --split_source_csv "$ROUNDTRIPS_CSV" \
  --output_dir meta_model/refined_cv \
  --n_folds 4 \
  --seed 17
```

Repair form aliases:

```bash
python meta_model/scripts/24_refined_cv_postprocess.py repair-fold-assignments \
  --fold_assignments_csv meta_model/refined_cv/fold_assignments.csv \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --output_csv meta_model/refined_cv/fold_assignments.repaired.csv \
  --audit_csv meta_model/refined_cv/fold_assignment_repair_audit.csv
```

Run fold induction:

```bash
for FOLD in 0 1 2 3; do
  python meta_model/scripts/23_refined_metamodel_cv_pipeline.py run-fold \
    --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
    --fold_assignments_csv meta_model/refined_cv/fold_assignments.repaired.csv \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --output_dir meta_model/refined_cv \
    --fold_id "$FOLD" \
    --min_sense_support 2 \
    --span_overlap_threshold 0.75 \
    --min_equivalence_weight 0.02 \
    --min_equivalence_positive_contexts 1 \
    --min_field_positive_mentions 5
done
```

Confirm no unassigned mentions:

```bash
grep n_unassigned_mentions meta_model/refined_cv/fold_*/fold_run_metadata.json
```

Run strict field selection:

```bash
python meta_model/scripts/24_refined_cv_postprocess.py select-fields \
  --fold_root meta_model/refined_cv \
  --output_dir meta_model/refined_cv/field_selection_strict \
  --signature_terms 10 \
  --stability_jaccard 0.55 \
  --core_min_folds 3 \
  --extension_min_folds 3 \
  --min_source_models 2 \
  --min_select_positive_mentions 40 \
  --core_min_total_positive_mentions 150 \
  --extension_min_total_positive_mentions 100
```

Create held-out roundtrip files:

```bash
python meta_model/scripts/25_make_heldout_roundtrips.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --fold_assignments_csv meta_model/refined_cv/fold_assignments.repaired.csv \
  --output_dir meta_model/refined_cv
```

## C. Build source-model crosswalk and evidence cards

```bash
python meta_model/scripts/26_build_functional_v1_crosswalk.py \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --output_dir meta_model/functional_v1/crosswalk

python meta_model/scripts/28_build_llm_schema_induction_cards.py \
  --fold_root meta_model/refined_cv \
  --selected_fields_csv meta_model/refined_cv/field_selection_strict/selected_fields_long.csv \
  --crosswalk_csv meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv \
  --output_dir meta_model/functional_v1/llm_induction_cards \
  --example_sentences_per_card 3 \
  --max_edges_per_card 12
```

## D. Manual Functional V1 round-trip assessment

Manual schema:

```text
meta_model/schemas/reduced_functional_v1_candidate.yaml
```

Smoke test first with the currently available/hosted model:

```bash
export RUN_MODEL=medgemma

python meta_model/scripts/27_run_functional_v1_roundtrip.py \
  --roundtrips_csv meta_model/refined_cv/fold_00/heldout_roundtrips.csv \
  --metamodel_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$RUN_MODEL" \
  --output_dir "$OUT_ROOT/manual_v1/fold_00" \
  --evidence_mode compact \
  --stage both \
  --limit 20

python meta_model/scripts/30_relabel_functional_v1_outputs.py \
  --output_dir "$OUT_ROOT/manual_v1/fold_00/${RUN_MODEL}/compact" \
  --condition functional_v1_manual \
  --information_model Functional_V1_Manual
```

Full manual V1 held-out evaluation for one currently hosted model:

```bash
export RUN_MODEL=medgemma

for FOLD in 0 1 2 3; do
  python meta_model/scripts/27_run_functional_v1_roundtrip.py \
    --roundtrips_csv meta_model/refined_cv/fold_0${FOLD}/heldout_roundtrips.csv \
    --metamodel_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$RUN_MODEL" \
    --output_dir "$OUT_ROOT/manual_v1/fold_0${FOLD}" \
    --evidence_mode compact \
    --stage both

  python meta_model/scripts/30_relabel_functional_v1_outputs.py \
    --output_dir "$OUT_ROOT/manual_v1/fold_0${FOLD}/${RUN_MODEL}/compact" \
    --condition functional_v1_manual \
    --information_model Functional_V1_Manual
done
```

Repeat this block after hosting each model by changing `RUN_MODEL`.

## E. LLM-induced schema generation with Mayo GPT-5.5

Use one fixed strong induction model for schema induction. For the Mayo Apigee GPT-5.5 config, use:

```bash
export INDUCTION_MODEL=mayo_gpt55
```

Do not use the manual V1 schema as an induction input.

```bash
export INDUCTION_MODEL=mayo_gpt55

for FOLD in 0 1 2 3; do
  python meta_model/scripts/29_induce_functional_schema_with_llm.py \
    --evidence_cards_jsonl meta_model/functional_v1/llm_induction_cards/fold_0${FOLD}/schema_induction_evidence_cards.jsonl \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$INDUCTION_MODEL" \
    --output_dir meta_model/functional_v1/llm_induced/fold_0${FOLD} \
    --stage all \
    --target_min_fields 16 \
    --target_max_fields 28 \
    --max_cards 80 \
    --max_spans_per_card 12
done
```

Inspect validation:

```bash
for FOLD in 0 1 2 3; do
  echo fold_0${FOLD}
  cat meta_model/functional_v1/llm_induced/fold_0${FOLD}/llm_induced_schema_validation.json
done
```

Output schema per fold:

```text
meta_model/functional_v1/llm_induced/fold_XX/llm_induced_functional_v1_candidate.yaml
```

## F. LLM-induced Functional V1 round-trip assessment

Smoke test one fold with the currently available/hosted model:

```bash
export RUN_MODEL=medgemma

python meta_model/scripts/27_run_functional_v1_roundtrip.py \
  --roundtrips_csv meta_model/refined_cv/fold_00/heldout_roundtrips.csv \
  --metamodel_yaml meta_model/functional_v1/llm_induced/fold_00/llm_induced_functional_v1_candidate.yaml \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$RUN_MODEL" \
  --output_dir "$OUT_ROOT/llm_induced_v1/fold_00" \
  --evidence_mode compact \
  --stage both \
  --limit 20

python meta_model/scripts/30_relabel_functional_v1_outputs.py \
  --output_dir "$OUT_ROOT/llm_induced_v1/fold_00/${RUN_MODEL}/compact" \
  --condition functional_v1_llm_induced \
  --information_model Functional_V1_LLM_Induced
```

Full LLM-induced V1 held-out evaluation for one currently hosted model:

```bash
export RUN_MODEL=medgemma

for FOLD in 0 1 2 3; do
  python meta_model/scripts/27_run_functional_v1_roundtrip.py \
    --roundtrips_csv meta_model/refined_cv/fold_0${FOLD}/heldout_roundtrips.csv \
    --metamodel_yaml meta_model/functional_v1/llm_induced/fold_0${FOLD}/llm_induced_functional_v1_candidate.yaml \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$RUN_MODEL" \
    --output_dir "$OUT_ROOT/llm_induced_v1/fold_0${FOLD}" \
    --evidence_mode compact \
    --stage both

  python meta_model/scripts/30_relabel_functional_v1_outputs.py \
    --output_dir "$OUT_ROOT/llm_induced_v1/fold_0${FOLD}/${RUN_MODEL}/compact" \
    --condition functional_v1_llm_induced \
    --information_model Functional_V1_LLM_Induced
done
```

Repeat this block after hosting each model by changing `RUN_MODEL`.

## G. Baseline outputs: individual models and Union V0

Use the existing individual/Union V0 runbooks to generate outputs. These should already use their existing controlled prompt families.

Expected directories, by model:

```text
meta_model/outputs/individual_model_roundtrip/<model_key>
meta_model/outputs/union_v0_roundtrip/<model_key>
```

## H. Standardize all conditions for scoring

Build comma-separated directory lists from whatever model outputs exist:

```bash
INDIV_DIRS=$(find meta_model/outputs/individual_model_roundtrip -mindepth 1 -maxdepth 1 -type d | paste -sd, -)
UNION_DIRS=$(find meta_model/outputs/union_v0_roundtrip -mindepth 1 -maxdepth 1 -type d | paste -sd, -)
MANUAL_DIRS=$(find "$OUT_ROOT/manual_v1" -path "*/compact" -type d | paste -sd, -)
LLM_INDUCED_DIRS=$(find "$OUT_ROOT/llm_induced_v1" -path "*/compact" -type d | paste -sd, -)
REDUCED_DIRS="${MANUAL_DIRS},${LLM_INDUCED_DIRS}"

python meta_model/scripts/07_standardize_roundtrip_outputs.py \
  --individual_model_dirs "$INDIV_DIRS" \
  --union_model_dirs "$UNION_DIRS" \
  --reduced_v1_model_dirs "$REDUCED_DIRS" \
  --output_dir "$OUT_ROOT/scoring_inputs" \
  --require_backward
```

Inspect:

```bash
cat "$OUT_ROOT/scoring_inputs/standardization_audit.csv"
cat "$OUT_ROOT/scoring_inputs/missing_pairs.csv"
```

## I. Train final meaning-preservation classifier

Use the final all-labeled-data classifier for proxy scoring. The held-out validation of the classifier itself should be reported from prior split-based classifier experiments.

Preferred semantic-feature run:

```bash
python meta_model/scripts/08_train_final_meaning_classifier.py \
  --labeled_roundtrips_csv "$ROUNDTRIPS_CSV" \
  --output_dir meta_model/outputs/final_classifier \
  --feature_set engineered_semantic \
  --embedding_model all-MiniLM-L6-v2 \
  --embedding_backend hf \
  --embedding_device cpu \
  --nli_model cross-encoder/nli-deberta-v3-base \
  --nli_device cuda
```

Fallback:

```bash
python meta_model/scripts/08_train_final_meaning_classifier.py \
  --labeled_roundtrips_csv "$ROUNDTRIPS_CSV" \
  --output_dir meta_model/outputs/final_classifier \
  --feature_set engineered_all
```

## J. Score all standardized outputs

```bash
python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv "$OUT_ROOT/scoring_inputs/standardized_roundtrips.csv" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir "$OUT_ROOT/scored_roundtrips"
```

Compile condition comparison:

```bash
python meta_model/scripts/31_compile_schema_condition_comparison.py \
  --scored_csv "$OUT_ROOT/scored_roundtrips/scored_roundtrips.csv" \
  --output_dir "$OUT_ROOT/comparison"
```

Primary outputs:

```text
$OUT_ROOT/scored_roundtrips/score_summary_by_condition.csv
$OUT_ROOT/comparison/schema_condition_summary.csv
$OUT_ROOT/comparison/paired_condition_scores_wide.csv
```

## K. Interpretation

Report four main schema conditions:

```text
individual_source_model_json
union_v0_full_dictionary
functional_v1_manual
functional_v1_llm_induced
```

Compare:

```text
meaning-preservation classifier score
content/cue preservation metrics
annotation count
unique field count
parse success
unmatched-language rate when available
qualitative relationship errors
```

Expected paper-facing interpretation:

```text
Manual V1 > LLM-induced V1:
  expert functional organization improved over automatic induction.

LLM-induced V1 > Manual V1:
  evidence-card induction found useful boundaries or labels missed manually.

Both similar:
  the reduced functional representation is robust to curation strategy.

Reduced models lower than Union V0 but much smaller:
  compactness/usability tradeoff with comparable preservation.
```
