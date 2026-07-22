# Expert Review Workbook Runbook

This runbook creates one Excel workbook for PI/domain-expert comparison of:

```text
1. Source information-model dictionaries: DUO, ICO, ODRL, FHIR
2. Manual Functional V1 data dictionary
3. LLM-induced Functional V1 data dictionary, fold-specific and consensus if available
4. Source-model -> Manual V1 crosswalk
5. Source-model -> LLM-induced V1 crosswalk
6. Combined source-model -> Manual/LLM V1 crosswalk
7. Modeling-strategy performance summaries
8. Classifier details and cue dictionary
9. Fixed review examples and qualitative error samples
```

The workbook is intended for expert adjudication of comprehensiveness, field boundaries, merges/splits/renames, and source-model coverage.

## 1. Build the corrected v2 package first

```bash
export OUT_ROOT=meta_model/functional_v1_experiments

python meta_model/scripts/37_rebuild_pi_expert_review_package_v2.py \
  --out_root "$OUT_ROOT" \
  --manual_crosswalk_csv meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced \
  --evidence_cards_root meta_model/functional_v1/llm_induction_cards \
  --n_source_examples 12 \
  --random_seed 17 \
  --zip
```

Do not use `--refresh_example_sample` once you want the same fixed examples for downstream review.

## 2. Build the all-in-one Excel workbook

```bash
python meta_model/scripts/38_build_expert_review_workbook.py \
  --package_dir "$OUT_ROOT/pi_expert_review_package_v2" \
  --source_inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --manual_schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced
```

The output is:

```text
$OUT_ROOT/pi_expert_review_package_v2/expert_review_data_dictionary_and_crosswalks.xlsx
```

## 3. Optional: include LLM-induced consensus fields

If consensus fields are available, pass them too:

```bash
python meta_model/scripts/38_build_expert_review_workbook.py \
  --package_dir "$OUT_ROOT/pi_expert_review_package_v2" \
  --source_inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --manual_schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced \
  --llm_consensus_fields_csv meta_model/functional_v1/llm_consensus/llm_induced_consensus_fields.csv
```

## 4. Workbook sheets

```text
README
Metric_Summary
Strategy_x_LLM
Strategy_x_SourceModel
Classifier_Details
Source_Dictionaries
Dict_DUO
Dict_ICO
Dict_ODRL
Dict_FHIR
Manual_V1_Dictionary
LLM_Induced_Dictionary
Crosswalk_Source_to_Manual
Crosswalk_Source_to_LLM
Combined_Crosswalk
Fixed_Examples
Qualitative_Errors
```

## 5. Suggested expert review workflow

1. Start with `Manual_V1_Dictionary` and `LLM_Induced_Dictionary` to compare definitions and field boundaries.
2. Use `Source_Dictionaries` and the individual `Dict_*` sheets to inspect the original source-model elements.
3. Use `Combined_Crosswalk` to assess whether each DUO/ICO/ODRL/FHIR source element is adequately represented by Manual V1 and/or LLM-induced V1.
4. Use `Fixed_Examples` alongside `expert_review_examples.html` to examine highlighted annotations and reconstructions.
5. Use `Metric_Summary` and `Strategy_x_LLM` to interpret performance differences by modeling strategy and downstream LLM.

Review-status dropdown columns are included in the workbook where appropriate.
