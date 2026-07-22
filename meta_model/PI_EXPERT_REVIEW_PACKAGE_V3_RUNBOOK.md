# PI Expert Review Package v3 Runbook

This is the corrected final workflow for preparing the package for PI/domain-expert review.

The v3 package addresses the limitations of the earlier packages:

```text
1. It explicitly documents meaning-preservation classifier development.
2. It includes feature families, model families tested, model-selection artifacts when available, and the finalized classifier.
3. It includes modeling-strategy comparison results across the available evaluation metrics.
4. It builds one main Excel workbook for expert review.
5. It restricts crosswalks to DUO/ICO/ODRL/FHIR source elements mapped to Manual V1 and LLM-induced V1.
6. It preserves source element IDs, labels, definitions, target V1 fields, mapping basis, and expert-review columns.
7. It reuses the corrected fixed-source annotation example HTML from the v2 package when available.
```

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/39_prepare_pi_expert_review_package_v3.py
```

## 2. Build v3 package

Set the experiment root:

```bash
export OUT_ROOT=meta_model/functional_v1_experiments
```

Run the v3 builder:

```bash
python meta_model/scripts/39_prepare_pi_expert_review_package_v3.py \
  --out_root "$OUT_ROOT" \
  --source_package_dir "$OUT_ROOT/pi_expert_review_package_v2" \
  --source_inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --manual_schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --manual_crosswalk_csv meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced \
  --evidence_cards_root meta_model/functional_v1/llm_induction_cards \
  --classifier_dir meta_model/outputs/final_classifier \
  --zip
```

This writes:

```text
$OUT_ROOT/pi_expert_review_package_v3/
$OUT_ROOT/pi_expert_review_package_v3.zip
```

## 3. If classifier experiment/model-selection outputs are stored elsewhere

The final classifier directory usually contains the final trained classifier and training summary. If the classifier-development experiment outputs such as `metrics_by_split.csv` and `threshold_metrics.csv` are stored in another directory, pass that directory explicitly:

```bash
python meta_model/scripts/39_prepare_pi_expert_review_package_v3.py \
  --out_root "$OUT_ROOT" \
  --source_package_dir "$OUT_ROOT/pi_expert_review_package_v2" \
  --source_inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --manual_schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --manual_crosswalk_csv meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv \
  --llm_induced_schema_root meta_model/functional_v1/llm_induced \
  --evidence_cards_root meta_model/functional_v1/llm_induction_cards \
  --classifier_dir meta_model/outputs/final_classifier \
  --classifier_experiments_dir path/to/classifier_experiment_outputs \
  --zip
```

## 4. Main files to send/review

```text
pi_expert_review_package_v3/
  README_PI_REVIEW_PACKAGE.md
  CLASSIFIER_DEVELOPMENT_SUMMARY.md
  expert_review_data_dictionary_and_crosswalks.xlsx

  expert_review_examples/
    expert_review_examples.html
    fixed_example_source_ids.csv
    expert_review_examples.csv

  crosswalks/
    v1_crosswalk_review_summary.html
    manual_v1_source_model_crosswalk_for_review.csv
    llm_induced_v1_source_model_crosswalk_by_fold_for_review.csv
    source_model_to_manual_and_llm_v1_crosswalk_for_review.csv

  comparison/
    schema_condition_overall.csv
    schema_condition_by_llm.csv
    schema_condition_by_information_model.csv

  diagnostics/
    roundtrip_diagnostic_metrics.csv
    condition_diagnostic_summary.csv
    cue_group_retention_summary_by_condition.csv
    qualitative_relationship_error_review_sample.csv

  plots/
    classifier and modeling-strategy result figures
```

## 5. Recommended expert review order

```text
1. Open README_PI_REVIEW_PACKAGE.md.
2. Read CLASSIFIER_DEVELOPMENT_SUMMARY.md.
3. Open expert_review_data_dictionary_and_crosswalks.xlsx.
4. Review Classifier_Method, Classifier_Final_Details, and Classifier_Model_Selection.
5. Review Results_Overall, Results_by_LLM, Results_by_SourceModel, and Cue_Retention.
6. Review Manual_V1_Dictionary and LLM_Induced_Dictionary.
7. Review Combined_Crosswalk and the source-specific dictionary tabs.
8. Open expert_review_examples/expert_review_examples.html for highlighted fixed examples.
```

## 6. Interpretation notes

- Use fold-specific LLM-induced schemas for primary held-out evaluation.
- Treat LLM-induced consensus fields, when included, as a post-CV expert-review artifact.
- The classifier score is a scalable evaluation signal, not the only truth.
- Expert review should prioritize field comprehensiveness, unsafe merges, missing roles, polarity/negation issues, and reconstruction errors.
