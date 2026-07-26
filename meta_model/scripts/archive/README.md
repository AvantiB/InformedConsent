# Archived meta-model scripts

This folder is for scripts from older exploratory or superseded meta-model workflows.

Current active workflow scripts should remain in `meta_model/scripts/`.

Active current workflow:

- `00_build_union_v0_inventory.py`
- `03_run_union_v0_roundtrip.py`
- `05_run_individual_model_roundtrip.py`
- `07_standardize_roundtrip_outputs.py`
- `08_train_final_meaning_classifier.py`
- `09_score_roundtrip_outputs.py`
- `32_compute_roundtrip_diagnostic_metrics.py`
- `41_filter_phase1_outputs_for_schema_induction.py`
- `42_run_phase1_union_v0_roundtrip.py`
- `43_run_phase1_union_v0_roundtrip_apigee.py`
- `44_run_phase1_individual_roundtrip.py`
- `45_run_phase1_individual_roundtrip_apigee.py`
- `46_prepare_direct_llm_schema_induction_inputs.py`
- `47_induce_direct_reduced_schema_with_llm.py`
- `48_run_direct_schema_roundtrip.py`
- `49_compile_schema_strategy_roundtrips.py`
- `50_prepare_data_driven_schema_induction_inputs.py`
- `51_induce_data_driven_schema_with_llm.py`

Run `meta_model/scripts/archive_obsolete_scripts.py` from the repository root to move known superseded scripts into this folder after reviewing the manifest printed by that script.
