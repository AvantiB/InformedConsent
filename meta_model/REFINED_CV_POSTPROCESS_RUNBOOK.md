# Refined CV post-processing runbook

This runbook covers the two steps to run after the initial `23_refined_metamodel_cv_pipeline.py run-fold` experiments.

## Why this step exists

The raw fold induction outputs are evidence dictionaries, not the final compact meta-model. Two post-processing checks are required before held-out LLM evaluation:

1. **Form-assignment repair:** the main `roundtrips.csv` may use a stable `form_key` with punctuation normalized differently from the expert workbooks. For example, `Alzheimer_s Disease...` and `Alzheimer's Disease...` should be treated as the same canonical consent form.
2. **Field selection:** the raw candidate schemas are intentionally permissive. They include many fold-specific candidate fields. A smaller selected schema should be generated using cross-fold stability, positive evidence, and source-model support.

## 1. Compile

```bash
python -m py_compile meta_model/scripts/23_refined_metamodel_cv_pipeline.py
python -m py_compile meta_model/scripts/24_refined_cv_postprocess.py
```

## 2. Repair fold assignments

Run this after creating `meta_model/refined_cv/fold_assignments.csv` from the main `roundtrips.csv`.

```bash
python meta_model/scripts/24_refined_cv_postprocess.py repair-fold-assignments \
  --fold_assignments_csv meta_model/refined_cv/fold_assignments.csv \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --output_csv meta_model/refined_cv/fold_assignments.repaired.csv \
  --audit_csv meta_model/refined_cv/fold_assignment_repair_audit.csv
```

Inspect:

```bash
column -s, -t < meta_model/refined_cv/fold_assignment_repair_audit.csv | less -S
```

Expected: the ADRD/American Samoa row should be matched by punctuation-insensitive key rather than remaining unmatched.

## 3. Rerun fold induction using repaired assignments

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

Check that no fold has unassigned mentions:

```bash
grep n_unassigned_mentions meta_model/refined_cv/fold_*/fold_run_metadata.json
```

Expected: `0` for all four folds.

## 4. Summarize raw candidate fields

```bash
python meta_model/scripts/23_refined_metamodel_cv_pipeline.py summarize-folds \
  --fold_root meta_model/refined_cv \
  --output_dir meta_model/refined_cv/stability
```

This is a broad evidence summary. It is not yet the selected compact schema.

## 5. Select stable core/extension candidate fields

```bash
python meta_model/scripts/24_refined_cv_postprocess.py select-fields \
  --fold_root meta_model/refined_cv \
  --output_dir meta_model/refined_cv/field_selection \
  --signature_terms 10 \
  --stability_jaccard 0.45 \
  --core_min_folds 3 \
  --extension_min_folds 2 \
  --min_source_models 2 \
  --min_select_positive_mentions 20 \
  --core_min_total_positive_mentions 80 \
  --extension_min_total_positive_mentions 40
```

Outputs:

```text
meta_model/refined_cv/field_selection/cross_fold_field_stability_groups.csv
meta_model/refined_cv/field_selection/selected_field_stability_summary.csv
meta_model/refined_cv/field_selection/selected_fields_long.csv
meta_model/refined_cv/field_selection/field_selection_metadata.json
meta_model/refined_cv/fold_XX/refined_selected_candidate_schema.yaml
meta_model/refined_cv/fold_XX/refined_selected_candidate_schema.json
```

## 6. Inspect selected schema size

```bash
python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('meta_model/refined_cv').glob('fold_*/refined_selected_candidate_schema.json')):
    obj = json.loads(p.read_text())
    print(p, len(obj.get('fields', [])))
PY
```

The selected schemas should be much smaller than the raw 290-320-field candidate dictionaries. Treat them as fold-specific candidate schemas for held-out evaluation, not final audited meta-model fields.

## 7. Use selected schemas for held-out round trips

Use `refined_selected_candidate_schema.yaml` rather than `refined_candidate_schema.yaml` for the first held-out LLM evaluation.

```bash
python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv meta_model/refined_cv/fold_${FOLD}/heldout_roundtrips.csv \
  --metamodel_yaml meta_model/refined_cv/fold_${FOLD}/refined_selected_candidate_schema.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/refined_cv/fold_${FOLD}/heldout_roundtrip_outputs \
  --evidence_mode compact \
  --stage both
```
