# Functional V1 reduced meta-model runbook

This is the active runbook for the reduced functional informed-consent meta-model. The earlier cluster-ID schemas are evidence artifacts, not the final annotation target.

## 1. Compile

```bash
python -m py_compile meta_model/scripts/23_refined_metamodel_cv_pipeline.py
python -m py_compile meta_model/scripts/24_refined_cv_postprocess.py
python -m py_compile meta_model/scripts/25_make_heldout_roundtrips.py
python -m py_compile meta_model/scripts/26_build_functional_v1_crosswalk.py
python -m py_compile meta_model/scripts/27_run_functional_v1_roundtrip.py
```

## 2. Review the functional schema

```text
meta_model/schemas/reduced_functional_v1_candidate.yaml
```

This schema is data-seeded and expert-review pending. It should be reviewed by the PI/domain expert before final claims.

## 3. Build the source-model crosswalk

The crosswalk maps all source elements in the source-element inventory into the proposed functional V1 fields. Broad elements are marked as context dependent rather than forced into one field.

```bash
python meta_model/scripts/26_build_functional_v1_crosswalk.py \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --output_dir meta_model/functional_v1/crosswalk
```

Outputs:

```text
meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv
meta_model/functional_v1/crosswalk/functional_v1_crosswalk_summary.csv
meta_model/functional_v1/crosswalk/functional_v1_model_field_matrix.csv
meta_model/functional_v1/crosswalk/functional_v1_context_dependent_review.csv
meta_model/functional_v1/crosswalk/functional_v1_source_to_field_edges.csv
meta_model/functional_v1/crosswalk/functional_v1_crosswalk_metadata.json
```

Use `functional_v1_model_field_matrix.csv` and `functional_v1_source_to_field_edges.csv` for plots showing which information-model elements overlap or complement each reduced functional field.

## 4. Prepare held-out round-trip files

Use the repaired form-level assignments.

```bash
python meta_model/scripts/25_make_heldout_roundtrips.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --fold_assignments_csv meta_model/refined_cv/fold_assignments.repaired.csv \
  --output_dir meta_model/refined_cv
```

Outputs:

```text
meta_model/refined_cv/fold_00/heldout_roundtrips.csv
meta_model/refined_cv/fold_01/heldout_roundtrips.csv
meta_model/refined_cv/fold_02/heldout_roundtrips.csv
meta_model/refined_cv/fold_03/heldout_roundtrips.csv
meta_model/refined_cv/heldout_roundtrips_metadata.json
```

## 5. Functional V1 held-out smoke test

Start with one fold and one model.

```bash
export MODEL_KEY=medgemma
export FOLD=0

python meta_model/scripts/27_run_functional_v1_roundtrip.py \
  --roundtrips_csv meta_model/refined_cv/fold_00/heldout_roundtrips.csv \
  --metamodel_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/functional_v1/heldout_roundtrip/fold_00 \
  --evidence_mode compact \
  --stage both \
  --limit 20
```

Inspect:

```bash
column -s, -t < meta_model/functional_v1/heldout_roundtrip/fold_00/${MODEL_KEY}/compact/functional_v1_roundtrip_outputs.csv | less -S
```

Check for these issues before scaling:

```text
participant/actor inversion
institution vs repository confusion
resource/action conflation
temporal phrase attached to wrong target
purpose vs research-domain confusion
decision cue used as sentence decision or vice versa
unmatched important content
```

## 6. Full held-out functional V1 evaluation

After smoke inspection:

```bash
for FOLD in 0 1 2 3; do
  for MODEL_KEY in medgemma qwen235b; do
    python meta_model/scripts/27_run_functional_v1_roundtrip.py \
      --roundtrips_csv meta_model/refined_cv/fold_0${FOLD}/heldout_roundtrips.csv \
      --metamodel_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
      --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
      --model_key "$MODEL_KEY" \
      --output_dir meta_model/functional_v1/heldout_roundtrip/fold_0${FOLD} \
      --evidence_mode compact \
      --stage both
  done
done
```

## 7. Scoring and comparison

Standardize the Functional V1 outputs using the same scoring pipeline used for individual models and Union V0, then compute:

```text
meaning-preservation classifier score
content-word preservation
cue preservation
annotation count
unique field count
unmatched-language rate
relationship/attachment error categories
```

The main comparison should be:

```text
Individual DUO / ICO / FHIR Consent / ODRL
Union V0
Reduced Functional V1
```

## 8. Expert-review package

For PI/domain-expert review, provide:

```text
meta_model/FUNCTIONAL_V1_METHODS.md
meta_model/schemas/reduced_functional_v1_candidate.yaml
meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv
meta_model/functional_v1/crosswalk/functional_v1_context_dependent_review.csv
examples from Functional V1 smoke test
```

The expert-reviewed revision should be saved as a new schema, e.g.:

```text
meta_model/schemas/reduced_functional_v1_1_expert_reviewed.yaml
```

Do not overwrite the data-seeded candidate schema.
