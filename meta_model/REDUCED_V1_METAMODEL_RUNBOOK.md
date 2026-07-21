# Retired reduced V1 cluster-development runbook

This older runbook is retained only for methodological provenance. It described the fold-specific cluster/sense induction workflow that generated the data-derived evidence used to seed the reduced meta-model.

The **active paper-facing workflow** is now the reduced functional V1 process:

```text
meta_model/FUNCTIONAL_V1_METHODS.md
meta_model/FUNCTIONAL_V1_ROUNDTRIP_RUNBOOK.md
meta_model/REFINED_CV_POSTPROCESS_RUNBOOK.md
meta_model/schemas/reduced_functional_v1_candidate.yaml
```

## Why this runbook was retired

The fold-specific selected schemas reduced the evidence space from roughly 295-317 raw candidate fields per fold to about 29-36 stable selected fields per fold. However, those selected fields remained cluster-like and partially overlapping. They were therefore treated as **data-derived seeds**, not final annotation labels.

The current method uses those seeds, typed near-equivalence evidence, co-occurrence/complementarity evidence, and cross-fold recurrence to construct a smaller functional schema whose fields are intended to be mostly non-overlapping and interpretable.

## Active next steps

```bash
python -m py_compile meta_model/scripts/26_build_functional_v1_crosswalk.py
python -m py_compile meta_model/scripts/27_run_functional_v1_roundtrip.py
```

Build the source-model crosswalk:

```bash
python meta_model/scripts/26_build_functional_v1_crosswalk.py \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --schema_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --output_dir meta_model/functional_v1/crosswalk
```

Run a Functional V1 held-out smoke test:

```bash
python meta_model/scripts/27_run_functional_v1_roundtrip.py \
  --roundtrips_csv meta_model/refined_cv/fold_00/heldout_roundtrips.csv \
  --metamodel_yaml meta_model/schemas/reduced_functional_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/functional_v1/heldout_roundtrip/fold_00 \
  --evidence_mode compact \
  --stage both \
  --limit 20
```
