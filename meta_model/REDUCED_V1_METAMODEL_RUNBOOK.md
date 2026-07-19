# Reduced V1 meta-model induction and validation runbook

This runbook starts after the corrected all-model analysis has been generated with `15_analyze_roundtrip_scored_outputs.py` after the sentence-level decision fix.

The goal is to induce and validate a reduced, provision-centered functional meta-model from Union V0 and individual-model evidence. Reduced V1 uses the same round-trip structure as Union V0:

```text
original sentence -> structured JSON
structured JSON only -> reconstructed sentence
```

## Design choice: run both V1 evidence variants

Run both variants using the same reduced schema:

1. `compact`: short evidence phrases only. This tests whether the reduced schema itself preserves meaning without long-clause evidence spans.
2. `permissive`: same reduced schema, but allows longer evidence phrases when needed. This controls for the annotation-granularity effect observed in Union V0, especially with GPT-5.5.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/07_standardize_roundtrip_outputs.py
python -m py_compile meta_model/scripts/16_audit_annotation_granularity.py
python -m py_compile meta_model/scripts/17_induce_reduced_v1_metamodel.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
```

## 2. Make sure corrected V0/individual analysis exists

```bash
python meta_model/scripts/15_analyze_roundtrip_scored_outputs.py \
  --scored_csv meta_model/outputs/scored_roundtrips_all4/scored_roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/outputs/roundtrip_meta_model_analysis_all4_v2
```

The co-occurrence output should no longer be dominated by sentence-level decision fields such as `ODRL::Rule_TestSentence` or `FHIR_Consent::Consent.provision.type`.

## 3. Induce the candidate reduced V1 schema

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --analysis_dir meta_model/outputs/roundtrip_meta_model_analysis_all4_v2 \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced
```

Key outputs:

```text
meta_model/v1_reduced/candidate_role_assignments.csv
meta_model/v1_reduced/candidate_role_evidence_summary.csv
meta_model/v1_reduced/candidate_role_cooccurrence_summary.csv
meta_model/v1_reduced/reduced_metamodel_v1_candidate.yaml
meta_model/v1_reduced/reduced_metamodel_v1_candidate.md
```

Review the markdown and assignment CSV for obviously unsafe mappings before a large run. This is an audit step, not manual schema design.

## 4. Smoke-test Reduced V1 compact and permissive variants

Example with GPT-5.5 through Apigee:

```bash
rm -rf meta_model/outputs/reduced_v1_roundtrip_smoke/mayo_gpt55

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode compact \
  --stage both \
  --limit 3

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode permissive \
  --stage both \
  --limit 3
```

## 5. Full Reduced V1 runs

Run both evidence modes for each model you want to compare.

```bash
for MODEL in medgemma qwen235b llama4 mayo_gpt55; do
  for MODE in compact permissive; do
    python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
      --roundtrips_csv /path/to/roundtrips.csv \
      --metamodel_yaml meta_model/v1_reduced/reduced_metamodel_v1_candidate.yaml \
      --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
      --model_key "$MODEL" \
      --output_dir meta_model/outputs/reduced_v1_roundtrip \
      --evidence_mode "$MODE" \
      --stage both
  done
done
```

For slow API models, run one model/mode at a time in `tmux`.

## 6. Standardize all model conditions, including Reduced V1

```bash
python meta_model/scripts/07_standardize_roundtrip_outputs.py \
  --union_model_dirs \
meta_model/outputs/union_v0_roundtrip/medgemma,meta_model/outputs/union_v0_roundtrip/qwen235b,meta_model/outputs/union_v0_roundtrip/llama4,meta_model/outputs/union_v0_roundtrip/mayo_gpt55 \
  --individual_model_dirs \
meta_model/outputs/individual_model_roundtrip/medgemma,meta_model/outputs/individual_model_roundtrip/qwen235b,meta_model/outputs/individual_model_roundtrip/llama4,meta_model/outputs/individual_model_roundtrip/mayo_gpt55 \
  --reduced_v1_model_dirs \
meta_model/outputs/reduced_v1_roundtrip/medgemma/compact,meta_model/outputs/reduced_v1_roundtrip/medgemma/permissive,meta_model/outputs/reduced_v1_roundtrip/qwen235b/compact,meta_model/outputs/reduced_v1_roundtrip/qwen235b/permissive,meta_model/outputs/reduced_v1_roundtrip/llama4/compact,meta_model/outputs/reduced_v1_roundtrip/llama4/permissive,meta_model/outputs/reduced_v1_roundtrip/mayo_gpt55/compact,meta_model/outputs/reduced_v1_roundtrip/mayo_gpt55/permissive \
  --output_dir meta_model/outputs/scoring_inputs_v0_individual_v1 \
  --require_backward
```

## 7. Score and analyze

```bash
python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv meta_model/outputs/scoring_inputs_v0_individual_v1/standardized_roundtrips.csv \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir meta_model/outputs/scored_roundtrips_v0_individual_v1

python meta_model/scripts/15_analyze_roundtrip_scored_outputs.py \
  --scored_csv meta_model/outputs/scored_roundtrips_v0_individual_v1/scored_roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/outputs/roundtrip_meta_model_analysis_v0_individual_v1
```

## 8. Annotation/role granularity audit including V1

```bash
python meta_model/scripts/16_audit_annotation_granularity.py \
  --union_model_dirs \
meta_model/outputs/union_v0_roundtrip/medgemma,meta_model/outputs/union_v0_roundtrip/qwen235b,meta_model/outputs/union_v0_roundtrip/llama4,meta_model/outputs/union_v0_roundtrip/mayo_gpt55 \
  --individual_model_dirs \
meta_model/outputs/individual_model_roundtrip/medgemma,meta_model/outputs/individual_model_roundtrip/qwen235b,meta_model/outputs/individual_model_roundtrip/llama4,meta_model/outputs/individual_model_roundtrip/mayo_gpt55 \
  --reduced_v1_model_dirs \
meta_model/outputs/reduced_v1_roundtrip/medgemma/compact,meta_model/outputs/reduced_v1_roundtrip/medgemma/permissive,meta_model/outputs/reduced_v1_roundtrip/qwen235b/compact,meta_model/outputs/reduced_v1_roundtrip/qwen235b/permissive,meta_model/outputs/reduced_v1_roundtrip/llama4/compact,meta_model/outputs/reduced_v1_roundtrip/llama4/permissive,meta_model/outputs/reduced_v1_roundtrip/mayo_gpt55/compact,meta_model/outputs/reduced_v1_roundtrip/mayo_gpt55/permissive \
  --output_dir meta_model/outputs/annotation_granularity_v0_individual_v1
```

## 9. Interpretation

A strong V1 result is not necessarily "V1 always beats Union V0." Union V0 can be high-recall and verbose. The desired evidence pattern is:

```text
Reduced V1 compact ≈ Union V0 on semantic preservation, with lower granularity burden.
Reduced V1 compact > individual models on semantic/cue preservation.
Reduced V1 permissive shows whether any V1 deficit is due to compact evidence-span constraints.
Union V0 remains the high-recall upper-bound baseline.
```

Report compact and permissive V1 side-by-side. If permissive V1 improves a lot over compact V1, evidence-span length is still contributing materially. If compact V1 performs close to Union V0, the reduced functional schema is doing the work.
