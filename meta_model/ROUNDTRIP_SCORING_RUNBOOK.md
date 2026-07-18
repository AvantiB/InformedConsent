# Round-Trip Standardization and Classifier Scoring Runbook

This runbook covers the next-stage workflow after running Union V0 and individual information-model round trips for MedGemma, Qwen, Llama, and GPT.

## Goal

The goal is to produce comparable meaning-preservation proxy scores for:

1. individual source-model prompts: DUO, ICO, ODRL, FHIR Consent;
2. the unreduced Union V0 meta-model baseline;
3. later reduced meta-model versions.

The split-based classifier experiments remain the evaluation evidence for the classifier. This workflow trains a final deployment/scoring classifier on all original human-labeled rows after model selection, then applies it to new LLM round-trip outputs.

## Outputs

The workflow creates:

```text
meta_model/outputs/scoring_inputs/standardized_roundtrips.csv
meta_model/outputs/scoring_inputs/standardization_audit.csv
meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib
meta_model/outputs/scored_roundtrips/scored_roundtrips.csv
meta_model/outputs/scored_roundtrips/score_summary_by_condition.csv
meta_model/outputs/scored_roundtrips/paired_union_vs_individual.csv
```

## Step A. Standardize and validate outputs

After MedGemma and Qwen runs finish, standardize Union V0 and individual-model outputs into one table:

```bash
python meta_model/scripts/07_standardize_roundtrip_outputs.py \
  --union_model_dirs meta_model/outputs/union_v0_roundtrip/medgemma,meta_model/outputs/union_v0_roundtrip/qwen235b \
  --individual_model_dirs meta_model/outputs/individual_model_roundtrip/medgemma,meta_model/outputs/individual_model_roundtrip/qwen235b \
  --output_dir meta_model/outputs/scoring_inputs \
  --require_backward
```

Inspect the audit:

```bash
cat meta_model/outputs/scoring_inputs/standardization_audit.csv
cat meta_model/outputs/scoring_inputs/missing_pairs.csv
```

Proceed only after the missing-pairs file is empty or the missing records are understood.

## Step B. Train final classifier on all original labeled data

The final scoring classifier is trained on all original human-labeled rows. This is separate from the earlier split-based evaluation runs.

Recommended default if embeddings/NLI are available:

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

Lightweight fallback without embeddings/NLI:

```bash
python meta_model/scripts/08_train_final_meaning_classifier.py \
  --labeled_roundtrips_csv "$ROUNDTRIPS_CSV" \
  --output_dir meta_model/outputs/final_classifier \
  --feature_set engineered_all
```

The split-based results should be reported as the classifier validation evidence. This all-data classifier is the final proxy scorer used on new round-trip outputs.

## Step C. Score Union V0 and individual-model outputs

```bash
python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv meta_model/outputs/scoring_inputs/standardized_roundtrips.csv \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir meta_model/outputs/scored_roundtrips
```

Review condition-level summaries:

```bash
cat meta_model/outputs/scored_roundtrips/score_summary_by_condition.csv
cat meta_model/outputs/scored_roundtrips/paired_union_vs_individual.csv
```

`paired_union_vs_individual.csv` reports whether Union V0 improves over each individual source model for the same LLM and sentence. This is the naive Union V0 comparison before reduced meta-model induction.

## Interpretation

Classifier scores are proxy preservation estimates, not ground-truth labels. Use them to prioritize reduction decisions and failure audits. Final claims should distinguish between:

- classifier validation results from human-labeled data;
- classifier-scored proxy results on new LLM outputs;
- later human audit of selected borderline or high-impact cases.

## After MedGemma/Qwen

After MedGemma and Qwen are scored, repeat the same round-trip generation and scoring workflow for Llama and GPT. Once all model conditions are scored, proceed to preservation-aware reduced meta-model induction.
