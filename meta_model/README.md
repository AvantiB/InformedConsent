# Data/Language-Driven Meta-Model Development

This folder contains the reduced consent/data-use meta-model development workflow.

The goal is not to manually design a new ontology. The goal is to induce a reduced, functional meta-model from evidence produced by existing information models, consent-language round trips, and meaning-preservation behavior.

## Starting point: Union V0

**Union V0** is the unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL. It is intentionally bulky and serves as the naive maximal baseline from which redundancy, missing concepts, and meaning-critical distinctions can be discovered.

Current comparison conditions:

1. **Individual source-model JSON prompts**: DUO, ICO, ODRL, and FHIR Consent run separately with their source prompt content/data dictionaries and standardized JSON output.
2. **Union V0 full dictionary**: all source elements in one combined prompt.
3. **Later reduced V1 meta-model**: induced after preservation-aware analysis.

## Core evidence-unit idea

Each round-trip example is converted into evidence units:

```text
consent sentence
→ extracted phrase / source-model element
→ information model used
→ forward mapping text
→ backward reconstruction
→ human label or classifier preservation score
→ cue features and failure patterns
```

The reduced meta-model is inferred from source-element usage, phrase/source-node co-occurrence, cue-group preservation/failure patterns, language embeddings, clustering, and preservation behavior. Human involvement should be limited to audit, naming, and interpretation of induced clusters, not manual construction of the schema.

## Main scripts

### Build Union V0 inventory

```bash
python meta_model/scripts/00_build_union_v0_inventory.py \
  --prompt_dir /path/to/source_model_forward_prompts \
  --output_dir meta_model/v0_union
```

### Run Union V0 round trips

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/union_v0_roundtrip
```

### Run individual information-model round trips

Use the JSON prompt copies under `meta_model/prompts/individual_json`.

```bash
python meta_model/scripts/05_run_individual_model_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --prompt_dir meta_model/prompts/individual_json \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/individual_model_roundtrip \
  --info_models all
```

### Validate and standardize outputs

```bash
python meta_model/scripts/07_standardize_roundtrip_outputs.py \
  --union_model_dirs meta_model/outputs/union_v0_roundtrip/medgemma,meta_model/outputs/union_v0_roundtrip/qwen235b \
  --individual_model_dirs meta_model/outputs/individual_model_roundtrip/medgemma,meta_model/outputs/individual_model_roundtrip/qwen235b \
  --output_dir meta_model/outputs/scoring_inputs \
  --require_backward
```

### Train the final scoring classifier

The split-based classifier experiments remain the validation evidence. This final scorer is trained on all original human-labeled rows after model selection.

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

A lightweight fallback without embeddings/NLI is available:

```bash
python meta_model/scripts/08_train_final_meaning_classifier.py \
  --labeled_roundtrips_csv "$ROUNDTRIPS_CSV" \
  --output_dir meta_model/outputs/final_classifier \
  --feature_set engineered_all
```

### Score new outputs and compare baselines

```bash
python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv meta_model/outputs/scoring_inputs/standardized_roundtrips.csv \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir meta_model/outputs/scored_roundtrips
```

Key outputs:

```text
score_summary_by_condition.csv       # condition/model-level preservation proxy summaries
paired_union_vs_individual.csv       # paired naive Union V0 vs individual-model comparisons
scored_roundtrips.csv                # row-level classifier scores
```

See the full runbook:

```text
meta_model/ROUNDTRIP_SCORING_RUNBOOK.md
```

## Reduced meta-model induction principle

Candidate meta-model units should be retained when they are frequent, cross-model, preservation-sensitive, and compositionally stable. Candidate units should be merged only when their collapse does not appear to harm meaning preservation. Candidate units should be split when subclusters show different language, source-model, or preservation behavior.

## Current next steps

1. Finish MedGemma/Qwen individual-model and Union V0 runs.
2. Standardize all MedGemma/Qwen outputs.
3. Train the final classifier on all original human-labeled rows.
4. Score MedGemma/Qwen outputs and compare individual baselines against naive Union V0.
5. Repeat the same pipeline for Llama and GPT.
6. Use the full scored evidence table for reduced meta-model development.
