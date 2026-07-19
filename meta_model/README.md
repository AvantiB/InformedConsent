# Data/Language-Driven Meta-Model Development

This folder contains the workflow for deriving and validating a reduced, functional informed-consent meta-model.

## Correct framing

Reduced V1 is **derived from the original expert-evaluated round-trip dataset**. New LLM outputs are used for validation and stress testing, not for primary schema induction.

```text
Derivation / induction corpus:
  original round-trip dataset with expert meaning-preservation labels

Validation / stress-test corpus:
  MedGemma, Qwen235B, Llama4, GPT-5.5 outputs
```

Expert-preserved rows are treated as functionally validated positive evidence: the forward representation contained enough structured information to support a meaning-preserving backward reconstruction. Expert-failed rows are boundary evidence: they weaken proposed merges and flag unsafe simplifications.

Human involvement should be limited to audit, naming, and unsafe-merge review, not manual schema construction.

## Baseline conditions

1. **Individual source-model JSON prompts**: DUO, ICO, ODRL, and FHIR Consent run separately.
2. **Union V0 full dictionary**: unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL.
3. **Reduced V1 compact**: expert-induced reduced schema with short evidence phrases.
4. **Reduced V1 permissive**: same reduced schema with longer evidence phrases allowed when needed.

The compact/permissive split lets us test whether V1 works because of the reduced functional schema itself, or because longer evidence spans carry forward source wording.

## Core evidence path

```text
expert-labeled round-trip row
→ original consent sentence
→ forward source-model annotations
→ expert meaning-preservation label
→ positive functional evidence or boundary evidence
→ source-element profiles
→ weighted element relationship graph
→ graph clusters
→ core/context/audit field selection
→ candidate Reduced V1 schema
→ validation on new LLM outputs
```

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

```bash
python meta_model/scripts/05_run_individual_model_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --prompt_dir meta_model/prompts/individual_json \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/individual_model_roundtrip \
  --info_models all
```

### Train the final scoring classifier

The final scorer is trained on all original expert-labeled rows after model selection. It is used for validation scoring of new LLM outputs, not for primary Reduced V1 induction.

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

### Induce Reduced V1 from expert-labeled rows

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --expert_roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_expert \
  --min_edge_weight 0.22 \
  --min_core_positive_sentences 15
```

Key outputs:

```text
meta_model/v1_reduced_expert/expert_element_profiles.csv
meta_model/v1_reduced_expert/expert_element_relationship_edges.csv
meta_model/v1_reduced_expert/expert_element_clusters.csv
meta_model/v1_reduced_expert/expert_cluster_evidence_summary.csv
meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml
meta_model/v1_reduced_expert/expert_validated_induction_methodology.md
```

### Run Reduced V1 validation round trips

```bash
python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/reduced_v1_roundtrip \
  --evidence_mode compact \
  --stage both
```

Run both `compact` and `permissive` modes for each validation LLM.

## Runbooks

```text
meta_model/ROUNDTRIP_SCORING_RUNBOOK.md
meta_model/ROUNDTRIP_EVALUATION_METAMODEL_RUNBOOK.md
meta_model/REDUCED_V1_METAMODEL_RUNBOOK.md
```

## Manuscript claim to preserve

The reduced meta-model is induced from expert-validated evidence, then validated on new LLMs. A successful V1 should preserve meaning better than individual source models and approach Union V0 while reducing annotation burden, redundancy, and model-dependent verbosity.
