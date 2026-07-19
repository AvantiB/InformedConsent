# Reduced V1 meta-model induction and validation runbook

This is the authoritative runbook for the reduced V1 informed-consent meta-model.

## Correct study framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset, not from the new MedGemma/Qwen/Llama/GPT outputs.

```text
Derivation / induction corpus:
  original round-trip dataset with expert meaning-preservation labels

Validation / stress-test corpus:
  new MedGemma, Qwen235B, Llama4, GPT-5.5 round-trip outputs
```

Expert-preserved rows are treated as functionally validated positive evidence: the forward representation contained enough information to reconstruct the original sentence meaning. Expert-failed rows are boundary evidence: they weaken proposed merges and flag unsafe simplifications.

The new all-model outputs are used later to test whether the expert-induced V1 schema generalizes across LLMs and whether compact/permissive evidence modes behave differently.

## Round-trip structure kept comparable to Union V0

Reduced V1 keeps the same basic evaluation structure as Union V0:

```text
original sentence -> structured JSON
structured JSON only -> reconstructed sentence
```

Two V1 evidence variants should be evaluated using the same schema:

1. `compact`: short evidence phrases only. Tests whether the reduced schema itself preserves meaning without long-clause evidence spans.
2. `permissive`: same reduced schema, but longer evidence phrases are allowed when needed. Controls for annotation-granularity effects observed in Union V0.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/07_standardize_roundtrip_outputs.py
python -m py_compile meta_model/scripts/16_audit_annotation_granularity.py
python -m py_compile meta_model/scripts/17_induce_reduced_v1_metamodel.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
```

## 2. Induce Reduced V1 from the expert-labeled corpus

Use the original expert-evaluated round-trip CSV. The script auto-detects common column names for original sentence, forward mapping, information model, LLM, and expert label. Use `--label_col` if the label column is not auto-detected.

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --expert_roundtrips_csv /path/to/expert_evaluated_roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_expert \
  --min_edge_weight 0.22 \
  --min_core_positive_sentences 15
```

If needed:

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --expert_roundtrips_csv /path/to/expert_evaluated_roundtrips.csv \
  --label_col meaning_preserved \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_expert
```

## 3. Evidence outputs from induction

```text
meta_model/v1_reduced_expert/expert_element_mentions_long.csv
meta_model/v1_reduced_expert/expert_sentence_level_decision_mentions_long.csv
meta_model/v1_reduced_expert/expert_element_profiles.csv
meta_model/v1_reduced_expert/expert_element_relationship_edges.csv
meta_model/v1_reduced_expert/expert_element_clusters.csv
meta_model/v1_reduced_expert/expert_cluster_evidence_summary.csv
meta_model/v1_reduced_expert/expert_sentence_level_decision_summary.csv
meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml
meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.md
meta_model/v1_reduced_expert/expert_validated_induction_methodology.md
```

## 4. Methodology implemented by script 17

1. Parse original forward mappings into source-element mentions.
2. Separate sentence-level decision fields:
   - `DUO.decision`
   - `ICO.decision`
   - `ODRL::Rule_TestSentence`
   - `FHIR_Consent::Consent.provision.type`
3. Treat annotations from expert-preserved rows as positive functional evidence.
4. Treat annotations from expert-failed rows as boundary/failure evidence.
5. Build source-element profiles containing frequency, sentence coverage, LLM support, source-model support, positive rate, and span examples.
6. Build weighted graph edges from:
   - expert-positive co-occurrence;
   - same-span expert-positive use;
   - label/definition/span similarity;
   - cross-source-model support.
7. Penalize edges when the same element pair occurs mainly in expert-failed rows.
8. Cluster the weighted graph.
9. Select clusters as:
   - `core_shared`: frequent expert-positive evidence, cross-source support, multi-LLM support;
   - `context_module`: coherent but less broadly shared evidence;
   - `audit_or_extension` or `failure_boundary_audit`: rare, uncertain, or failure-associated evidence.
10. Generate the candidate Reduced V1 YAML schema with field-level selection evidence.

UMAP may be added later for visualization only. It is not the primary grouping method.

## 5. Audit before running V1

Before running a large V1 experiment, inspect:

```bash
less meta_model/v1_reduced_expert/expert_validated_induction_methodology.md
less meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.md
head -30 meta_model/v1_reduced_expert/expert_cluster_evidence_summary.csv
head -30 meta_model/v1_reduced_expert/expert_element_relationship_edges.csv
```

This audit is for unsafe merges, bad names, or parsing failures. It is not manual schema design.

## 6. Smoke-test Reduced V1 compact and permissive variants

```bash
rm -rf meta_model/outputs/reduced_v1_roundtrip_smoke/mayo_gpt55

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode compact \
  --stage both \
  --limit 3

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode permissive \
  --stage both \
  --limit 3
```

## 7. Full Reduced V1 validation runs

Run both evidence modes for each validation model.

```bash
for MODEL in medgemma qwen235b llama4 mayo_gpt55; do
  for MODE in compact permissive; do
    python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
      --roundtrips_csv /path/to/roundtrips.csv \
      --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
      --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
      --model_key "$MODEL" \
      --output_dir meta_model/outputs/reduced_v1_roundtrip \
      --evidence_mode "$MODE" \
      --stage both
  done
done
```

## 8. Standardize all validation conditions

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

## 9. Score and analyze validation outputs

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

## 10. Annotation granularity audit including V1

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

## Manuscript framing

The reduced V1 meta-model is induced from expert-validated evidence. New LLM outputs are validation conditions. A successful V1 does not need to beat Union V0 on every raw score; it should preserve meaning better than individual source models and approach Union V0 while reducing annotation burden, redundancy, and model-dependent verbosity.
