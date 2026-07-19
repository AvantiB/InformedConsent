# Reduced V1 meta-model induction and validation runbook

This runbook starts after the corrected all-model analysis has been generated with `15_analyze_roundtrip_scored_outputs.py` after the sentence-level decision fix.

The goal is to induce and validate a reduced, provision-centered functional meta-model from Union V0 and individual-model evidence. Reduced V1 uses the same round-trip structure as Union V0:

```text
original sentence -> structured JSON
structured JSON only -> reconstructed sentence
```

## Methodology: how Reduced V1 is selected

Reduced V1 is induced from a source-element evidence graph. The fields are not selected only from a hand-written recommendation. The role names are used only after clustering to label and describe empirically supported clusters.

### Inputs

The induction script uses corrected script-15 outputs:

```text
source_element_evidence_summary.csv
source_element_mentions_long.csv
source_element_cooccurrence_pairs.csv
sentence_level_decision_summary.csv
```

Sentence-level decision fields are handled separately and are not used as span-level graph nodes:

```text
DUO.decision
ICO.decision
ODRL::Rule_TestSentence
FHIR_Consent::Consent.provision.type
```

These become the V1 `decision` field.

### Node construction

Each remaining span-level source-model element becomes a node. Each node stores:

```text
source model
source element ID/name/definition
number of mentions
number of source sentences
number of LLMs using it
preservation score when present
content/cue preservation metrics
top evidence spans
top cue groups
```

### Edge construction

Two nodes are connected when the data suggests that they are related. Edge weight combines:

```text
source-sentence co-occurrence
same exact evidence span use
profile similarity from labels/definitions/span examples/cue groups
cross-source-model support
```

This is why corrected co-occurrence matters: sentence-level decision labels would otherwise dominate the graph.

### Graph clustering

The script clusters the weighted graph using NetworkX greedy modularity when available. If NetworkX is unavailable, it falls back to thresholded connected components. UMAP should be used only for visualization, not as the primary grouping method.

### Cluster selection

Clusters are summarized using:

```text
number of source elements
number of source models represented
number of LLMs represented
source-sentence coverage
mean classifier preservation score
mean content-token recall
mean cue-group recall
top span examples
top source elements
```

Clusters are labeled:

```text
core_shared: cross-source, multi-LLM, frequent, and functionally central
context_module: lower-coverage but recurrent/context-specific
signal_for_audit_or_extension: rare or unstable but potentially important
```

The resulting candidate YAML includes the sentence-level `decision` field plus selected graph-induced cluster fields and audit fields. Human review is used only for unsafe merges, naming, and edge-case audit, not for manual schema design.

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

## 3. Induce the graph-based candidate reduced V1 schema

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --analysis_dir meta_model/outputs/roundtrip_meta_model_analysis_all4_v2 \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_graph \
  --min_edge_weight 0.22 \
  --min_core_sentences 15
```

Key outputs:

```text
meta_model/v1_reduced_graph/element_nodes.csv
meta_model/v1_reduced_graph/element_relationship_edges.csv
meta_model/v1_reduced_graph/element_clusters.csv
meta_model/v1_reduced_graph/cluster_evidence_summary.csv
meta_model/v1_reduced_graph/cluster_cooccurrence_summary.csv
meta_model/v1_reduced_graph/sentence_level_decision_fields.csv
meta_model/v1_reduced_graph/reduced_metamodel_v1_candidate.yaml
meta_model/v1_reduced_graph/reduced_metamodel_v1_candidate.md
meta_model/v1_reduced_graph/reduced_v1_graph_induction_methodology.md
```

Review the cluster summary, edge table, and markdown for obviously unsafe merges before a large run. This is an audit step, not manual schema design.

## 4. Smoke-test Reduced V1 compact and permissive variants

Example with GPT-5.5 through Apigee:

```bash
rm -rf meta_model/outputs/reduced_v1_roundtrip_smoke/mayo_gpt55

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_graph/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode compact \
  --stage both \
  --limit 3

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_graph/reduced_metamodel_v1_candidate.yaml \
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
      --metamodel_yaml meta_model/v1_reduced_graph/reduced_metamodel_v1_candidate.yaml \
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
