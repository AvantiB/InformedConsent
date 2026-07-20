# Reduced V1 meta-model discovery, smoke testing, specificity control, and validation runbook

This is the working runbook for the reduced informed-consent meta-model.

## Current framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset. New MedGemma/Qwen/Llama/GPT outputs are validation and stress-test material, not the primary source for schema induction.

```text
Derivation / discovery corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Validation / stress-test corpus:
  new model round-trip outputs
```

Expert-preserved rows are positive functional evidence. Expert-failed rows are boundary evidence that should weaken unsafe merges.

## Finding from the first smoke tests

The first provisional cluster-ID smoke tests showed that the empirical clusters contain real signal but are too permissive for a complementary reduced schema. The current clusters can behave like broad bundles: the same cluster may cover participant, organization, database, action, decision cue, or constraint-like spans. This can preserve meaning in simple cases, but it also risks actor inversion, over-overlap, and V0-like redundancy.

Therefore, do not treat the first empirical clusters as final V1 fields. The next step is a specificity-control exploration.

## Methodological correction

V1 should not be created by simply clustering source elements that co-occur or annotate overlapping text. Same-span evidence can indicate any of the following:

```text
near-equivalence
broader/narrower relation
complementary facets of the same phrase
related but unsafe-to-merge roles
```

Only near-equivalence should drive field merging. Complementary evidence belongs in the provision-bundle graph, not the merge graph.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/12_build_expert_roundtrip_corpus.py
python -m py_compile meta_model/scripts/17_induce_reduced_v1_metamodel.py
python -m py_compile meta_model/scripts/19_visualize_v1_discovery.py
python -m py_compile meta_model/scripts/21_build_provisional_v1_schema_from_clusters.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
python -m py_compile meta_model/scripts/22_diagnose_v1_cluster_specificity.py
python -m py_compile meta_model/scripts/20_build_reduced_v1_schema_from_audit.py
```

## 2. Build the clean expert corpus

```bash
python meta_model/scripts/12_build_expert_roundtrip_corpus.py \
  --workbook_dir /path/to/original_annotation_workbooks \
  --output_csv meta_model/outputs/expert_roundtrips_clean.csv
```

## 3. Discover initial empirical cluster evidence

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_expert \
  --min_semantic_edge_weight 0.28 \
  --span_overlap_jaccard 0.50 \
  --min_core_positive_sentences 15
```

Key outputs:

```text
semantic_equivalence_edges.csv
semantic_equivalence_clusters.csv
semantic_cluster_evidence_summary.csv
semantic_cluster_audit_template.csv
provision_bundle_edges.csv
provision_bundle_summary_by_semantic_cluster.csv
```

## 4. Visualize empirical evidence

```bash
python meta_model/scripts/19_visualize_v1_discovery.py \
  --discovery_dir meta_model/v1_reduced_expert \
  --output_dir meta_model/v1_reduced_expert/visual_report
```

Use these outputs to distinguish merge evidence from composition evidence:

```text
semantic_equivalence_edges/clusters = candidate merges
provision_bundle_edges = compositional co-occurrence, not merge evidence
source-model heatmap = cross-source support
provision-bundle network = how fields combine in consent provisions
```

## 5. Build provisional cluster-ID schema for smoke testing only

```bash
python meta_model/scripts/21_build_provisional_v1_schema_from_clusters.py \
  --semantic_cluster_summary_csv meta_model/v1_reduced_expert/semantic_cluster_evidence_summary.csv \
  --semantic_clusters_csv meta_model/v1_reduced_expert/semantic_equivalence_clusters.csv \
  --decision_summary_csv meta_model/v1_reduced_expert/expert_sentence_level_decision_summary.csv \
  --output_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_provisional_empirical.yaml \
  --output_json meta_model/v1_reduced_expert/reduced_metamodel_v1_provisional_empirical.json
```

Describe this condition as:

```text
Provisional empirical V1: data-driven semantic clusters evaluated before expert naming/organization.
```

## 6. Smoke test, one hosted vLLM model at a time

Do not run the full dataset yet. Start one model server, run a few examples, stop the server, then repeat for the next hosted model.

```bash
export MODEL_KEY=medgemma   # or qwen235b, matching the currently hosted server
export N_SMOKE=5

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_provisional_empirical.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/outputs/reduced_v1_smoke \
  --evidence_mode compact \
  --stage both \
  --limit "$N_SMOKE"
```

Inspect:

```bash
column -s, -t < meta_model/outputs/reduced_v1_smoke/${MODEL_KEY}/compact/reduced_v1_roundtrip_outputs.csv | less -S
```

Then stop the current vLLM server, host the next model, update `MODEL_KEY`, and rerun the same smoke command.

## 7. Inspect smoke outputs

For each example, check:

```text
1. sentence_decision is provision-level only.
2. permit/deny/mixed/unclear are not span annotations.
3. decision cues are stored in sentence_level_elements.
4. span annotations use semantic_cluster_C### cluster IDs only.
5. unmatched_language preserves important uncaptured content.
6. interpretation_units explain how clusters combine.
7. reconstruction preserves meaning without seeing the original sentence.
8. clusters are not being used as overly broad catch-all buckets.
```

## 8. Diagnose cluster specificity and over-merge risk

Run the specificity diagnostic after smoke testing MedGemma and Qwen compact mode:

```bash
python meta_model/scripts/22_diagnose_v1_cluster_specificity.py \
  --discovery_dir meta_model/v1_reduced_expert \
  --smoke_dirs meta_model/outputs/reduced_v1_smoke/medgemma/compact,meta_model/outputs/reduced_v1_smoke/qwen235b/compact \
  --output_dir meta_model/v1_reduced_expert/specificity_diagnostics
```

Key outputs:

```text
cluster_membership_specificity_metrics.csv
smoke_cluster_annotations_long.csv
smoke_same_span_multi_cluster_cases.csv
smoke_cluster_specificity_metrics.csv
cluster_split_review_candidates.csv
cluster_specificity_report.md
```

The diagnostic flags clusters for split review when they show high span-type entropy, high lexical-head entropy, repeated same-span multi-cluster overlap, broad hub-like behavior, or evidence that one cluster is being used for multiple complementary roles.

## 9. Next induction target: specificity-controlled clusters

Use the specificity diagnostics to guide a second induction pass:

```text
source element mentions
→ span/context sense induction
→ source-element-sense nodes
→ typed pairwise relationships
   - near-equivalent
   - broader/narrower
   - complementary
   - unsafe merge
→ specificity-controlled equivalence clusters
→ provision-bundle graph among clusters
→ revised provisional V1 fields
→ smoke test again
→ PI naming/audit
```

The immediate goal is not fewer clusters. The goal is more specific, complementary clusters that reduce V0-style overlap.

## 10. PI handoff after specificity pass

Do not send the first coarse clusters as final fields. Send the PI:

```text
1. semantic cluster evidence summary
2. specificity diagnostics and split-review candidates
3. visual report
4. smoke-test examples showing successes and failures
5. proposed split/merge questions
```

Ask the PI:

```text
Which clusters represent real informed-consent functions?
Which clusters are too broad and should be split?
Which distinctions should remain extensions rather than core fields?
What names should be assigned after the specificity pass?
```

## Manuscript framing

The reduced schema is discovered from expert-validated evidence, but the smoke tests show that high-support clusters alone are not sufficient. A defensible reduced model must balance coverage with specificity. The revised pipeline therefore treats broad co-occurring bundles as a discovery signal, then applies specificity diagnostics and expert review before final V1 schema construction.