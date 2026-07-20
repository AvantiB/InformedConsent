# Reduced V1 meta-model discovery, audit, and validation runbook

This is the authoritative runbook for the reduced V1 informed-consent meta-model.

## Correct study framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset, not from the new MedGemma/Qwen/Llama/GPT outputs.

```text
Derivation / discovery corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Validation / stress-test corpus:
  new MedGemma, Qwen235B, Llama4, GPT-5.5 round-trip outputs
```

Expert-preserved rows are positive functional evidence. Expert-failed rows are boundary evidence. New LLM outputs are used only after V1 is defined, to test generalization.

## Important methodological correction

V1 is **not** induced by hard-coding fields such as action/resource/actor. The workflow separates two empirical graphs:

1. **Semantic-equivalence graph**: asks which source-model elements may express the same semantic field. Edges come from same/overlapping evidence spans, cross-information-model support, cross-LLM support, expert-positive evidence, and profile similarity.
2. **Provision-bundle graph**: asks which elements co-occur compositionally in consent sentences. This graph is used to understand provision structure, not to merge fields.

The final V1 schema is not generated directly by script 17. Script 17 writes empirical clusters and an audit template. Humans only name/select clusters and flag unsafe merges/splits. Script 20 then converts the audited cluster table into the V1 YAML schema.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/12_build_expert_roundtrip_corpus.py
python -m py_compile meta_model/scripts/17_induce_reduced_v1_metamodel.py
python -m py_compile meta_model/scripts/20_build_reduced_v1_schema_from_audit.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
```

## 2. Build the clean expert round-trip corpus

Use the original researcher handoff workbooks.

```bash
python meta_model/scripts/12_build_expert_roundtrip_corpus.py \
  --workbook_dir /path/to/original_annotation_workbooks \
  --output_csv meta_model/outputs/expert_roundtrips_clean.csv
```

Check:

```bash
cat meta_model/outputs/expert_roundtrip_corpus_summary.json
column -s, -t < meta_model/outputs/expert_roundtrip_corpus_summary.csv | less -S
```

Raw repeated annotations are retained as salience/frequency evidence.

## 3. Discover empirical V1 evidence

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
expert_element_mentions_long.csv
expert_element_profiles.csv
semantic_equivalence_edges.csv
semantic_equivalence_clusters.csv
semantic_cluster_evidence_summary.csv
semantic_cluster_audit_template.csv
provision_bundle_edges.csv
provision_bundle_summary_by_semantic_cluster.csv
expert_sentence_level_decision_summary.csv
expert_v1_discovery_methodology.md
semantic_cluster_discovery_report.md
```

Interpretation:

```text
semantic_equivalence_edges/clusters = candidate field merges
provision_bundle_edges = composition/co-occurrence, not merge evidence
semantic_cluster_audit_template.csv = the file to audit/name before schema creation
```

## 4. Audit semantic clusters

Open:

```bash
column -s, -t < meta_model/v1_reduced_expert/semantic_cluster_audit_template.csv | less -S
less meta_model/v1_reduced_expert/semantic_cluster_discovery_report.md
```

Fill these columns in `semantic_cluster_audit_template.csv`:

```text
include_in_v1        yes/no
final_field_name     audited field name, e.g., resource, purpose, temporal_scope
audit_decision       keep / split / merge_with_Cxx / exclude
unsafe_merge_notes   why a merge is unsafe, if applicable
split_or_merge_notes notes for downstream schema construction
```

This is the human audit/naming step, not manual schema design. Cluster membership and support are data-derived.

## 5. Build the audited V1 schema

After auditing the template:

```bash
python meta_model/scripts/20_build_reduced_v1_schema_from_audit.py \
  --audit_csv meta_model/v1_reduced_expert/semantic_cluster_audit_template.csv \
  --clusters_csv meta_model/v1_reduced_expert/semantic_equivalence_clusters.csv \
  --decision_summary_csv meta_model/v1_reduced_expert/expert_sentence_level_decision_summary.csv \
  --output_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.yaml \
  --output_json meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.json
```

The schema builder will fail if no clusters are explicitly selected with `include_in_v1=yes` and `final_field_name` filled in.

## 6. Smoke-test Reduced V1 compact/permissive variants

```bash
python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode compact \
  --stage both \
  --limit 3

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode permissive \
  --stage both \
  --limit 3
```

## 7. Full validation

Run both evidence modes for each validation model, then standardize, score, and compare against individual-model and Union V0 baselines as before.

## Manuscript framing

The reduced V1 meta-model is discovered from expert-validated semantic-equivalence evidence, finalized by limited audit/naming, and validated on new LLM outputs. A good V1 should approach Union V0 meaning preservation while reducing redundancy and annotation burden.
