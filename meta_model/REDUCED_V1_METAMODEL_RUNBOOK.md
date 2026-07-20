# Reduced V1 meta-model discovery, provisional evaluation, audit, visualization, and validation runbook

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

## Methodological correction

V1 is not induced by hard-coding fields such as action/resource/actor. The workflow separates two empirical graph views:

1. Semantic-equivalence graph: candidate field merges from same/overlapping evidence spans, cross-information-model support, cross-LLM support, expert-positive evidence, failure penalties, and profile similarity.
2. Provision-bundle graph: compositional co-occurrence in consent sentences. This supports a provision-centered schema, but is not used directly as merge evidence.

Script 17 writes empirical clusters and an audit template. Script 19 visualizes the evidence. Script 21 builds a provisional cluster-ID schema so the data-driven model can be evaluated as-is. Script 20 converts a PI/expert-audited cluster table into the final V1 YAML schema.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/12_build_expert_roundtrip_corpus.py
python -m py_compile meta_model/scripts/17_induce_reduced_v1_metamodel.py
python -m py_compile meta_model/scripts/19_visualize_v1_discovery.py
python -m py_compile meta_model/scripts/21_build_provisional_v1_schema_from_clusters.py
python -m py_compile meta_model/scripts/20_build_reduced_v1_schema_from_audit.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
```

## 2. Build the clean expert round-trip corpus

```bash
python meta_model/scripts/12_build_expert_roundtrip_corpus.py \
  --workbook_dir /path/to/original_annotation_workbooks \
  --output_csv meta_model/outputs/expert_roundtrips_clean.csv
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

## 4. Visualize and defend the discovery evidence

```bash
python meta_model/scripts/19_visualize_v1_discovery.py \
  --discovery_dir meta_model/v1_reduced_expert \
  --output_dir meta_model/v1_reduced_expert/visual_report
```

Use the visual report to defend the empirical discovery step:

```text
semantic_equivalence_edges/clusters = candidate field merges
provision_bundle_edges = composition/co-occurrence, not merge evidence
source-model heatmap = whether clusters span DUO, ICO, FHIR Consent, ODRL
semantic cluster network = merge pressure from same/overlapping spans
provision-bundle network = how fields compose in consent provisions
```

## 5. Build a provisional empirical V1 schema for performance testing

This step intentionally uses the discovered clusters as-is. Field names are provisional cluster IDs such as `semantic_cluster_C001`, so performance can be measured before the PI names or reorganizes the fields.

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

## 6. Run provisional V1 compact/permissive tests with one hosted vLLM model at a time

Do not loop over all models unless all models are simultaneously hosted. For the local vLLM setup, start one model server, run compact and permissive for that model, stop the server, then repeat for the next model.

### A. Start the vLLM server for one model

Use the model-specific vLLM command from your local environment/configuration. Keep the served model name and port aligned with `meta_model/configs/union_v0_models.local.yaml`.

Example pattern:

```bash
# Terminal 1: start exactly one model server
# Replace with the correct model path/name/port for the current model.
vllm serve /path/to/current/model \
  --served-model-name CURRENT_SERVED_MODEL_NAME \
  --port 8000
```

Wait until the health check responds before running the round trips.

### B. Run compact and permissive for the hosted model only

Set `MODEL_KEY` to the key in `meta_model/configs/union_v0_models.local.yaml` that points to the currently hosted vLLM endpoint.

```bash
export MODEL_KEY=medgemma   # change to qwen235b, llama4, or the current hosted model key

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_provisional_empirical.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/outputs/reduced_v1_provisional_roundtrip \
  --evidence_mode compact \
  --stage both

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_provisional_empirical.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/outputs/reduced_v1_provisional_roundtrip \
  --evidence_mode permissive \
  --stage both
```

Expected output directories:

```text
meta_model/outputs/reduced_v1_provisional_roundtrip/<MODEL_KEY>/compact
meta_model/outputs/reduced_v1_provisional_roundtrip/<MODEL_KEY>/permissive
```

### C. Repeat model-by-model

After the two modes finish for the current hosted model:

```text
1. stop current vLLM server
2. start next model with vLLM
3. update MODEL_KEY
4. run compact
5. run permissive
```

The V1 runner is schema-dynamic and reads field names from the YAML, so provisional `semantic_cluster_C###` fields and final audited fields can be evaluated with the same code.

## 7. Standardize, score, and compare

Use the existing standardization/scoring pipeline, adding only the provisional V1 output directories that have completed under `--reduced_v1_model_dirs`.

Compare:

```text
individual DUO / ICO / ODRL / FHIR source-model prompts
Union V0 full meta-model
Provisional empirical V1 compact
Provisional empirical V1 permissive
```

Primary comparison dimensions:

```text
meaning-preservation classifier score
lexical/content recall
cue preservation
annotation burden / number of extracted fields
parse success
residual/unmatched language rate
```

## 8. Prepare PI handoff package

Include:

```text
1. visual_report/v1_discovery_visual_audit_report.md
2. semantic_cluster_evidence_summary.csv
3. semantic_cluster_audit_template.csv
4. cluster_support_corrected.csv
5. semantic_cluster_support.png
6. semantic_cluster_source_model_heatmap.png
7. semantic_cluster_network.png
8. provision_bundle_cluster_network.png
9. provisional V1 performance comparison vs individual and Union V0
10. representative examples where V1 succeeds/fails
```

Ask the PI to review:

```text
Which clusters represent real informed-consent semantic functions?
Which clusters should be split or merged?
What should each included cluster be named?
Which source-model distinctions should remain extensions rather than core fields?
```

## 9. Build the final audited V1 schema after PI review

After auditing `semantic_cluster_audit_template.csv`:

```bash
python meta_model/scripts/20_build_reduced_v1_schema_from_audit.py \
  --audit_csv meta_model/v1_reduced_expert/semantic_cluster_audit_template.csv \
  --clusters_csv meta_model/v1_reduced_expert/semantic_equivalence_clusters.csv \
  --decision_summary_csv meta_model/v1_reduced_expert/expert_sentence_level_decision_summary.csv \
  --output_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.yaml \
  --output_json meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.json
```

Then rerun V1 validation with the audited schema, again one hosted vLLM model at a time.

## Manuscript framing

The reduced V1 meta-model is discovered from expert-validated semantic-equivalence evidence, evaluated once as a provisional data-driven schema, finalized by limited expert audit/naming, and validated on new LLM outputs. A good V1 should approach Union V0 meaning preservation while reducing redundancy and annotation burden, and should outperform individual source models by combining complementary consent-language coverage from DUO, ICO, FHIR Consent, and ODRL.
