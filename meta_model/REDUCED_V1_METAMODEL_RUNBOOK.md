# Reduced V1 meta-model discovery, smoke testing, audit, visualization, and validation runbook

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

Script 17 writes empirical clusters and an audit template. Script 19 visualizes the evidence. Script 21 builds a provisional cluster-ID schema so the data-driven model can be smoke-tested as-is. Script 20 converts a PI/expert-audited cluster table into the final V1 YAML schema.

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

## 5. Build a provisional empirical V1 schema for smoke testing

This step intentionally uses the discovered clusters as-is. Field names are provisional cluster IDs such as `semantic_cluster_C001`, so behavior can be inspected before the PI names or reorganizes the fields.

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

## 6. V1 reduced smoke tests only

Do **not** run the full dataset yet. First inspect a small number of examples from two currently hosted models, MedGemma and Qwen. The V1 runner now uses a V0-like prompt structure:

```text
sentence_decision
sentence_level_elements
annotations
interpretation_units
unmatched_language
```

The prompt uses neutral cluster-dictionary wording, does not say “V1 meta-model” to the LLM, does not use named roles, and treats permit/deny/mixed/unclear as sentence/provision-level decisions only.

### A. Start one hosted vLLM model

Use the model-specific command from your local environment/configuration. Keep served model name and port aligned with `meta_model/configs/union_v0_models.local.yaml`.

```bash
# Terminal 1: start exactly one model server
# Replace with the correct model path/name/port.
vllm serve /path/to/current/model \
  --served-model-name CURRENT_SERVED_MODEL_NAME \
  --port 8000
```

Wait for the health check to respond.

### B. Smoke test MedGemma or Qwen, one hosted model at a time

Set `MODEL_KEY` to the currently hosted model key. Use `--limit 5` or `--limit 10` for the first inspection.

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

Inspect the CSV:

```bash
column -s, -t < meta_model/outputs/reduced_v1_smoke/${MODEL_KEY}/compact/reduced_v1_roundtrip_outputs.csv | less -S
```

Then repeat after stopping the current vLLM server and starting the next model:

```text
1. stop current vLLM server
2. start the other currently available vLLM model
3. update MODEL_KEY
4. rerun the compact smoke test
5. inspect outputs before running permissive or full tests
```

Only after compact smoke outputs look reasonable should permissive smoke tests be run:

```bash
python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_provisional_empirical.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/outputs/reduced_v1_smoke \
  --evidence_mode permissive \
  --stage both \
  --limit "$N_SMOKE"
```

Expected output directories:

```text
meta_model/outputs/reduced_v1_smoke/<MODEL_KEY>/compact
meta_model/outputs/reduced_v1_smoke/<MODEL_KEY>/permissive
```

## 7. What to inspect in smoke outputs

For each example, check:

```text
1. sentence_decision is sentence/provision-level only.
2. permit/deny/mixed/unclear are not used as span-level annotations.
3. decision cues such as “agree” or “allow” are stored in sentence_level_elements.
4. span-level annotations use only semantic_cluster_C### cluster IDs.
5. unmatched_language preserves important content not captured by clusters.
6. interpretation_units explain how clusters combine for reconstruction.
7. reconstructed_sentence preserves the original meaning without seeing the original sentence.
```

If the smoke outputs look acceptable, then proceed to broader provisional V1 evaluation.

## 8. Later full comparison, after smoke tests pass

Use the existing standardization/scoring pipeline, adding only completed provisional V1 output directories under `--reduced_v1_model_dirs`.

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

## 9. Prepare PI handoff package

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
9. provisional V1 smoke/performance comparison vs individual and Union V0
10. representative examples where V1 succeeds/fails
```

Ask the PI to review:

```text
Which clusters represent real informed-consent semantic functions?
Which clusters should be split or merged?
What should each included cluster be named?
Which source-model distinctions should remain extensions rather than core fields?
```

## 10. Build final audited V1 schema after PI review

After auditing `semantic_cluster_audit_template.csv`:

```bash
python meta_model/scripts/20_build_reduced_v1_schema_from_audit.py \
  --audit_csv meta_model/v1_reduced_expert/semantic_cluster_audit_template.csv \
  --clusters_csv meta_model/v1_reduced_expert/semantic_equivalence_clusters.csv \
  --decision_summary_csv meta_model/v1_reduced_expert/expert_sentence_level_decision_summary.csv \
  --output_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.yaml \
  --output_json meta_model/v1_reduced_expert/reduced_metamodel_v1_audited.json
```

Then rerun validation with the audited schema, again one hosted vLLM model at a time.

## Manuscript framing

The reduced V1 model is discovered from expert-validated semantic-equivalence evidence, smoke-tested as a provisional data-driven cluster dictionary, finalized by limited expert audit/naming, and validated on new LLM outputs. A good V1 should approach Union V0 meaning preservation while reducing redundancy and annotation burden, and should outperform individual source models by combining complementary consent-language coverage from DUO, ICO, FHIR Consent, and ODRL.
