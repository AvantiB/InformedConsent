# Union V0 Full-Dictionary Round-Trip Runbook

This runbook describes how to run the Union V0 baseline with one LLM deployed at a time. Union V0 is the unreduced combined inventory of ICO, DUO, FHIR Consent, and ODRL source-model elements.

Do not commit local absolute paths, user names, API keys, private endpoints, or cluster-specific directories. Use environment variables or local config files for anything machine-specific.

## Files for this phase

```text
meta_model/configs/union_v0_models_template.yaml
meta_model/scripts/03_run_union_v0_roundtrip.py
meta_model/scripts/04_validate_union_v0_outputs.py
```

The runner uses the full Union V0 dictionary in the prompt, performs forward mapping and backward reconstruction, and writes append-only JSONL outputs so interrupted runs can resume.

## Method note: overlap-aware Union V0 mapping

Union V0 is a naive union of multiple source information models. Because it is not reduced, the same or similar phrase may legitimately map to multiple elements. A broader phrase may also map to one source-model element while a nested phrase maps to a narrower element from another source model.

The runner asks the LLM to produce two layers:

```text
raw annotations
  - all clear span labels, including same-span, overlapping, and nested labels

interpretation_units
  - the LLM's decision about how related annotations should be considered together for backward reconstruction
```

Backward reconstruction uses `interpretation_units` as the primary meaning-preserving layer, while preserving the raw annotation layer as evidence for redundancy, complementarity, broad/narrow nesting, and conflicts.

The runner validates `union_element_id` values against the Union V0 inventory. Common unambiguous formatting errors are repaired, for example a single-colon output such as `ICO:0000108` can be normalized to the exact inventory ID `ICO::ICO:0000108`. Remaining invalid IDs are moved to `invalid_annotations` and are not treated as primary dictionary-grounded evidence for backward reconstruction.

## 0. Set local paths outside the repo

Set paths in your shell or job script. Replace the placeholders with local paths on your machine or cluster.

```bash
export REPO_DIR=/path/to/InformedConsent
export PROMPT_DIR=/path/to/source_model_prompts
export ROUNDTRIPS_CSV=/path/to/roundtrips.csv

cd "$REPO_DIR"
```

## 1. Pull and install dependencies

```bash
git pull origin main
pip install -r meta_model/requirements.txt
```

## 2. Rebuild Union V0 inventory if needed

```bash
python -m py_compile meta_model/scripts/00_build_union_v0_inventory.py
python -m py_compile meta_model/scripts/03_run_union_v0_roundtrip.py
python -m py_compile meta_model/scripts/04_validate_union_v0_outputs.py

python meta_model/scripts/00_build_union_v0_inventory.py \
  --prompt_dir "$PROMPT_DIR" \
  --output_dir meta_model/v0_union
```

Check the inventory:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv('meta_model/v0_union/source_element_inventory.csv')
print(df.groupby(['source_model', 'element_scope']).size())
print('Total:', len(df))
PY
```

Expected count for the current prompts is approximately:

```text
DUO: 23 span
ICO: 49 span
ODRL: 7 span + 1 sentence_level
FHIR_Consent: 18 span + 1 sentence_level
Total: 99
```

## 3. Create local model config

```bash
mkdir -p meta_model/configs
cp meta_model/configs/union_v0_models_template.yaml \
   meta_model/configs/union_v0_models.local.yaml
```

Edit `meta_model/configs/union_v0_models.local.yaml` if the served model names or ports differ. Do not commit the local file if it contains private endpoints or keys.

For vLLM, use a dummy non-empty key:

```bash
export VLLM_API_KEY=EMPTY
```

For GPT-5.5 through the OpenAI API:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
```

## 4. Smoke test one model at a time

Use one terminal/job for the vLLM server and another for the runner.

### 4A. Start vLLM server for MedGemma

Adjust the model path/name to your local model location.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/MedGemma \
  --served-model-name medgemma \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192
```

### 4B. Run a 20-sentence smoke test

```bash
rm -rf meta_model/outputs/union_v0_roundtrip_smoke/medgemma

python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/union_v0_roundtrip_smoke \
  --limit 20
```

### 4C. Validate the smoke test

```bash
python meta_model/scripts/04_validate_union_v0_outputs.py \
  --model_output_dir meta_model/outputs/union_v0_roundtrip_smoke/medgemma \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv
```

Inspect:

```bash
cat meta_model/outputs/union_v0_roundtrip_smoke/medgemma/validation.summary.json
head -n 20 meta_model/outputs/union_v0_roundtrip_smoke/medgemma/validation.invalid_annotations.csv
```

Smoke-test gates:

```text
n_forward_records == 20
n_backward_records == 20
n_failed_requests == 0
n_forward_jsonl_parse_errors == 0
n_records_with_interpretation_units == 20
n_invalid_ids_in_primary_annotations == 0
n_repairable_ids_not_yet_repaired == 0
ready_for_full_run_pragmatic == true
```

`ready_for_full_run_strict` additionally requires zero quarantined invalid IDs. This is ideal but not mandatory. A small number of `invalid_annotations_from_runner` is acceptable if the IDs are quarantined, the invalid rate is low, and the spans are reviewed. By default, the pragmatic gate allows quarantined invalid IDs up to 5% of primary annotations.

## 5. Current MedGemma smoke-test status and immediate next action

The patched MedGemma smoke test produced 20 forward mappings and 20 backward reconstructions with no failed requests or parse errors. All 20 records included interpretation units. Primary annotations had exact valid Union V0 IDs after runner repair. Five invalid IDs were quarantined into `invalid_annotations`; this is below the 5% pragmatic threshold but should still be inspected.

Next action: pull the latest validator, rerun validation, inspect the quarantined invalid annotations, and proceed to the full MedGemma run if `ready_for_full_run_pragmatic` is true.

## 6. Full MedGemma run

After the pragmatic smoke gate is passed:

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/union_v0_roundtrip
```

Validate the full run:

```bash
python meta_model/scripts/04_validate_union_v0_outputs.py \
  --model_output_dir meta_model/outputs/union_v0_roundtrip/medgemma \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv
```

The runner deduplicates repeated source sentences by default. To run every row rather than unique source sentences, add:

```bash
--no_dedupe_sentences
```

Outputs for each model are written to:

```text
meta_model/outputs/union_v0_roundtrip/<model_key>/
  run_metadata.json
  union_v0_forward_mappings.jsonl
  union_v0_backward_reconstructions.jsonl
  union_v0_roundtrip_outputs.csv
  failed_requests.jsonl
  validation.summary.json
  validation.invalid_annotations.csv
```

The JSONL files are append-only. If the job stops, rerun the same command and completed `source_id`s will be skipped.

## 7. Stop server, deploy the next open-source model, rerun with the next key

Stop the current vLLM server, then start the next model using the same port and matching served model name.

### Qwen235B

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen235B \
  --served-model-name qwen235b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192
```

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key qwen235b \
  --output_dir meta_model/outputs/union_v0_roundtrip

python meta_model/scripts/04_validate_union_v0_outputs.py \
  --model_output_dir meta_model/outputs/union_v0_roundtrip/qwen235b \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv
```

### Llama-4 Scout

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --served-model-name meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192
```

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key llama4_scout \
  --output_dir meta_model/outputs/union_v0_roundtrip

python meta_model/scripts/04_validate_union_v0_outputs.py \
  --model_output_dir meta_model/outputs/union_v0_roundtrip/llama4_scout \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv
```

## 8. GPT-5.5 run

No vLLM server is needed for this condition.

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE

python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key gpt55 \
  --output_dir meta_model/outputs/union_v0_roundtrip

python meta_model/scripts/04_validate_union_v0_outputs.py \
  --model_output_dir meta_model/outputs/union_v0_roundtrip/gpt55 \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv
```

## 9. Individual-model replication with the new LLMs

Because the LLM panel changes relative to the original researchers' experiments, the clean design is:

```text
A. New LLMs + original individual-model prompts
B. New LLMs + Union V0 full-dictionary prompt
C. New LLMs + reduced meta-model prompt, later
```

Condition A is needed to separate the effect of the LLM from the effect of the information model. Union V0 should not be compared only to historical outputs from different LLMs.

## 10. After all four Union V0 runs

Archive or upload:

```text
meta_model/outputs/union_v0_roundtrip/
```

The next analysis step will score the Union V0 reconstructions with the meaning-preservation classifier and use both raw annotations and interpretation units to build the evidence graph for data/language-driven reduction.
