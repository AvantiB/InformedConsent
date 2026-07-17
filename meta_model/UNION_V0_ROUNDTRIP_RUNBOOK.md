# Union V0 Full-Dictionary Round-Trip Runbook

This runbook describes how to run the Union V0 baseline with one LLM deployed at a time. Union V0 is the unreduced combined inventory of ICO, DUO, FHIR Consent, and ODRL source-model elements.

## Files added for this phase

```text
meta_model/configs/union_v0_models_template.yaml
meta_model/scripts/03_run_union_v0_roundtrip.py
```

The runner uses the full Union V0 dictionary in the prompt, performs forward mapping and backward reconstruction, and writes append-only JSONL outputs so interrupted runs can resume.

## 0. Pull and install dependencies

```bash
cd /dgx1data/aii/tao/m338824/R03-InformedConsent/InformedConsent

git pull origin main

pip install openai pyyaml pandas
```

## 1. Rebuild Union V0 inventory if needed

```bash
python -m py_compile meta_model/scripts/00_build_union_v0_inventory.py

python meta_model/scripts/00_build_union_v0_inventory.py \
  --prompt_dir /dgx1data/aii/tao/m338824/R03-InformedConsent/source_model_prompts \
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

## 2. Create local model config

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

## 3. Smoke test one model at a time

Use one terminal for the vLLM server and another for the runner.

### 3A. Start vLLM server for MedGemma

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

### 3B. Run a 20-sentence smoke test

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv /dgx1data/aii/tao/m338824/R03-InformedConsent/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/union_v0_roundtrip_smoke \
  --limit 20
```

Inspect:

```bash
ls -lh meta_model/outputs/union_v0_roundtrip_smoke/medgemma
head -n 2 meta_model/outputs/union_v0_roundtrip_smoke/medgemma/union_v0_forward_mappings.jsonl
head -n 2 meta_model/outputs/union_v0_roundtrip_smoke/medgemma/union_v0_backward_reconstructions.jsonl
```

If JSON parsing looks good, proceed to the full run for that model.

## 4. Full run for one deployed open-source model

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv /dgx1data/aii/tao/m338824/R03-InformedConsent/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/union_v0_roundtrip
```

The runner deduplicates repeated source sentences by default. With the current round-trip dataset this should run the unique consent sentences rather than all prior model/source-model rows. To run every row, add:

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
```

The JSONL files are append-only. If the job stops, rerun the same command and completed `source_id`s will be skipped.

## 5. Stop server, deploy the next model, rerun with the next key

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
  --roundtrips_csv /dgx1data/aii/tao/m338824/R03-InformedConsent/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key qwen235b \
  --output_dir meta_model/outputs/union_v0_roundtrip
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
  --roundtrips_csv /dgx1data/aii/tao/m338824/R03-InformedConsent/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key llama4_scout \
  --output_dir meta_model/outputs/union_v0_roundtrip
```

## 6. GPT-5.5 run

No vLLM server is needed for this condition.

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE

python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv /dgx1data/aii/tao/m338824/R03-InformedConsent/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key gpt55 \
  --output_dir meta_model/outputs/union_v0_roundtrip
```

## 7. After all four runs

Archive or upload:

```text
meta_model/outputs/union_v0_roundtrip/
```

The next analysis step will score the Union V0 reconstructions with the meaning-preservation classifier and use the forward mappings to build the evidence graph for data/language-driven reduction.
