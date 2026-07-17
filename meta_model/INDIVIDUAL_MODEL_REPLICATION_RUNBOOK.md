# Individual Source-Model Replication Runbook

This runbook covers the replication condition:

```text
new LLMs + original individual source-model prompts
```

This condition is needed because the LLM panel changes relative to the original researchers' experiments. It separates the effect of the LLM from the effect of the information model.

## Files for this phase

```text
meta_model/scripts/05_run_individual_model_roundtrip.py
```

The script runs one hosted model against the original individual source-model forward prompts for:

```text
DUO
ICO
ODRL
FHIR_Consent
```

Forward outputs use the original prompt text. Backward reconstruction uses a matching backward prompt if you provide `--backward_prompt_dir`; otherwise it uses a generic reconstruction prompt that does not see the original sentence.

## 0. Required local paths

Do not commit local absolute paths. Set these in your shell or job script:

```bash
export REPO_DIR=/path/to/InformedConsent
export PROMPT_DIR=/path/to/source_model_prompts
export ROUNDTRIPS_CSV=/path/to/roundtrips.csv

cd "$REPO_DIR"
```

The prompt directory should contain the original individual source-model prompt files. The script discovers files by names containing `DUO`, `ICO`, `ODRL`, and `FHIR`.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/05_run_individual_model_roundtrip.py
```

## 2. Smoke test MedGemma individual prompts

Keep the current MedGemma vLLM server running. Then run a small smoke test across all four individual information models:

```bash
python meta_model/scripts/05_run_individual_model_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --prompt_dir "$PROMPT_DIR" \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/individual_model_roundtrip_smoke \
  --info_models all \
  --limit 5
```

This runs 5 unique consent sentences for each of DUO, ICO, ODRL, and FHIR_Consent.

Inspect:

```bash
find meta_model/outputs/individual_model_roundtrip_smoke/medgemma -maxdepth 2 -type f -print

head -n 2 meta_model/outputs/individual_model_roundtrip_smoke/medgemma/DUO/forward_mappings.jsonl
head -n 2 meta_model/outputs/individual_model_roundtrip_smoke/medgemma/DUO/backward_reconstructions.jsonl
```

## 3. Full MedGemma individual-prompt run

After the smoke test works, run the full individual-prompt condition while MedGemma is still hosted:

```bash
python meta_model/scripts/05_run_individual_model_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --prompt_dir "$PROMPT_DIR" \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/individual_model_roundtrip \
  --info_models all
```

By default, the script deduplicates repeated source sentences. To run every row rather than unique source sentences, add:

```bash
--no_dedupe_sentences
```

Outputs are written to:

```text
meta_model/outputs/individual_model_roundtrip/medgemma/<INFO_MODEL>/
  prompt_files.json
  forward_mappings.jsonl
  backward_reconstructions.jsonl
  roundtrip_outputs.csv
  failed_requests.jsonl
```

The JSONL files are append-only. If the job stops, rerun the same command and completed `source_id`s will be skipped.

## 4. Suggested execution order while one model is hosted

For each hosted LLM, run both conditions before stopping the server:

```text
1. Union V0 full-dictionary run
2. Individual source-model prompt run
3. Validate/archive outputs
4. Stop server and deploy next LLM
```

For the currently hosted MedGemma, finish the Union V0 full run first if it is already processing. Then run the individual-prompt smoke test and full individual-prompt run before switching to Qwen235B.

## 5. Caveat about backward prompts

If original individual-source-model backward prompts are available, pass them with:

```bash
--backward_prompt_dir /path/to/backward_prompts
```

If not provided, the script uses a generic backward reconstruction prompt. In that case, report the experiment as:

```text
new LLMs + original individual forward prompts + standardized backward reconstruction prompt
```

rather than a perfect replication of the original backward-prompt setup.
