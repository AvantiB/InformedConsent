# Phase 1 strict round-trip completion runbook

This runbook is the current paper-facing Phase 1 instruction set.

## Core design

Phase 1 uses a constant forward/backward protocol so that differences reflect the information model/dictionary, not prompt wording.

Sentence-level consent force is **not** an annotation label and is **not** an information-model element for Phase 1 evaluation. Fields such as `FHIR_Consent::Consent.provision.type` and `ODRL::Rule_TestSentence` are removed from the annotation dictionaries and are not allowed as span labels. Their allowed values are represented only through the universal top-level field:

```text
sentence_decision = permit | deny | mixed | unclear
```

Backward reconstruction may receive this controlled `sentence_decision` only when valid span annotations exist.

## Scripts to use

Use the Phase 1 wrapper scripts below, not the legacy entry points, for the final Phase 1 runs.

### Union V0

- Local/vLLM script: `meta_model/scripts/42_run_phase1_union_v0_roundtrip.py`
- Mayo GPT/Apigee script: `meta_model/scripts/43_run_phase1_union_v0_roundtrip_apigee.py`
- Dictionary: all non-sentence-level rows in `meta_model/v0_union/source_element_inventory.csv`

### Individual source models

- Local/vLLM script: `meta_model/scripts/44_run_phase1_individual_roundtrip.py`
- Mayo GPT/Apigee script: `meta_model/scripts/45_run_phase1_individual_roundtrip_apigee.py`
- Dictionary: non-sentence-level rows from `source_element_inventory.csv`, filtered to one source model at a time
- `--prompt_dir` is deprecated/reference-only and is not used for primary Phase 1 prompting.

### Backward prompt for all conditions

All Phase 1 conditions use the same universal backward prompt. Backward reconstruction receives only:

- valid span annotations
- static label metadata/definitions
- sanitized relationship links
- the controlled universal `sentence_decision`, when valid span evidence exists

Backward reconstruction does not receive:

- original sentence
- unmatched language
- evidence text from interpretation units
- combined meanings
- backward mapping decisions
- rationales
- raw forward responses
- source-model-specific sentence-level label rows

Rows with no valid span annotation evidence are not sent to the backward LLM. They receive a blank reconstruction and are flagged as excluded from schema induction.

## Compile check

```bash
python -m py_compile \
  meta_model/scripts/42_run_phase1_union_v0_roundtrip.py \
  meta_model/scripts/43_run_phase1_union_v0_roundtrip_apigee.py \
  meta_model/scripts/44_run_phase1_individual_roundtrip.py \
  meta_model/scripts/45_run_phase1_individual_roundtrip_apigee.py \
  meta_model/scripts/41_filter_phase1_outputs_for_schema_induction.py
```

## Smoke tests

Union V0 local model:

```bash
export PHASE1_SMOKE=meta_model/phase1_smoke/final_sentence_decision_policy
export MODEL_KEY=medgemma

rm -rf "$PHASE1_SMOKE/union_v0/$MODEL_KEY"

python meta_model/scripts/42_run_phase1_union_v0_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_SMOKE/union_v0" \
  --stage both \
  --limit 10
```

Individual source models local model:

```bash
export PHASE1_SMOKE=meta_model/phase1_smoke/final_sentence_decision_policy
export MODEL_KEY=medgemma

rm -rf "$PHASE1_SMOKE/individual/$MODEL_KEY"

python meta_model/scripts/44_run_phase1_individual_roundtrip.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_SMOKE/individual" \
  --info_models all \
  --stage both \
  --limit 10
```

Mayo GPT/Apigee equivalents:

```bash
export PHASE1_SMOKE=meta_model/phase1_smoke/final_sentence_decision_policy
export MODEL_KEY=mayo_gpt55

rm -rf "$PHASE1_SMOKE/union_v0/$MODEL_KEY"
python meta_model/scripts/43_run_phase1_union_v0_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_SMOKE/union_v0" \
  --stage both \
  --limit 10

rm -rf "$PHASE1_SMOKE/individual/$MODEL_KEY"
python meta_model/scripts/45_run_phase1_individual_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_SMOKE/individual" \
  --info_models all \
  --stage both \
  --limit 10
```

## Full Phase 1 runs

Union V0 local/vLLM models:

```bash
export PHASE1_ROOT=meta_model/phase1_strict

for MODEL_KEY in medgemma qwen235b llama4_scout; do
  rm -rf "$PHASE1_ROOT/union_v0/$MODEL_KEY"

  python meta_model/scripts/42_run_phase1_union_v0_roundtrip.py \
    --roundtrips_csv "$ROUNDTRIPS_CSV" \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$MODEL_KEY" \
    --output_dir "$PHASE1_ROOT/union_v0" \
    --stage both

done
```

Union V0 Mayo GPT/Apigee:

```bash
export PHASE1_ROOT=meta_model/phase1_strict
export MODEL_KEY=mayo_gpt55

rm -rf "$PHASE1_ROOT/union_v0/$MODEL_KEY"

python meta_model/scripts/43_run_phase1_union_v0_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_ROOT/union_v0" \
  --stage both
```

Individual source-model local/vLLM models:

```bash
export PHASE1_ROOT=meta_model/phase1_strict

for MODEL_KEY in medgemma qwen235b llama4_scout; do
  rm -rf "$PHASE1_ROOT/individual/$MODEL_KEY"

  python meta_model/scripts/44_run_phase1_individual_roundtrip.py \
    --roundtrips_csv "$ROUNDTRIPS_CSV" \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$MODEL_KEY" \
    --output_dir "$PHASE1_ROOT/individual" \
    --info_models all \
    --stage both

done
```

Individual source-model Mayo GPT/Apigee:

```bash
export PHASE1_ROOT=meta_model/phase1_strict
export MODEL_KEY=mayo_gpt55

rm -rf "$PHASE1_ROOT/individual/$MODEL_KEY"

python meta_model/scripts/45_run_phase1_individual_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_ROOT/individual" \
  --info_models all \
  --stage both
```

## Quick structural summary

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path('meta_model/phase1_strict')
for csv_path in sorted(root.glob('**/*roundtrip_outputs.csv')):
    df = pd.read_csv(csv_path).fillna('')
    print('\n', csv_path)
    print('rows', len(df))
    for col in [
        'annotation_count',
        'n_annotations_valid',
        'n_annotations_invalid',
        'n_annotations_routed_to_unmatched',
        'n_sentence_level_annotations_backward_eligible',
        'n_sentence_level_elements_dropped_by_policy',
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
            print(col, 'sum=', vals.sum(), 'mean=', round(vals.mean(), 2))
PY
```

## Backward leakage check

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

banned = [
    'unmatched_language',
    'evidence_span_text',
    'combined_meaning',
    'backward_mapping_decision',
    'rationale',
    'raw_response',
    'original_sentence',
    'Consent.provision.type',
    'Rule_TestSentence',
]

for csv_path in sorted(Path('meta_model/phase1_strict').glob('**/*roundtrip_outputs.csv')):
    df = pd.read_csv(csv_path).fillna('')
    packet_col = 'backward_packet_json' if 'backward_packet_json' in df.columns else 'sanitized_forward_material_json'
    if packet_col not in df.columns:
        continue
    text = '\n'.join(df[packet_col].astype(str).tolist())
    hits = [x for x in banned if x in text]
    print(csv_path, 'BANNED_HITS=', hits)
PY
```

Expected: `BANNED_HITS=[]` for every output file.

## Filter Phase 1 outputs for meta-model development

Before co-occurrence, schema induction, or functionally validated evidence construction, remove NA-only/deferred-tagging rows and rows without valid span evidence:

```bash
python meta_model/scripts/41_filter_phase1_outputs_for_schema_induction.py \
  --inputs 'meta_model/phase1_strict/**/*.csv' \
  --output_dir meta_model/phase1_strict/schema_induction_inputs \
  --prefix phase1_strict
```

Use only:

```text
meta_model/phase1_strict/schema_induction_inputs/phase1_strict_filtered_rows.csv
```

for meta-model development.

Keep the unfiltered Phase 1 outputs for coverage-inclusive evaluation. Zero-annotation rows, including true DUO non-coverage rows, are useful for coverage analysis but must not contribute positive schema evidence.
