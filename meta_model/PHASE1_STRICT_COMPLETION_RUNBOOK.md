# Phase 1 strict round-trip completion runbook

This runbook supersedes legacy prompt-specific Phase 1 instructions.

## Core design

Phase 1 uses a constant forward/backward protocol so that differences reflect the information model/dictionary, not prompt wording.

### Union V0

- Script: `meta_model/scripts/03_run_union_v0_roundtrip.py`
- Mayo GPT/Apigee wrapper: `meta_model/scripts/12_run_union_v0_roundtrip_apigee.py`
- Forward prompt: universal Union V0 dictionary prompt
- Dictionary: all rows in `meta_model/v0_union/source_element_inventory.csv`

### Individual source models

- Script: `meta_model/scripts/05_run_individual_model_roundtrip.py`
- Mayo GPT/Apigee wrapper: `meta_model/scripts/13_run_individual_model_roundtrip_apigee.py`
- Forward prompt: one universal source-model prompt
- Dictionary: `source_element_inventory.csv` filtered to one source model at a time
- `--prompt_dir` is deprecated/reference-only and is not used for primary Phase 1 forward prompting.

### Backward prompt for all conditions

All Phase 1 conditions use the same universal backward prompt. Backward reconstruction receives only:

- valid span annotations
- static label metadata/definitions
- sanitized relationship links
- controlled sentence-level decisions

Backward reconstruction does not receive:

- original sentence
- unmatched language
- evidence text from interpretation units
- combined meanings
- backward mapping decisions
- rationales
- raw forward responses

Rows with no valid span annotation evidence are not sent to the backward LLM. They receive a blank reconstruction and are flagged as excluded from schema induction.

## Compile check

```bash
python -m py_compile \
  meta_model/scripts/03_run_union_v0_roundtrip.py \
  meta_model/scripts/05_run_individual_model_roundtrip.py \
  meta_model/scripts/12_run_union_v0_roundtrip_apigee.py \
  meta_model/scripts/13_run_individual_model_roundtrip_apigee.py \
  meta_model/scripts/41_filter_phase1_outputs_for_schema_induction.py
```

## Union V0 full runs

Local/vLLM models:

```bash
export PHASE1_ROOT=meta_model/phase1_strict

for MODEL_KEY in medgemma qwen235b llama4_scout; do
  rm -rf "$PHASE1_ROOT/union_v0/$MODEL_KEY"

  python meta_model/scripts/03_run_union_v0_roundtrip.py \
    --roundtrips_csv "$ROUNDTRIPS_CSV" \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$MODEL_KEY" \
    --output_dir "$PHASE1_ROOT/union_v0" \
    --stage both

done
```

Mayo GPT/Apigee:

```bash
export PHASE1_ROOT=meta_model/phase1_strict
export MODEL_KEY=mayo_gpt55

rm -rf "$PHASE1_ROOT/union_v0/$MODEL_KEY"

python meta_model/scripts/12_run_union_v0_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$PHASE1_ROOT/union_v0" \
  --stage both
```

## Individual source-model full runs

Local/vLLM models:

```bash
export PHASE1_ROOT=meta_model/phase1_strict

for MODEL_KEY in medgemma qwen235b llama4_scout; do
  rm -rf "$PHASE1_ROOT/individual/$MODEL_KEY"

  python meta_model/scripts/05_run_individual_model_roundtrip.py \
    --roundtrips_csv "$ROUNDTRIPS_CSV" \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$MODEL_KEY" \
    --output_dir "$PHASE1_ROOT/individual" \
    --info_models all \
    --stage both

done
```

Mayo GPT/Apigee:

```bash
export PHASE1_ROOT=meta_model/phase1_strict
export MODEL_KEY=mayo_gpt55

rm -rf "$PHASE1_ROOT/individual/$MODEL_KEY"

python meta_model/scripts/13_run_individual_model_roundtrip_apigee.py \
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
