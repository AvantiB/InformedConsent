# Direct LLM-induced reduced schema runbook

This runbook describes the conservative/direct reduced-schema arm.

## Purpose

The direct LLM-induced schema arm creates fold-specific reduced informed-consent dictionaries using:

1. DUO, ICO, ODRL, and FHIR Consent source-model dictionary rows.
2. Optional external requirements/guidance text, such as NIH repository/directive guidance.
3. Representative training-fold consent sentences.

This arm should stay close to the governed/expert source information models. The later data-driven + LLM induction arm is the more flexible/creative schema-composition pipeline.

## How the NIH guidance is given to GPT-5.5

The Mayo GPT-5.5 Apigee path currently uses chat-completions style messages, so the guidance is passed as text inside the induction prompt, not uploaded as a separate API file.

Recommended workflow:

```bash
# PDF input is supported if pypdf is installed; otherwise convert to text first.
python meta_model/scripts/46_prepare_direct_llm_schema_induction_inputs.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --guidance_files path/to/nih_guidance_or_directive.txt \
  --output_dir meta_model/direct_llm_reduced_schema/inputs \
  --n_folds 4 \
  --examples_per_fold 60 \
  --max_guidance_chars_per_file 12000
```

The script writes one packet per fold:

```text
meta_model/direct_llm_reduced_schema/inputs/fold_00/direct_induction_input.json
meta_model/direct_llm_reduced_schema/inputs/fold_01/direct_induction_input.json
meta_model/direct_llm_reduced_schema/inputs/fold_02/direct_induction_input.json
meta_model/direct_llm_reduced_schema/inputs/fold_03/direct_induction_input.json
```

Each packet contains training-fold examples only. Held-out form keys are recorded but their sentences are not shown to the induction LLM.

## Fold-specific schemas

Yes: for cross-validation, create four fold-specific direct schemas.

For each fold:

- training forms are used for examples;
- held-out forms are excluded from schema induction;
- source-model dictionaries remain the same;
- guidance text remains the same;
- resulting schema is evaluated on that fold's held-out sentences.

After CV, a consensus schema can be built separately for expert review. The fold-specific schemas are the primary evaluation artifacts.

## Schema shape

For now, the output is a flat span-level dictionary with optional modifiers.

Important design constraints:

- `sentence_decision` is universal and separate from labels.
- Sentence-level source-model rows are not dictionary labels.
- Flat fields should remain close to DUO, ICO, ODRL, and FHIR Consent.
- Modifiers are allowed only when they preserve meaning without creating excessive fields.
- Any requirement-driven extension must be explicitly marked and justified.

## Run induction with Mayo GPT-5.5

```bash
export MODEL_CONFIG=meta_model/configs/union_v0_models.local.yaml
export MODEL_KEY=mayo_gpt55

python meta_model/scripts/47_induce_direct_reduced_schema_with_llm.py \
  --input_dir meta_model/direct_llm_reduced_schema/inputs \
  --output_dir meta_model/direct_llm_reduced_schema/outputs \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --granularity both \
  --folds all
```

This creates:

```text
meta_model/direct_llm_reduced_schema/outputs/fold_00/high/schema.json
meta_model/direct_llm_reduced_schema/outputs/fold_00/low/schema.json
...
meta_model/direct_llm_reduced_schema/outputs/fold_03/high/schema.json
meta_model/direct_llm_reduced_schema/outputs/fold_03/low/schema.json
```

## Recommended first smoke

Run one fold and one granularity first:

```bash
python meta_model/scripts/47_induce_direct_reduced_schema_with_llm.py \
  --input_dir meta_model/direct_llm_reduced_schema/inputs \
  --output_dir meta_model/direct_llm_reduced_schema/smoke_outputs \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key mayo_gpt55 \
  --granularity low \
  --folds fold_00
```

Check the output JSON for:

- no sentence-level source rows as fields;
- fields trace back to source information models;
- no raw guideline headings copied as field names;
- modifiers are optional and justified;
- crosswalk covers source elements with mapping types;
- requirement-driven extensions are few and explicit.

## Compile check

```bash
python -m py_compile \
  meta_model/scripts/46_prepare_direct_llm_schema_induction_inputs.py \
  meta_model/scripts/47_induce_direct_reduced_schema_with_llm.py
```
