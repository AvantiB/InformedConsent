# Strict annotation-only Phase 1 baseline runbook

This runbook documents the agreed recovery plan for finalizing the baseline phase after identifying leakage in earlier backward reconstruction packets.

## Agreed methodological decision

Existing forward mappings for the baselines can be reused:

```text
individual source models x LLMs
Union V0 x LLMs
```

Only the backward reconstruction and downstream scoring/diagnostics must be rerun for Phase 1.

## Strict backward input policy

The backward prompt is universal across all experiments. It may receive only:

```text
valid span-level annotations
annotation labels / source-model IDs / schema fields
annotation-attached canonical modifiers when present
sentence-level annotations only when at least one valid span annotation exists
```

It must not receive:

```text
original sentence
raw forward response
unmatched_language / residual text
interpretation_units
combined_meaning
rationales
previous reconstruction
unanchored sentence_decision
```

Rows with no backward-eligible annotations are not sent to the LLM. Their reconstruction is intentionally blank.

## Fresh output root

Do not write corrected outputs into the old experiment root. Start a fresh root:

```bash
export OLD_ROOT=meta_model/functional_v1_experiments
export STRICT_ROOT=meta_model/strict_annotation_only_experiments
mkdir -p "$STRICT_ROOT"
```

## Preserve old outputs instead of deleting

Recommended approach: archive old outputs rather than deleting them.

```bash
export ARCHIVE_ROOT=meta_model/archive/leakage_contaminated_$(date +%Y%m%d)
mkdir -p "$ARCHIVE_ROOT"

# Move only generated outputs, not scripts or source files.
# Edit these paths based on what exists locally.
for d in \
  "$OLD_ROOT/pi_expert_review_package_v2" \
  "$OLD_ROOT/pi_expert_review_package_v3" \
  "$OLD_ROOT/scored_roundtrips" \
  "$OLD_ROOT/diagnostics" \
  "$OLD_ROOT/comparison" \
  "$OLD_ROOT/plots"; do
  if [ -e "$d" ]; then
    mv "$d" "$ARCHIVE_ROOT/"
  fi
done
```

Keep a short note:

```bash
cat > "$ARCHIVE_ROOT/README.md" <<'EOF'
# Archived exploratory outputs

These outputs were generated before the strict annotation-only backward policy.
They may include backward packets that exposed interpretation_units and/or
unmatched_language to the backward LLM. Preserve for provenance only. Do not use
for final performance claims.
EOF
```

## Import existing forward outputs into the fresh root

The strict backward rerun scripts expect forward JSONL files to exist in the new output directory.

### Union V0

```bash
export MODEL_KEY=<model_key>
mkdir -p "$STRICT_ROOT/union_v0/$MODEL_KEY"
cp "$OLD_ROOT/union_v0/$MODEL_KEY/union_v0_forward_mappings.jsonl" \
   "$STRICT_ROOT/union_v0/$MODEL_KEY/"
```

Use the actual old path that contains `union_v0_forward_mappings.jsonl`.

### Individual source models

```bash
export MODEL_KEY=<model_key>
for INFO_MODEL in DUO ICO ODRL FHIR_Consent; do
  mkdir -p "$STRICT_ROOT/individual/$MODEL_KEY/$INFO_MODEL"
  cp "$OLD_ROOT/individual/$MODEL_KEY/$INFO_MODEL/forward_mappings.jsonl" \
     "$STRICT_ROOT/individual/$MODEL_KEY/$INFO_MODEL/"
done
```

Use the actual old path that contains the per-model `forward_mappings.jsonl` files.

## Rerun strict backward only

### Union V0

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv path/to/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models_template.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir "$STRICT_ROOT/union_v0" \
  --stage backward
```

For Mayo Apigee configs:

```bash
python meta_model/scripts/12_run_union_v0_roundtrip_apigee.py \
  --roundtrips_csv path/to/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models_template.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir "$STRICT_ROOT/union_v0" \
  --stage backward
```

### Individual source models

```bash
python meta_model/scripts/05_run_individual_model_roundtrip.py \
  --roundtrips_csv path/to/roundtrips.csv \
  --prompt_dir meta_model/prompts/individual_source_models \
  --model_config_yaml meta_model/configs/union_v0_models_template.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir "$STRICT_ROOT/individual" \
  --info_models all \
  --stage backward
```

For Mayo Apigee configs:

```bash
python meta_model/scripts/13_run_individual_model_roundtrip_apigee.py \
  --roundtrips_csv path/to/roundtrips.csv \
  --prompt_dir meta_model/prompts/individual_source_models \
  --model_config_yaml meta_model/configs/union_v0_models_template.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir "$STRICT_ROOT/individual" \
  --info_models all \
  --stage backward
```

## Standardize, score, and diagnose

Collect the strict baseline CSVs into the standardization input.

Expected strict baseline output files:

```text
$STRICT_ROOT/union_v0/<model_key>/union_v0_roundtrip_outputs.csv
$STRICT_ROOT/individual/<model_key>/DUO/roundtrip_outputs.csv
$STRICT_ROOT/individual/<model_key>/ICO/roundtrip_outputs.csv
$STRICT_ROOT/individual/<model_key>/ODRL/roundtrip_outputs.csv
$STRICT_ROOT/individual/<model_key>/FHIR_Consent/roundtrip_outputs.csv
```

Then run the same scoring and diagnostics workflow used elsewhere:

```bash
python meta_model/scripts/07_standardize_roundtrip_outputs.py \
  --input_root "$STRICT_ROOT" \
  --output_csv "$STRICT_ROOT/scoring_inputs/standardized_roundtrips.csv"

python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv "$STRICT_ROOT/scoring_inputs/standardized_roundtrips.csv" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir "$STRICT_ROOT/scored_roundtrips"

python meta_model/scripts/32_compute_roundtrip_diagnostic_metrics.py \
  --roundtrips_csv "$STRICT_ROOT/scored_roundtrips/scored_roundtrips.csv" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir "$STRICT_ROOT/diagnostics" \
  --review_sample_per_condition 25

python meta_model/scripts/31_compile_schema_condition_comparison.py \
  --scored_csv "$STRICT_ROOT/diagnostics/roundtrip_diagnostic_metrics.csv" \
  --output_dir "$STRICT_ROOT/comparison"
```

## Completion criteria for Phase 1

Phase 1 is complete only when all of the following are checked off in `PIPELINE_STAGE_TRACKER.csv`:

```text
old leakage-contaminated outputs archived
fresh strict output root created
Union V0 forward outputs copied/imported
Individual forward outputs copied/imported
Union V0 strict backward rerun complete
Individual strict backward rerun complete
strict baseline outputs standardized
strict baseline outputs scored
strict diagnostics generated
strict comparison summaries generated
```
