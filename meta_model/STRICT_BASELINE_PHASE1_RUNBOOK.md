# Structured annotation-only Phase 1 baseline runbook

This runbook documents the agreed recovery plan for finalizing the baseline phase after identifying leakage in earlier backward reconstruction packets.

## Agreed methodological decision

Existing forward mappings for the baselines can be reused:

```text
individual source models x LLMs
Union V0 x LLMs
```

Only the backward reconstruction and downstream scoring/diagnostics must be rerun for Phase 1.

## Phase 1 backward evidence protocol

Phase 1 now uses one corrected baseline protocol:

```text
valid annotation spans
+ static label names/definitions from the source/union dictionary
+ sanitized relationship links derived from interpretation-unit structure
+ supported sentence-level annotations only when valid span annotations exist
```

The backward packet must not include:

```text
unmatched_language
raw evidence_span_text from interpretation_units
combined_meaning
forward rationale
backward_mapping_decision
raw forward response
original source sentence
```

The backward packet may include exactly these top-level keys:

```text
backward_input_policy
ordered_reconstruction_items
relationship_links
sentence_level_annotations
```

Each `ordered_reconstruction_items` entry includes the annotation span plus static metadata such as `label_id`, `label_name`, `label_definition`, `source_model`, and `source_element_id`. These are dictionary/schema metadata, not forward-generated rationales.

Each `relationship_links` entry includes only:

```text
relationship_id
relationship_type
annotation_ids
```

It must not include free-text interpretation-unit evidence, combined meanings, or rationales.

## Universal backward prompt

The backward prompt is universal across baseline arms and future schema arms. The current prompt is:

```text
Task: reconstruct one concise natural-language consent sentence using only the annotation-only mapping below.

Instructions:
- Use only information explicitly present in the annotation-only mapping.
- Use label_name and label_definition to interpret annotation labels.
- Use relationship_links only as structural cues for how listed annotations relate to each other.
- Relationship links do not add source wording beyond the annotation spans and static label metadata.
- Preserve the order indicated by sentence_order_index when available.
- You may add minimal grammar/function words needed to make the reconstruction readable, but do not add unsupported content.
- If the annotation evidence is empty or insufficient, return an empty reconstructed_sentence and explain that annotation evidence was insufficient.

Relationship link types:
- same_span_multiple_labels: the listed annotations describe the same evidence span using multiple labels.
- same_span_multiple_fields: the listed annotations describe the same evidence span using multiple fields.
- nested_broad_narrow: the listed annotations describe overlapping or nested spans where one is broader and another is narrower.
- complementary_roles: the listed annotations describe different parts of one local meaning unit.
- complementary_fields: the listed annotations describe different fields that should be considered together.
- single: the source forward output marked this as a one-annotation unit.
- conflicting_or_uncertain: the relationship among the listed annotations is uncertain or potentially conflicting.

Annotation-only mapping:
{mapping_text}

Return JSON with exactly this structure:
{
  "reconstructed_sentence": "...",
  "reconstruction_notes": "brief note or empty string"
}
```

Rows with no backward-eligible annotations are not sent to the LLM. Their reconstruction is intentionally blank.

## Fresh output root

Do not write corrected outputs into the old experiment root. Start a fresh root:

```bash
export ROUNDTRIPS_CSV=path/to/roundtrips.csv
export MODEL_CONFIG=meta_model/configs/union_v0_models_template.yaml
export OLD_ROOT=meta_model/functional_v1_experiments
export STRICT_ROOT=meta_model/strict_annotation_only_experiments
mkdir -p "$STRICT_ROOT"
```

## Preserve old outputs instead of deleting

Archive old summaries/packages rather than deleting them:

```bash
export ARCHIVE_ROOT=meta_model/archive/leakage_contaminated_$(date +%Y%m%d)
mkdir -p "$ARCHIVE_ROOT"

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

cat > "$ARCHIVE_ROOT/README.md" <<'EOF'
# Archived exploratory outputs

These outputs were generated before the corrected structured annotation-only backward policy.
Preserve for provenance only. Do not use for final performance claims.
EOF
```

## Import existing forward outputs into the fresh root

### Union V0

```bash
for f in $(find "$OLD_ROOT" -path "*/union_v0_forward_mappings.jsonl" | sort); do
  MODEL_KEY=$(basename "$(dirname "$f")")
  mkdir -p "$STRICT_ROOT/union_v0/$MODEL_KEY"
  cp "$f" "$STRICT_ROOT/union_v0/$MODEL_KEY/"
done
```

### Individual source models

```bash
for f in $(find "$OLD_ROOT" -path "*/forward_mappings.jsonl" | sort); do
  INFO_MODEL=$(basename "$(dirname "$f")")
  MODEL_KEY=$(basename "$(dirname "$(dirname "$f")")")

  case "$INFO_MODEL" in
    DUO|ICO|ODRL|FHIR_Consent)
      mkdir -p "$STRICT_ROOT/individual/$MODEL_KEY/$INFO_MODEL"
      cp "$f" "$STRICT_ROOT/individual/$MODEL_KEY/$INFO_MODEL/"
      ;;
  esac
done
```

Check copied files:

```bash
find "$STRICT_ROOT" -name "*forward_mappings.jsonl" | sort -print -exec wc -l {} \;
```

Remove any stale backward outputs from the strict root:

```bash
find "$STRICT_ROOT" \( \
  -name "*backward*.jsonl" -o \
  -name "*roundtrip_outputs.csv" \
\) -delete
```

## Verify prompt identity before running Phase 1 baselines

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

scripts = [
    Path("meta_model/scripts/03_run_union_v0_roundtrip.py"),
    Path("meta_model/scripts/05_run_individual_model_roundtrip.py"),
]
texts = []
for p in scripts:
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    texts.append((str(p), mod.UNIVERSAL_BACKWARD_SYSTEM, mod.UNIVERSAL_BACKWARD_USER_TEMPLATE))

base = texts[0][1:]
for path, system, user in texts:
    assert (system, user) == base, f"Backward prompt mismatch: {path}"
    prompt_text = system + "\n" + user
    banned = ["unmatched_language", "interpretation_units", "combined_meaning", "backward_mapping_decision", "raw forward", "original source sentence"]
    hits = [x for x in banned if x in prompt_text.lower()]
    assert not hits, f"Banned prompt terms in {path}: {hits}"
print("PASSED: Phase 1 baseline backward prompt is identical and structured.")
PY
```

## Rerun backward only

### Union V0, local/OpenAI-compatible configs

```bash
for MODEL_KEY in $(find "$STRICT_ROOT/union_v0" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort); do
  rm -f "$STRICT_ROOT/union_v0/$MODEL_KEY/union_v0_backward_reconstructions.jsonl" \
        "$STRICT_ROOT/union_v0/$MODEL_KEY/union_v0_roundtrip_outputs.csv" \
        "$STRICT_ROOT/union_v0/$MODEL_KEY/failed_requests.jsonl"

  python meta_model/scripts/03_run_union_v0_roundtrip.py \
    --roundtrips_csv "$ROUNDTRIPS_CSV" \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$MODEL_KEY" \
    --output_dir "$STRICT_ROOT/union_v0" \
    --stage backward
done
```

### Union V0, Mayo Apigee/GPT-5.5

```bash
export MODEL_KEY=mayo_gpt55
rm -f "$STRICT_ROOT/union_v0/$MODEL_KEY/union_v0_backward_reconstructions.jsonl" \
      "$STRICT_ROOT/union_v0/$MODEL_KEY/union_v0_roundtrip_outputs.csv" \
      "$STRICT_ROOT/union_v0/$MODEL_KEY/failed_requests.jsonl"

python meta_model/scripts/12_run_union_v0_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$STRICT_ROOT/union_v0" \
  --stage backward
```

### Individual source models, local/OpenAI-compatible configs

```bash
for MODEL_KEY in $(find "$STRICT_ROOT/individual" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort); do
  find "$STRICT_ROOT/individual/$MODEL_KEY" \( -name "backward_reconstructions.jsonl" -o -name "roundtrip_outputs.csv" -o -name "failed_requests.jsonl" \) -delete

  python meta_model/scripts/05_run_individual_model_roundtrip.py \
    --roundtrips_csv "$ROUNDTRIPS_CSV" \
    --prompt_dir meta_model/prompts/individual_source_models \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --model_config_yaml "$MODEL_CONFIG" \
    --model_key "$MODEL_KEY" \
    --output_dir "$STRICT_ROOT/individual" \
    --info_models all \
    --stage backward
done
```

### Individual source models, Mayo Apigee/GPT-5.5

```bash
export MODEL_KEY=mayo_gpt55
find "$STRICT_ROOT/individual/$MODEL_KEY" \( -name "backward_reconstructions.jsonl" -o -name "roundtrip_outputs.csv" -o -name "failed_requests.jsonl" \) -delete

python meta_model/scripts/13_run_individual_model_roundtrip_apigee.py \
  --roundtrips_csv "$ROUNDTRIPS_CSV" \
  --prompt_dir meta_model/prompts/individual_source_models \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml "$MODEL_CONFIG" \
  --model_key "$MODEL_KEY" \
  --output_dir "$STRICT_ROOT/individual" \
  --info_models all \
  --stage backward
```

## Verify structured backward packets

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("meta_model/strict_annotation_only_experiments")
allowed_keys = {"backward_input_policy", "ordered_reconstruction_items", "relationship_links", "sentence_level_annotations"}
forbidden_anywhere = ["unmatched_language", "combined_meaning", "backward_mapping_decision", "raw_forward_response"]
bad = []
for p in root.rglob("*backward*.jsonl"):
    with p.open() as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            packet = obj.get("backward_packet") or obj.get("sanitized_forward_material") or {}
            extra = set(packet.keys()) - allowed_keys
            text = json.dumps(packet, ensure_ascii=False)
            hits = [x for x in forbidden_anywhere if x in text]
            if extra or hits:
                bad.append((str(p), i, sorted(extra), hits))
if bad:
    print("FAILED: invalid structured backward packet")
    for row in bad[:50]:
        print(row)
    raise SystemExit(1)
print("PASSED: structured backward packet top-level keys and forbidden text checks passed.")
PY
```

## Verify zero-annotation rows are blank

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("meta_model/strict_annotation_only_experiments")
bad = []
for p in root.rglob("*roundtrip_outputs.csv"):
    df = pd.read_csv(p).fillna("")
    if "annotation_count" not in df.columns or "reconstructed_sentence" not in df.columns:
        continue
    n = pd.to_numeric(df["annotation_count"], errors="coerce").fillna(0)
    sub = df[(n == 0) & (df["reconstructed_sentence"].astype(str).str.strip() != "")]
    if len(sub):
        bad.append((str(p), len(sub)))
if bad:
    print("FAILED: zero-annotation rows with reconstructions")
    for x in bad:
        print(x)
    raise SystemExit(1)
print("PASSED: zero-annotation rows have blank reconstructions.")
PY
```

## Standardize, score, diagnose, and compile

```bash
mkdir -p "$STRICT_ROOT/scoring_inputs"

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
