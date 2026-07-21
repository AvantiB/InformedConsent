# Diagnostic Evaluation Runbook

This runbook adds interpretable quantitative and qualitative diagnostics after the meaning-preservation classifier has scored standardized round-trip outputs.

The diagnostic layer is meant to complement, not replace, the classifier score.

## Inputs

```text
$OUT_ROOT/scored_roundtrips/scored_roundtrips.csv
meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib
```

The classifier bundle is optional but preferred because it contains the cue dictionary used by the selected meaning-preservation classifier.

## Run diagnostics

```bash
python meta_model/scripts/32_compute_roundtrip_diagnostic_metrics.py \
  --roundtrips_csv "$OUT_ROOT/scored_roundtrips/scored_roundtrips.csv" \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir "$OUT_ROOT/diagnostics" \
  --review_sample_per_condition 25
```

## Recompile paper-facing comparison with diagnostic metrics

Use the enriched row-level diagnostic CSV as the input to the comparison compiler:

```bash
python meta_model/scripts/31_compile_schema_condition_comparison.py \
  --scored_csv "$OUT_ROOT/diagnostics/roundtrip_diagnostic_metrics.csv" \
  --output_dir "$OUT_ROOT/comparison"
```

## Main outputs

```text
$OUT_ROOT/diagnostics/roundtrip_diagnostic_metrics.csv
$OUT_ROOT/diagnostics/condition_diagnostic_summary.csv
$OUT_ROOT/diagnostics/condition_llm_diagnostic_summary.csv
$OUT_ROOT/diagnostics/condition_information_model_diagnostic_summary.csv
$OUT_ROOT/diagnostics/cue_group_retention_summary_by_condition.csv
$OUT_ROOT/diagnostics/cue_group_retention_summary_by_condition_llm.csv
$OUT_ROOT/diagnostics/qualitative_relationship_error_review_sample.csv
$OUT_ROOT/diagnostics/evaluation_dictionary_used.json
$OUT_ROOT/comparison/schema_condition_summary.csv
$OUT_ROOT/comparison/schema_condition_overall.csv
$OUT_ROOT/comparison/schema_condition_by_llm.csv
$OUT_ROOT/comparison/schema_condition_by_information_model.csv
```

## Metrics added

### Lexical/content preservation

```text
content_word_recall
content_word_precision
content_word_f1
content_word_jaccard
missing_content_word_count
added_content_word_count
dropped_content_word_rate
added_content_word_rate
missing_content_words
added_content_words
```

These are calculated over unique non-stopword content terms from the original and reconstructed sentence.

### Cue and important-category preservation

The script loads the classifier cue dictionary from the final classifier bundle when available. It reports per-category cue retention for the same kinds of consent cues used by the classifier, including:

```text
permission
obligation
prohibition
negation
condition
exception
restriction
withdrawal
action
resource
actor
purpose
```

It also adds evaluation-only groups for paper-facing diagnostics:

```text
temporal
privacy_identifiability
results_feedback
storage_lifecycle
contact_recontact
```

For each cue group, it reports original count, reconstruction count, exact cue recall, jaccard overlap, missing count, added count, category retained, missing cue strings, and added cue strings.

### Modal preservation

The modal layer reports:

```text
modal_orig
modal_recon
modal_category_changed
modal_word_recall
modal_word_jaccard
modal_word_change_ratio
modal_missing_count
modal_added_count
modal_missing_cues
modal_added_cues
```

The modal priority is:

```text
prohibition > obligation > permission > none
```

`modal_word_change_ratio` is `1 - modal_word_jaccard`, so higher values indicate larger modal cue change.

### Unmatched-language rate

When forward mappings contain `unmatched_language`, `unmatched_spans`, `unmapped_language`, or similar fields, the script reports:

```text
unmatched_language_available
unmatched_language_count
unmatched_language_token_count
unmatched_language_rate
unmatched_language_text
```

This is most useful for Functional V1 outputs where the prompt explicitly requests unmatched-language accounting.

### Qualitative relationship-error review sheet

The script creates a heuristic review sheet:

```text
qualitative_relationship_error_review_sample.csv
```

It includes rows likely to contain relationship or semantic shifts, such as:

```text
modal_or_permission_category_changed
prohibition_cue_dropped
negation_cue_dropped
condition_scope_or_exception_cue_dropped
withdrawal_choice_cue_dropped
governed_action_changed
governed_resource_changed
actor_or_recipient_changed
purpose_or_use_context_changed
temporal_expression_changed
privacy_identifiability_changed
substantial_content_loss
substantial_added_content
low_classifier_score
```

These flags are not adjudicated truth labels. They are a structured triage mechanism for qualitative manual review and examples in the paper.

## Paper-facing interpretation

Use the classifier score as the primary automated proxy for meaning preservation, and use the diagnostic tables to explain why models/schemas differ:

```text
Meaning-preservation classifier score:
  overall proxy outcome.

Content-word recall / dropped content rate:
  how much original semantic content was retained.

Cue and modal preservation:
  whether permission, prohibition, obligation, negation, withdrawal, condition, restriction, action, resource, actor, and purpose cues survived reconstruction.

Annotation count / unique field count:
  annotation burden and schema compactness.

Parse success:
  output reliability.

Unmatched-language rate:
  how much meaning-critical text the schema/prompt left uncovered when available.

Qualitative review sample:
  relationship-level error analysis and paper examples.
```
