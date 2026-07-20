# Refined informed-consent meta-model cross-validation runbook

This runbook is the paper-facing workflow from this point forward. The objective is to develop a compact but complementary informed-consent meta-model that generalizes to unseen consent forms and is evaluated with multiple preservation signals, not only the meaning-preservation classifier.

## Design principles

```text
1. Split by consent form, not by sentence.
2. Derive candidate fields only from training forms in each fold.
3. Use source-element-in-context mentions, not raw source elements, as the unit of analysis.
4. Split broad source elements into context-specific sense nodes before clustering.
5. Distinguish near-equivalence from broader/narrower, complementary, and unsafe-to-merge relations.
6. Use only near-equivalence edges to form candidate fields.
7. Preserve co-occurrence/provision-bundle edges as complementarity evidence, not merge evidence.
8. Preserve provenance at every step: form, sentence, source model, LLM, annotation span, source row, and fold.
```

The coarse cluster-ID smoke-test path is no longer the main development path and should not be treated as the paper result.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/12_build_expert_roundtrip_corpus.py
python -m py_compile meta_model/scripts/23_refined_metamodel_cv_pipeline.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
python -m py_compile meta_model/scripts/07_standardize_roundtrip_outputs.py
python -m py_compile meta_model/scripts/20_build_reduced_v1_schema_from_audit.py
```

## 2. Build the clean expert corpus

Use the original researcher/evaluator workbooks as the derivation corpus.

```bash
python meta_model/scripts/12_build_expert_roundtrip_corpus.py \
  --workbook_dir /path/to/original_annotation_workbooks \
  --output_csv meta_model/outputs/expert_roundtrips_clean.csv
```

This file is used for schema development only. It is not the same as the held-out evaluation round-trip input.

## 3. Create form-level cross-validation splits

Use consent forms as the split unit to avoid sentence-level leakage.

```bash
python meta_model/scripts/23_refined_metamodel_cv_pipeline.py make-folds \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --output_dir meta_model/refined_cv \
  --n_folds 4 \
  --seed 17
```

Output:

```text
meta_model/refined_cv/fold_assignments.csv
meta_model/refined_cv/fold_metadata.json
```

Recommended design: 4 folds, each with 15 derivation forms and 5 held-out test forms.

## 4. Run refined induction for each fold

Each fold uses only its training forms to derive the candidate schema. Held-out forms are retained only for provenance and later evaluation.

```bash
for FOLD in 0 1 2 3; do
  python meta_model/scripts/23_refined_metamodel_cv_pipeline.py run-fold \
    --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
    --fold_assignments_csv meta_model/refined_cv/fold_assignments.csv \
    --inventory_csv meta_model/v0_union/source_element_inventory.csv \
    --output_dir meta_model/refined_cv \
    --fold_id "$FOLD" \
    --min_sense_support 2 \
    --span_overlap_threshold 0.75 \
    --min_equivalence_weight 0.02 \
    --min_equivalence_positive_contexts 1 \
    --min_field_positive_mentions 5
done
```

Key outputs per fold:

```text
fold_XX/evidence_mentions_all.csv
fold_XX/evidence_mentions_train.csv
fold_XX/evidence_mentions_test_provenance_only.csv
fold_XX/source_element_sense_mentions_train.csv
fold_XX/source_element_sense_nodes.csv
fold_XX/typed_relationship_edges.csv
fold_XX/provision_bundle_edges.csv
fold_XX/candidate_field_clusters.csv
fold_XX/refined_candidate_schema.yaml
fold_XX/refined_candidate_schema.json
fold_XX/fold_run_metadata.json
```

## 5. Interpret the fold outputs

The key distinction is:

```text
typed_relationship_edges.csv
  near_equivalent          -> can merge into candidate field
  broader_narrower         -> hierarchy/scope evidence, not direct merge
  related_distinct         -> related but not safe to merge
  complementary            -> provision structure, not merge evidence

provision_bundle_edges.csv
  co-occurrence/composition evidence only
```

Candidate fields are generated from strict near-equivalence communities among context-specific source-element senses. Co-occurrence is never used as direct merge evidence.

## 6. Summarize field stability across folds

```bash
python meta_model/scripts/23_refined_metamodel_cv_pipeline.py summarize-folds \
  --fold_root meta_model/refined_cv \
  --output_dir meta_model/refined_cv/stability
```

Outputs:

```text
stability/fold_candidate_fields_long.csv
stability/field_recurrence_across_folds.csv
```

Use this to determine which candidate functions are stable across training folds. Stable fields become consensus-core candidates; unstable fields become extension or audit candidates.

## 7. Held-out forward/backward evaluation

For each fold, evaluate on that fold's held-out forms using the fold-specific schema. The exact held-out input file should be created from the same sentence universe as previous individual-model and Union V0 experiments, filtered to the held-out form IDs in `fold_assignments.csv`.

Run one hosted vLLM model at a time.

```bash
export MODEL_KEY=medgemma  # or qwen235b, matching the currently hosted server
export FOLD=0

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv meta_model/refined_cv/fold_${FOLD}/heldout_roundtrips.csv \
  --metamodel_yaml meta_model/refined_cv/fold_${FOLD}/refined_candidate_schema.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key "$MODEL_KEY" \
  --output_dir meta_model/refined_cv/fold_${FOLD}/heldout_roundtrip_outputs \
  --evidence_mode compact \
  --stage both
```

Repeat for each fold/model after hosting the corresponding vLLM server.

## 8. Standardize and score held-out outputs

Use the existing standardization/scoring pipeline for classifier-based meaning-preservation evaluation. The classifier result is one evaluation signal, not the only endpoint.

Compare held-out forms across:

```text
individual DUO / ICO / ODRL / FHIR Consent
Union V0 full dictionary
fold-specific refined candidate meta-model
```

## 9. Multi-layer preservation evaluation

After standardization, compute lexical/cue/coverage-complexity metrics.

```bash
python meta_model/scripts/23_refined_metamodel_cv_pipeline.py evaluate-roundtrips \
  --standardized_roundtrips_csv meta_model/refined_cv/heldout_standardized/standardized_roundtrips.csv \
  --output_dir meta_model/refined_cv/heldout_multilayer_eval
```

Outputs:

```text
lexical_cue_preservation_long.csv
coverage_complexity_summary.csv
```

Report these alongside the classifier:

```text
meaning-preservation classifier score
content-word recall / precision
cue preservation
annotation count
unique field count
same-span overlap rate
unmatched-language rate
qualitative relationship errors
```

## 10. Qualitative held-out audit

Audit a small held-out sample across folds and conditions. Focus on relationship preservation and missing semantic categories.

Suggested error taxonomy:

```text
actor inversion
resource omitted
action changed
decision polarity changed
decision cue lost
temporal phrase attached to wrong target
purpose attached to wrong action
condition/exception lost
restriction weakened
withdrawal meaning lost
study lifecycle confused with data lifecycle
privacy/identifiability omitted
hallucinated content
```

Use this audit to decide whether fields should be split, merged, moved to extension status, or clarified before PI review.

## 11. Build the consensus refined meta-model

After fold-stability and held-out evaluation:

```text
stable across folds + meaning-critical + low redundancy -> consensus core
less frequent but useful -> extension
unstable or ambiguous -> audit/split/merge decision
rare and not meaning-critical -> remove
```

The consensus schema should be built from cross-fold evidence, not from a single full-dataset derivation.

## 12. Final full-corpus characterization

After the consensus schema is frozen, run a final descriptive evaluation on all 20 forms. Label this as full-corpus characterization, not primary generalization evidence. The primary generalization evidence is the form-level held-out CV.

## Paper claims supported by this workflow

```text
1. Individual information models cover different parts of informed-consent meaning but are incomplete alone.
2. Union V0 improves coverage but is bulky and redundant.
3. A defensible reduced meta-model must separate equivalence from complementarity.
4. Context-specific source-element senses produce more specific, complementary candidate fields.
5. Fold-level schema stability and held-out evaluation address overfitting concerns.
6. Multi-layer evaluation shows preservation of content, cues, relationships, and annotation efficiency.
```
