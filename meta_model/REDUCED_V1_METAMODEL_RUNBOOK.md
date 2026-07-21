# Refined informed-consent meta-model cross-validation runbook

This runbook is the paper-facing workflow for developing a compact but complementary informed-consent meta-model that generalizes to unseen consent forms. The primary generalization evidence is form-level held-out cross-validation, not full-corpus reuse.

## Design principles

```text
1. Split by consent form, not by sentence.
2. Prefer stable `form_key` from the main roundtrips.csv for fold creation.
3. Derive candidate fields only from training forms in each fold.
4. Use source-element-in-context mentions, not raw source elements, as the unit of analysis.
5. Treat annotation-format decisions such as (permit), (deny), (Permission), and (Prohibition) as decision metadata, not span text.
6. Strip decision markers from evidence spans before sense induction.
7. Aggregate annotation-level decisions separately to sentence/provision-level decision evidence.
8. Split broad source elements into context-specific sense nodes before clustering.
9. Distinguish near-equivalence from broader/narrower, related-distinct, and complementary relations.
10. Use only near-equivalence edges to form candidate fields.
11. Preserve co-occurrence/provision-bundle edges as complementarity evidence, not merge evidence.
12. Preserve provenance at every step: form, sentence, source model, LLM, raw span, cleaned span, annotation decision, source row, and fold.
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

This file is used for schema development. The folds are created from the main `roundtrips.csv` because its `form_key` is the stable consent-form identifier.

## 3. Create form-level cross-validation splits

```bash
python meta_model/scripts/23_refined_metamodel_cv_pipeline.py make-folds \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --split_source_csv /path/to/roundtrips.csv \
  --output_dir meta_model/refined_cv \
  --n_folds 4 \
  --seed 17
```

Outputs:

```text
meta_model/refined_cv/fold_assignments.csv
meta_model/refined_cv/form_grouping_audit.csv
meta_model/refined_cv/fold_metadata.json
meta_model/refined_cv/excluded_empty_form_rows.csv   # only when exclusions occur
```

Before running induction, check that metadata shows about 20-21 canonical consent forms, not hundreds of sentence-level copy/output files:

```bash
cat meta_model/refined_cv/fold_metadata.json
column -s, -t < meta_model/refined_cv/fold_assignments.csv | less -S
```

## 4. Run refined induction for each fold

Each fold uses only its training forms to derive candidate fields. Held-out forms are retained only for provenance and later evaluation.

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
fold_XX/evidence_mentions_unassigned_review.csv              # only when needed
fold_XX/annotation_decision_marker_cleaning_audit.csv
fold_XX/sentence_decision_evidence.csv
fold_XX/source_element_sense_mentions_train.csv
fold_XX/source_element_sense_nodes.csv
fold_XX/typed_relationship_edges.csv
fold_XX/provision_bundle_edges.csv
fold_XX/candidate_field_clusters.csv
fold_XX/refined_candidate_schema.yaml
fold_XX/refined_candidate_schema.json
fold_XX/fold_run_metadata.json
```

## 5. Inspect decision-marker cleaning before trusting clusters

The original researcher annotation format often attaches decisions to each annotation, for example:

```text
blood [NRES] (permit)
urine [NRES] (permit)
stored at the All of Us biobank [RTN] (permit)
```

For meta-model induction, `(permit)` is not part of the evidence span. The pipeline therefore strips such markers and stores them separately.

Check:

```bash
column -s, -t < meta_model/refined_cv/fold_00/annotation_decision_marker_cleaning_audit.csv | less -S
column -s, -t < meta_model/refined_cv/fold_00/sentence_decision_evidence.csv | less -S
```

The regenerated candidate schemas should not contain positive span examples such as `(permit) research`, `(deny) researchers`, or `(Permission) data`. Clean examples should look like `research`, `researchers`, `data`, `blood`, `urine`, or `stored at the All of Us biobank`.

## 6. Interpret typed relationship outputs

```text
typed_relationship_edges.csv
  near_equivalent          -> can merge into candidate field
  broader_narrower         -> hierarchy/scope evidence, not direct merge
  related_distinct         -> related but unsafe or uncertain to merge
  complementary            -> provision structure, not merge evidence

provision_bundle_edges.csv
  co-occurrence/composition evidence only
```

Candidate fields are generated from strict near-equivalence communities among context-specific source-element senses. Co-occurrence is never used as direct merge evidence.

## 7. Summarize field stability across folds

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

Use this to identify consensus-core candidates versus extension/audit candidates.

## 8. Held-out forward/backward evaluation

For each fold, evaluate on that fold's held-out forms using the fold-specific schema. The held-out input file should be filtered from the same sentence universe as the individual-model and Union V0 experiments, using the held-out `form_key`s from `fold_assignments.csv`.

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

## 9. Standardize and score held-out outputs

Use the existing standardization/scoring pipeline for classifier-based meaning-preservation evaluation. The classifier is one signal, not the only endpoint.

Compare held-out forms across:

```text
individual DUO / ICO / ODRL / FHIR Consent
Union V0 full dictionary
fold-specific refined candidate meta-model
```

## 10. Multi-layer preservation evaluation

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

## 11. Qualitative held-out audit

Audit a small held-out sample across folds and conditions. Focus on relationship preservation and missing semantic categories:

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

## 12. Build the consensus refined meta-model

After fold-stability and held-out evaluation:

```text
stable across folds + meaning-critical + low redundancy -> consensus core
less frequent but useful -> extension
unstable or ambiguous -> audit/split/merge decision
rare and not meaning-critical -> remove
```

The consensus schema should be built from cross-fold evidence, not from a single full-dataset derivation.

## 13. Final full-corpus characterization

After the consensus schema is frozen, run a final descriptive evaluation on all forms. Label this as full-corpus characterization, not primary generalization evidence. The primary generalization evidence is the form-level held-out CV.
