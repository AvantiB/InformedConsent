# Refined informed-consent meta-model development

This folder contains the workflow for developing, validating, and auditing a reduced informed-consent meta-model that is compact, complementary, and evaluated on held-out consent forms.

## Current framing

The reduced meta-model is derived from original expert-evaluated round-trip annotations, but final claims should not rely on a schema developed and tested on the same consent forms. The current workflow therefore uses form-level cross-validation.

```text
Derivation / discovery corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Generalization test:
  held-out consent forms within 4-fold form-level cross-validation

Validation / stress-test material:
  MedGemma, Qwen235B, Llama4, GPT-5.5 round-trip outputs
```

## Methodological position

The goal is not to find the smallest number of clusters. The goal is to derive semantic fields that are:

```text
meaning-critical
specific enough to avoid unsafe role overlap
complementary rather than redundant
supported by source-model and round-trip evidence
generalizable to unseen consent forms
```

The pipeline therefore does **not** cluster raw source elements directly. It first builds source-element-in-context mentions, splits broad source elements into context-specific sense nodes, and then constructs typed relationship evidence.

Only strict near-equivalence edges can merge candidate fields. Co-occurrence/provision-bundle edges are retained as complementarity evidence and are not used directly for merging.

## Core evidence path

```text
original researcher workbooks
→ clean expert round-trip corpus
→ form-level CV splits
→ training-fold mention evidence
→ context-specific source-element senses
→ typed relationships
   - near-equivalent
   - broader/narrower
   - complementary
   - related distinct / unsafe-to-merge
→ fold-specific candidate schemas
→ field-stability analysis across folds
→ held-out forward/backward evaluation
→ multi-layer preservation evaluation
→ qualitative held-out error audit
→ consensus refined meta-model
→ final full-corpus characterization
```

## Baselines and comparisons

The refined meta-model should be compared against:

```text
individual DUO
individual ICO
individual ODRL
individual FHIR Consent
Union V0 full dictionary
fold-specific refined candidate schema
```

Evaluation should include the meaning-preservation classifier, but not rely on it alone. Additional metrics include content-word preservation, consent cue preservation, annotation coverage, annotation burden, same-span overlap, unmatched-language rate, and relationship/attachment errors.

## Main scripts

```text
12_build_expert_roundtrip_corpus.py             # workbooks -> clean expert corpus
23_refined_metamodel_cv_pipeline.py             # folds, evidence, senses, typed graph, fold schemas, stability, multilayer metrics
18_run_reduced_v1_roundtrip.py                  # forward/backward evaluation with a compact schema dictionary
07_standardize_roundtrip_outputs.py             # standardize conditions for scoring
16_audit_annotation_granularity.py              # compare annotation burden/granularity
20_build_reduced_v1_schema_from_audit.py        # audited consensus schema -> final YAML/JSON
```

Older coarse cluster-ID scripts were removed from the active workflow because the paper-facing method now uses specificity-controlled, form-level cross-validated schema induction.

See the full runbook:

```text
meta_model/REDUCED_V1_METAMODEL_RUNBOOK.md
```
