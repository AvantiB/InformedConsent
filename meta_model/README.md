# Refined informed-consent meta-model development

This folder contains the workflow for developing, crosswalking, and evaluating a reduced informed-consent meta-model.

## Current paper-facing framing

The current reduced model is **data-seeded and expert-refined**. Form-level cross-validated analyses identify recurring semantic neighborhoods and complementary provision structures, but the final reduced annotation labels are functional roles rather than raw cluster IDs.

```text
Derivation evidence:
  original expert-evaluated round-trip annotation workbooks

Data-driven seed analysis:
  form-level CV, source-element-in-context senses, typed relationship graphs,
  near-equivalence, co-occurrence/complementarity, and cross-fold recurrence

Reduced model:
  functional V1 candidate schema reviewed by PI/domain expert

Generalization test:
  held-out consent forms in form-level CV, plus optional external stress-test forms
```

## Key methodological principle

The goal is not to find the smallest number of clusters. The goal is to derive semantic fields that are:

```text
meaning-critical
specific enough to avoid unsafe role overlap
complementary rather than redundant
supported by source-model and round-trip evidence
translatable from ICO / DUO / FHIR Consent / ODRL
generalizable to unseen consent forms
```

The pipeline therefore distinguishes:

```text
near-equivalence      -> can support field consolidation
co-occurrence         -> provision-bundle/complementarity evidence, not direct merging
broader/narrower      -> hierarchy/scope evidence
related-but-distinct  -> unsafe or uncertain merge
```

## Active evidence path

```text
original researcher workbooks
→ clean expert round-trip corpus
→ form-level CV splits using stable form_key
→ training-fold mention evidence
→ decision-marker stripping and sentence_decision evidence
→ context-specific source-element senses
→ typed relationships and provision-bundle co-occurrence
→ strict selected semantic neighborhoods
→ reduced functional V1 candidate schema
→ source-model-to-V1 crosswalk
→ PI/domain-expert review
→ held-out forward/backward evaluation
→ multi-layer preservation evaluation
→ consensus/expert-reviewed V1.1
```

## Active scripts

```text
12_build_expert_roundtrip_corpus.py      # original workbooks -> clean expert corpus
23_refined_metamodel_cv_pipeline.py      # folds, mention evidence, sense nodes, typed graph, fold schemas
24_refined_cv_postprocess.py             # repair form aliases and select stable fold candidates
25_make_heldout_roundtrips.py            # create held-out roundtrip CSVs by fold
26_build_functional_v1_crosswalk.py      # map ICO/DUO/FHIR/ODRL source elements to functional V1 fields
27_run_functional_v1_roundtrip.py        # held-out forward/backward assessment using functional V1 schema
07_standardize_roundtrip_outputs.py      # standardize conditions for scoring
16_audit_annotation_granularity.py       # compare annotation burden/granularity
```

Older coarse cluster-ID schema builders and cluster-ID smoke-test runners are no longer part of the active workflow. They were useful for diagnosis, but the paper-facing model is the functional V1 schema.

## Core schema and methods documents

```text
meta_model/FUNCTIONAL_V1_METHODS.md
meta_model/FUNCTIONAL_V1_ROUNDTRIP_RUNBOOK.md
meta_model/REFINED_CV_POSTPROCESS_RUNBOOK.md
meta_model/schemas/reduced_functional_v1_candidate.yaml
```

## Baselines and comparisons

The functional V1 model should be compared against:

```text
individual DUO
individual ICO
individual ODRL
individual FHIR Consent
Union V0 full dictionary
Reduced Functional V1 / expert-reviewed V1.1
```

Evaluation should include the meaning-preservation classifier, but not rely on it alone. Additional metrics include content-word preservation, consent cue preservation, annotation coverage, annotation burden, same-span overlap, unmatched-language rate, and relationship/attachment errors.
