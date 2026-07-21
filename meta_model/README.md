# Refined informed-consent meta-model development

This folder contains the workflow for developing, crosswalking, and evaluating reduced informed-consent meta-models.

## Current paper-facing framing

The current reduced-model experiments compare two data-seeded functional schema derivation strategies:

```text
Manual Functional V1:
  form-level CV evidence -> manually organized functional schema -> expert review

LLM-Induced Functional V1:
  same form-level CV evidence -> evidence cards -> fixed strong LLM induction/critique/revision -> expert review
```

Both reduced schemas are compared with individual source models and Union V0 under controlled round-trip evaluation. For all round-trip experiments, the prompt template should remain constant within the experiment family; only the schema/dictionary changes.

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
-> clean expert round-trip corpus
-> form-level CV splits using stable form_key
-> training-fold mention evidence
-> decision-marker stripping and sentence_decision evidence
-> context-specific source-element senses
-> typed relationships and provision-bundle co-occurrence
-> strict selected semantic neighborhoods
-> Manual Functional V1 candidate schema
-> source-model-to-V1 crosswalk
-> LLM schema-induction evidence cards
-> LLM-induced Functional V1 candidate schema
-> held-out forward/backward evaluation for both reduced schemas
-> meaning-preservation classifier scoring and multi-layer preservation metrics
-> PI/domain-expert review and consensus V1.1
```

## Active scripts

```text
12_build_expert_roundtrip_corpus.py      # original workbooks -> clean expert corpus
23_refined_metamodel_cv_pipeline.py      # folds, mention evidence, sense nodes, typed graph, fold schemas
24_refined_cv_postprocess.py             # repair form aliases and select stable fold candidates
25_make_heldout_roundtrips.py            # create held-out roundtrip CSVs by fold
26_build_functional_v1_crosswalk.py      # map ICO/DUO/FHIR/ODRL source elements to functional V1 fields
27_run_functional_v1_roundtrip.py        # held-out forward/backward assessment using any functional V1 schema
28_build_llm_schema_induction_cards.py   # selected fold evidence -> evidence cards for LLM induction
29_induce_functional_schema_with_llm.py  # induce, critique, revise, and validate LLM-induced schema
30_relabel_functional_v1_outputs.py      # relabel output metadata without changing prompts
31_compile_schema_condition_comparison.py # summarize scored schema-condition results
07_standardize_roundtrip_outputs.py      # standardize conditions for classifier scoring
16_audit_annotation_granularity.py       # compare annotation burden/granularity
```

Older coarse cluster-ID schema builders and cluster-ID smoke-test runners are no longer part of the active workflow. They were useful for diagnosis, but the paper-facing models are functional V1 schemas.

## Core schema and methods documents

```text
meta_model/FUNCTIONAL_V1_METHODS.md
meta_model/FUNCTIONAL_V1_ROUNDTRIP_RUNBOOK.md
meta_model/MANUAL_VS_LLM_INDUCED_V1_EXPERIMENT_RUNBOOK.md
meta_model/REFINED_CV_POSTPROCESS_RUNBOOK.md
meta_model/schemas/reduced_functional_v1_candidate.yaml
```

## Baselines and comparisons

The final comparison should include:

```text
individual DUO
individual ICO
individual ODRL
individual FHIR Consent
Union V0 full dictionary
Manual Functional V1
LLM-Induced Functional V1
expert-reviewed consensus V1.1, if finalized
```

Evaluation should include the meaning-preservation classifier, but not rely on it alone. Additional metrics include content-word preservation, consent cue preservation, annotation coverage, annotation burden, same-span overlap, unmatched-language rate, and relationship/attachment errors.
