# Refined informed-consent meta-model development

This folder contains the workflow for developing, crosswalking, and evaluating reduced informed-consent meta-models.

## Current paper-facing framing

The active evaluation has been reset around a strict annotation-only backward reconstruction rule. Earlier round-trip outputs that exposed `interpretation_units`, `unmatched_language`, residual text, or free-text combined meanings to the backward LLM are treated as exploratory/leakage-contaminated and should not be used for final performance claims.

The next paper-facing comparison is organized as:

```text
Phase 1 baselines:
  existing individual DUO / ICO / ODRL / FHIR forward mappings -> strict annotation-only backward reconstruction
  existing Union V0 forward mappings -> strict annotation-only backward reconstruction

Phase 2 schema induction:
  strict-preserved valid annotation evidence -> source-element senses -> typed pairwise relationships -> evidence cards
  direct LLM schema induction from source dictionaries + examples
  data-driven LLM schema induction from empirical evidence cards

Phase 3 evaluation and expert review:
  constant universal strict backward prompt across all schema strategies
  expert assessment of schema dictionaries, source-model crosswalks, and examples
```

## Non-negotiable strict backward rule

For every experiment family, the backward prompt must be universal and must receive only backward-eligible annotation evidence:

```text
Allowed:
  valid span-level annotations
  annotation labels / field IDs
  canonical modifiers attached to valid annotations, when present
  sentence-level annotations only when at least one valid span annotation exists

Excluded:
  original sentence
  raw forward response
  unmatched_language / residual text
  interpretation_units
  combined_meaning
  rationales
  previous reconstruction
  unanchored sentence_decision
```

Rows with no backward-eligible span annotations are not sent to the LLM. Their reconstruction is intentionally blank. This prevents residual-only mappings from receiving inflated meaning-preservation scores.

## Key methodological principle

The goal is not to find the smallest number of clusters. The goal is to derive semantic fields that are:

```text
meaning-critical
specific enough to avoid unsafe role overlap
complementary rather than redundant
supported by source-model and strict round-trip evidence
translatable from ICO / DUO / FHIR Consent / ODRL
generalizable to unseen consent forms
```

The data-driven pipeline therefore distinguishes:

```text
near-equivalence      -> can support seed clustering
co-occurrence         -> provision-bundle/complementarity evidence, not direct merging
proximity             -> local relationship evidence, not direct merging
broader/narrower      -> hierarchy/scope evidence
related-but-distinct  -> unsafe or uncertain merge
modifier attributes   -> characterize annotations/frames, not source-model nodes to cluster
```

## Active evidence path

```text
existing individual-model and Union V0 forward mappings
-> strict annotation-only backward rerun for baselines
-> classifier scoring and diagnostic metrics for baselines
-> strict-preserved valid annotation evidence table
-> source-element-sense induction
-> pairwise evidence features: same-span, overlap, nesting, proximity, PMI/lift, semantic similarity, role signatures
-> typed pairwise relationships
-> near-equivalence seed clusters only
-> complementary/proximity graph for functional bundles
-> LLM schema-induction evidence cards
-> direct LLM high/low granularity reduced schemas
-> data-driven LLM high/low granularity reduced schemas
-> strict forward/backward evaluation under a constant universal backward prompt
-> PI/domain-expert review and consensus schema
```

## Active scripts

```text
03_run_union_v0_roundtrip.py              # Union V0; backward is strict annotation-only
05_run_individual_model_roundtrip.py      # individual source models; backward is strict annotation-only
12_run_union_v0_roundtrip_apigee.py       # Apigee wrapper; inherits strict Union V0 backward policy
13_run_individual_model_roundtrip_apigee.py # Apigee wrapper; inherits strict individual backward policy
27_run_functional_v1_roundtrip.py         # functional schemas; backward is strict annotation-only
09_score_roundtrip_outputs.py             # classifier scoring
32_compute_roundtrip_diagnostic_metrics.py # diagnostic metrics
31_compile_schema_condition_comparison.py # summarize scored schema-condition results
```

Older coarse cluster-ID schema builders, residual-enabled backward packets, and leakage-contaminated package outputs are no longer part of the paper-facing workflow. Preserve them only in an archive folder for provenance.

## Core current documents

```text
meta_model/STRICT_BASELINE_PHASE1_RUNBOOK.md
meta_model/PIPELINE_STAGE_TRACKER.md
meta_model/PIPELINE_STAGE_TRACKER.csv
meta_model/FUNCTIONAL_V1_METHODS.md
meta_model/DIAGNOSTIC_EVALUATION_RUNBOOK.md
```

## Baselines and comparisons

The corrected final comparison should include:

```text
individual DUO strict baseline
individual ICO strict baseline
individual ODRL strict baseline
individual FHIR Consent strict baseline
Union V0 strict baseline
Direct LLM induced schema, high granularity
Direct LLM induced schema, low granularity
Data-driven LLM induced schema, high granularity
Data-driven LLM induced schema, low granularity
expert-reviewed consensus schema, if finalized
```

Evaluation should include the meaning-preservation classifier, but not rely on it alone. Additional metrics include content-word preservation, consent cue preservation, annotation coverage, annotation burden, strict backward-eligible annotation counts, full-sentence span drops, and relationship/attachment errors.