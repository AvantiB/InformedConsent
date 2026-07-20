# Data/Language-Driven Meta-Model Development

This folder contains the workflow for deriving, visualizing, smoke testing, refining, auditing, and validating a reduced informed-consent meta-model.

## Correct framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset. New LLM outputs are used for validation and stress testing, not for primary schema induction.

```text
Derivation / discovery corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Validation / stress-test corpus:
  MedGemma, Qwen235B, Llama4, GPT-5.5 outputs
```

Expert-preserved rows are treated as functionally validated positive evidence. Expert-failed rows are boundary evidence that weaken proposed merges and flag unsafe simplifications.

## Current finding

The first provisional cluster-ID smoke tests showed that the empirical clusters contain useful signal, but several clusters are too permissive/broad for a complementary reduced schema. A high-support cluster may still be unsafe if it combines participant, organization, repository, action, purpose, temporal, or constraint-like spans into one field.

The next exploration therefore focuses on **specificity control**, not simply fewer clusters.

## Current V1 methodology

The reduced schema is not selected from a hand-written list of fields. The empirical step separates two graph views:

1. **Semantic-equivalence graph**: candidate field merges based on same/overlapping evidence spans, cross-source-model support, cross-LLM support, expert-positive evidence, failure penalties, and profile similarity.
2. **Provision-bundle graph**: co-occurrence/compositional structure used to understand how fields combine in consent provisions. Bundle edges are not merge evidence.

Smoke tests showed that same-span evidence must be typed more carefully. Same-span or overlapping evidence can indicate near-equivalence, broader/narrower relations, complementary facets, or unsafe merges. Only near-equivalence should directly support field merging.

## Baseline, smoke-test, and validation conditions

1. **Individual source-model JSON prompts**: DUO, ICO, ODRL, and FHIR Consent run separately.
2. **Union V0 full dictionary**: unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL.
3. **Provisional empirical V1 smoke test**: data-driven semantic clusters run as cluster IDs on a few examples first, with a V0-like annotations/interpretion-units prompt.
4. **Specificity diagnostics**: cluster-level and smoke-output diagnostics to identify over-broad, hub-like, or overlapping clusters.
5. **Revised provisional V1**: later second-pass clusters after specificity control.
6. **Audited Reduced V1**: PI/expert-named and organized schema using the same round-trip protocol.

## Core evidence path

```text
original researcher workbooks
→ clean expert round-trip corpus
→ raw source-element mentions
→ initial semantic-equivalence graph
→ provision-bundle graph
→ provisional empirical V1 smoke tests
→ cluster specificity diagnostics
→ specificity-controlled second induction pass
→ PI semantic cluster review and naming
→ audited Reduced V1 YAML schema
→ compact/permissive validation on new LLM outputs
```

## Main scripts

```text
12_build_expert_roundtrip_corpus.py             # workbooks -> clean expert corpus
17_induce_reduced_v1_metamodel.py               # initial empirical semantic + bundle graphs
19_visualize_v1_discovery.py                    # visual report for cluster support and co-occurrence
21_build_provisional_v1_schema_from_clusters.py # clusters -> provisional cluster-ID schema
18_run_reduced_v1_roundtrip.py                  # V0-style cluster prompt for smoke/full V1 round trips
22_diagnose_v1_cluster_specificity.py           # over-broad cluster and smoke-overlap diagnostics
20_build_reduced_v1_schema_from_audit.py        # audited clusters -> final V1 YAML schema
07_standardize_roundtrip_outputs.py             # standardize conditions for scoring
16_audit_annotation_granularity.py              # compare annotation burden/granularity
```

See the full runbook:

```text
meta_model/REDUCED_V1_METAMODEL_RUNBOOK.md
```