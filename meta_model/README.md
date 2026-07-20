# Data/Language-Driven Meta-Model Development

This folder contains the workflow for deriving, visualizing, smoke-testing, auditing, and validating a reduced informed-consent representation.

## Correct framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset. New LLM outputs are used for validation and stress testing, not for primary schema induction.

```text
Derivation / discovery corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Validation / stress-test corpus:
  MedGemma, Qwen235B, Llama4, GPT-5.5 outputs
```

Expert-preserved rows are treated as functionally validated positive evidence. Expert-failed rows are boundary evidence that weaken proposed merges and flag unsafe simplifications.

## Current V1 methodology

The reduced schema is not selected from a hand-written list of fields. The empirical step separates two graph views:

1. **Semantic-equivalence graph**: candidate field merges, based on same/overlapping evidence spans, cross-source-model support, cross-LLM support, expert-positive evidence, failure penalties, and profile similarity.
2. **Provision-bundle graph**: co-occurrence/compositional structure, used to understand how fields combine in consent provisions. Bundle edges are not merge evidence.

Script 17 writes empirical evidence and an audit template. Script 19 visualizes this evidence. Script 21 builds a **provisional empirical V1** directly from clusters so the data-driven model can be smoke-tested before PI naming. Script 20 builds the final V1 YAML only after a human fills the audit template with include/exclude decisions and final field names.

## Baseline, smoke-test, and validation conditions

1. **Individual source-model JSON prompts**: DUO, ICO, ODRL, and FHIR Consent run separately.
2. **Union V0 full dictionary**: unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL.
3. **Provisional empirical V1 smoke test**: data-driven semantic clusters run as cluster IDs on a few examples first, with a V0-like annotations/interpretion-units prompt.
4. **Provisional empirical V1 compact/permissive**: run more broadly only after smoke examples look reasonable.
5. **Audited Reduced V1 compact/permissive**: PI/expert-named and organized schema using the same round-trip protocol.

The smoke-test step is intentionally small. It checks whether sentence-level decisions, cluster annotations, interpretation units, and reconstructions behave sensibly before running the full dataset.

## Core evidence path

```text
original researcher workbooks
→ clean expert round-trip corpus
→ raw source-element mentions
→ semantic-equivalence graph
→ provision-bundle graph
→ visualization/audit report
→ provisional empirical V1 smoke tests
→ broader provisional V1 performance test, if smoke tests pass
→ PI semantic cluster review and naming
→ audited Reduced V1 YAML schema
→ compact/permissive validation on new LLM outputs
```

## Main scripts

```text
12_build_expert_roundtrip_corpus.py             # workbooks -> clean expert corpus
17_induce_reduced_v1_metamodel.py               # empirical semantic + bundle graphs; no final schema
19_visualize_v1_discovery.py                    # visual report for cluster support and co-occurrence
21_build_provisional_v1_schema_from_clusters.py # clusters -> provisional cluster-ID V1 schema
18_run_reduced_v1_roundtrip.py                  # V0-style cluster prompt for smoke/full V1 round trips
20_build_reduced_v1_schema_from_audit.py        # audited clusters -> final V1 YAML schema
07_standardize_roundtrip_outputs.py             # standardize conditions for scoring
16_audit_annotation_granularity.py              # compare annotation burden/granularity
```

See the full runbook:

```text
meta_model/REDUCED_V1_METAMODEL_RUNBOOK.md
```
