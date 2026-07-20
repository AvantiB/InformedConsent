# Data/Language-Driven Meta-Model Development

This folder contains the workflow for deriving, visualizing, auditing, and validating a reduced, functional informed-consent meta-model.

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

Script 17 writes empirical evidence and an audit template. Script 19 visualizes the discovery evidence. Script 20 builds the final V1 YAML only after a human fills the audit template with include/exclude decisions and final field names. Human involvement is limited to naming, unsafe-merge review, and split/merge audit.

## Baseline and validation conditions

1. **Individual source-model JSON prompts**: DUO, ICO, ODRL, and FHIR Consent run separately.
2. **Union V0 full dictionary**: unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL.
3. **Reduced V1 compact**: audited expert-induced schema with short evidence phrases.
4. **Reduced V1 permissive**: same audited schema with longer evidence phrases allowed when needed.

The compact/permissive split tests whether V1 works because of the reduced functional schema itself or because longer evidence spans carry forward source wording.

## Core evidence path

```text
original researcher workbooks
→ clean expert round-trip corpus
→ raw source-element mentions
→ semantic-equivalence graph
→ provision-bundle graph
→ visual audit report
→ semantic cluster audit template
→ audited Reduced V1 YAML schema
→ compact/permissive validation on new LLM outputs
```

## Main scripts

```text
12_build_expert_roundtrip_corpus.py       # workbooks -> clean expert corpus
17_induce_reduced_v1_metamodel.py         # empirical semantic + bundle graphs; no final schema
19_visualize_v1_discovery.py              # evidence figures/tables for audit and manuscript defense
20_build_reduced_v1_schema_from_audit.py  # audited clusters -> V1 YAML schema
18_run_reduced_v1_roundtrip.py            # compact/permissive V1 validation
07_standardize_roundtrip_outputs.py       # standardize conditions for scoring
16_audit_annotation_granularity.py        # compare annotation burden/granularity
```

See the full runbook:

```text
meta_model/REDUCED_V1_METAMODEL_RUNBOOK.md
```
