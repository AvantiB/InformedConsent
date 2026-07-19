# InformedConsent

Code for the informed-consent data-use and sharing project.

## Repository structure

```text
meaning_preservation/   # Expert-labeled round-trip classifier experiments and final proxy scorer
meta_model/             # Union V0, individual-model round trips, expert-induced Reduced V1, and validation
```

## Current workflow

The project has two linked components.

1. **Meaning-preservation classifier**: trains and evaluates a proxy classifier for whether an LLM backward reconstruction preserves the meaning of the original consent sentence.
2. **Meta-model development**: derives a reduced consent/data-use meta-model from the original expert-evaluated round-trip dataset, then validates it against individual models and the unreduced Union V0 baseline using new LLM outputs.

## Key principle

```text
Derivation / induction corpus:
  original expert-evaluated round-trip dataset

Validation / stress-test corpus:
  MedGemma, Qwen235B, Llama4, GPT-5.5 generated outputs
```

Expert-preserved rows are treated as functionally validated positive evidence. Expert-failed rows are boundary evidence. Classifier scores on new LLM outputs are proxy validation outcomes, not human gold labels.

## Key runbooks

```text
meaning_preservation/README.md
meta_model/README.md
meta_model/UNION_V0_ROUNDTRIP_RUNBOOK.md
meta_model/INDIVIDUAL_MODEL_REPLICATION_RUNBOOK.md
meta_model/ROUNDTRIP_SCORING_RUNBOOK.md
meta_model/REDUCED_V1_METAMODEL_RUNBOOK.md
```

## Current stage

```text
A. Use the original expert-labeled rows to train the final scoring classifier.
B. Use the original expert-labeled rows to induce Reduced V1 via source-element graph evidence.
C. Run Reduced V1 compact/permissive round trips on validation LLMs.
D. Compare individual models vs Union V0 vs Reduced V1 compact vs Reduced V1 permissive.
E. Report meaning preservation, cue/content preservation, parse stability, and annotation granularity.
```
