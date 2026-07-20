# InformedConsent

Code for the informed-consent data-use and sharing project.

## Repository structure

```text
meaning_preservation/   # Expert-labeled round-trip classifier experiments and proxy scoring
meta_model/             # Individual models, Union V0, and refined cross-validated meta-model development
```

## Current workflow

The project has two linked components.

1. **Meaning-preservation classifier**: develops a proxy classifier for whether an LLM backward reconstruction preserves the meaning of the original consent sentence.
2. **Refined meta-model development**: derives a compact, complementary consent-language schema using form-level cross-validation, source-element-in-context evidence, typed relationship graphs, and held-out forward/backward evaluation.

## Key principle

```text
Derivation / induction corpus:
  original expert-evaluated round-trip dataset

Generalization test:
  held-out consent forms in form-level cross-validation

Validation / stress-test corpus:
  MedGemma, Qwen235B, Llama4, GPT-5.5 generated outputs
```

Expert-preserved rows are treated as functionally validated positive evidence. Expert-failed rows are boundary evidence. Classifier scores are proxy validation outcomes and are reported alongside lexical, cue, annotation-coverage, annotation-burden, and qualitative relationship-preservation analyses.

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
A. Build the clean expert round-trip corpus from original workbooks.
B. Create form-level CV splits to avoid training/testing leakage.
C. For each training fold, derive source-element-in-context senses and typed relationship evidence.
D. Merge only strict near-equivalence relationships into candidate fields; keep co-occurrence as complementarity evidence.
E. Evaluate fold-specific schemas on held-out consent forms.
F. Compare individual models, Union V0, and refined candidate schemas using classifier and non-classifier preservation metrics.
G. Use fold stability and qualitative error analysis to construct the consensus refined meta-model.
```
