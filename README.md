# InformedConsent

Code for the informed-consent data-use and sharing project.

## Repository structure

```text
meaning_preservation/   # Human-labeled round-trip classifier experiments and final proxy scorer
meta_model/             # Union V0, individual-model round trips, scoring, and reduced meta-model development
```

## Current workflow

The project has two linked components.

1. **Meaning-preservation classifier**: trains and evaluates a proxy classifier for whether an LLM backward reconstruction preserves the meaning of the original consent sentence.
2. **Meta-model development**: compares original individual information-model prompts against an unreduced Union V0 baseline, scores new LLM round trips with the classifier, and then uses those results to induce a reduced consent/data-use meta-model.

## Key runbooks

```text
meaning_preservation/README.md
meta_model/README.md
meta_model/UNION_V0_ROUNDTRIP_RUNBOOK.md
meta_model/INDIVIDUAL_MODEL_REPLICATION_RUNBOOK.md
meta_model/ROUNDTRIP_SCORING_RUNBOOK.md
```

## Current stage

MedGemma and Qwen are being run for Union V0 and individual information-model conditions. The next stage is:

```text
A. Validate and standardize all round-trip outputs.
B. Train the final scoring classifier on all original human-labeled rows.
C. Score MedGemma/Qwen Union V0 and individual-model outputs.
D. Compare individual-model baselines against naive Union V0.
E. Repeat for Llama/GPT, then proceed to reduced meta-model induction.
```

Classifier scores on new LLM outputs are proxy preservation estimates, not human gold labels. The split-based classifier experiments remain the classifier validation evidence.
