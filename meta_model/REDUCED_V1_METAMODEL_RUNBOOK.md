# Reduced V1 meta-model discovery, provisional evaluation, audit, visualization, and validation runbook

This is the authoritative runbook for the reduced V1 informed-consent meta-model.

## Correct study framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset, not from the new MedGemma/Qwen/Llama/GPT outputs.

```text
Derivation / discovery corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Validation / stress-test corpus:
  new MedGemma, Qwen235B, Llama4, GPT-5.5 round-trip outputs
```

Expert-preserved rows are positive functional evidence. Expert-failed rows are boundary evidence. New LLM outputs are used only after V1 is defined, to test generalization.

## Important methodological correction

V1 is **not** induced by hard-coding fields such as action/resource/actor. The workflow separates two empirical graph views:

1. **Semantic-equivalence graph**: asks which source-model elements may express the same semantic field. Edges come from same/overlapping evidence spans, cross-information-model support, cross-LLM support, expert-positive evidence, failure penalties, and profile similarity.
2. **Provision-bundle graph**: asks