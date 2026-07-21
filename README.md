# InformedConsent

Code for the informed-consent data-use and sharing project.

## Repository structure

```text
meaning_preservation/   # Expert-labeled round-trip classifier experiments and proxy scoring
meta_model/             # Individual models, Union V0, and reduced functional meta-model development
```

## Current workflow

The project has two linked components.

1. **Meaning-preservation classifier**: develops a proxy classifier for whether an LLM backward reconstruction preserves the meaning of the original consent sentence.
2. **Reduced functional meta-model development**: uses form-level cross-validated evidence from expert-preserved round trips to seed a compact, complementary consent-language schema, then crosswalks ICO/DUO/FHIR Consent/ODRL elements to that schema and evaluates held-out forward/backward reconstruction.

## Key principle

```text
Derivation / induction evidence:
  original expert-evaluated round-trip dataset

Data-driven seed analysis:
  form-level CV, source-element-in-context senses, typed relationship graphs,
  near-equivalence, co-occurrence/complementarity, and cross-fold recurrence

Reduced model:
  data-seeded functional V1 candidate schema, finalized through PI/domain-expert review

Generalization test:
  held-out consent forms in form-level cross-validation

Optional stress test:
  additional public unseen biobank/data-sharing/data-reuse consent forms
```

Expert-preserved rows are treated as functionally validated positive evidence. Expert-failed rows are boundary evidence. Classifier scores are proxy validation outcomes and are reported alongside lexical, cue, annotation-coverage, annotation-burden, and qualitative relationship-preservation analyses.

## Key runbooks and method documents

```text
meaning_preservation/README.md
meta_model/README.md
meta_model/FUNCTIONAL_V1_METHODS.md
meta_model/FUNCTIONAL_V1_ROUNDTRIP_RUNBOOK.md
meta_model/REFINED_CV_POSTPROCESS_RUNBOOK.md
meta_model/ROUNDTRIP_SCORING_RUNBOOK.md
```

## Current stage

```text
A. Build the clean expert round-trip corpus from original workbooks.
B. Create form-level CV splits to avoid training/testing leakage.
C. For each training fold, derive source-element-in-context senses and typed relationship evidence.
D. Use strict selected fold clusters as data-derived seeds, not final labels.
E. Consolidate recurring semantic neighborhoods into a reduced functional V1 schema.
F. Crosswalk ICO/DUO/FHIR Consent/ODRL elements to the functional V1 fields.
G. Review field names, boundaries, and context-dependent mappings with PI/domain expert.
H. Evaluate the functional V1/V1.1 schema on held-out consent forms.
I. Compare individual models, Union V0, and reduced functional V1/V1.1 using classifier and non-classifier preservation metrics.
```
