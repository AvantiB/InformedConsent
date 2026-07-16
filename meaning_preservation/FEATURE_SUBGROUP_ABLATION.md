# Literature-Informed Feature Subgroup Ablation Design

This note documents the language/data-use feature design used by
`run_feature_subgroup_ablation_experiments.py`.

## Why split the engineered features?

Earlier experiments showed that broad hand-engineered features and sentence
embeddings both predict human binary meaning-preservation labels, while BOW
TF-IDF is a weaker lexical baseline. The next question is more specific:
which kinds of language drive the preservation judgment?

The subgroup runner separates the broad engineered feature set into:

1. **Lexical similarity**: length ratio, absolute length difference, token
   Jaccard, and TF-IDF cosine.
2. **Mapping/annotation complexity**: annotation count, unique element count,
   mapping length, and bracket/parenthesis counts from the forward mapping.
3. **Dictionary/modal cue features**: literature-informed cue groups for
   deontic modality, privacy/data-practice categories, and biobank-specific
   consent concepts.
4. **Semantic features**: embedding cosine/distance and optional NLI features.
5. **Combined sets**: dictionary + semantic, all engineered, all engineered +
   semantic, and a full TF-IDF + engineered + semantic ablation.

This design lets us test whether consent-relevant language categories add
signal beyond generic lexical similarity and beyond sentence-level semantic
similarity.

## Literature-informed cue groups

The cue dictionary is stored in:

```text
meaning_preservation/literature_informed_consent_cues.json
```

It is intentionally an editable seed resource rather than a complete validated
lexicon.

The categories are motivated by three literatures:

### 1. Deontic modality and legal-language NLP

Consent forms change meaning through permission, obligation, prohibition,
conditionality, exception, and revocation/withdrawal language. Legal NLP work
on deontic modality, including LEXDEMOD, treats modal triggers and agent-linked
permissions/obligations as explicit annotation targets. This motivates cue
groups such as `permission`, `obligation`, `prohibition`, `condition`,
`exception`, `restriction_scope`, and `choice_withdrawal_control`.

### 2. Privacy-policy and data-practice NLP

Biobank consent language overlaps with privacy/data-practice disclosures:
collection, use, sharing/disclosure, retention/deletion, access, user
choice/control, security, recipient, and purpose. OPP-115-style privacy-policy
classification and later privacy-policy QA/ontology work motivate feature
groups such as `collection_obtaining`, `use_processing`,
`sharing_disclosure_transfer`, `storage_retention_destruction`,
`access_contact_return_results`, `recipient_actor`, and
`security_safeguards`.

### 3. Biomedical data-use and biobank consent

Biobank/resource-use consent also includes data-use restrictions and biomedical
resource concepts: health data, biospecimens, genetic/genomic data,
identifiability/de-identification, future research, commercial use,
publication/public release, return of results, and governance/IRB oversight.
These motivate groups such as `data_object_health_record`,
`data_object_biospecimen`, `genetic_genomic`, `identifiability_privacy`,
`purpose_research`, `commercialization`, `publication_public_release`, and
`governance_oversight`.

## Feature analysis outputs

The subgroup runner writes:

```text
cue_dictionary_terms.csv
cue_group_frequency_by_label.csv
dictionary_modal_lr_coefficients.csv
engineered_all_lr_coefficients.csv
dictionary_modal_semantic_lr_coefficients.csv
engineered_semantic_lr_coefficients.csv
*_rf_importances.csv
```

These outputs are meant to support two analyses:

1. **Frequency analysis**: which cue groups appear in preserved vs
   non-preserved examples?
2. **Predictive importance**: which cue categories or engineered features are
   most influential for classification?

Coefficient and feature-importance analyses should be interpreted alongside
error analysis because engineered features can be correlated. For example,
a positive coefficient on a missing-count feature does not automatically mean
that omission preserves meaning; it may reflect sentence complexity or other
correlated features.

## Implication for meta-model development

The same categories that predict meaning-preservation failures are candidate
dimensions for a reduced consent meta-model. In particular, the meta-model
should pay attention to:

- normative force: permission, obligation, prohibition;
- conditionality and exceptions;
- withdrawal/control;
- action type: collect, use/process, share/disclose, store/retain/destroy;
- resource/data object: health record, biospecimen, genetic/genomic data;
- identifiability and privacy safeguards;
- recipient/actor;
- purpose/scope, including future research;
- commercial use;
- return of results/contact;
- governance/oversight;
- temporal duration and retention.

These categories are not proposed as the final meta-model. They are a
language- and data-driven starting point for empirical reduction from the
round-trip evidence.
