# Reduced functional informed-consent meta-model development

This document describes the paper-facing method for developing the first candidate reduced informed-consent meta-model. The process is **data-seeded and expert-refined**: empirical evidence from form-level cross-validated round-trip analyses identifies recurring semantic neighborhoods and provision structures, but the final reduced labels are organized as functional, complementary consent roles rather than raw cluster IDs.

## Objective

The objective is to construct a compact representation that preserves the meaning of informed-consent language while reducing the redundancy of a naive union of ICO, DUO, FHIR Consent, and ODRL. The target model should support forward annotation, backward reconstruction, source-model translation, and held-out evaluation on unseen consent forms.

## Methodological position

The reduced model is not claimed to be fully automatically discovered. Direct clustering of ontology/source-model elements produced useful evidence, but it also created overlapping lexical groups. Informed-consent language often expresses several complementary functions within the same phrase: for example, `stored in the All of Us databases` contains both a governed action (`stored`) and a repository (`All of Us databases`). Therefore, the reduction process distinguishes **near-equivalence** from **complementarity**.

The paper-facing claim is:

> We used form-level cross-validated, data-driven evidence from expert-preserved round-trip annotations to identify stable semantic neighborhoods and complementary provision structures. These data-derived seeds were consolidated into a smaller functional schema through expert-guided review and crosswalked to ICO, DUO, FHIR Consent, and ODRL.

## Input evidence

The derivation corpus consists of original researcher annotation workbooks with expert meaning-preservation labels. The evaluation universe is the main `roundtrips.csv`, using stable `form_key` values for consent-form-level splits.

The cleaned mention-level evidence stores:

```text
form_id
sentence_id
sentence_text
information_model
source_element_id / union_element_id
source_element_label
raw_span_text
cleaned span_text
annotation_decision
meaning_preserved
reconstructed_sentence
LLM
fold_id
split
provenance_key
```

## Form-level cross-validation

The split unit is the consent form, not the sentence. The main `roundtrips.csv` contains 21 stable `form_key` values. Form-level splitting avoids leakage from having sentences from the same consent document in both derivation and held-out evaluation.

A punctuation-insensitive repair step is used only to align expert-workbook form identifiers to the stable `form_key` split. For example, `Alzheimer_s Disease...` and `Alzheimer's Disease...` are treated as aliases for the same consent form. After repair, the fold runs showed `n_unassigned_mentions = 0` for all folds.

## Decision-marker separation

The original annotation workbooks often attach decisions to each annotation, e.g.:

```text
blood [NRES] (permit)
urine [NRES] (permit)
stored at the All of Us biobank [RTN] (permit)
```

For the reduced model, `(permit)` and `(deny)` are provision-level decision metadata, not span text. The pipeline strips these markers from evidence spans and stores them separately as `annotation_decision`. Annotation-level decisions are aggregated to `sentence_decision_evidence.csv`.

This prevents false fields such as `(permit) research` or `(deny) researchers` from contaminating candidate schemas.

## Source-element-in-context sense induction

The unit of induction is not a raw source element. The unit is a **source-element-in-context mention**. Broad elements such as ODRL `Party`, ODRL `Constraint`, FHIR `Consent.provision.actor`, and FHIR `Consent.provision.action` are split into contextual senses using observed evidence-span usage.

Examples:

```text
ODRL::Party used for "you"                 -> participant-like sense
ODRL::Party used for "researchers"         -> actor-like sense
ODRL::Party used for "Mayo Clinic"         -> institution/custodian-like sense
ODRL::Party used for "All of Us databases" -> repository-like sense
```

This step is essential because many source-model elements are intentionally broad and cannot be used as reduced fields without context.

## Typed relationship graph

Within each training fold, pairwise evidence between sense nodes is typed as:

```text
near_equivalent     -> can support field merging
broader_narrower    -> hierarchy/scope evidence, not direct merging
related_distinct    -> related but unsafe or uncertain to merge
complementary       -> co-occurring provision structure, not merging evidence
```

Only strict near-equivalence edges can merge source-element senses into candidate fields. Co-occurrence is retained separately as provision-bundle evidence.

## Co-occurrence and complementarity

Repeated co-occurrence identifies how consent meanings are composed. For example, many provisions contain bundles like:

```text
participant_or_subject + governed_action + governed_resource + repository_or_registry
participant_or_subject + choice_or_withdrawal_right + temporal_scope
authorized_actor + governed_action + governed_resource + purpose_or_use_context
restriction_or_prohibition + condition_or_exception + governed_action
```

These bundles show complementarity, not equivalence. They motivate functional roles rather than merged clusters.

## Intermediate empirical results

The initial raw fold-specific candidate schemas contained approximately 295-317 candidate fields per fold. These were intentionally permissive evidence dictionaries.

A stricter cross-fold selection pass retained only stable, supported candidate neighborhoods:

| Fold | Strict selected fields |
|---|---:|
| fold_00 | 29 |
| fold_01 | 32 |
| fold_02 | 36 |
| fold_03 | 32 |

This showed that the data-driven process consistently reduced the search space to a manageable set of recurring semantic neighborhoods. However, these selected fields remained cluster-like and partially overlapping, so they were used as evidence rather than as final labels.

## From data-derived neighborhoods to reduced functional fields

The transition from clusters to the candidate functional meta-model used the following logic:

1. **Recurrence across folds** identified stable semantic neighborhoods.
2. **Near-equivalence edges** suggested which source-element senses could be consolidated.
3. **Co-occurrence/provision-bundle edges** showed which concepts were complementary and should remain separate.
4. **Broad source-element splits** identified places where existing models were too coarse.
5. **Reconstruction error patterns** motivated additional boundaries, especially temporal attachment, repository vs institution, and decision cue vs sentence decision.
6. **Expert/domain review** is used for final naming, boundary decisions, missing functions, and unsafe merges.

Example consolidations:

| Data-derived evidence | Functional field |
|---|---|
| blood, urine, saliva, DNA, samples, specimens, health information, records | governed_resource |
| researchers, study team, doctors, investigators | authorized_actor |
| Mayo Clinic, KP Research Bank, All of Us, Cincinnati Children's | institution_or_custodian |
| database, registry, biobank, tissue bank | repository_or_registry |
| use, collect, store, share, disclose, access, destroy, return | governed_action |
| research, future research, clinical purposes, public health purposes | purpose_or_use_context |
| at any time, indefinitely, no expiration, until FDA approval | temporal_scope |
| study duration vs data-storage duration vs permission duration | temporal_target |
| de-identified, identifiable, coded, confidential | privacy_or_identifiability |
| withdraw, say no, stop taking part, remove samples | choice_or_withdrawal_right |

## Candidate reduced functional V1 fields

The first candidate reduced model contains one sentence/provision-level decision and a set of span-level fields:

```text
sentence_decision
decision_cue_or_consent_act
participant_or_subject
authorized_actor
institution_or_custodian
repository_or_registry
governed_resource
governed_action
purpose_or_use_context
research_domain_or_study_topic
temporal_scope
temporal_target
condition_or_exception
restriction_or_prohibition
choice_or_withdrawal_right
privacy_or_identifiability
study_or_data_lifecycle
consequence_or_protection
return_of_results_or_feedback
contact_or_request
residual_important_content
provenance
```

The YAML version is stored at:

```text
meta_model/schemas/reduced_functional_v1_candidate.yaml
```

## Source-model crosswalk

Each ICO, DUO, FHIR Consent, and ODRL element should be mapped into the reduced functional fields. Mapping is many-to-one and sometimes one-to-many. Broad source elements are marked as context-dependent.

Example crosswalk logic:

```text
FHIR Consent.subject       -> participant_or_subject
FHIR Consent.provision.data -> governed_resource
FHIR Consent.provision.action -> governed_action
FHIR Consent.provision.purpose -> purpose_or_use_context
FHIR Consent.provision.period / dataPeriod -> temporal_scope + temporal_target
FHIR Consent.provision.type / Consent.decision -> sentence_decision
ODRL Asset_DO -> governed_resource
ODRL Action_Verb -> governed_action
ODRL Party -> context-dependent split across participant, actor, institution, repository
ODRL Constraint -> context-dependent split across condition, temporal, purpose, restriction, privacy
DUO disease/purpose restrictions -> purpose_or_use_context / research_domain_or_study_topic / restriction_or_prohibition
ICO consent/withdrawal/action elements -> decision_cue_or_consent_act / choice_or_withdrawal_right / governed_action
```

The crosswalk enables visualizations such as:

```text
source information model -> source element -> reduced functional field
source model by functional field matrix
functional field recurrence across source models
context-dependent source elements requiring expert audit
near-equivalence and complementarity evidence supporting each field
```

## Expert review

The PI/domain expert should review:

```text
field names
field definitions
include/exclude boundaries
source-model crosswalk
context-dependent mappings
examples assigned to each field
missing consent functions
fields that should be core vs extension
```

The output of that review becomes V1.1.

## Evaluation plan

The main evaluation should be run on the expert-reviewed reduced functional model, not on raw cluster IDs. The comparison conditions are:

```text
individual DUO
individual ICO
individual FHIR Consent
individual ODRL
Union V0
Reduced Functional V1/V1.1
```

The held-out round-trip evaluation should measure:

```text
meaning-preservation classifier score
content-word recall/precision
cue preservation
annotation count
unique field count
unmatched-language rate
same-span/multi-label rate
relationship/attachment preservation
qualitative error categories
```

## Optional external generalization set

A small external set of publicly available, unseen consent forms can be used after internal held-out CV. This should be labeled as external stress testing rather than primary validation unless sampling and annotation are formalized.

Recommended scope:

```text
5-10 public biobank/data-sharing/data-reuse consent forms
10-30 sentences sampled per form
no use in schema design
report as external qualitative or exploratory generalization
```

This can strengthen the paper if time allows, but internal form-level CV remains the primary generalization test.
