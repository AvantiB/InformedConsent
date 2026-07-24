# Summarized NIH requirements pointers for direct reduced-schema induction

Source summarized: NIH Office of Science Policy / Office of Extramural Research, *Informed Consent for Secondary Research with Data and Biospecimens: Points to Consider and Sample Language for Future Use and/or Sharing* (May 2022).

Use this summary as requirements guidance only. Do not copy these headings directly as schema fields unless they correspond to source-model-supported consent meaning.

## Scope

- The guidance addresses informed consent language for storing and sharing data and biospecimens collected during a primary research protocol for future secondary research.
- It covers both individually identifiable and deidentified data and biospecimens.
- It does not cover data or biospecimens initially collected outside a research context, such as ordinary clinical care.
- It is not itself a comprehensive consent form and does not replace IRB review, local requirements, or study-specific tailoring.

## Requirements implications for a machine-interpretable consent meta-model

A reduced consent schema should be able to represent, when stated in consent language:

1. **Resource being governed**
   - data, biospecimens, extracted DNA, blood, tissue, urine, medical images, surveys, EHR-derived information, wearable-device data, coded data, deidentified data, identifiable data, or linked data.

2. **Governed action**
   - collect, store, maintain, share, distribute, access, use, reuse, analyze, link, code, deidentify, reidentify/relink, retrieve, destroy, return, disclose, inspect, or commercialize.

3. **Purpose or use context**
   - primary study use, future secondary research, similar disease research, unrelated disease research, genomic research, FDA-regulated research, investigational drug/device research, commercial product development, or general research.

4. **Actor / recipient / controller**
   - participant, investigator, study team, institution, repository, honest broker, access committee, IRB/Privacy Board, external researchers, researchers at other institutions, researchers around the world, commercial entities, FDA/regulators, or entity controlling access/code keys.

5. **Consent force / governance decision**
   - permission, prohibition/denial, obligation, restriction, participant choice/right, protection, mixed, or unclear. In our experiments, global sentence force should be represented only through `sentence_decision`; source-model fields should remain span-level semantic fields.

6. **Future use and sharing scope**
   - whether future use is allowed, whether sharing is internal or external, whether sharing is worldwide, whether unrestricted access is allowed, whether future researchers must obtain approval, and whether future use is inside or outside the primary protocol/consent scope.

7. **Identifiability and privacy state**
   - identifiable, individually identifiable, deidentified, coded, anonymized, linked to identifiers, code-key retained, code-key holder, ability or inability to re-link, reidentification risk, access controls, confidentiality safeguards, and possibility of privacy breach.

8. **Temporal scope**
   - storage duration, indefinite storage, limited storage timeframe, future use, ongoing use, after withdrawal, age-of-majority/re-consent timing, and already-shared/already-used data or biospecimens.

9. **Withdrawal / choice / opt-in or opt-out**
   - whether storage/sharing is optional or required for participation, whether the participant may say yes/no, whether they may change their mind later, how to withdraw, who to contact, what can be retrieved, and what cannot be retrieved or undone.

10. **Restrictions, exceptions, and limitations**
    - explicit limits on future use, limits due to identifiability, inability to retrieve already shared or already used materials, FDA-regulated retention requirements, cultural/Tribal/sovereign-group restrictions, vulnerable population considerations, and local/federal/international legal requirements.

11. **Broad consent / waiver / re-consent**
    - whether future secondary research is covered by existing consent, whether IRB approval, re-consent, or waiver may be needed for secondary research outside the primary protocol, and whether broad consent requirements are satisfied or not.

12. **Risk/benefit and protections**
    - privacy/confidentiality risks, unauthorized access risk, reidentification risk, lack of direct benefit, possible societal/scientific benefit, commercial value, patents/licensing, and whether participants will receive payment.

## Design guidance for direct LLM-induced schema

- Stay close to DUO, ICO, ODRL, and FHIR Consent source-model elements.
- Consolidate source-model elements only when they are near-equivalent in consent function.
- Prefer a flat span-level dictionary for annotation.
- Allow modifiers only when they are needed to preserve consent meaning, such as identifiability state, temporal attachment, conditionality, negation, or optionality.
- Do not create a separate schema field solely because the NIH guidance mentions a topic. Use the guidance to check coverage and boundary cases.
- Keep `sentence_decision` separate from span-level dictionary fields.
- Preserve distinctions among actor, resource, action, purpose, condition, restriction, temporal scope, identifiability, sharing scope, withdrawal/choice, consequence/protection, and provenance when those distinctions are supported by source-model elements or necessary for faithful reconstruction.
