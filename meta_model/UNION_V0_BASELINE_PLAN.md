# Union V0 Baseline Plan

The first meta-model baseline is **Union V0**: the unreduced union of source-model elements from ICO, DUO, FHIR Consent, and ODRL.

Union V0 is intentionally bulky. It is not the final meta-model. It is the maximal evidence inventory from which redundancy, missing concepts, and meaning-critical distinctions will be discovered.

## Why Union V0?

The original round-trip prompts used each information model's full data dictionary in the prompt. To remain comparable, the original individual-model baselines should be preserved as full-dictionary prompts.

For the combined model, however, a literal full prompt containing ICO + DUO + FHIR Consent + ODRL may exceed practical context limits or create poor LLM behavior. Therefore Union V0 has two possible operational forms:

1. **Union V0 full dictionary**: all combined elements are included in one prompt, if token budget permits.
2. **Union V0 retrieval-augmented dictionary**: the full combined inventory is stored externally, but only top candidate element cards are provided for a given sentence.

These should be reported as different experimental conditions.

## Data-driven reduction logic

Union V0 will be reduced using evidence, not manual schema design.

For each source element, estimate:

- frequency in forward mappings;
- co-selection/co-occurrence with other source elements;
- overlap in evidence spans and sentence contexts;
- source-model coverage;
- association with meaning-preserved versus not-preserved reconstructions;
- relationship to cue groups and NLI/semantic features;
- unmatched language around it.

Reduction decisions are then induced:

- **merge** elements that behave redundantly across language, co-occurrence, and preservation behavior;
- **retain** distinctions whose loss is associated with meaning failure;
- **split** clusters that contain subgroups with different language or preservation behavior;
- **add** concepts repeatedly seen in language but not adequately captured by source models.

## Immediate pipeline

```bash
# 1. Build unreduced source-element inventory from original full-dictionary prompts
python meta_model/scripts/00_build_union_v0_inventory.py \
  --prompt_dir /path/to/source_model_prompts \
  --output_dir meta_model/v0_union

# 2. Retrieve candidate source elements for each sentence from Union V0
python meta_model/scripts/00_retrieve_union_v0_candidates.py \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --sentences_csv /path/to/roundtrips.csv \
  --output_csv meta_model/v0_union/sentence_candidate_elements.csv \
  --top_k 40

# 3. Continue with evidence-unit construction and clustering
python meta_model/scripts/01_build_evidence_units.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --output_dir meta_model/outputs/evidence_units \
  --cue_dictionary meaning_preservation/literature_informed_consent_cues.json
```

## Expected outputs

```text
meta_model/v0_union/source_element_inventory.csv
meta_model/v0_union/element_cards.jsonl
meta_model/v0_union/source_model_prompt_sizes.csv
meta_model/v0_union/sentence_candidate_elements.csv
meta_model/v0_union/retrieval_summary.csv
```

## Next planned analysis

After candidate retrieval works, evaluate retrieval recall against historical forward mappings. This will tell us whether Union V0-RAG is a defensible approximation of full-dictionary Union V0.
