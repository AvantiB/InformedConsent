# Data/Language-Driven Meta-Model Development

This folder contains the reduced consent/data-use meta-model development workflow.

The goal is not to manually design a new ontology. The goal is to induce a reduced, functional meta-model from evidence produced by existing information models and round-trip data.

## Starting point: Union V0

The first baseline is **Union V0**: the unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL.

Union V0 is intentionally bulky. It is used as the maximal evidence inventory from which redundancy, missing concepts, and meaning-critical distinctions can be discovered.

Because the original per-model round-trip prompts used each information model's full data dictionary, the original individual-model results should remain the primary replication/reference baseline. For the combined union model, there are two possible conditions:

1. **Union V0 full dictionary**: include all combined elements in one prompt if token budget allows.
2. **Union V0 retrieval-augmented dictionary**: store the full union externally, retrieve top candidate element cards per sentence, and prompt only those candidates.

These are different experimental conditions and should be reported separately.

## Core evidence-unit idea

Each round-trip example is converted into evidence units:

```text
consent sentence
→ extracted phrase / source-model element
→ information model used
→ forward mapping text
→ backward reconstruction
→ human meaning-preservation label
→ optional classifier score / cue features
```

The reduced meta-model is then inferred from:

- source-model element usage;
- phrase/source-node co-occurrence;
- cue-group preservation/failure patterns;
- language embeddings and clustering;
- preservation behavior from human labels and the meaning-preservation classifier.

Human involvement should be limited to audit, naming, and interpretation of induced clusters, not manual construction of the schema.

## Current scripts

### 0a. Build Union V0 inventory

```bash
python meta_model/scripts/00_build_union_v0_inventory.py \
  --prompt_dir /path/to/source_model_forward_prompts \
  --output_dir meta_model/v0_union
```

Outputs:

```text
source_element_inventory.csv
element_cards.jsonl
source_model_prompt_sizes.csv
```

### 0b. Retrieve Union V0 candidates per sentence

```bash
python meta_model/scripts/00_retrieve_union_v0_candidates.py \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --sentences_csv /path/to/roundtrips.csv \
  --output_csv meta_model/v0_union/sentence_candidate_elements.csv \
  --top_k 40
```

Outputs:

```text
sentence_candidate_elements.csv
retrieval_summary.csv
```

### 1. Build evidence units

```bash
python meta_model/scripts/01_build_evidence_units.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --output_dir meta_model/outputs/evidence_units \
  --cue_dictionary meaning_preservation/literature_informed_consent_cues.json
```

Outputs:

```text
evidence_units.csv
phrase_node_graph_edges.csv
source_node_frequency.csv
source_node_cooccurrence.csv
cue_group_frequency.csv
extraction_audit.json
```

### 2. Cluster evidence units

```bash
python meta_model/scripts/02_cluster_evidence_units.py \
  --evidence_units_csv meta_model/outputs/evidence_units/evidence_units.csv \
  --output_dir meta_model/outputs/clusters \
  --embedding_model all-MiniLM-L6-v2 \
  --embedding_backend hf \
  --embedding_device cpu
```

Outputs:

```text
cluster_assignments.csv
cluster_summary.csv
cluster_pair_cooccurrence.csv
cluster_source_model_coverage.csv
```

## Induction principle

Candidate meta-model units should be retained when they are frequent, cross-model, preservation-sensitive, and compositionally stable. Candidate units should be merged only when their collapse does not appear to harm meaning preservation. Candidate units should be split when subclusters show different language, source-model, or preservation behavior.

## Next planned steps

1. Collect the original ICO, DUO, FHIR Consent, and ODRL forward prompts into one folder.
2. Build the Union V0 source-element inventory.
3. Measure full Union V0 prompt size and decide whether a full-union prompt is feasible.
4. Retrieve Union V0 candidates for each sentence.
5. Evaluate candidate recall against historical forward mappings.
6. Generate evidence units from the existing round-trip CSV.
7. Cluster evidence units and inspect cluster summaries.
8. Add preservation-aware merge/split/add recommendation script.
9. Generate induced meta-model v0.1 from cluster evidence.
10. Evaluate v0.1 through forward/backward round-trip reconstruction.
