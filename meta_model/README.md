# Data/Language-Driven Meta-Model Development

This folder contains the reduced consent/data-use meta-model development workflow.

The goal is not to manually design a new ontology. The goal is to induce a reduced, functional meta-model from evidence produced by existing information models and round-trip data.

## Starting point: Union V0

The first baseline is **Union V0**: the unreduced union of source elements from ICO, DUO, FHIR Consent, and ODRL.

Union V0 is intentionally bulky. It is used as the maximal evidence inventory from which redundancy, missing concepts, and meaning-critical distinctions can be discovered.

Because the original per-model round-trip prompts used each information model's full data dictionary, the original individual-model results should remain the primary replication/reference baseline. For the combined union model, there are two possible conditions:

1. **Union V0 full dictionary**: include all combined elements in one prompt if token budget allows.
2. **Union V0 retrieval-augmented dictionary**: store the full union externally, retrieve top candidate element cards per sentence, and prompt only those candidates.

These are different experimental conditions and should be reported separately. With the current prompt files, Union V0 is small enough to run the full-dictionary condition first.

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
parse_audit.csv
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

### 0c. Run Union V0 full-dictionary round trip

```bash
python meta_model/scripts/03_run_union_v0_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key medgemma \
  --output_dir meta_model/outputs/union_v0_roundtrip
```

The runner is designed for one deployed model at a time. It supports vLLM OpenAI-compatible endpoints and OpenAI API models through the same config template:

```text
meta_model/configs/union_v0_models_template.yaml
```

Outputs per model:

```text
run_metadata.json
union_v0_forward_mappings.jsonl
union_v0_backward_reconstructions.jsonl
union_v0_roundtrip_outputs.csv
failed_requests.jsonl
```

See the full runbook:

```text
meta_model/UNION_V0_ROUNDTRIP_RUNBOOK.md
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

1. Run Union V0 full-dictionary round trips for each LLM, one model at a time.
2. Score Union V0 reconstructions with the meaning-preservation classifier.
3. Generate evidence units from Union V0 plus the existing historical round-trip CSV.
4. Cluster evidence units and inspect cluster summaries.
5. Add preservation-aware merge/split/add recommendation script.
6. Generate induced meta-model v0.1 from cluster evidence.
7. Evaluate v0.1 through the same forward/backward round-trip setup.
