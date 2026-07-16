# Data/Language-Driven Meta-Model Development

This folder contains the reduced consent/data-use meta-model development workflow.

The goal is not to manually design a new ontology. The goal is to induce a reduced, functional meta-model from evidence produced by the existing information models and round-trip data.

## Core idea

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

1. Generate evidence units from the existing round-trip CSV.
2. Inspect extraction audit and cluster summaries.
3. Add preservation-aware merge/split recommendation script.
4. Generate induced meta-model v0.1 from cluster evidence.
5. Evaluate v0.1 through forward/backward round-trip reconstruction.
