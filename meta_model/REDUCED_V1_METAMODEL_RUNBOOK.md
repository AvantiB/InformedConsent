# Reduced V1 meta-model induction and validation runbook

This is the authoritative runbook for the reduced V1 informed-consent meta-model.

## Correct study framing

Reduced V1 is derived from the original expert-evaluated round-trip dataset, not from the new MedGemma/Qwen/Llama/GPT outputs.

```text
Derivation / induction corpus:
  original researcher annotation workbooks with expert meaning-preservation labels

Validation / stress-test corpus:
  new MedGemma, Qwen235B, Llama4, GPT-5.5 round-trip outputs
```

Expert-preserved rows are treated as functionally validated positive evidence: the forward representation contained enough information to reconstruct the original sentence meaning. Expert-failed rows are boundary evidence: they weaken proposed merges and flag unsafe simplifications.

## Comparable V1 round-trip evaluation

Reduced V1 keeps the same basic evaluation structure as Union V0:

```text
original sentence -> structured JSON
structured JSON only -> reconstructed sentence
```

Run both V1 evidence variants using the same schema:

1. `compact`: short evidence phrases only.
2. `permissive`: same schema, but longer evidence phrases are allowed when needed to preserve condition, exception, temporal, or privacy meaning.

## 1. Pull and compile

```bash
git pull origin main

python -m py_compile meta_model/scripts/12_build_expert_roundtrip_corpus.py
python -m py_compile meta_model/scripts/07_standardize_roundtrip_outputs.py
python -m py_compile meta_model/scripts/16_audit_annotation_granularity.py
python -m py_compile meta_model/scripts/17_induce_reduced_v1_metamodel.py
python -m py_compile meta_model/scripts/18_run_reduced_v1_roundtrip.py
```

## 2. Build the clean expert round-trip corpus

Use the original researcher handoff workbooks. The corpus builder reads the workbook columns:

```text
source_file, ID, full_text, annotations_combined, backward_mapping, Results/results, Notes
```

It produces a normalized CSV with one row per source sentence / information model / LLM workbook row. It also parses `annotations_combined` into `annotations_json`, which is the clean input used by the V1 induction script.

```bash
python meta_model/scripts/12_build_expert_roundtrip_corpus.py \
  --workbook_dir /path/to/original_annotation_workbooks \
  --output_csv meta_model/outputs/expert_roundtrips_clean.csv
```

Check the corpus summary:

```bash
cat meta_model/outputs/expert_roundtrip_corpus_summary.json
column -s, -t < meta_model/outputs/expert_roundtrip_corpus_summary.csv | less -S
```

Important: raw repeated annotations are retained in the clean corpus. Frequency within a sentence, across LLMs, and across information models is evidence for salience. Later graph-edge construction counts co-occurrence at the context level so repeated labels in one row do not create artificial Cartesian-product edge inflation.

## 3. Induce Reduced V1 from the clean expert corpus

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_expert \
  --min_edge_weight 0.22 \
  --min_core_positive_sentences 15
```

If the label column is not auto-detected:

```bash
python meta_model/scripts/17_induce_reduced_v1_metamodel.py \
  --expert_roundtrips_csv meta_model/outputs/expert_roundtrips_clean.csv \
  --label_col meaning_preserved \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/v1_reduced_expert
```

## 4. Evidence outputs from induction

```text
meta_model/v1_reduced_expert/expert_element_mentions_long.csv
meta_model/v1_reduced_expert/expert_sentence_level_decision_mentions_long.csv
meta_model/v1_reduced_expert/expert_element_profiles.csv
meta_model/v1_reduced_expert/expert_element_relationship_edges.csv
meta_model/v1_reduced_expert/expert_element_clusters.csv
meta_model/v1_reduced_expert/expert_cluster_evidence_summary.csv
meta_model/v1_reduced_expert/expert_sentence_level_decision_summary.csv
meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml
meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.md
meta_model/v1_reduced_expert/expert_validated_induction_methodology.md
```

## 5. Methodology implemented by script 17

1. Parse clean `annotations_json` into source-element mentions.
2. Separate sentence-level decision fields: `DUO.decision`, `ICO.decision`, `ODRL::Rule_TestSentence`, and `FHIR_Consent::Consent.provision.type`.
3. Treat annotations from expert-preserved rows as positive functional evidence.
4. Treat annotations from expert-failed rows as boundary/failure evidence.
5. Build source-element profiles containing raw mention counts, sentence coverage, LLM support, information-model support, positive rate, and span examples.
6. Build weighted graph edges from expert-positive co-occurrence by source sentence / LLM / information model context, same-span expert-positive use, label/definition/span similarity, and cross-source-model support.
7. Penalize edges when the same element pair occurs mainly in expert-failed rows.
8. Cluster the weighted graph.
9. Select clusters as `core_shared`, `context_module`, `audit_or_extension`, or `failure_boundary_audit`.
10. Generate the candidate Reduced V1 YAML schema with field-level selection evidence.

UMAP may be added later for visualization only. It is not the primary grouping method.

## 6. Audit before running V1

```bash
less meta_model/v1_reduced_expert/expert_validated_induction_methodology.md
less meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.md
head -30 meta_model/v1_reduced_expert/expert_cluster_evidence_summary.csv
head -30 meta_model/v1_reduced_expert/expert_element_relationship_edges.csv
```

This audit is for unsafe merges, bad names, or parsing failures. It is not manual schema design.

## 7. Smoke-test Reduced V1 compact and permissive variants

```bash
rm -rf meta_model/outputs/reduced_v1_roundtrip_smoke/mayo_gpt55

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode compact \
  --stage both \
  --limit 3

python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
  --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
  --model_key mayo_gpt55 \
  --output_dir meta_model/outputs/reduced_v1_roundtrip_smoke \
  --evidence_mode permissive \
  --stage both \
  --limit 3
```

## 8. Full Reduced V1 validation runs

Run both evidence modes for each validation model.

```bash
for MODEL in medgemma qwen235b llama4 mayo_gpt55; do
  for MODE in compact permissive; do
    python meta_model/scripts/18_run_reduced_v1_roundtrip.py \
      --roundtrips_csv /path/to/roundtrips.csv \
      --metamodel_yaml meta_model/v1_reduced_expert/reduced_metamodel_v1_candidate.yaml \
      --model_config_yaml meta_model/configs/union_v0_models.local.yaml \
      --model_key "$MODEL" \
      --output_dir meta_model/outputs/reduced_v1_roundtrip \
      --evidence_mode "$MODE" \
      --stage both
  done
done
```

## 9. Standardize, score, and audit validation outputs

After Reduced V1 validation runs, use the existing standardization, scoring, round-trip analysis, and annotation-granularity audit scripts to compare:

```text
individual source-model prompts
Union V0 full dictionary
Reduced V1 compact
Reduced V1 permissive
```

## Manuscript framing

The reduced V1 meta-model is induced from expert-validated evidence. New LLM outputs are validation conditions. A successful V1 does not need to beat Union V0 on every raw score; it should preserve meaning better than individual source models and approach Union V0 while reducing annotation burden, redundancy, and model-dependent verbosity.
