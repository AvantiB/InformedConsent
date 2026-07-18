# Round-trip evaluation and meta-model evidence runbook

This runbook starts after the model-generation jobs finish. It assumes the following model folders are present for both Union V0 and individual-model round trips:

- `medgemma`
- `qwen235b`
- `llama4`
- `mayo_gpt55`

The goal is to produce one combined evaluation set with classifier-based meaning preservation, lexical/content coverage, cue-preservation diagnostics, and source-element evidence for reduced meta-model development.

## 1. Check model-output counts

```bash
python - <<'PY'
from pathlib import Path
models = ["medgemma", "qwen235b", "llama4", "mayo_gpt55"]

print("Union V0")
for m in models:
    d = Path(f"meta_model/outputs/union_v0_roundtrip/{m}")
    print("\n", m)
    for f in ["union_v0_forward_mappings.jsonl", "union_v0_backward_reconstructions.jsonl", "failed_requests.jsonl"]:
        p = d / f
        print(f, sum(1 for x in p.open() if x.strip()) if p.exists() else "MISSING")

print("\nIndividual models")
for m in models:
    print("\n", m)
    for info in ["DUO", "ICO", "ODRL", "FHIR_Consent"]:
        d = Path(f"meta_model/outputs/individual_model_roundtrip/{m}/{info}")
        vals = []
        for f in ["forward_mappings.jsonl", "backward_reconstructions.jsonl", "failed_requests.jsonl"]:
            p = d / f
            vals.append(f"{f}=" + (str(sum(1 for x in p.open() if x.strip())) if p.exists() else "MISSING"))
        print(info, "; ".join(vals))
PY
```

Target is approximately 187 forward and 187 backward rows per completed condition, with zero or very few failures.

## 2. Standardize all four model families

```bash
python meta_model/scripts/07_standardize_roundtrip_outputs.py \
  --union_model_dirs \
meta_model/outputs/union_v0_roundtrip/medgemma,meta_model/outputs/union_v0_roundtrip/qwen235b,meta_model/outputs/union_v0_roundtrip/llama4,meta_model/outputs/union_v0_roundtrip/mayo_gpt55 \
  --individual_model_dirs \
meta_model/outputs/individual_model_roundtrip/medgemma,meta_model/outputs/individual_model_roundtrip/qwen235b,meta_model/outputs/individual_model_roundtrip/llama4,meta_model/outputs/individual_model_roundtrip/mayo_gpt55 \
  --output_dir meta_model/outputs/scoring_inputs_all4 \
  --require_backward
```

This writes:

- `meta_model/outputs/scoring_inputs_all4/standardized_roundtrips.csv`
- `meta_model/outputs/scoring_inputs_all4/standardization_audit.csv`
- `meta_model/outputs/scoring_inputs_all4/missing_pairs.csv`

## 3. Score with the final meaning-preservation classifier and lexical metrics

```bash
python meta_model/scripts/09_score_roundtrip_outputs.py \
  --standardized_csv meta_model/outputs/scoring_inputs_all4/standardized_roundtrips.csv \
  --classifier_bundle meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib \
  --output_dir meta_model/outputs/scored_roundtrips_all4
```

This produces:

- `scored_roundtrips.csv`
- `score_summary_by_condition.csv`
- `paired_union_vs_individual.csv`
- `high_classifier_low_overlap_audit.csv`
- `lowest_content_coverage_top200.csv`

Interpretation:

- Classifier score is a proxy meaning-preservation score.
- Lexical/content metrics are guardrails for omissions and heavy compression.
- High classifier score plus low lexical/content coverage should be manually audited.

## 4. Add cue-preservation and meta-model evidence analysis

```bash
python meta_model/scripts/15_analyze_roundtrip_scored_outputs.py \
  --scored_csv meta_model/outputs/scored_roundtrips_all4/scored_roundtrips.csv \
  --inventory_csv meta_model/v0_union/source_element_inventory.csv \
  --output_dir meta_model/outputs/roundtrip_meta_model_analysis_all4
```

This produces:

- `scored_roundtrips_with_cue_audit.csv`
- `evaluation_summary_by_condition.csv`
- `cue_group_preservation_long.csv`
- `modal_transition_summary.csv`
- `high_score_cue_loss_audit.csv`
- `lowest_cue_preservation_top250.csv`
- `source_element_mentions_long.csv`
- `source_element_evidence_summary.csv`
- `source_element_cooccurrence_pairs.csv`
- `meta_model_evidence_summary.md`

The cue analysis checks whether clinically and ethically important consent cues survive reconstruction, including permission, obligation, prohibition, condition, constraint/exception, time/duration, withdrawal, data/specimen, sharing/access, privacy/identifiability, actor/recipient, and risk/benefit cues.

## 5. What to inspect first

Start with these files:

```bash
column -s, -t < meta_model/outputs/scored_roundtrips_all4/score_summary_by_condition.csv | less -S
column -s, -t < meta_model/outputs/scored_roundtrips_all4/paired_union_vs_individual.csv | less -S
column -s, -t < meta_model/outputs/roundtrip_meta_model_analysis_all4/evaluation_summary_by_condition.csv | less -S
column -s, -t < meta_model/outputs/roundtrip_meta_model_analysis_all4/cue_group_preservation_long.csv | less -S
less meta_model/outputs/roundtrip_meta_model_analysis_all4/meta_model_evidence_summary.md
```

Then manually audit:

```bash
column -s, -t < meta_model/outputs/scored_roundtrips_all4/high_classifier_low_overlap_audit.csv | less -S
column -s, -t < meta_model/outputs/roundtrip_meta_model_analysis_all4/high_score_cue_loss_audit.csv | less -S
```

## 6. How this feeds reduced meta-model development

Use the outputs as follows:

1. `evaluation_summary_by_condition.csv`: establishes which model/information-model conditions preserve meaning and content most reliably.
2. `cue_group_preservation_long.csv`: identifies modal and consent-governance cues that are frequently lost. These cues should become explicit meta-model dimensions or constraints.
3. `source_element_evidence_summary.csv`: identifies source elements that are frequent, cross-model supported, and associated with high or low preservation.
4. `source_element_cooccurrence_pairs.csv`: proposes candidate merge groups, but merge only after checking cue preservation and examples.
5. `high_score_cue_loss_audit.csv` and `high_classifier_low_overlap_audit.csv`: targeted human-review queue for classifier/lexical disagreement.

Paper story at this stage:

> We evaluate each source information model, the unreduced Union V0, and multiple LLMs using a common round-trip protocol. Meaning preservation is estimated with the final classifier, while lexical/content coverage and cue-preservation diagnostics provide independent guardrails against omission. The resulting source-element and cue-level evidence is then used to induce a reduced consent meta-model, with human review reserved for audit, naming, and coherence checks.
