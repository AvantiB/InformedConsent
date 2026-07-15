# Meaning-Preservation Classifier MVP

This folder contains the minimum viable experiments for a proxy meaning-preservation classifier for informed-consent LLM round trips.

The classifier is **not** the main manuscript contribution. It is a supporting evaluator that approximates existing binary human labels:

```text
original consent sentence
+ forward mapping into an information model
+ backward reconstruction
→ meaning_preserved = 0/1
```

## What it tests

The runner compares:

1. majority baseline;
2. bag-of-words TF-IDF + logistic regression;
3. consent-aware linguistic features + logistic regression;
4. optional embedding similarity features;
5. optional NLI entailment features;
6. hybrid bag-of-words + consent-aware + optional semantic features.

It evaluates generalization under:

- random round-trip split;
- leave-sentence-out grouped cross-validation;
- leave-one-LLM-out;
- leave-one-information-model-out.

## Setup

Using conda:

```bash
cd meaning_preservation
conda env create -f environment.yml
conda activate meaning-preservation
```

Or with pip:

```bash
cd meaning_preservation
pip install -r requirements.txt
```

## Minimum run

From the repository root:

```bash
python meaning_preservation/run_classifier_experiments.py \
  --roundtrips_csv /path/to/existing_informed_consent_repo/step1_output/roundtrips.csv \
  --output_dir meaning_preservation/outputs/mvp
```

## Optional embedding features

The script now supports two embedding backends:

- `sentence_transformers`: uses `sentence-transformers` directly;
- `hf`: uses plain Hugging Face `transformers` with mean pooling;
- `auto`: tries `sentence-transformers` first, then falls back to `hf`.

On the HPC, if `sentence-transformers` fails with an import error such as `AutoProcessor`, use the HF backend explicitly:

```bash
python meaning_preservation/run_classifier_experiments.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --output_dir meaning_preservation/outputs/with_embeddings_hf \
  --embedding_model all-MiniLM-L6-v2 \
  --embedding_backend hf \
  --embedding_device cuda
```

The shorthand `all-MiniLM-L6-v2` is automatically resolved to `sentence-transformers/all-MiniLM-L6-v2` unless it is a local path. If compute nodes do not have internet access, pass a local cached model path:

```bash
python meaning_preservation/run_classifier_experiments.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --output_dir meaning_preservation/outputs/with_embeddings_hf \
  --embedding_model /path/to/local/all-MiniLM-L6-v2 \
  --embedding_backend hf \
  --embedding_device cuda
```

## Optional NLI features

```bash
python meaning_preservation/run_classifier_experiments.py \
  --roundtrips_csv /path/to/step1_output/roundtrips.csv \
  --output_dir meaning_preservation/outputs/with_nli \
  --nli_model cross-encoder/nli-deberta-v3-base \
  --nli_device cuda
```

## Outputs

The script writes:

```text
processed/roundtrip_dataset.csv
processed/dataset_audit.csv
features/features.csv
results/metrics_by_split.csv
results/threshold_metrics.csv
results/feature_lr_coefficients.csv
results/final_feature_lr.joblib
results/meaning_preservation_classifier_summary.md
```

## Interpretation

Use this classifier as a **proxy evaluator**, not as a definitive semantic or ontology-correctness judge. The current labels are 0/1 human meaning-preservation judgments from previous round-trip review. The most important outputs for downstream filtering are high-confidence precision and retained sample size.
