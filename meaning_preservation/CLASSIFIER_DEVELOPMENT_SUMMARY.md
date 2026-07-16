# Meaning-Preservation Classifier Development Summary

This document summarizes the meaning-preservation classifier work completed before moving to the reduced consent/data-use meta-model development phase.

The classifier predicts whether a backward reconstruction preserves the meaning of an original informed-consent sentence, given the original sentence, forward mapping, reconstructed sentence, source information model, and LLM metadata. It is intended as a proxy evaluator for the existing binary human meaning-preservation labels and as a high-confidence filtering/scoring component for downstream meta-model induction. It is not intended to prove ontology correctness.

## Dataset

The classifier development used the existing LLM round-trip dataset:

| Quantity | Value |
|---|---:|
| Round-trip rows | 2,068 |
| Meaning-preserved labels | 1,358 |
| Not-preserved labels | 710 |
| LLMs | 4 |
| Information models | 4 |
| Source sentences | 188 |
| Consent forms | 21 |

Evaluation included random split, leave-sentence-out, leave-one-LLM-out, and leave-one-information-model-out settings. Leave-sentence and leave-model/LLM settings are the most important for generalization.

## Model development stages

### 1. MVP classifier

The first MVP compared:

- majority baseline;
- TF-IDF bag-of-words logistic regression;
- engineered feature logistic regression;
- metadata/no-metadata variants;
- optional embedding and NLI hooks.

The MVP showed that bag-of-words alone was weak, while engineered features and semantic similarity were much stronger.

### 2. Embedding experiments

Two embedding models were tested as frozen feature generators, not fine-tuned classifiers:

- `all-MiniLM-L6-v2`, resolved as `sentence-transformers/all-MiniLM-L6-v2`;
- `nlpaueb/legal-bert-base-uncased`.

The embedding features were:

- `embedding_cosine`;
- `embedding_distance`.

MiniLM performed better than LegalBERT for embedding-only logistic regression. LegalBERT did not provide a clear domain advantage. This suggests that the task behaves more like sentence-level semantic equivalence than specialized legal-domain classification.

### 3. Feature-set ablation experiments

The broad engineered feature set was split into interpretable subgroups:

| Feature set | Purpose |
|---|---|
| `bow_tfidf_lr_baseline` | lexical baseline only |
| `lexical_similarity` | length ratio, absolute length difference, token Jaccard, TF-IDF cosine |
| `mapping_complexity` | annotation count, unique element count, mapping length, bracket/parenthesis counts |
| `dictionary_modal` | literature-informed deontic/privacy/biobank cue groups |
| `semantic` | embedding and optional NLI features |
| `dictionary_modal_semantic` | dictionary/modal cues plus semantic features |
| `engineered_all` | lexical + mapping complexity + dictionary/modal cues |
| `engineered_semantic` | all engineered features plus semantic features |
| `full_hybrid_tfidf_engineered_semantic` | TF-IDF + engineered + semantic, retained as an ablation only |

Classifiers tested over the relevant feature sets:

- logistic regression;
- random forest;
- XGBoost, if installed.

### 4. NLI robustness experiment

A single NLI experiment was run as a final robustness check using MiniLM embeddings plus NLI features. The semantic feature set then included:

- `embedding_cosine`;
- `embedding_distance`;
- `nli_entail_o2r`;
- `nli_contra_o2r`;
- `nli_neutral_o2r`;
- `nli_entail_r2o`;
- `nli_contra_r2o`;
- `nli_neutral_r2o`;
- `nli_min_bidirectional_entail`;
- `nli_max_contradiction`.

## Final key results

### Selected MiniLM + NLI results

| Model | AUROC | AUPRC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `engineered_semantic_rf` | 0.8863 | 0.9322 | 0.8076 | 0.8421 | 0.8481 | 0.8436 |
| `dictionary_modal_semantic_rf` | 0.8826 | 0.9299 | 0.8053 | 0.8515 | 0.8327 | 0.8403 |
| `semantic_rf` | 0.8673 | 0.9254 | 0.7942 | 0.8390 | 0.8331 | 0.8341 |
| `engineered_semantic_xgb` | 0.8813 | 0.9297 | 0.7976 | 0.8585 | 0.8071 | 0.8302 |
| `engineered_all_rf` | 0.8702 | 0.9236 | 0.7890 | 0.8332 | 0.8294 | 0.8295 |
| `engineered_semantic_lr` | 0.8563 | 0.8944 | 0.7856 | 0.8326 | 0.8153 | 0.8220 |
| `semantic_lr` | 0.8593 | 0.9206 | 0.7830 | 0.8471 | 0.7951 | 0.8161 |
| `bow_tfidf_lr_baseline` | 0.6858 | 0.7784 | 0.6604 | 0.7501 | 0.7041 | 0.7217 |

### Effect of NLI versus MiniLM without NLI

| Model | F1 without NLI | F1 with NLI | Delta F1 | AUROC without NLI | AUROC with NLI | Delta AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_rf` | 0.7799 | 0.8341 | +0.0542 | 0.8052 | 0.8673 | +0.0621 |
| `dictionary_modal_semantic_rf` | 0.8182 | 0.8403 | +0.0221 | 0.8630 | 0.8826 | +0.0196 |
| `engineered_semantic_xgb` | 0.8128 | 0.8302 | +0.0174 | 0.8713 | 0.8813 | +0.0100 |
| `engineered_semantic_rf` | 0.8315 | 0.8436 | +0.0121 | 0.8740 | 0.8863 | +0.0123 |
| `engineered_semantic_lr` | 0.8253 | 0.8220 | -0.0033 | 0.8525 | 0.8563 | +0.0038 |
| `semantic_lr` | 0.8272 | 0.8161 | -0.0110 | 0.8570 | 0.8593 | +0.0023 |

NLI is useful enough to retain, especially for random forest models. It improves the best model's average F1 by about 0.012 and AUROC by about 0.012. NLI does not uniformly improve logistic regression, likely because it adds nonlinear interaction signals that RF can use more effectively.

## Main conclusions

1. **Bag-of-words is weak.** TF-IDF BOW should remain a baseline, not a main feature source.
2. **Hand-engineered features are valid and predictive.** Dictionary/modal cues and lexical similarity features carry strong signal.
3. **Semantic similarity is useful.** MiniLM embedding similarity is strong, and LegalBERT does not clearly outperform MiniLM.
4. **NLI adds value.** Bidirectional entailment, neutrality, and contradiction features improve the strongest RF models.
5. **Best operational model:** `engineered_semantic_rf` with MiniLM embeddings and NLI features.
6. **Best interpretation models:** `dictionary_modal_lr`, `engineered_all_lr`, and `engineered_semantic_lr`, because coefficients support language/feature analysis.

## Important cue-group findings

Preserved examples showed substantially higher preservation of several consent/data-use cue groups. The largest positive Jaccard differences between preserved and not-preserved examples were:

| Cue group | Mean Jaccard, not preserved | Mean Jaccard, preserved | Difference |
|---|---:|---:|---:|
| `permission` | 0.4291 | 0.8386 | +0.4095 |
| `condition` | 0.6754 | 0.9303 | +0.2549 |
| `negation` | 0.6831 | 0.9046 | +0.2215 |
| `use_processing` | 0.7636 | 0.9539 | +0.1903 |
| `data_object_health_record` | 0.8148 | 0.9759 | +0.1612 |
| `prohibition` | 0.7746 | 0.9300 | +0.1554 |
| `data_object_biospecimen` | 0.8484 | 0.9896 | +0.1412 |
| `choice_withdrawal_control` | 0.8619 | 0.9753 | +0.1135 |
| `purpose_research` | 0.8717 | 0.9745 | +0.1028 |

These results suggest that meaning-preservation failures are concentrated around preservation of consent-relevant language categories, not just general paraphrase quality.

## Feature-importance signal

Top random-forest features in the final MiniLM + NLI `engineered_semantic_rf` model included:

- `embedding_distance`;
- `token_jaccard`;
- `embedding_cosine`;
- `tfidf_cosine`;
- `nli_min_bidirectional_entail`;
- `length_ratio`;
- `abs_length_diff`;
- `nli_entail_r2o`;
- `nli_entail_o2r`;
- `nli_neutral_o2r`;
- `nli_neutral_r2o`;
- `nli_contra_o2r`;
- `nli_max_contradiction`;
- `permission_jaccard`.

Top logistic-regression coefficients in dictionary/modal models highlighted permission, condition, negation, withdrawal/control, security safeguards, sharing/disclosure, storage/retention/destruction, commercialization, and modal-category features. These should be interpreted alongside error analysis because engineered features are correlated.

## Literature and repository-governance grounding

The cue dictionary is intentionally a seed dictionary, not a complete validated lexicon. It was motivated by:

- deontic modality and legal-language NLP: permission, obligation, prohibition, condition, exception, restriction, withdrawal/control;
- privacy/data-practice NLP: collection, use/processing, sharing/disclosure, retention/destruction, access, recipient, security safeguards;
- biomedical data-use and biobank consent: biospecimens, health records, genetic/genomic data, identifiability, future research, commercialization, return of results, governance/oversight.

The NCCN biorepository guidance also emphasized operational dimensions that should inform the meta-model: broad versus specific consent, waiver/authorization, internal/external/national/international sharing, DUA/MTA/LDS agreements, honest broker and de-identification processes, dbGaP/GWAS/future genetic data sharing, withdrawal/destruction, access governance, consent verification, return of results/incidental findings, and retention/destruction documentation.

## Final classifier recommendation

For downstream high-confidence filtering and scoring, use:

```text
engineered_semantic_rf with MiniLM embeddings + NLI features
```

For interpretable reporting and meta-model cue discovery, use:

```text
dictionary_modal_lr
engineered_all_lr
engineered_semantic_lr
cue_group_frequency_by_label.csv
*_lr_coefficients.csv
*_rf_importances.csv
```

Suggested use in the next phase:

1. Score new forward/backward round trips.
2. Use high-confidence meaning-preserved examples to identify stable consent/data-use units.
3. Use not-preserved examples and false positives to identify semantic distinctions that must not be collapsed.
4. Use cue-group frequencies, coefficients, and RF importances to prioritize candidate meta-model dimensions.

## Transition to meta-model development

The classifier stage motivates a language- and data-driven meta-model with dimensions for:

- normative force: permission, obligation, prohibition;
- conditions and exceptions;
- negation and restriction scope;
- collection/use/processing/sharing/disclosure/storage/retention/destruction actions;
- data object: health record, biospecimen, genetic/genomic data;
- identifiability, coding, de-identification, anonymization, limited data set status;
- recipient/actor and institutional/external/national/international sharing scope;
- purpose/scope, including future research and genomic data sharing;
- withdrawal/control and downstream effect of withdrawal;
- return of results/incidental findings/contact;
- governance/oversight: IRB, privacy board, honest broker, biospecimen use committee;
- agreement requirements: DUA, MTA, LDS, confidentiality agreement;
- temporal scope, duration, retention, destruction, and audit requirements.

This completes the classifier development stage. The next stage should focus on deriving, organizing, and evaluating the reduced consent/data-use meta-model rather than continuing classifier model search.

## Reproducibility pointers

Final subgroup/NLI runner:

```bash
python meaning_preservation/run_feature_subgroup_ablation_experiments.py \
  --roundtrips_csv /path/to/roundtrips.csv \
  --output_dir meaning_preservation/outputs/feature_subgroup_minilm_nli \
  --embedding_model all-MiniLM-L6-v2 \
  --embedding_backend hf \
  --embedding_device cpu \
  --nli_model cross-encoder/nli-deberta-v3-base \
  --nli_device cpu \
  --batch_size 16
```

If NLI or XGBoost is too slow or unavailable, the ablation runner can be run with `--skip_xgb` or without `--nli_model`.
