# Binary classifier trained on the corrected full dataset

This experiment prepares new grouped splits and trains TF-IDF + Logistic Regression from scratch on the full corrected team-data export. It preserves source labels, corrections, vendor research, earlier datasets and saved model experiments. It does not train a new category model, call Jev, or change governance scoring.

## Preparation and accounting

Input: `data/processed/corrected_labels_2026-11-05/all_corrected.csv`, 17,292 source records. All input IDs appear exactly once either in a split or in `excluded_records.csv`.

| Preparation outcome | Records |
| --- | ---: |
| Unresolved labels set aside | 36 |
| Missing names/descriptions | 528 |
| Invalid domains | 19 |
| Duplicate normalized name and host | 35 |
| Retained across splits | 16,674 |
| Total input | 17,292 |

Before deduplication, opposite labels sharing an exact normalized name or description are quarantined. None remained in this actual input after the approved review. Same-name/host duplicates with the same label retain one representative: prefer assistant-source-reviewed records, then the longer description, then deterministic ID order. Different names with shared descriptions are grouped rather than automatically collapsed into a single product.

Split groups connect exact normalized names, descriptions, hosts, conservative host families, and known normalized vendor strings. Connections are transitive. Grouping does not assert that unrelated hosted products have the same vendor; grouping shared hosts conservatively keeps them together. Host-family handling includes common multi-part suffixes but is not a full public-suffix or corporate-ownership database. Uncached ownership relationships and unrecognized aliases may remain. Vendor strings are used only for grouping and never as predictor inputs.

Groups are allocated toward 70/15/15 class-count targets using a fixed seed (20261109), based only on counts and group structure—not model scores.

| Split | AI | Conventional | Total |
| --- | ---: | ---: | ---: |
| Training | 11,329 | 343 | 11,672 |
| Validation | 2,428 | 73 | 2,501 |
| Test | 2,428 | 73 | 2,501 |

`data/processed/classification_corrected_full/manifest.json` records input/output hashes, grouping policy, counts and exclusions. The split command refuses to overwrite a changed frozen manifest; use a new version for later source changes. These are newly allocated splits, not newly collected products or independent human-reviewed test labels. Many products were seen during earlier experiments and source review.

## Training and model selection

The predictor reads **description only**, not name, URL, vendor, category, source/review fields, supplied label or governance fields. The vectorizer learns word unigrams/bigrams, sublinear TF-IDF, at most 50,000 features, from training data only. Logistic Regression uses max_iter=2000 and random_state=42. Sample weights balance the two classes based on training frequencies only.

Compare three regularization settings, `C = 0.1, 1, 10`, using validation macro F1. Exact ties prefer smaller C; binary threshold is fixed at 0.5. No additional tuning is performed. Save the selected model and its configuration before a separate evaluation command opens the test split. The selected model is not refitted on validation or test.

| C | Validation macro F1 |
| --- | ---: |
| 0.1 | 0.8892 |
| 1 | 0.9253 |
| **10 (selected)** | **0.9427** |

The training command loads train/validation only. The evaluation command checks the saved model and dataset hashes and refuses an existing test report. These prevent accidental overwrites/re-evaluation, not deliberate test reuse. Reproducing fixed results is acceptable; do not use known test mistakes to tune this experiment.

## Frozen test results

| Metric | Selected classifier | Always predict AI |
| --- | ---: | ---: |
| Accuracy | 99.04% (2,477/2,501) | 97.08% |
| Macro F1 | 0.9243 | 0.4926 |

| Class | Precision | Recall | F1 | Test examples |
| --- | ---: | ---: | ---: | ---: |
| Conventional | 76.92% | 95.89% | 0.8537 | 73 |
| AI | 99.88% | 99.14% | 0.9950 | 2,428 |

Confusion matrix, rows=true and columns=predicted, order conventional then AI:

```text
                 Predicted conventional   Predicted AI
Conventional               70                  3
AI                         21              2,407
```

The detector identifies 70 of 73 conventional examples. Conventional precision is lower because 21 AI-labeled records are predicted conventional. High overall accuracy alone is misleading given the class imbalance. Brier score is 0.00750 and log loss 0.03464; scores remain uncalibrated, and prevalence affects these aggregate probability metrics.

These results measure agreement with primarily source-assigned labels, supplemented by limited assistant source review. They do not establish real-world AI-detection accuracy. Conventional-source descriptions tend to be written differently and may be longer than AI-source descriptions, allowing source-style artifacts to influence classification. Exact grouping does not prove all semantic duplicates/aliases are removed. The 73 conventional test examples still limit precision. Scores are not governance-risk probabilities or proof that conventional products lack AI features.

## Manual diagnostic checks

Two additional handwritten descriptions were tried after evaluation without changing the model. The coding-assistant description is predicted AI (score 0.9876). However, **a plain conventional file-storage description is also predicted AI (score 0.9142), incorrectly**. See `artifacts/corrected_full_binary/smoke_examples.json` for the exact text and outputs.

This failure is material: high agreement with held-out catalog labels does not establish reliable generalization to everyday descriptions. The model remains experimental and requests review. The examples were not used to change settings or labels after test evaluation. More representative independently reviewed descriptions and source-style checks are needed before relying on these predictions.

## Run

```bash
# Frozen split preparation/check, Python standard library only:
python3 preprocessing_scripts/prepare_corrected_classification.py --freeze
python3 preprocessing_scripts/prepare_corrected_classification.py

# Install modeling/requirements.txt in a virtual environment first.
# A new empty output directory is required for reproduction:
python modeling/train_corrected_binary.py train --out /tmp/cyera-corrected-reproduction
python modeling/train_corrected_binary.py evaluate --out /tmp/cyera-corrected-reproduction

# Use the checked-in saved model:
python modeling/train_corrected_binary.py predict \
  --description 'A coding assistant that generates and edits source code from instructions.'

python -m unittest discover -s tests
```

Saved artifacts under `artifacts/corrected_full_binary/` contain the model, versioned selection configuration, all validation candidate metrics, test metrics and per-record test predictions. Empty descriptions are rejected. Descriptions with no vocabulary recognized by TF-IDF abstain and request review; all returned model predictions request review as well.

## Next comparison

Jev has **not** been called. A fair follow-up should use exactly the same description-only inputs and fixed splits, develop Jev instructions/thresholds on train/validation, freeze its configuration, then compare on the same test IDs. Do not expose labels or provenance in its requests. Preserve this Logistic Regression selection and report class-specific metrics, macro F1, confidence reliability, cost and latency. Since this test has now been evaluated, do not adjust either approach using its known errors and describe the outcome as an untouched independent benchmark.
