# Full classification dataset preparation — issue #5

This implements dataset preparation using the available team data. It does not train a model or implement governance risk scoring. All labels remain weak supervision; dataset preparation does not make them independently verified ground truth.

## Dataset accounting

The full catalog contains **16,762** records. The negative seed contributes another **30** conventional enterprise applications recovered from the team's Git history.

| Catalog outcome | Rows |
|---|---:|
| Included as weak AI examples | 14,762 |
| Missing name/description | 530 |
| Invalid domain | 19 |
| Flagged dead site | 1,247 |
| Existing review rejected | 90 |
| Existing review needs review | 105 |
| Unreviewed overlap with negative seed | 7 |
| Related to held-out applications; quarantined | 2 |
| **Original catalog total** | **16,762** |

Reasons are mutually exclusive and follow the ordering in the preparation script. For example, a rejected review is excluded before website status is considered. No source row is silently lost. Both the complete audit and the exclusion/quarantine subset are included.

| Split | Model-reviewed AI | Provisional catalog AI | Team non-AI | Total |
|---|---:|---:|---:|---:|
| Train | 213 | 14,457 | 22 | 14,692 |
| Validation | 46 | 0 | 4 | 50 |
| Test | 46 | 0 | 4 | 50 |
| **Total** | **305** | **14,457** | **30** | **14,792** |

The 70/15/15 split is applied to independent groups containing model-reviewed/seed records. Provisional-only groups are used for training, not evaluation. Records related to held-out applications are quarantined, preventing their aliases or duplicate descriptions from entering training. Thus the full training share is deliberately greater than 70%.

The 23 model-suggested non-AI records from the catalog review are not used as new confirmed negatives: absent AI language is not sufficient proof that a product is non-AI. The negative seed remains the team's 30 labeled examples.

## Meaning of the columns

The dataset retains the starter schema documented in `classification_dataset.md`: application and group IDs, name, domain, description, text_for_model, is_ai_tool, category, category_source, label_source, label_quality, source_record_id, source_file and split.

- `is_ai_tool`: `1` for an AI-focused tool; `0` for conventional primary application purpose.
- `category`: a business-function annotation from the accepted review. It is blank for unreviewed catalog entries and non-AI rows. No full-catalog categories were invented.
- `label_source`: `gemini_text_review`, `provisional_catalog_membership` or `team_non_ai_seed`.
- `label_quality`: `weak_supervision` for all records.
- `text_for_model`: name plus description, composed identically for positives and negatives. This is the only classifier input.

Risk, approval, SSO, retention and customer-data training policies are not predictors or targets in this pipeline. Historical raw negative records contain governance fields; these are not verified vendor facts and never enter the prepared classification dataset.

## Cleaning and splitting

The script normalizes whitespace and URL hostnames, requires names and descriptions, checks domain validity and source status flags, deduplicates normalized name/hostname application identities, and rejects contradictory accepted AI/negative identities. Existing review exclusions override catalog membership; unreviewed records overlapping a negative seed name, hostname or description are excluded as ambiguous.

Applications sharing a normalized name, hostname or exact normalized description are linked transitively. Entire connected groups stay in one split. Approximate 70/15/15 allocation applies to groups containing reviewed/seed anchors. Provisional-only groups enter training; provisional records connected to validation/test applications are quarantined. Seed 42 and stable sorting make outputs reproducible.

These checks do not resolve every rebrand, alias, parent/subdomain relationship or semantic near-duplicate. `site_status_group` is an inherited historical flag, not a fresh availability check. Missing descriptions are excluded, not invented. Unknown categories remain blank.

## Run and verify

Python 3.9+ and the standard library are sufficient. From the repository root:

```sh
python3 preprocessing_scripts/prepare_full_classification.py
python3 -m unittest discover -s tests -v
```

The scripts find default input paths relative to their location. To reproduce in a new directory:

```sh
python3 preprocessing_scripts/prepare_full_classification.py --out /path/to/new-directory
```

Use `--overwrite` only to deliberately replace generated outputs. `--catalog`, `--reviews`, `--non-ai` and `--seed` are configurable. Inputs are not modified. The starter builder `prepare_classification.py` is also included; it uses only the 305 accepted AI and 30 negative records and reproduces the older `data/processed/classification/` outputs.

## Files and provenance

Inputs:

- Existing `data/processed/kaggle_clean.csv` (16,762 records).
- Existing `data/processed/dataset_b/sample/review_results.csv` (500 reviews).
- Restored `data/raw/non_ai_enterprise_tools.csv` (30 records), exactly recovered from commit `87f760769e39e89ba3bcb1691d8f0eb32c60e4b2:data/non_ai_enterprise_tools.csv`.

Outputs under `data/processed/classification_full/`:

- `classification.csv`, `train.csv`, `validation.csv`, `test.csv`.
- `catalog_audit.csv`: all 16,762 source catalog records with dispositions and reasons, including the 2,000 excluded/quarantined records.
- `summary.json`: counts, checks and target definition.
- `manifest.json`: SHA-256 input hashes, seed, feature and split policies.

The restored raw negative source includes historical governance fields. They are not verified vendor policies and are excluded from prepared classification data. Categories, label provenance, filenames and risk metadata must not be used as binary model features. Use only `text_for_model` for the initial baseline and fit learned preprocessing on training records only.

## Validation and limits

Fifteen passing preparation tests cover the real source counts, input-order-independent reproducibility, all-row accounting, heldout isolation and quarantine, duplicate removal, conflicting identities, missing/invalid fields, review exclusions, negative-overlap exclusions, minimum class availability, and separate feature/target provenance. All generated full-data artifacts also reproduce byte-for-byte.

Both evaluation sets contain only four negative applications, and their positive labels come from Gemini, not independent human review. One error changes non-AI recall by 25 percentage points. The corpus is heavily imbalanced and source writing styles may differ. These are development splits for a baseline, not proof of reliable real-world detection. A conventional platform's non-AI label concerns primary purpose and does not assert that it has no AI features.

The dataset is prepared for binary classification and retains existing category annotations for future categorization. Independent label audits and a broader negative class remain important improvements, while issue #6 handles baseline modeling separately.
