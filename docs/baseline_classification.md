# Baseline classification models — issue #6

This implements the first binary AI-detection and business-purpose category baselines on the dataset merged in PR #11. The classifiers in this change were trained from the existing data; the earlier Gemini pipeline supplied model-reviewed labels, not these trained classifiers.

## Models and targets

- **Binary detection:** word TF-IDF + Logistic Regression. The target is `is_ai_tool`: `1` for an AI-focused tool and `0` for conventional primary application purpose.
- **Business-purpose categorization:** a separate TF-IDF + Logistic Regression classifier trained on the existing 12-category annotations for AI tools.
- **Comparison:** an always-majority-class predictor reports what accuracy looks like without useful discrimination.

Risk is a separate task under issue #8. No approval status, vendor security metadata, employee usage or risk label is used as a feature or predicted here.

Two binary training variants are compared:

| Variant | Reviewed AI training rows | Provisional AI training rows | Non-AI training rows | Total |
|---|---:|---:|---:|---:|
| `reviewed_only` | 213 | 0 | 22 | 235 |
| `full_catalog` | 213 | 14,457 | 22 | 14,692 |

“Reviewed” means accepted by the repository's existing Gemini text review, not independently human-verified. The 30 non-AI examples come from the team's restored seed dataset and label conventional primary purpose; they do not establish that a platform has no AI features.

Both variants use the same 50 validation and 50 test applications (46 AI and four non-AI per split). Group isolation is checked again before fitting. Provisional catalog membership never supplies evaluation labels.

## Training and selection

The only input feature is `text_for_model`, composed of name and description in the preparation step. Each variant fits its own TF-IDF vocabulary on its own training rows. Categories, filenames, source labels, risk metadata and split membership are excluded from input text.

Fixed settings:

- TF-IDF word unigrams/bigrams, sublinear term frequency, minimum document frequency 1 and at most 50,000 features.
- Logistic Regression: C=1.0, max_iter=2000, random_state=42.
- Binary row weights: 1.0 for reviewed/seed records and 0.25 for provisional catalog members, then normalized to equal total effective weight per binary class.
- Category row weights: inverse category frequency on the 213 annotated AI training records.
- Binary threshold: 0.5. Model scores have not been calibrated.

Select the binary variant with the highest validation macro F1. Ties prefer `reviewed_only`. Test results are generated after selection; test records never fit the vectorizer or classifier. The final files keep the training-only fitted models, without refitting on evaluation sets. The earlier exploratory run used the same split and fixed settings; these are reproducibility/development results, not a fresh independent benchmark.

## Results

| Binary variant | Validation macro F1 | Test macro F1 | Test accuracy | Test non-AI recall |
|---|---:|---:|---:|---:|
| Majority-class predictor | 0.479 | 0.479 | 92% | 0% (0/4) |
| Reviewed-only Logistic Regression | **0.823** | **0.770** | 94% | 50% (2/4) |
| Full-catalog Logistic Regression | 0.684 | 0.684 | 94% | 25% (1/4) |

The default prediction variant is `reviewed_only`, chosen on validation macro F1. Additional provisional positives did not improve this fixed baseline on validation; the comparison does not establish why.

Selected model's test confusion matrix (rows = supplied labels, columns = predictions):

| | Predicted non-AI | Predicted AI |
|---|---:|---:|
| Supplied non-AI | 2 | 2 |
| Supplied AI | 1 | 45 |

The initial category model has 34.8% validation accuracy and 30.4% test accuracy, with 0.235 test macro F1 across all 12 training categories. Its category evaluation uses 46 supplied AI examples per split. Meeting / Transcription Tools has no support in either held-out category set. These results are conditional on an AI label, not end-to-end detection-plus-category accuracy.

`artifacts/baseline/metrics.json` contains per-class precision, recall, F1, support, balanced accuracy and confusion matrices for both binary variants and category evaluation. Test prediction CSVs identify individual binary errors. `model_config.json` records exact settings, library versions and hashes of training/validation/test inputs.

## Run or reproduce

Use Python 3.12 and the pinned modeling dependencies. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r modeling/requirements.txt

# Reproduce in a fresh directory using the merged dataset:
python modeling/train_baseline.py --out artifacts/reproduced-baseline
python -m unittest discover -s tests -v
```

The default training output is `artifacts/baseline`. Existing nonempty output directories are protected unless `--overwrite` is supplied. `--data` can select another prepared directory containing train.csv, validation.csv and test.csv.

Predict with the included models:

```sh
python modeling/predict.py --name "Example Writing Assistant" --description "An AI assistant that generates marketing copy and drafts articles from prompts."

python modeling/predict.py --name "Example File Archive" --description "A cloud file storage service for sharing documents and organizing team folders."
```

Use `--models artifacts/reproduced-baseline` for a different model directory. `--binary full_catalog` explicitly selects the alternate binary model. The default comes from `model_config.json`.

Prediction returns `is_ai_tool`, an uncalibrated `model_ai_score`, and, for predicted AI tools, a separate category suggestion and category model score. Results are marked `needs_review=true`. Text with no known vocabulary produces a null detection and an explanation. A concrete description is required; a bare hostname does not reveal unseen-tool functionality.

`notebooks/full_classification_baseline.ipynb` runs the same preparation/training/prediction scripts. Repeated notebook runs overwrite only the named `artifacts/notebook_run/` outputs. It does not duplicate the model implementation.

## Verification and limitations

The full suite includes 15 preparation tests and model tests for source-aware class weighting, train-only vocabulary, validation-based selection, saved-model metric reproduction, missing/OOV input handling, and separate detection/category outputs. All model input hashes match the committed CSVs. The preparation step already accounts for all source catalog records; its full audit remains in `classification_full/catalog_audit.csv`.

The labels are weak supervision, not independently verified truth. Four non-AI examples per evaluation split make recall unstable: one example changes it by 25 percentage points. A majority predictor already gets 92% test accuracy, so the selected model's 94% accuracy should be read alongside its macro F1 and per-class errors. Different source writing styles may confound classification, and identity grouping does not resolve all aliases or semantic near-duplicates.

Category performance is weak and predictions require review. This change establishes a reproducible starting point for issue #7's embedding comparison; expanding/auditing the negative class and an independent unseen-tool evaluation remain important before claiming reliable real-world detection.

Technical references: [Logistic Regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html), [classification metrics](https://scikit-learn.org/1.7/modules/model_evaluation.html), [avoiding data leakage](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage).
