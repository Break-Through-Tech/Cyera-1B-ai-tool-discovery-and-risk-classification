# Classifiers trained on dataset v2

The new v2 training path fits TF-IDF and frozen sentence-embedding logistic regression classifiers from scratch. It uses only the 200 v2 training examples (152 conventional, 48 AI), not the old catalog or old fitted classifiers. All 12 category heads' labels come from the 48 AI training examples, four per category.

## Fixed training and selection

Predictor input is the whitespace-normalized **description only**. Product names, review metadata, sources, family IDs, feature annotations and labels never enter the features. This reduces direct product-name memorization, but does not remove author-style confounding.

Both model families use `LogisticRegression(C=1, max_iter=2000, random_state=42)` with training-only inverse-frequency class weights and a fixed binary decision threshold of 0.5. TF-IDF uses word unigrams/bigrams, sublinear term frequency and at most 50,000 features. Embeddings use the same frozen `sentence-transformers/all-MiniLM-L6-v2` encoder and pinned revision as v1; encoder weights are never fitted here. Requirements are pinned in `modeling/requirements-embeddings.txt`.

Family selection uses validation macro F1 independently for AI detection and category classification. A tie prefers TF-IDF. There was no hyperparameter search. Training opens train and validation CSVs only; it records the dataset manifest, model hashes and selected families before a separate command opens the test CSV. No refit uses validation or test examples.

| Family | Validation binary macro F1 | Validation category macro F1 |
| --- | ---: | ---: |
| TF-IDF | 1.0000 | 0.8889 |
| Embeddings | 0.9161 | 0.7778 |

**TF-IDF is the selected default for both tasks.** Validation has only 24 examples and just one AI example per category, so this selection is uncertain.

## Frozen test evaluation

One evaluation run compared both pre-fitted families after default selection. The test has 72 examples (36 conventional, 36 AI), including three AI examples per category.

| Family | Binary accuracy | Binary macro F1 | Category accuracy on true AI examples | Category macro F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF (selected) | 98.61% (71/72) | 0.9861 | 86.11% (31/36) | 0.8627 |
| Embeddings | 98.61% (71/72) | 0.9861 | 94.44% (34/36) | 0.9345 |
| Majority conventional baseline | 50.00% (36/72) | 0.3333 | — | — |

Each binary confusion matrix is `[[36, 0], [1, 35]]`, with rows=true and columns=predicted, label order conventional then AI. Embeddings' higher test category score does not change the validation-selected default. Choosing again using these results would tune on the test set.

The selected pipeline gets both detection and, for true AI products, category correct on **66/72 (91.67%)** examples. Conditional category accuracy scores the category model on all true AI rows, including any that the detector misses; this is different from end-to-end accuracy.

The selected detector misses Databricks Genie. The selected pipeline also miscategorizes Taskade AI Agents, Conversica, Recraft, HyperWrite and Polymer. These observations are recorded for review, not used to change the frozen labels, feature definitions or parameters in this release.

These figures measure agreement with assistant source reviews and assistant-authored descriptions. They are **not independently human-verified enterprise accuracy**. Both training and evaluation descriptions were written through the same review process, category support is tiny, and the pretrained encoder may know the products. V1 and v2 use different descriptions, label composition, input features and splits: their scores are not a controlled improvement comparison. No AI-feature availability or governance-risk model is trained here. Model probabilities are uncalibrated.

## Run and predict

```bash
# Install in a separate virtual environment using modeling/requirements-embeddings.txt.
# Train into an empty versioned output directory; existing model files are preserved.
python modeling/train_v2.py train --out /tmp/cyera-v2-reproduction

# Run only after selection; evaluation refuses an existing test_metrics.json.
python modeling/train_v2.py evaluate --out /tmp/cyera-v2-reproduction

# Use saved v2 defaults without loading the encoder when TF-IDF is selected.
python modeling/predict_v2.py --name 'Example storage app' \
  --description 'Store, synchronize and share files across devices.'

python -m unittest discover -s tests
```

Checked-in outputs under `artifacts/classification_v2/` include four fitted classifier files, selection configuration, validation/test reports and per-test predictions. The encoder remains in the ignored local cache; an embedding-selected reproduction requires the pinned encoder download or cache. Model hashes guard against accidentally mixing modified fitted files with saved selection metadata. The dataset's manifest checks frozen input bytes; training rejects protected evaluation families and duplicate descriptions.

The training command refuses nonempty output directories and the evaluation command refuses a second run in the same directory. These are accidental-overwrite guards, not an access-control barrier against deliberately reusing a test set. Reproducing fixed results is acceptable; future feature or parameter development should use training/validation and reserve a new independent benchmark if this test is used to guide changes.

The CLI predicts primary-purpose status and, only for a predicted AI example, its category. Empty descriptions are rejected, and TF-IDF descriptions with no known vocabulary abstain. The provided name is a display field only. Predictions always request review; they do not establish vendor approval or governance risk.
