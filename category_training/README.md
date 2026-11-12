# Component 3: local category-model training

Use Python 3.11 or 3.12 on Windows. This package supplies training code, not trained
models. It consumes the unchanged output of `prepare_category_data.py`.

## What we are comparing

1. TF-IDF + multiclass logistic regression: learns category associations from words
   and short phrases. TF-IDF is fitted separately inside each training fold.
2. Frozen MiniLM embeddings + multiclass logistic regression: uses a pretrained text
   encoder and learns a small classifier on its vectors. It does not train or
   fine-tune the language model.
3. A most-frequent-category baseline, fitted using development labels only.

Both candidate models use exactly the same prepared `model_text` (description +
functionality) and the same groups/partitions. Names, categories, synthetic governance
fields, risk labels, and AI-detection explanations are not added as features.
Category is the target. Naturally occurring product names in text remain present.

## Install and run on Windows

Put this folder at `category_training` in your project root. Use the preparation
output folder from the previous step. Run in PowerShell from your project root:

```powershell
# Use the project-root .venv (create it once if needed)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Or for the pinned category-training set:
# .\.venv\Scripts\python.exe -m pip install -r category_training/requirements-embeddings.txt

.\.venv\Scripts\python.exe category_training/train_categories.py `
  --prepared "data/processed/category_preparation_v1" `
  --out "models/categories/kaggle14_run1" `
  --description-ablation
```

A virtual environment does not need PowerShell activation when invoking its Python
executable directly.
The first embedding run downloads the public MiniLM encoder; no API key or paid API
calls are involved. Training uses CPU by default. `--device cuda` is optional if you
have a working CUDA-compatible PyTorch installation.

For a faster first experiment with only TF-IDF, install `requirements.txt` instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r category_training/requirements.txt
.\.venv\Scripts\python.exe category_training/train_categories.py `
  --prepared "data/processed/category_preparation_v1" `
  --out "models/categories/kaggle14_tfidf_run1" `
  --models tfidf `
  --description-ablation
```

To verify prepared files before installing packages or training:

```powershell
python category_training/train_categories.py --prepared "data/processed/category_preparation_v1" --out "models/categories/check" --validate-only
```

Validation checks file hashes against the preparation manifest, taxonomy hash,
feature contract, text hashes, eligibility, unique IDs, and identical-text group
consistency. It writes nothing and does not create the supplied output directory.

## Evaluation design

- The first fold of a shuffled, seeded 5-fold `StratifiedGroupKFold` reserves about
  20% of products/groups as the held-out test partition. Exact size depends on groups.
  No test fold is chosen by its performance.
- The remaining development rows use 3-fold grouped cross-validation.
- Each model compares C = 0.5 and 2.0, with and without balanced class weights.
- The winner and each model's hyperparameters are locked using pooled development
  out-of-fold macro-F1, before computing test predictions. `selection.json` records
  this decision. CV scores used for selection are not unbiased final scores.
- Each chosen configuration is then fitted on development rows and evaluated once
  on the same test partition. Both test results are reported for comparison; the
  winner is NOT changed based on them.
- Reports include per-category precision/recall/F1/support, confusion matrices,
  frequent confusion pairs, individual errors, task-only versus description-bearing
  subsets, and macro-F1 with inverse-group-size weights as a secondary diagnostic.
  Standard row-level macro-F1 remains the selection metric.
- The optional description ablation uses development data only and fixed TF-IDF
  hyperparameters selected for combined text. It compares description-only versus
  combined predictions on the same description-bearing validation rows; blank
  descriptions are excluded. It is a diagnostic, not another tuned contender.
- `tfidf.joblib`, `embeddings.joblib`, and `winner.joblib` are fitted on development
  data only. The test rows are not used in a hidden final refit.

If a partition lacks a category, the script stops rather than silently changing the
evaluation. Inspect groups or reduce `--cv-folds`/`--outer-folds`. A convergence warning
also stops training. Increase `--max-iter` and use a new output folder if needed.

## Outputs in the selected --out directory

| File | Purpose |
|---|---|
| `run_config.json` | Configuration, input hashes, script hash, Python/package versions. Written at start. |
| `split_manifest.csv` | Every source ID, group, label, development/test assignment and development validation fold. |
| `embedding_manifest.json` | Encoder source/content fingerprint, cache location, encoding version and count of chunked products. Only for embedding runs. |
| `model_comparison.csv` | Development CV results for all tested configurations. |
| `selection.json` | Winner and chosen hyperparameters, locked before test evaluation. |
| `cv_predictions.csv` | Out-of-fold development predictions for each model's selected configuration. |
| `metrics.json` | Complete test metrics, majority baseline, subsets, optional description ablation and limitations. |
| `per_category.csv` | Test precision, recall, F1 and support for each model/category. |
| `tfidf_confusion_matrix.csv`, `embeddings_confusion_matrix.csv` | Test confusion counts; rows are actual labels, columns predicted labels. Only requested models appear. |
| `confusion_pairs.csv` | Non-diagonal mistakes sorted by count. |
| `test_predictions.csv` | Every test prediction, actual label, input text, group, top-three suggestions and uncalibrated scores. |
| `test_errors.csv` | Incorrect test predictions for inspection. |
| `tfidf.joblib`, `embeddings.joblib` | Fitted model bundles for requested models; embedding bundle also records its encoder location/hash. |
| `winner.joblib` | Copy of the CV-selected model bundle. |
| `taxonomy.json` | Exact Kaggle-14 taxonomy copied from preparation. |
| `manifest.json` | Completion marker, run fingerprint and artifact hashes. Written last. |

The separate `data/cache/category_embeddings` directory contains `encoder/`,
`encoder_source.json`, and fingerprinted `.npy` embedding matrices plus `.json`
metadata. Encoder files and matrices are hashed before reuse. Do not run concurrent
writers against one cache. Choose a new `--embedding-cache` for a different encoder
or to recover from an interrupted encoder download. Trained `.joblib` outputs,
encoder weights, and this cache are gitignored — train locally; GitHub keeps the
code and prepared inputs, not the model binaries.

Long inputs are token-chunked at the encoder's context length, embedded, averaged
and normalized, rather than silently dropping the end of a functionality report.
Future embedding inference must reuse `encode_texts()` and the same saved encoder.
The pretrained encoder is frozen; precomputing embeddings does not fit anything
to test labels. MiniLM pretraining exposure to public products is unknown.

## Reuse and inference boundary

An identical completed training command verifies artifact hashes and exits without
retraining. A different or incomplete run requires a new output directory. No source
CSV, discovery result, or preparation output is modified. Dependencies' exact versions
are recorded; embedding transitive dependencies are resolved by pip, not fully locked.

Routine inference should load the trusted local `winner.joblib` once and call its
estimator. For embeddings, load the fingerprinted encoder locally and run the same
chunking code. Moving an embedding bundle requires moving its encoder or updating
the future inference loader's explicit path; the `.joblib` alone is not sufficient.
Only load joblib files you created/trust, in a compatible environment.

This delivery does not yet merge discovery/catalog decisions, implement an
acceptance threshold, or call an API fallback. Uncalibrated top scores are not
verified confidence. Before operational categorization, the adapter must preserve
non-AI/unresolved cases, resolve identity, reuse catalog categories, and preserve
category provenance. Training remains a separate command.

## What results can establish

These experiments test prediction of the supplied Kaggle taxonomy. They do not
prove accuracy on independently reviewed discovery reports. Duplicate-name label
conflicts, unknown aliases and near-duplicate paraphrases remain documented data
limitations; grouped splits reduce known leakage but do not eliminate every form.
Do not tune using test errors and continue calling that test set untouched.

After running, share `model_comparison.csv`, `metrics.json`, `per_category.csv`, and
`confusion_pairs.csv`. Those are enough to assess the first comparison. A local-vs-API
decision additionally needs reviewed discovery-style examples across categories.

## Validation status of this package

The script's syntax, prepared-input validation against all 16,751 supplied eligible
rows, and dependency-free guard tests were checked. The delivery environment lacks
scikit-learn/sentence-transformers and cannot download packages, so classifier
fitting and the real MiniLM path could not be executed here. No measured model
accuracy or successful end-to-end training run is claimed. The included smoke test
exercises a small real TF-IDF run after you install the dependencies:

```powershell
.\.venv\Scripts\python.exe category_training/smoke_test.py
```
