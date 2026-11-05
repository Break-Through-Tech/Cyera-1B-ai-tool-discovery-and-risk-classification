# Sentence embeddings and semantic similarity — issue #7

This adds a frozen sentence-transformer experiment, trained classification heads and similarity-based matching on top of the issue #6 baselines. It uses the same prepared data, primary-purpose target, four source-review exclusions, sample-weight policy and held-out applications. Issue #6 does not need retraining to use this experiment.

## What runs

`sentence-transformers/all-MiniLM-L6-v2` maps the existing name-plus-description text to 384-dimensional unit-normalized vectors. The exact model revision is pinned to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The encoder is frozen; it is not fine-tuned on project records. Logistic Regression classification heads are fitted only on project training vectors.

The binary target remains `is_ai_tool`: 1 means an AI-focused application under its described primary purpose; 0 means a conventional primary-purpose application. It does not establish whether an ordinary platform has AI capabilities or whether an employee used an AI feature. Business-purpose categories remain a separate prediction; no governance risk is calculated.

Experiments:

- Reviewed-only head: 209 model-reviewed AI examples plus 22 team negatives (231 total).
- Full-catalog head: 14,688 training rows, including provisional catalog positives.
- Category head: the 209 model-reviewed AI training examples with the existing 12-category taxonomy.
- Cosine neighbors: match query descriptions against the 231 reviewed/seed training examples. No validation/test records are in this reference index.

Classifier settings match the baseline: C=1.0, max_iter=2000, random_state=42. Binary source weights and inverse-category weights are reused. The reviewed-only/full-catalog variant is selected on validation macro F1. Then the default family is selected by comparing its validation macro F1 with the selected TF-IDF baseline. Ties preserve TF-IDF. Category family selection uses validation category macro F1 separately. Test metrics never determine these choices.

## Results on the existing development holdout

Each split contains 50 applications: 46 AI and four non-AI. The same evaluation set was inspected in earlier work, so these are development comparisons, not a fresh independent benchmark.

| Binary method | Validation macro F1 | Test macro F1 | Test accuracy | Non-AI test recall |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.479 | 0.479 | 92% | 0/4 |
| TF-IDF, reviewed-only | 0.823 | 0.770 | 94% | 2/4 |
| Frozen embeddings, reviewed-only | **0.939** | **0.939** | **98%** | **4/4** |
| Frozen embeddings, full catalog | 0.823 | 0.923 | 98% | 3/4 |
| One nearest training neighbor | 0.939 | 0.770 | 94% | See per-class metrics |

The selected default binary model is **embeddings / reviewed_only**, chosen on validation. Its test confusion matrix is:

| Supplied label | Predicted non-AI | Predicted AI |
|---|---:|---:|
| Non-AI | 4 | 0 |
| AI | 1 | 45 |

| Category method | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF | 34.8% | 0.210 | 28.3% | 0.205 |
| Frozen embeddings | **60.9%** | **0.545** | **54.3%** | **0.451** |

The default category family is also embeddings, selected on validation macro F1. Category evaluation is conditional on supplied AI labels (46 examples per split), not end-to-end discovery accuracy. Macro F1 includes all 12 training categories; Meeting / Transcription Tools has no support in either held-out category set. This can produce a scikit-learn warning when predictions include that absent class; it does not mean evaluation rows entered training.

## Unseen-tool test and semantic matches

All validation/test application groups are absent from the classifier training and similarity index. Here, **unseen means unseen by the project classifiers**. The pretrained encoder may have encountered product names or public descriptions during pretraining; its pretraining corpus was not audited for overlap.

For every held-out query, `heldout_neighbors.csv` records the top three training references and cosine scores. A one-neighbor binary agreement metric is reported as an additional similarity experiment. It is not a measured human relevance score or evidence of AI functionality.

Three fictitious-name demonstrations, excluded from training and evaluation:

| Description supplied | Detection / category | Closest training reference |
|---|---|---|
| AI code completion, explanations and debugging | AI / Coding / Developer Tools | Code Fundi, cosine 0.629 |
| AI meeting transcription, summaries and actions | AI / Meeting / Transcription Tools | Metaview, cosine 0.640 |
| Cloud file storage, sync and document sharing | Conventional primary purpose | Dropbox, cosine 0.718 |

These show the pipeline works beyond exact name/domain lookup. They are demonstrations, not independent test examples. Full returned results are in `example_predictions.json`. Cosine similarity measures relatedness, not a calibrated probability, approval status or risk.

## Run

Python 3.12 is recommended. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r modeling/requirements-embeddings.txt

# The first use downloads the pinned public encoder; no paid API key is needed.
python modeling/train_embeddings.py --out artifacts/reproduced-embeddings
python -m unittest discover -s tests -v

# Use the included classifier heads and reference index:
python modeling/predict_embeddings.py --name "ExampleCodeHelper" --description "An AI coding assistant that suggests code completions and helps debug software."
```

`--family auto` (default) uses the validation-selected binary and category families; `--family embeddings` or `--family tfidf` explicitly runs that family for comparison. `--top-k` controls the number of similar training references. Use `--models artifacts/reproduced-embeddings` for reproduced artifacts.

The default baseline dependency is `artifacts/baseline/`; `--baseline` can select another directory. The baseline dataset/review hashes must match the embedding experiment exactly. A changed baseline config is rejected at prediction time until the comparison is reproduced. This prevents silently comparing incompatible runs.

The encoder and full-catalog vectors are cached under `artifacts/embedding_cache/`, which is ignored by Git. Public pretrained weights are not committed to this repository. Inference runs locally after the first download. `--cache` chooses another writable location. `--overwrite` is required to replace an existing nonempty model output directory.

The runnable notebook `notebooks/embedding_classification.ipynb` invokes these same scripts. It writes its model outputs to `artifacts/notebook_embeddings/` and can reuse the encoder/vector cache.

## Files and verification

`artifacts/embeddings/` contains the two binary heads, category head, reference vectors and records, model config, metrics, held-out predictions, neighbor matches and demonstration outputs. The config pins the encoder revision, package versions, dataset hashes, review version, exclusions and validation-selected defaults. The reference index includes only eligible reviewed/seed training applications.

33 tests pass, including existing preparation/model checks and new tests for cosine normalization/ranking, malformed or zero vectors, cache invalidation, validation-only selection, pinned model/input provenance, classifier dimensions and held-out identity exclusion. Three real-encoder inference demonstrations also ran successfully. No dataset labels were changed for this experiment.

## Interpretation and remaining limits

Embeddings improved both detection and category metrics in this fixed development experiment. Four negatives per split are too few to estimate reliable enterprise performance; one error changes negative recall by 25 percentage points. Labels remain weak, source styles may differ, some categories are sparse and predictions are uncalibrated. No performance guarantee or statistical significance is established.

The small reviewed reference index limits similarity coverage. It is a functional-similarity component for discovery, not the vector-database/RAG pipeline in issue #9. The next project component is issue #8's explainable governance risk engine, which must combine tool metadata with enterprise usage context separately from AI detection.

References: [Sentence Transformers similarity documentation](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). The model card describes its 384-dimensional representation, intended use and truncation limits.
