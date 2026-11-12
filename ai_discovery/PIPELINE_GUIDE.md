# AI discovery pipeline: implementation and next steps

**Project:** AI Tool Discovery and Risk Classification  
**Status:** November 11, 2026  
**Suggested location:** `ai_discovery/PIPELINE_GUIDE.md`

This guide documents the scripts delivered and the runs reviewed so far. Commands
assume the project root is the working directory and the Python virtual environment
is active. Paths describe the intended local project layout, not a verified inventory
of every file on the Windows machine.

## 1. What we have built

The current discovery workflow extracts applications from enterprise activity,
looks for known product identities, researches unresolved products, and uses Gemini
to decide whether each researched product is primarily an AI tool.

The operational scripts are `ingest.py`, `catalog.py`, `enrich.py`, and `classify.py`.
Local-model scripts are retained for experiments and evaluation; they are no longer
required for normal discovery.

The successful latest classification run produced three AI labels and six non-AI
labels for the nine researched applications. An identical rerun returned
`cache=True` for all nine. That verifies decision-cache reuse for that run; it does
not establish accuracy on arbitrary unseen products.

**Current boundary:** `classify.py` processes entries already in the enrichment
cache. It does not extract new activity, rerun catalog matching, or merge known and
newly classified products into a complete enterprise report. That integration is
still to be completed.

## 2. Definitions and project decisions

### AI versus non-AI

For discovery, an AI tool is a product whose primary purpose depends centrally on
AI/ML. A general-purpose product does not become an AI tool merely because it offers
optional AI features.

The current project therefore distinguishes an AI coding assistant from its broader
code-hosting platform. It also distinguishes a general productivity application
from an optional AI assistant available inside that application.

`non_ai` means **not primarily an AI tool**. It does not mean the product has no AI
features, that an employee never used those features, or that the product has no
security/governance risk. Activity duration and transfer volume do not resolve those
questions by themselves.

### Unresolved decisions

`unknown` is an unresolved decision, not a third product type.

| Label | `is_ai_tool` | `classification_status` | Meaning |
|---|---|---|---|
| `ai` | `true` | `classified` | Evidence supports an AI-focused product |
| `non_ai` | `false` | `classified` | Evidence supports a general-purpose product |
| `unknown` | `null` | `unresolved` | Evidence, identity, or a technical issue prevents a decision |

Classified model decisions are not human verification. The script retains
`identity_verified: false`; `identity_supported` records the model's assessment of
available evidence. `needs_review: false` means no unresolved decision was flagged,
not that someone independently checked every source claim.

### Data and taxonomy

The raw Kaggle catalog examined contained **16,762 rows**. References to “13k” refer
to an earlier cleaned catalog, not the raw row count. Raw rows, cleaned rows, loaded
records, effective identities, and distinct products are different counts.

For this project, the supplied Kaggle catalog is treated as an AI catalog and its
categories as project ground truth. This is a project convention, not independent
verification of every listed product.

The combined finalized dataset contains **697 rows: 306 AI and 391 non-AI**. All 306
AI categories were mapped back to Kaggle categories; the 391 non-AI categories were
preserved. The resulting dataset is:

`data/processed/final/dataset_b_final_kaggle_categories.csv`

Non-AI categories need not follow the AI taxonomy for binary classification because
category is not a binary-model input. Synthetic governance fields are also not
inputs to discovery and should not be presented as verified vendor policies.

## 3. Operational scripts

### `ai_discovery/ingest.py` — extract observed applications

**Input:** an enterprise activity CSV, such as the Novatech usage dataset. Required
columns are `application_name` and `domain`. Employee, department, event IDs, and
supported usage metrics are optional.

The script normalizes names/endpoints and groups activity by application identity.
It preserves subdomains and product paths rather than collapsing everything to a
parent domain. It aggregates event counts, employees, departments, and available
usage metrics. Invalid identity rows are reported separately. Events are counted;
this is not event-ID deduplication.

Internally it returns applications, an event-to-application mapping, rejected rows,
and a summary. Missing metrics are not silently treated as measured zero.

**Output:** terminal summary and application listing. **No files are written.**
It does not classify products, perform web searches, or infer AI status from usage.

```powershell
python ai_discovery/ingest.py --input "PATH_TO_ACTIVITY.csv"
```

### `ai_discovery/catalog.py` — match known identities

**Inputs:** activity CSV, AI catalog, and finalized known AI/non-AI dataset.
The AI catalog can use raw Kaggle columns or Dataset B columns. An optional
`--catalog-context` CSV with `tool_id` and `website_url_clean` preserves original
product URLs when a Dataset B domain column lost the path.

The script calls ingestion, indexes catalog identities, and applies product-path,
name/host, and restricted host-only matching. Shared platforms require particular
care: GitHub and GitHub Copilot must not become the same identity merely because
they share a hostname. Conflicting or insufficient matches stay unresolved.

A finalized label can override the AI-catalog assumption for an exact matching
name/authority/path identity. It does not override every product on that host.
Invalid endpoints and unresolved catalog labels are reported as skipped rows.

Possible results include `known_ai`, `known_non_ai`, `unknown`, and `ambiguous`.
Each result includes match method, reason, and available matched catalog metadata,
including category. A present host does not guarantee the specific product is
resolved; there may be multiple products or an incompatible path/name.

**Output:** terminal results and counts; an in-memory result for downstream code.
**No files are written and no API calls are made.**

```powershell
python ai_discovery/catalog.py --activity "PATH_TO_ACTIVITY.csv" --ai-catalog "PATH_TO_AI_CATALOG.csv" --known-tools "data/processed/final/dataset_b_final_kaggle_categories.csv"
```

Use the corrected `catalog.py` from the current discovery bundle. The earlier
matcher could incorrectly associate Excel with Microsoft Office through a suite
root host; suite-host restrictions were subsequently tightened.

### `ai_discovery/enrich.py` — retrieve product evidence

**Inputs:** the same activity and catalog arguments as `catalog.py`. The script
runs known matching and selects unresolved/ambiguous applications plus questionable
suite-host matches for research.

Gemini with Google Search retrieves a report containing:

1. Product identity and correspondence with the observed endpoint.
2. Product description and business purpose.
3. AI functionality and its role in the product.
4. Alternate names/domains.
5. Uncertainty and conflicting evidence.

The report retains source citations, original text offsets, and available search
metadata. It is evidence for later classification, not an independently verified
identity or final AI label. Only product identity information is submitted for
research; employee identifiers and usage metrics are not needed.

**Automatic persistent output:**

`data/cache/enrichment_cache.json`

The version-1 JSON cache has an `entries` mapping keyed by normalized application
name and endpoint. Each entry contains model, prompt hash, retrieval timestamp,
and report. Successful cited reports are cached; request failures do not replace
successful entries. Saves are atomic, but concurrent writers are not supported.

**Other output:** reports and a summary print to terminal. The source catalogs
remain unchanged.

```powershell
python ai_discovery/enrich.py --activity "PATH_TO_ACTIVITY.csv" --ai-catalog "PATH_TO_AI_CATALOG.csv" --known-tools "data/processed/final/dataset_b_final_kaggle_categories.csv"
```

Useful flags: `--dry-run`, `--limit`, `--cache`, `--model`, and `--refresh`.
A matching cache hit reuses research. `--refresh` deliberately researches again.
The project-root `.env` supplies `GEMINI_API_KEY` when a request is needed.

### `ai_discovery/classify.py` — classify researched applications

**Input:** the enrichment cache. This script does not require local model files,
retraining, or a JEV key. It uses Gemini with the project's AI definition.

The classifier considers identity, description, AI functionality, and uncertainty.
It sends compact report text and deduplicated source references instead of large,
repeated citation objects. Full original reports/citations remain in saved results.
Long source URLs may be represented by bounded descriptors in the API input; the
original URLs remain in the stored reports. All report text is preserved.

Two identity modes are available:

| Mode | Behavior |
|---|---|
| Default: `strict_endpoint` | Application name is an unverified hint; ambiguous shared hosts may remain unresolved |
| `--trust-application-names` | Reliable enterprise product names disambiguate compatible shared hosts; contradictory identity evidence still prevents classification |

The trusted-name flag is an explicit assumption about the telemetry source, not a
claim that the domain independently proves which product was used. Do not use it
for names guessed solely from ambiguous hostnames.

For an unresolved decision, `next_action` directs the response:

- `web_research`: make at most one targeted search, then classify again.
- `activity_data`: missing observed paths or other private telemetry require better logs; no web retry.
- `human_review`: retain unresolved status for review.
- `retry_after_error`: resolve an API/setup/response error and rerun.
- `none`: a classification was produced.

A still-unresolved result after the one research attempt is retained for review.
No arbitrary probability threshold is used as a replacement for evidence checks.

**Automatic persistent outputs:**

| Path | Contents |
|---|---|
| `data/cache/ai_classification_cache.json` | Reusable decisions, model/policy fingerprint, evidence, reasons, timestamps, and attempt metadata |
| `data/processed/ai_discovery_classification.json` | Current run's review report, summary, classifications, evidence, API usage, and attempt history |

The report is updated during processing and overwritten by the next run. A limited
run produces a report for that subset, while the decision cache retains other
entries. These are product-level results, not a complete employee-activity report.

```powershell
python ai_discovery/classify.py --trust-application-names
```

Additional flags include `--retry-unresolved`, `--refresh`, `--limit`, `--model`,
`--cache`, `--decision-cache`, `--output`, and `--reviews`.

For reviewed corrections, `--reviews` accepts a CSV with
`application_name,endpoint,is_ai_tool,reason`. Use exact identities printed by the
script, a true/false or 1/0 label, and a nonempty reason. Blank labels are ignored.
Human decisions take precedence and persist across refreshes. They are not written
back into the source catalogs. If the underlying identity changes materially, those
persistent human decisions need deliberate re-review.

## 4. Caching: what is and is not repeated

There are separate caches for research, operational classification, experimental
API comparisons, and local model training. Reusing research does not automatically
mean a classification decision exists for the current policy.

`classify.py` fingerprints source evidence, model, policy/version, and identity
mode. A change invalidates the previous model decision so the new policy is applied.
That is why the first run after the update made nine calls despite cached research.
The subsequent identical run showed nine cache hits and made no classification calls.

Technical failures are included in the review report but not cached as successful
decisions. Unresolved evidence decisions are cached; use `--retry-unresolved` when
ready to retry. `--refresh` intentionally bypasses model-decision reuse. Human
reviews remain authoritative. There is no automatic time-based expiry currently;
stale evidence needs deliberate refresh. Use only one writer per cache file.

`[gemini; cache=True]` means the decision originally came from Gemini and was now
read locally. It does not mean Gemini was contacted on that rerun. Saved API usage
inside a reused record describes its original request, not newly billed usage.

## 5. Experimental and evaluation scripts

### `preprocessing_scripts/compare_ai_models.py`

Trains and compares five binary classifiers on descriptions from the finalized
697-row dataset. Target: `is_ai_tool`. Categories, usage metrics, and synthetic
risk/governance columns are not model features.

TF-IDF transforms text into weighted word/phrase counts; logistic regression learns
a linear binary decision from those features. The embedding alternatives use a
frozen pretrained MiniLM sentence encoder and train a smaller classifier on its
numeric representations. They do not train a language model from scratch.

Related normalized names, hosts, and identical descriptions are grouped to reduce
leakage. The observed split was 556 development rows and 141 held-out test rows.
Five-fold grouped cross-validation on development data selected the winner. Only
that winner was evaluated on the held-out set. Saved estimators were fitted on
development rows, not refitted on the test rows.

**Writes:** fingerprinted runs under `models/ai_detection/runs/`, including
`manifest.json`, `metrics.json`, `comparison.csv`, `cv_predictions.csv`,
`test_predictions.csv`, `split_manifest.csv`, `excluded_rows.csv`, all five named
`.joblib` estimators, and `winner.joblib`. A separate embedding cache stores encoder
and embeddings; its path is recorded in metrics. `--force-retrain` repeats model
training; `--recompute-embeddings` refreshes embeddings. `--validate-only` checks
inputs/splits without writing training artifacts.

The observed run directory was `models/ai_detection/runs/662c36e8dbc264496c68`.
Only load serialized models from a trusted source and compatible environment.

### `ai_discovery/review_ai_predictions.py`

Loads the saved winner and exact training CSV, inspects held-out errors, and predicts
from explicitly extracted Product Description sections in the enrichment cache.
It reports potential overlap with training/test products and available uncalibrated
scores. It does not fall back to feeding the entire report as a description.

**Output:** terminal only. No retraining, API requests, or files written. This was
the diagnostic step that exposed poor predictions on some web-enriched products.

### `ai_discovery/compare_discovery_methods.py`

Compares all five saved models on description-only and description-plus-AI-functionality
inputs. Optional Gemini and JEV methods receive the same evidence variants plus a
classification policy. It imports the helper from `review_ai_predictions.py`, so
keep the two files together. This is an experiment, not the operational classifier.

**Output:** terminal by default; an optional `--output` JSON report. Successful API
responses are automatically cached in `data/cache/discovery_comparison_api.json`.
`--refresh-api` repeats those requests. This cache is separate from the operational
classification cache. There is no retraining or catalog mutation.

Gemini uses `GEMINI_API_KEY`; JEV uses `TYPESAFE_API_KEY`. Optional reviewed labels
permit metrics, including coverage and accuracy among decided cases. Agreement by
itself is not accuracy. Local probabilities are uncalibrated, JEV's probabilities
are provider outputs, and Gemini's explanations are not confidence measurements.

## 6. Model results and why we chose API classification

### Binary classification on the finalized dataset

| Model | Development CV macro-F1 | AI precision | AI recall |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 0.972 | 0.991 | 0.947 |
| TF-IDF + linear SVM | 0.971 | 0.991 | 0.943 |
| Embeddings + logistic regression | 0.958 | 0.974 | 0.930 |
| Embeddings + KNN | 0.938 | 0.920 | 0.943 |
| Embeddings + random forest | 0.937 | 0.982 | 0.877 |

The selected TF-IDF logistic model achieved held-out accuracy **95.74%**, macro-F1
**0.957**, AI precision **93.75%**, and AI recall **96.77%**. Its confusion matrix
contained 75 true negatives, 4 false positives, 2 false negatives, and 60 true positives.

These are strong results on that dataset. They are not a guarantee for a different
input source. Web-generated descriptions differ in wording and content from the
training descriptions; groups can still share broader source/style patterns.

### Comparison on nine web-enriched applications

With description-only input, every tested method missed Cursor, whose description
omitted the decisive AI context. Adding functionality changed the result for the
embedding models and both APIs. TF-IDF still predicted non-AI for Cursor and Midjourney.

Combined-input results were:

| Application | Both TF-IDF models | Embedding models | Gemini and JEV |
|---|---|---|---|---|
| Cursor | Non-AI | AI | AI |
| GitHub | Non-AI | Non-AI | Non-AI |
| GitHub Copilot | AI | AI | AI |
| Google Docs | Non-AI | AI | Non-AI |
| Google Drive | Non-AI | Non-AI | Non-AI |
| Microsoft Excel | Non-AI | AI | Non-AI |
| Microsoft Teams | Non-AI | AI | Non-AI |
| Midjourney | Non-AI | AI | AI |
| VS Code | Non-AI | Mixed: KNN non-AI; others AI | Non-AI |

The embedding models appeared sensitive to AI features without consistently
separating optional features from primary purpose. They were also trained on
descriptions alone, so the combined-input comparison changed their input distribution.

Both API methods followed the intended policy more consistently in this small
sample. Gemini was selected for the prototype because it was already integrated
and supplied inspectable explanations. This was not evidence that Gemini is more
accurate than JEV, or that either is universally superior to local models.

Important limits: only nine applications were compared, some overlap prior data,
and enrichment already includes conclusions such as “AI is central” or “optional.”
Agreement can therefore reflect shared upstream evidence rather than independent
verification. JEV's reported 1.0 probabilities are not guaranteed correctness.

### Operational improvements and latest run

The first operational classifier resolved seven applications and left Docs and
Excel unknown because it required product-specific endpoint evidence. Targeted
searches could not recover missing private URL paths. We added the explicit
trusted-name mode rather than hardcoding either product's label.

The latest trusted-name run classified all nine: three AI and six non-AI, with no
technical errors. Compaction reduced initial input tokens from **92,392 to 20,898**
across nine classifications, approximately **77%**. This comparison concerns initial
classification input tokens; it is not a measured total-billing or latency reduction.
The old run also made extra research/reclassification calls for two unresolved cases.

## 7. Data preparation and older category experiments

These utilities explain how the inputs were built; they are not daily discovery steps.

| Script | Purpose and saved output |
|---|---|
| `prepare_kaggle_dataset_b.py` | Earlier conversion of the raw catalog to Dataset B. The established output was `data/processed/kaggle_dataset_b/kaggle_dataset_b.csv` via `--out-dir`. Its source was not available for this documentation review, so additional sidecar outputs are not asserted here. |
| `fix_final_ai_labels.py` | One-time repair of six missing Boolean labels; writes the specified new CSV and `.label_fixes.json` audit. Do not rerun against already-fixed data. |
| `map_kaggle_categories.py` | Maps AI products to original Kaggle categories while preserving non-AI rows; writes the specified CSV plus `.category_mapping.csv` and `.category_mapping.json` audits. All 306 AI rows mapped in the observed run. |
| `fill_catalog_synthetic.py` | Fills demonstration governance values consistently with matching reference entries; excludes vendor from saved columns. Prints preview by default. Explicit `--output` writes a new CSV and `.synthetic_provenance.json`. Synthetic values are not researched policies. |
| `train_category_model.py` | Historical TF-IDF category classifier. Writes metrics, classification report, CV predictions, exclusions, confusion matrix, top features, and `category_model.joblib` in `--out-dir`. |
| `train_category_embeddings.py` | Historical embedding category classifier; writes classifier, local encoder, metrics/comparison, CV predictions, confusion matrix and report in `--out-dir`. |
| `tune_category_embeddings.py` | Historical category-tuning experiment; retain as experimental. No final tuning metrics are documented here, and its outputs are not consumed by current discovery. |

The earlier category task used **306 AI rows and the outline's 12 categories**, not
the full Kaggle taxonomy. TF-IDF achieved accuracy **0.4085**, macro-F1 **0.3173**,
and weighted-F1 **0.4001**. Embeddings improved those to **0.5556**, **0.5040**, and
**0.5458**, respectively. These were category results, not binary AI-detection results.
They do not evaluate a classifier trained on all Kaggle entries or its 14 categories.
The old category scripts use the original taxonomy and need adaptation before reuse.

## 8. Next steps: Component 3 and integration

### A. Keep the adopted 14-category taxonomy

The vision's 12 labels were examples. We already adopted the original Kaggle labels
and mapped the finalized dataset accordingly. Do not silently switch back.

| Kaggle category | Raw catalog rows |
|---|---:|
| Ai And Machine Learning | 1,320 |
| Business And Finance | 1,053 |
| Communication And Support | 2,022 |
| Creative And Entertainment | 580 |
| Data And Analytics | 765 |
| Education And Learning | 277 |
| Industry Specific | 328 |
| Marketing And Sales | 1,389 |
| Media And Content Creation | 4,289 |
| Operations And Management | 1,567 |
| Personal And Lifestyle | 262 |
| Security And Privacy | 204 |
| Specialized Technologies | 967 |
| Technology And Development | 1,739 |

These counts were supplied in the raw-catalog analysis; all category cells were
nonempty. Define short category descriptions and tie-breaking rules before asking
an API to assign new products, especially for overlapping broad categories.

### B. Implement category assignment after AI detection

Proposed next script: `categorize.py` — **not yet built**.

- Run only for AI-classified products. Non-AI products are not forced into the AI taxonomy.
- For a resolved catalog identity, reuse its existing Kaggle category and record `category_source: catalog`.
- For newly discovered AI products without a resolved category, use the enriched evidence and an explicit 14-label taxonomy to assign a category, with a reason and provenance.
- Preserve unresolved category cases separately from unresolved AI detection; knowing a product is AI does not guarantee its business purpose is clear.
- Cache categories separately with an evidence/taxonomy/model fingerprint. A taxonomy change must invalidate incompatible cached category decisions.

An API categorizer is a practical initial candidate because evidence retrieval and
Gemini integration already exist. It still needs its own evaluation; good binary
classification does not establish good category accuracy.

If revisiting local category training, use **all eligible labeled Kaggle entries**,
not only the 306 sampled AI rows. Retain original descriptions and other appropriate
product text before lookup-only cleaning; empty descriptions cannot train a text
classifier. Validate duplicates and related-product grouping, and reserve a held-out
set. Category labels are the target, not an input feature.

### C. Combine discovery results for the complete activity set

Build an orchestration/reporting step that joins catalog matches and classification
results back to the applications extracted from a particular activity file. Use
stable identity matching and preserve provenance and event mappings. Avoid treating
all historical cache entries as current enterprise activity.

The final report should distinguish known catalog labels, API classifications,
human-reviewed decisions, and unresolved identities, then attach categories only
where applicable. It should retain usage counts and make unknown coverage visible.
This integration is needed before claiming every observed application is handled.

### D. Validate before governance and dashboard use

Review a broader independently labeled set, including unseen AI tools, optional-AI
enterprise products, shared hosts, contradictory names, and insufficient evidence.
Measure AI precision/recall, unresolved coverage, identity errors, category quality,
latency, and actual API usage. Keep evaluation separate from repeatedly inspected
training/test examples.

Risk classification and dashboard reporting remain later work. Synthetic governance
fields must retain their provenance and must not be represented as verified approval,
retention, training-on-customer-data, or security-control facts.

## 9. Practical operating checklist

1. For new activity, run enrichment with that activity and the intended catalogs;
   it performs ingestion and lookup internally before researching selected products.
2. Run `python ai_discovery/classify.py --trust-application-names` when that trust
   assumption is appropriate. Omit the flag for strict endpoint evidence.
3. Inspect the classification report, especially unresolved decisions and next actions.
4. Repeat with unchanged settings to verify cache hits. Refresh only deliberately.
5. Keep the local training/comparison scripts for reproducibility; they are not
   prerequisites for the normal API workflow.
6. Next build categorization and the merged activity report. Do not rerun old
   12-category models as if they implement the adopted Kaggle taxonomy.

Dependencies for operational API calls are `google-genai` and `python-dotenv`.
The current configured Gemini model is `gemini-3.8-flash`, matching the successful
project runs; `--model` permits a different supported model. The project-root `.env`
contains `GEMINI_API_KEY`. No SQL database or separate local model server is required.
