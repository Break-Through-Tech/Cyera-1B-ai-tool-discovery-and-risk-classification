# AI discovery pipeline: implementation and next steps

**Project:** AI Tool Discovery and Risk Classification  
**Status:** November 15, 2026 - Components 1-4 integrated in the Component 4 delivery.

## Latest validated status (Component 4 v2)

Components 1-3: 30,000 events, 21 applications, 9 AI (4 catalog categories and
5 provisional model predictions), 12 non-AI uncategorized, no unresolved detections.
Prediction caching was verified in the original local run.

Component 4 now accepts synthetic controls and activity in its PRIMARY prototype
score, per VISION.md. It reuses explicitly supplied governance catalogs by exact
name/endpoint, applies fictional-company scenarios, and optionally researches
missing vendor controls with citations. Approval remains company-scoped. Existing
risk_label values are never reused as assessments. AI-only checks stay conditional.

52 targeted tests pass. Offline NovaTech with both catalogs, synthetic approval
adoption and the NovaTech scenario yields 9 complete and 12 incomplete assessments
(13 low, 5 medium, 3 high), preserving all upstream records and ingestion. These
are fictional scenario scores. Online research is implemented and fake-tested;
a live Gemini run remains to be validated. Grammarly's Security And Privacy
category is still flagged for review, not silently changed.

See [Component 4 v2 contract and commands](../docs/component4_integration.md).
This section supersedes historical Component 4 boundaries and verified-only
scoring descriptions elsewhere in older notes.

This guide documents the scripts delivered and the runs reviewed so far. Commands
assume the project root is the working directory and the Python virtual environment
is active. Paths describe the intended local project layout, not a verified inventory
of every file on the Windows machine.

## 1. What we have built

The current discovery workflow extracts applications from enterprise activity,
looks for known product identities, researches unresolved products, and uses Gemini
to decide whether each researched product is primarily an AI tool.

The operational scripts are `ingest.py`, `catalog.py`, `enrich.py`, and `classify.py`.
Binary local-model scripts remain experimental and are not required for normal
discovery. Component 3 now has separate data-preparation and local-training scripts;
these do not change the operational binary AI-detection policy. `categorize.py` adds purpose categories; `run_pipeline.py` connects Components 1-4.

The successful latest classification run produced three AI labels and six non-AI
labels for the nine researched applications. An identical rerun returned
`cache=True` for all nine. That verifies decision-cache reuse for that run; it does
not establish accuracy on arbitrary unseen products.

**Current boundary:** `classify.py` processes entries already in the enrichment
cache. It does not extract new activity, rerun catalog matching, or merge known and
newly classified products into a complete enterprise report. The current `run_pipeline.py` combines Components 1-4 for an activity input. `report.py` retains its legacy governance workflow; use the new runner or `governance.assess_pipeline` for the new evidence contract.

### Team status at a glance

| Area | Current state | Next action |
|---|---|---|
| Application extraction and catalog lookup | Implemented in the supplied scripts; results are returned in memory. | Preserve identity and event mappings when integrating. |
| Evidence retrieval and AI detection | Implemented; latest supplied report has 3 AI and 6 non-AI decisions. | Broaden validation; preserve unknown decisions and provenance. |
| Category data preparation | Run completed: 16,751 eligible rows using Kaggle's exact 14 categories. | Keep the prepared files and manifest together. |
| Category model training | Completed locally; TF-IDF won development selection, held-out macro-F1 0.735. | Broaden independent discovery-evidence evaluation. |
| Category inference and prediction cache | Implemented in `categorize.py`; catalog, model and human provenance. | Run locally against saved discovery/model artifacts. |
| Combined activity run | `run_pipeline.py` connects Components 1-4 and preserves ingestion. | Collect verified governance evidence; dashboard integration remains separate. |

Section 8 is the current Component 3 handoff. Sections 6–7 retain historical results
for context; they are not performance claims for the newly delivered Kaggle-14 models.

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

## 8. Component 3: category preparation and training

**Status as of November 12, 2026:** preparation has been run and its outputs reviewed.
The new training package is delivered, but no completed Kaggle-14 training run or
model-quality results have been reviewed. Operational category inference and the
merged activity report are still integration work.

### A. Deliverables and setup

- [category_preparation.zip](https://chatgpt.com/library?file_id=libfile_02dfa2af7ca881918bae2e8e0bb0c149&account_id=personal): `prepare_category_data.py` and its README. Put the script in `preprocessing_scripts/`.
- [category_training.zip](https://chatgpt.com/library?file_id=libfile_e1e025888b288191948bb04c305ff177&account_id=personal): extract the enclosed `category_training/` folder into the project root. Contains `train_categories.py`, dependency files, `smoke_test.py`, and a README.

These links identify the supplied packages; teammates need access or a copy of the
packages in the shared project repository. The commands below use project-relative
paths. The catalog path matches the reviewed local preparation manifest. Do not
commit API keys, virtual environments, or downloaded encoder caches.

### B. Adopted taxonomy and actual preparation results

The vision's 12 labels were examples. The original **14 Kaggle categories** are
authoritative, with spelling preserved exactly. Older 12-category models are not
compatible with this task.

| Category | Raw rows | Eligible training rows |
|---|---:|---:|
| Ai And Machine Learning | 1,320 | 1,320 |
| Business And Finance | 1,053 | 1,053 |
| Communication And Support | 2,022 | 2,020 |
| Creative And Entertainment | 580 | 580 |
| Data And Analytics | 765 | 763 |
| Education And Learning | 277 | 277 |
| Industry Specific | 328 | 328 |
| Marketing And Sales | 1,389 | 1,387 |
| Media And Content Creation | 4,289 | 4,288 |
| Operations And Management | 1,567 | 1,565 |
| Personal And Lifestyle | 262 | 261 |
| Security And Privacy | 204 | 204 |
| Specialized Technologies | 967 | 966 |
| Technology And Development | 1,739 | 1,739 |
| **Total** | **16,762** | **16,751** |

The uploaded preparation `audit.json` and `manifest.json` confirm:

- The working Dataset B catalog has **16,743 rows**; all passed normalized-name/host checks against raw row-derived IDs. This check does not independently verify product identity.
- Training preparation starts from the **raw Kaggle data**, not just the 306 sampled AI tools. It excludes **11 exact name-and-endpoint matches labeled non-AI** in the catalog/reference.
- **16,221** eligible rows contain description plus primary-task text; **530** have only primary-task text. Task-only rows remain explicitly marked.
- There are **160 repeated normalized-name groups**, including **88 with conflicting categories**, and **19 exact cleaned-description duplicate groups**. These signals drive conservative split grouping, not automatic alias resolution.
- The preparation audit counts **16,372 groups before exclusions**. The earlier validated eligible input has **16,361 groups**; group counts and row counts are different.
- The discovery preview has **3 AI cases ready for a category-model input preview** and **6 non-AI cases with no model input**. A preview is not a category prediction or proof that catalog lookup is unnecessary.

The 697-row reference file is used for audit and exact-identity non-AI exclusions,
not appended to category training. Most of its AI products overlap Kaggle, so it is
not an independent category test set. Counts above are preparation findings, not
model scores.

### C. What the category classifier will actually receive

| Purpose evidence | Training source | Discovery source |
|---|---|---|
| Description | `Short Description` | Explicit Product Description report section |
| Functionality | `Primary Task` | Explicit AI Functionality report section |

`model_text` is cleaned description followed by cleaned functionality, separated by
a newline. `description_only_text` is retained for a controlled comparison. The
category is the target, never an input feature. Keywords, synthetic governance,
risk labels, usage metrics, and AI-detection explanations are excluded. Product
names naturally present inside descriptive prose are not removed; names/domains
are not separately added as features.

The preparation script's `build_model_text()` and `extract_report()` define the
shared text contract: `purpose-description-functionality-v1`. Future inference must
use the same cleaning/extraction logic. It extracts explicit purpose sections and
retains evidence offsets/citations separately; it never substitutes the entire
research report when a section is missing or ambiguous.

Kaggle task phrases and web-generated functionality reports are only approximate
equivalents. Training does not synthesize longer descriptions to hide this source
difference. Discovery-style validation is still required. No category model is
being used to decide whether an application is AI.

### D. Prepare or reproduce the dataset

Preparation requires Python 3.10+ and no additional packages. Run from the project
root; use a new output directory if preparation already exists:

```powershell
python preprocessing_scripts/prepare_category_data.py `
  --raw "data/raw/kaggle.csv" `
  --catalog "data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv" `
  --reference "data/processed/final/dataset_b_final_kaggle_categories.csv" `
  --discovery "data/processed/ai_discovery_classification.json" `
  --out "data/processed/category"
```

`--discovery` expects the classification report containing a `results` array, not
the enrichment cache. It can be omitted when unavailable. Raw and output paths are
required; catalog/reference arguments should be supplied for the reviewed audits.
The default `--min-words 1` retains meaningful short tasks such as “Image upscaling”;
very short text is flagged. No source datasets or discovery decisions are changed.

| Preparation output | Contents |
|---|---|
| `training_rows.csv` | Eligible rows, feature text, target, source IDs, quality flags, hashes and `group_id`. |
| `excluded_rows.csv` | Excluded rows and explicit reasons. |
| `group_links.csv` | Pairwise signals connecting leakage-control groups. |
| `label_conflicts.csv` | Label disagreements, with exclusion versus grouping/review actions. |
| `reference_overlap.csv` | Catalog/reference overlap candidates, distinguishing name, host and endpoint evidence. |
| `catalog_join_audit.csv` | Checks catalog IDs against raw names/hosts and records paths/category agreement. |
| `discovery_input_preview.json` | Extracted purpose text, original offsets/citations, preliminary detection gate and potential training overlap; no assigned categories. |
| `taxonomy.json` | Exact 14 labels, taxonomy version and hash. |
| `audit.json` | Counts, category balance, duplicate/conflict summaries and limitations. |
| `manifest.json` | Input/output hashes, preparation settings and text contract. |

Confirmed same-identity or identical-complete-text category conflicts are excluded.
Name-only category conflicts are retained within the same split group and flagged;
a repeated name is not proof of an identical product. Shared-platform roots do not
automatically group unrelated products. Unknown aliases and near-duplicate
paraphrases are not automatically detected in preparation v1 and remain evaluation
risks. Do not split these rows randomly without respecting `group_id`.

### E. Train and compare models locally

The new training script compares **TF-IDF + multiclass logistic regression** and
**frozen MiniLM embeddings + multiclass logistic regression**, plus a majority-label
sanity baseline. Neither option fine-tunes a language model. Both use the same
prepared text and product-grouped partitions.

Use Python 3.11 or 3.12. PowerShell commands from the project root:

```powershell
python -m pip install -r category_training/requirements-embeddings.txt

python category_training/train_categories.py `
  --prepared "data/processed/category" `
  --out "models/categories/kaggle14_run1" `
  --description-ablation
```

Use your already activated Python environment (3.11 or 3.12 for the pinned training dependencies). The first embedding run downloads the public MiniLM encoder;
there is no paid API call or API key. CPU is the default.

For input validation without packages, downloads, or file writes:

```powershell
python category_training/train_categories.py --prepared "data/processed/category" --out "models/categories/check" --validate-only
```

After dependency installation, the small synthetic TF-IDF smoke test is:

```powershell
python category_training/smoke_test.py
```

For TF-IDF only, install `category_training/requirements.txt`, pass `--models tfidf`,
and choose a separate output directory. The full package README includes that command.

**Evaluation implemented and run locally; reviewed results follow below:**

1. Reserve the first fold of seeded, shuffled 5-fold `StratifiedGroupKFold` as the test set (about 20%). No group crosses the development/test boundary.
2. Use 3-fold grouped cross-validation on development rows. Compare C = 0.5 and 2.0, with and without balanced class weights. Fit TF-IDF within each training fold.
3. Select both model configurations and the winner using development out-of-fold macro-F1. Lock this choice before computing test predictions.
4. Fit the chosen configurations on development data and report both on the same held-out test set. Do not change the winner based on those test results.
5. Report per-category precision/recall/F1/support, confusion patterns, individual errors, task-only subsets, and group-weighted macro-F1 as a secondary diagnostic.

`--description-ablation` adds a development-only TF-IDF diagnostic comparing
description-only versus combined evidence on the same description-bearing validation
rows, using fixed combined-selected hyperparameters. It is not a separately tuned
third contender. Saved models use development rows only; there is no hidden refit
on the held-out test set.

Long embedding inputs are token-chunked, embedded, averaged and normalized so the
end of a functionality report is not silently discarded. Future embedding inference
must reuse `encode_texts()` and the same saved encoder.

| Training output in `--out` | Contents |
|---|---|
| `run_config.json` | Settings, input/script hashes, Python and package versions. |
| `split_manifest.csv` | Row/group IDs, labels, development/test partitions and CV folds. |
| `embedding_manifest.json` | Encoder fingerprint/location and chunking information; embedding runs only. |
| `model_comparison.csv` | Development CV metrics for all candidate configurations. |
| `selection.json` | Winner/configurations chosen before test evaluation. |
| `cv_predictions.csv` | Selected models' out-of-fold development predictions. |
| `metrics.json` | Test metrics, baseline, subsets, optional ablation and limitations. |
| `per_category.csv` | Test precision, recall, F1 and support per model/category. |
| `tfidf_confusion_matrix.csv`, `embeddings_confusion_matrix.csv` | Actual-label rows and predicted-label columns for requested models. |
| `confusion_pairs.csv` | Non-diagonal confusions sorted by frequency. |
| `test_predictions.csv`, `test_errors.csv` | Predictions/errors with source identity, text and uncalibrated top-three scores. |
| `tfidf.joblib`, `embeddings.joblib` | Requested models fitted on development data. |
| `winner.joblib` | Copy of the CV-selected model. |
| `taxonomy.json` | Taxonomy copied from preparation. |
| `manifest.json` | Completion marker, run fingerprint and artifact hashes; written last. |

The separate `data/cache/category_embeddings/` contains the local encoder,
`encoder_source.json`, and fingerprinted `.npy` matrices/JSON metadata. These are
feature caches, not category-decision caches. Keep the encoder with an embedding
model; its `.joblib` alone is insufficient. Do not run concurrent cache writers.

An identical completed training command verifies its artifacts and exits without
retraining. Changed or incomplete runs require a new output directory. Normal
inference must load a saved model and must not invoke training. Operational inference and caching are now delivered separately in `categorize.py`.

The delivered scripts were checked here with standard-library guards. The user
subsequently ran full training locally: 13,403 development rows and 3,348
held-out rows, with product grouping. TF-IDF was selected by development CV;
held-out macro-F1 was **0.735**, accuracy **0.761**. MiniLM + logistic regression
achieved macro-F1 **0.603**, accuracy **0.650**. The description-only diagnostic
was substantially weaker than description plus task, so transfer to discovery
evidence still needs broader evaluation. These are the new Kaggle-14 results,
not the older 12-category experiments.

The subsequent discovery review predicted Technology And Development for Cursor
and GitHub Copilot, and Media And Content Creation for Midjourney. All six inputs
marked non-AI stayed uncategorized. Cursor had development overlap; none was
detected for the other two. There were no human category labels, so this run is
an integration check, not an independent accuracy estimate.

### F. Operational categorization and combined runner

Copy the `ai_discovery/` and `tests/` contents into the same folders in your repository.
This adds `categorize.py`, `run_pipeline.py`, and tests; includes the unchanged
`review_discovery_categories.py` dependency; and updates `PIPELINE_GUIDE.md`.
It does not replace `classify.py`, `catalog.py`, `enrich.py`, `report.py`,
`pipeline.json`, models, source CSVs, or existing discovery decisions.

Run every command from the repository root with your selected environment.
Training remains a separate operation. Neither command below trains a model.

#### Categorize the discovery results you already have

```powershell
python -m ai_discovery.categorize `
  --discovery "data/processed/ai_discovery_classification.json" `
  --model-dir "models/categories/kaggle14_run1" `
  --out "data/processed/category_results/run1"
```

Defaults use these catalog files:

- `data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv`
- `data/processed/final/dataset_b_final_kaggle_categories.csv`

Use `--catalog` and `--known-tools` to supply different files. The first accepts
raw Kaggle or Dataset B; the second is the finalized reference and overrides
the first only for the exact same normalized product name AND endpoint.
Dataset B `is_ai_tool` labels are respected. Raw Kaggle is AI by the project's
catalog assumption. Unrecognized labels remain unresolved.

`--catalog-context PATH` optionally supplies `tool_id,website_url_clean` for the
first catalog, restoring product paths lost from Dataset B domains. No shared
domain, fuzzy name, alias guess, or URL-prefix match alone establishes identity.
Query/fragment identities remain pending. Non-default ports and path case remain
significant. This policy favors missed catalog reuse over wrong-product reuse.

The standalone command covers only records in its discovery input. Catalog-known
products absent from that file appear when you use the activity runner below.

#### Run Components 1-3 together using saved research decisions

```powershell
python -m ai_discovery.run_pipeline `
  --activity "data/raw/novatech_usage.csv" `
  --discovery "data/processed/ai_discovery_classification.json" `
  --model-dir "models/categories/kaggle14_run1" `
  --out "data/processed/pipeline_runs/novatech_run1"
```

This is offline: extracts activity, resolves exact catalog identities, joins saved
discovery by name AND endpoint, and categorizes confirmed AI. All extracted apps,
usage aggregates, event mappings and rejected rows are preserved. Historical cache
entries not in this activity do not become current usage. Without `--discovery`,
offline detection uses catalog identities only; other apps stay pending.

Existing discovery decisions remain authoritative input, including unresolved
decisions; the runner does not silently replace them with catalog labels. A
catalog/discovery label conflict is recorded and blocks categorization.

#### Allow online discovery in the same command

```powershell
python -m ai_discovery.run_pipeline `
  --activity "data/raw/novatech_usage.csv" `
  --discovery "data/processed/ai_discovery_classification.json" `
  --online `
  --trust-application-names `
  --out "data/processed/pipeline_runs/novatech_online1"
```

Use `--trust-application-names` only if the activity names reliably identify
products; omit it for strict endpoint identity assessment. `--online` may incur
Gemini/search costs. Only product names and sanitized endpoints/evidence go to
the existing services, not activity users or usage totals.

The runner uses existing `enrich.enrich_applications` and invokes
`python -m ai_discovery.classify` on a scoped cache of current unresolved targets.
Resolved catalog labels and saved classified decisions avoid AI-detection calls.
Confirmed AI with no catalog category and no reports can receive purpose research
without replacing its detection decision. Failed research remains pending.
Multiple reports require an explicit selection; no last-report-wins policy.

Set `GEMINI_API_KEY` in your existing `.env` or environment. API model defaults to
the value used by the supplied scripts; override with `--api-model` if needed.
Online dependencies: `google-genai` and `python-dotenv` already used by discovery.

Successful enrichment updates `data/cache/enrichment_cache.json` through the
existing cache implementation. AI decision caching operates on a new snapshot,
not by overwriting your existing decision cache. To reuse a completed combined
run next time, supply its `categorized_applications.json` as `--discovery` and,
if it was produced, its `discovery_decision_cache.json` as `--decision-cache`.
There is no automatic refresh of accepted saved detection decisions; explicit
reassessment remains in `classify.py`.

#### Category rules and provenance

| Case | Result |
|---|---|
| Confirmed non-AI | Null category; `not_applicable` |
| Unresolved detection / technical error | Null category; `pending_ai_detection` |
| Unsupported identity | Null category; `pending_identity_review` |
| Catalog disagrees with discovery AI decision | Null category; `pending_detection_conflict` |
| Valid explicit human category review | `assigned`, source `human_review` |
| Exact product match with one valid catalog category | `assigned`, source `catalog` |
| Conflicting or out-of-taxonomy catalog categories | `pending_catalog_category_review` |
| Confirmed AI, no catalog category, sufficient purpose evidence | `provisional_prediction`, source `model` |
| Missing/ambiguous evidence or no known TF-IDF vocabulary | Explicit pending/review status, null category |

All category fields live inside `categorization`; original detection fields and
reports remain in the output. The taxonomy preserves the original 14 spellings.
There is no category API fallback in this release. An API category adapter, if
later evaluated and implemented, must use source `api`; the Gemini AI-detection
source must not be confused with the source of a purpose category.

Purpose features are only the existing explicit Product Description + AI
Functionality extraction contract. Governance, risk and category labels are not
prediction features. Top-three scores are uncalibrated; no arbitrary acceptance
threshold is imposed. Human reviews take precedence over catalog *categories*,
but cannot override non-AI/unresolved detection or catalog detection conflicts.

Fill a copy of `category_review_template.csv`, using exact category spellings,
reviewer and reason, then run again with `--reviews PATH` and a new `--out`.
Supply the review file on future runs too; accepted reviews are not silently
stored as model predictions. Reviews are bound to the full reports hash and are
rejected when evidence changes. For multiple reports, fill zero-based
`report_index` in `report_selection_template.csv` and supply
`--report-selections PATH`; selection is also bound to that reports hash.
Use `review_discovery_categories.py` for blind model evaluation: the operational
review template can contain catalog/model text already seen by the reviewer.

#### Every output

Both commands require a new `--out` directory and write:

| File | Contents |
|---|---|
| `categorized_applications.json` | Full input decisions/reports plus separate category status, provenance, catalog record IDs/source hashes, selected evidence, model run/hash and top-three scores. Activity runner also includes ingestion and online outcomes. |
| `categories.csv` | Flat application/detection/category/status/provenance table with evidence hash and cache-hit flag. |
| `category_review_template.csv` | Editable evidence-bound category review fields; blank labels are ignored. |
| `report_selection_template.csv` | Only cases needing a choice among reports, with evidence hash. |
| `taxonomy.json` | Exact 14 labels and taxonomy version. |
| `manifest.json` | Input/script hashes and metadata for the saved model if prediction was needed. |

The activity runner also writes `pipeline_summary.json` (ingestion, detection
and category counts). When online AI classification runs it additionally writes:

| File | Contents |
|---|---|
| `scoped_enrichment.json` | Existing-service cache format restricted to current research targets submitted for detection. |
| `discovery_classification.json` | Original `classify.py` output for those targets, including unresolved/error records. |
| `discovery_decision_cache.json` | New decision-cache snapshot, seeded from `--decision-cache` if present; input cache is untouched. |

Outside the output directory, `data/cache/purpose_categories.json` caches only
model predictions. Override with `--cache`. Fingerprints include product
identity, complete research evidence, extracted text, detection context, catalog
hashes, model bytes, taxonomy and runtime/extraction/encoding code. Changed
inputs invalidate reuse. No simultaneous writers to either JSON cache.

#### Dependencies and checks

Existing required helpers: `preprocessing_scripts/prepare_category_data.py`,
`category_training/train_categories.py`, and the included review helper.
The training module is imported only for encoding helpers; its training entry
point is never called. Routine inference needs no prepared training CSVs.

TF-IDF uses the saved training environment. Your model was saved with
scikit-learn 1.7.2; if necessary:

```powershell
python -m pip install "scikit-learn==1.7.2"
python -m unittest discover -s tests -p test_component3_pipeline.py
```

The runtime validates model/taxonomy hashes and text contract before inference,
and sklearn version before deserialization. Load only trusted local joblib files.
Embedding inference additionally requires the existing sentence-transformers
dependencies and local encoder snapshot; `--encoder-dir` can relocate it, with
content verification. No encoder download or retraining happens here.

Validation in the delivery environment: standard-library tests cover catalog
identity restrictions, conflicting categories, detection gates, human review,
evidence selection, cache reuse/invalidation, activity joining and input/output
protection. Prediction tests use a test double. Real saved-model inference and
live Gemini calls must be exercised locally; they were not rerun here.

#### Existing release boundary

`classify.py` handles AI detection; it was never the entire activity pipeline.
`report.py` is the older combined governance/release command. This delivery
adds a complete Components 1-3 runner without replacing its governance behavior.
The new runner does not consume `pipeline.json`, its corrections file, binary
model, synthetic policy overlays or older category artifact. Pass intended
catalogs explicitly when defaults differ from your active release. Governance,
dashboard and deployment integration with this new result schema remain work
for a later step. Do not claim the new command recomputes risk.

The independently uploaded `catalog.py` and `report.py` have different historical
function signatures; this runner calls ingestion directly and does not depend
on that interface. This observation is about uploaded snapshots, not a claim
that your current repository is broken.

### G. Next integration work

1. Run the new offline command with the actual saved model and review the full
   activity output, especially pending identities and conflicting categories.
2. Use the optional online command for missing evidence and unresolved detection;
   preserve saved snapshots and do not treat failures as non-AI.
3. Collect independently reviewed discovery examples across the 14 categories.
   Do not tune repeatedly against the original held-out products.
4. Integrate this result schema with the existing governance/report/dashboard
   path, preserving risk evidence and keeping synthetic controls visibly marked.
5. Evaluate a category API fallback only if broader review demonstrates value.

No production acceptance threshold is set. Model predictions remain provisional.
Human reviews resolve category questions; category review never changes the
underlying AI-detection decision. Training and preparation remain separate from
routine inference. Existing datasets and detection decisions are not rewritten.

## November 18 extension: report retrieval and analyst questions

Saved Component 4 reports can now be indexed locally with `python -m ai_discovery.retrieval` and queried with `python -m ai_discovery.ask`. Company scope is required; exact metadata filters are separate from semantic ranking. Default answers use report evidence and citations without paid calls. Optional Gemini drafting is explicitly enabled and not live-validated in this extension. This does not change Components 1–4 scores or predictions. See [the report-assistant guide](../docs/governance_assistant.md). The dashboard remains paused.
