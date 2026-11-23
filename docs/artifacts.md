# Prototype and artifact guide

This guide labels the artifacts added or used by PR #22. Filenames and the team's original results are preserved. **Model predictions, source labels, synthetic governance settings and functional test fixtures serve different purposes and must not be presented as independently verified truth.**

## Working prototype

**Artifact: AI Governance Console — runnable synthetic enterprise prototype.**

- Entry point: [`dashboard/app.py`](../dashboard/app.py).
- Setup and acceptance walkthrough: [`dashboard/README.md`](../dashboard/README.md).
- Recording sequence: [`docs/dashboard_demo.md`](dashboard_demo.md).
- Before PR merge, use branch `recording-dashboard`; after merge, use the corresponding `main` version.
- Core modes: saved company investigation/overview, CSV activity analysis, filters and company-scoped exports. Core viewing and upload analysis need no API key or trained category model.
- Optional local questions: ChromaDB and a local MiniLM encoder retrieve evidence from the selected company report. The documented setup includes a pinned, explicit encoder download. Gemini-generated answers are disabled in the dashboard.
- Unfamiliar products without saved evidence remain unresolved in upload mode. The newly connected online Jev adapter is separately enabled and was tested with simulated responses; a new live end-to-end provider run is not claimed.

## Data, predictions and company demonstrations

| Artifact label | Location | What it contains and how to interpret it |
|---|---|---|
| **Input data — synthetic company activity** | [`data/raw/novatech_usage.csv`](../data/raw/novatech_usage.csv), [`data/raw/summit_financial_usage (1).csv`](../data/raw/summit_financial_usage%20%281%29.csv) | Fictional employee/application events. NovaTech: 30,000 events / 21 apps. Summit: 25,120 events / 22 apps. These are simulations, not real enterprise telemetry. |
| **Reference data — source-assigned application labels** | [`data/processed/kaggle_dataset_b/`](../data/processed/kaggle_dataset_b/), [`data/processed/final/dataset_b_final_kaggle_categories.csv`](../data/processed/final/dataset_b_final_kaggle_categories.csv) | Operational product metadata, source labels and category annotations. Security/approval fields can be fictional or missing. These are not a trusted binary test set. |
| **Predictions — full-catalog Jev AI/non-AI** | [`data/processed/jev_catalog/all_classified.csv`](../data/processed/jev_catalog/all_classified.csv) | 17,107 normalized identities: 9,814 AI, 1,412 conventional, 5,881 unknown/review. Preserves original labels separately, plus probabilities, model version and review indicators. The 1,052 resolved source-label disagreements are review candidates, not automatic corrections or measured errors. |
| **Run metadata — catalog scope and usage** | [`input_manifest.json`](../data/processed/jev_catalog/input_manifest.json), [`summary.json`](../data/processed/jev_catalog/summary.json) | Source hashes, classification policy, 0.70 prototype review threshold, completion counts and reported tokens. The cost field is a public-price estimate, not an account billing statement. These files contain no model weights. |
| **Predictions — researched 21-product sample** | [`data/processed/pipeline_runs/novatech_jev_classification.json`](../data/processed/pipeline_runs/novatech_jev_classification.json) | Saved Jev decisions: 9 AI / 12 conventional. Inputs include earlier Gemini-assisted product research, unlike the description-only catalog run. This is not an independent Jev-versus-Gemini comparison. |
| **Company report — reviewed synthetic demo** | [`data/processed/dashboard_runs/novatech_reviewed/`](../data/processed/dashboard_runs/novatech_reviewed/), [`summit_reviewed/`](../data/processed/dashboard_runs/summit_reviewed/) | Team reports with usage, reviewed purposes, governance rules, evidence gaps and scenario provenance. The dashboard preserves their original detection sources. Real product names do not make fictional security settings real vendor claims. Other `dashboard_runs` directories retain earlier team-generated snapshots. |
| **Review records — purpose categories** | [`data/reviews/categories_2026-11-21/`](../data/reviews/categories_2026-11-21/) | The team's provided category review records. A `human_review` source identifies those records; it does not establish an independent audit of every field in the application report. |
| **Advisory predictions — Jev purpose-category opinions** | [`data/processed/dashboard_runs/jev_second_opinion.json`](../data/processed/dashboard_runs/jev_second_opinion.json) | Category suggestions for saved company applications. These are advisory and do not replace assigned categories or determine AI/non-AI status or governance scores. |

### Jev catalog CSV field labels

| Fields | Meaning |
|---|---|
| `jev_label`, `jev_is_ai_tool` | Effective prediction: `ai` / 1, `non_ai` / 0, or `unknown` / blank. Unknown is not a negative example. |
| `raw_choice`, `p_ai`, `p_non_ai`, `p_unknown` | Provider choice and its probabilities. A binary raw choice below the prototype 0.70 threshold becomes an effective unknown. These are not governance-risk probabilities. |
| `needs_review`, `review_reason` | Marks an unknown, a below-threshold prediction, or a resolved disagreement requiring review. |
| `source_labels`, `source_label_conflict`, `disagrees_with_source` | Preserved source annotations and comparison flags. Source disagreement does not establish which answer is correct. |
| `application_id`, `application_name`, `endpoint`, `source_row_count` | Normalized identity and number of source records represented by the deduplicated row. |
| `returned_model`, `classified_at`, `input_fingerprint` | Model version, run timestamp and fingerprint of the model/policy/evidence input. |

## Evaluation, models and report search

| Artifact label | Location | What it establishes and what it does not |
|---|---|---|
| **Evaluation — purpose categorization** | [`artifacts/jev_category_eval/`](../artifacts/jev_category_eval/), [`docs/jev_category_evaluation.md`](jev_category_evaluation.md) | Team comparison on 154 held-out tools using noisy reference categories: macro-F1 about 0.59 for Jev and 0.48 for the local category model. This evaluates purpose labels, not binary AI detection, and is not real-world security accuracy. |
| **Test fixtures — synthetic answer contracts** | [`data/reviews/assistant_validation_2026-11-18/`](../data/reviews/assistant_validation_2026-11-18/) | Two fictional reports plus 26 developer-authored expected behaviors. Used to test counts, company boundaries, missing evidence and abstention. Not an independently reviewed answer-quality benchmark. |
| **Test fixtures — semantic search relevance** | [`data/reviews/retrieval_2026-11-18/queries.json`](../data/reviews/retrieval_2026-11-18/queries.json) | Six named-application search checks against a fixed report. A 6/6 result is a small smoke check, not evidence of broad retrieval quality. |
| **Database — local derived evidence store** | [`ai_discovery/retrieval.py`](../ai_discovery/retrieval.py); local `data/cache/governance_retrieval/` | ChromaDB schema-3 collection, evidence snapshots, embeddings and manifest. Built from completed company reports, not all catalog products. CLI databases can persist; dashboard question stores are temporary and closed after answering. Derived database files are excluded from Git. |
| **Model — frozen text encoder** | Local `artifacts/category_embeddings/encoder`; setup in [`dashboard/README.md`](../dashboard/README.md) | Pretrained MiniLM generates embeddings. We do not train or fine-tune its weights in this feature. The pinned model is downloaded explicitly if needed and is not committed. |
| **Model — trained purpose classifier** | [`category_training/`](../category_training/), local `models/categories/`; [`PIPELINE_GUIDE.md`](../ai_discovery/PIPELINE_GUIDE.md) | Team category-training/inference package. Saved company views and exact saved-purpose reuse work without its local weight bundle. Regeneration involving new local category inference requires that bundle and its documented environment. |
| **Historical experiments** | Existing `artifacts/` subdirectories and linked method guides | Earlier binary/category metrics and experiment outputs retain their original scope. Do not substitute them for the current model's evaluation or reinterpret old hashes as a new ChromaDB run. |

## Labels in the interface and exports

- The interface identifies the run as a **synthetic prototype**, distinguishes saved reviewed reports from fresh upload analysis, and displays detection/category sources.
- **Jev AI/non-AI probabilities** describe product classification. **Jev category opinions** describe purpose. Neither is a governance risk score or independently measured accuracy.
- Company CSV exports include **AI label source**, **Purpose label source**, **Usage data provenance**, **Contains synthetic evidence** and **Assessment**.
- Missing provenance remains unknown; a blank synthetic-evidence flag does not establish real-world data.
- Complete report JSON retains per-application reasons, company scope, provenance and missing-evidence fields. Company downloads do not include the full reference catalog.
- Tests are executable behavior checks, not trained models. Report-search examples and code-test totals must not be called model accuracy.

## Reproducibility and remaining validation

On November 23, 2026, the core dashboard was verified in a fresh Python 3.12 virtual environment installed only from `dashboard/requirements.txt`: both saved company views and a 21-application sample upload passed without retrieval packages, category weights, or provider calls. The full development environment also passed the 205-check suite.

The prototype's checked-in demonstrations and catalog predictions can be inspected without regenerating paid results. See the setup guide for a fresh environment and sample-upload walkthrough; see the method guides for the explicit commands that regenerate particular artifacts.

Stronger independently reviewed binary Jev/local-model evaluation, representative real-world validation and live validation of the newly connected provider path remain separate work. Gemini-generated report answers remain disabled in this dashboard. No claim of production readiness or verified vendor security is made.
