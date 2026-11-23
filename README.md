# Cyera — AI Tool Discovery & Risk Classification

Investigate application usage at a fictional enterprise: identify AI tools, suggest their purpose, summarize usage, explain governance concerns, and retrieve the evidence behind findings. The unified dashboard preserves the team’s reviewed company reports, investigation views, heatmaps and trends, and adds offline CSV analysis and local evidence questions. Jev has processed all 17,107 unique identities in the current catalog (9,814 AI-focused, 1,412 conventional, 5,881 unknown/review). The separate 21-app enterprise demo remains available; viewing saved results makes no paid calls.

## Try the dashboard

For PR #22, run from `recording-dashboard`; use `main` after the PR is merged.

```bash
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

See [dashboard setup and modes](dashboard/README.md) and the [dashboard recording guide](docs/dashboard_demo.md) for input requirements, a walkthrough, and the distinction between saved decisions and fresh activity analysis. No API key is needed for this offline demo.

## Prototype and artifact labels

This PR contains a runnable Streamlit prototype. Start with the [setup and acceptance walkthrough](dashboard/README.md), then use the [artifact guide](docs/artifacts.md) to distinguish synthetic demo reports, source-assigned labels, Jev predictions, purpose-category evaluation and local ChromaDB evidence. The core demo works without paid provider calls; optional evidence questions require local retrieval dependencies and an encoder.

## Current workflow

```mermaid
flowchart LR
    A[Synthetic enterprise activity] --> B[Ingestion and catalog matching]
    B --> C[Optional evidence retrieval and AI classification]
    C --> D[Purpose categorization]
    D --> E[Governance evidence and explainable scoring]
    E --> F[Saved Component 4 report]
    F --> G[Local ChromaDB semantic search]
    G --> H[Company-scoped analyst questions and citations]
    H --> I[Optional generated RAG draft]
```

- **Components 1–4:** Sadit's current operational workflow is `ai_discovery/run_pipeline.py`. Read [the pipeline guide](ai_discovery/PIPELINE_GUIDE.md) and [Component 4 contract](docs/component4_integration.md). The report preserves exact identities, accepted/rejected events, category provenance, governance evidence and missingness. Online discovery/research are separately enabled; they are not automatic defaults.
- **Grounded answer validation:** [26 functional cases and strict fact selection](docs/answer_validation.md), including unknown scores, control predicates and abstention.
- **Evidence search and analyst questions:** `ai_discovery/retrieval.py` and `ai_discovery/ask.py` consume saved Component 4 reports. They do not rerun paid discovery, retrain models or change scores. Read [the assistant guide](docs/governance_assistant.md).
- **Earlier local-model backend:** `ai_discovery/report.py` and `pipeline.json` retain the November 11 experimental binary/category workflow. Its compatibility with current catalog code is restored, but it is not the latest operational Components 1–4 runner. Read [that historical release](docs/backend_release.md).

## Where code lives

| Location | Job |
|---|---|
| `ai_discovery/ingest.py`, `catalog.py` | Extract observed applications and match exact known identities |
| `ai_discovery/jev.py`, `jev_catalog.py` | Jev product classification for enterprise samples and the full current catalog |
| `ai_discovery/enrich.py`, `classify.py` | Optional cited product research and AI/non-AI decisions |
| `ai_discovery/categorize.py`, `run_pipeline.py` | Category inference/cache and integrated Components 1–4 |
| `governance/risk.py`, `evidence.py`, `research.py` | Scoring, company-scoped evidence and optional control research |
| `ai_discovery/retrieval.py`, `ask.py`, `evaluate_retrieval.py` | Index report evidence, answer questions and run labeled search checks |
| `category_training/` | Current larger Kaggle-14 category training package; see the pipeline guide |
| `modeling/` | Earlier binary/category experiments and optional dependency lists |
| `preprocessing_scripts/` | Data preparation, correction, enrichment and comparison utilities |
| `data/` | Source data, prepared tables, review evidence and saved pipeline runs |
| `artifacts/` | Historical model metrics/reports and small validation examples |
| `dashboard/app.py`, `data.py`, `workflow.py` | Unified interface, team report views, saved Jev lookup, uploads and report questions |
| `ai_discovery/dashboard_data.py` | Operational ingestion/catalog/governance adapter for fresh uploaded activity |
| `tests/` | Automated behavior and consistency checks |
| `docs/`, `notebooks/` | Methods, limitations, history and exploratory analysis |

Trained weights, encoders, virtual environments and derived vector indexes are **local artifacts**, excluded from Git. Do not force-add them. Preserve run manifests and metrics; train locally or supply your existing encoder/model bundles. Different training experiments have different environments and data contracts.

## Try report search without paid calls

Install the optional search dependencies in a dedicated environment. You need an existing local Sentence Transformers encoder bundle; index building does not download one.

```bash
python -m pip install -r modeling/requirements-retrieval.txt

python -m ai_discovery.retrieval \
  --report data/processed/pipeline_runs/novatech_component4_v2.json \
  --encoder /path/to/local/MiniLM/encoder \
  --out data/cache/governance_retrieval/novatech-chroma-v3

python -m ai_discovery.ask \
  --index data/cache/governance_retrieval/novatech-chroma-v3 \
  --company NovaTech --question 'Why was Salesforce flagged?'
```

Default answers are local evidence summaries with citations, not LLM-generated prose. Optional `--generate --api-model YOUR_MODEL` enables a Gemini RAG draft and may incur cost. Live generation remains unvalidated in this delivery. Company/department/status filters, synthetic controls, incomplete evidence and whole-application user-count scope remain visible. Similarity is relevance, not verification or security confidence.

## Results

**Shadow-AI case (VISION.md §4).** In Summit Financial's 25,120 events, **DocuMind AI** was in no catalog. Online discovery researched it and classified it as an AI document-analysis tool. Governance scored it **high (55)**: unapproved, used only by Finance (84 employees), and uploading 2.6× more per event than the company's typical app. Saved run: `data/processed/dashboard_runs/summit_reviewed/`.

**AI detection.** Jev processed all 17,107 catalog identities: 9,814 AI-focused, 1,412 conventional and 5,881 unknown/review. It disagreed with 1,052 source labels ([details](docs/jev_classification.md)). The local binary model's 99.4% test score is measured against source-assigned labels, not human review.

**Purpose categories (14 Kaggle categories).** All scores are measured against Kaggle's own labels, which are noisy:

| Model and input | Macro-F1 |
|---|---:|
| Local TF-IDF, description **plus** Kaggle task phrase (held-out, 3,348 tools) | 0.734 |
| Local TF-IDF, description only (same 154 held-out tools as below) | 0.477 |
| **Jev, description only** (154 held-out tools) | **0.591** |

The 0.734 is inflated. Kaggle's task phrase maps to a category almost one-to-one, and a plain lookup of it scores 85.8% accuracy. Description-only text is what a newly discovered app actually has. On that text Jev beats the local model (McNemar p = 0.004; better in 12 of 14 categories). Below 0.5 probability, Jev was right only 29% of the time, so those cases go to human review ([evaluation](docs/jev_category_evaluation.md)).

- **In the pipeline:** `--category-api jev` makes Jev the categorizer for AI apps without a catalog category. The local model is the offline fallback.
- **Catalog audit:** `--audit-catalog-categories` flags catalog categories Jev disputes. It flagged all four well-known catalog-categorized tools: ChatGPT ("Communication And Support"), Claude, Perplexity and Grammarly ("Security And Privacy"). An independent human review had judged all four catalog labels wrong.
- **Spot check:** on the 10 AI tools in the usage logs, Jev agreed with the human-reviewed category for 9 (`data/reviews/categories_2026-11-21/`).

**Governance.** Every score lists the rules and points behind it, plus any missing evidence. Example: ChatGPT at Summit scores 95 (unapproved, may train on customer data, no SSO, indefinite retention). Weights are a demonstration policy (`governance/policy.json`, demo-1.0) and have not been reviewed against independent judgments ([advisor worksheet](docs/component4_advisor_evaluation.md)).

**Not yet validated against people.** No detection or category score above has been checked against human labels. A 150-tool trusted test set is built and ready to label: blind labelling sheet, hidden model predictions and a scorer reporting labeller agreement ([evaluation/README.md](evaluation/README.md)). That, and the advisor worksheet, are the next steps that turn these measurements into validated results.

**Reproducing.** The trained category model can be downloaded instead of retrained: `python category_training/fetch_category_model.py` (GitHub Release, hash-verified). CI (`.github/workflows/tests.yml`) runs the core tests on every pull request.

## Validation and limitations

The unified dashboard integration passes **205 automated tests** in the local validation environment, including 14 retrieval/answer checks and 9 additional grounding/QA checks. Six named-application search smoke queries returned the intended application first; this small developer-authored set is not a broad or independent retrieval benchmark. The 26 developer-authored functional cases also pass; this is not a general answer-accuracy benchmark. Live answer-quality evaluation is still needed.

Model accuracy figures refer to specific datasets and contracts; do not present earlier 306-row category results as results of the newer full-Kaggle category pipeline. Source-assigned labels, provisional predictions and generated controls are not independently verified truth. Synthetic governance settings demonstrate the project workflow and are labeled; missing evidence never establishes safety.

## Guides and remaining work

- [Jev classification and completed catalog/sample runs](docs/jev_classification.md)
- [Current discovery/category/governance pipeline](ai_discovery/PIPELINE_GUIDE.md)
- [Component 4 evidence and scoring](docs/component4_integration.md)
- [New report retrieval and analyst assistant](docs/governance_assistant.md)
- [Original requirements](Challenge-Project-Overview.md) and [vision](VISION.md)
- [November 11 model experiment](docs/backend_release.md), [older baseline](docs/baseline_classification.md), [older embeddings](docs/embedding_classification.md)
- [Dataset corrections](docs/corrected_labels.md), [historical project notes](docs/project_history.md)

Next: broader independently reviewed retrieval/answer evaluation, optional live RAG-provider validation, stronger discovery evidence/model validation, and a held-out Jev/local-model comparison. The offline dashboard is available; Jev catalog and sample classification runs are complete; new identities require a separate API classification run.

Challenge advisor: Shubham Arya. AI coach: Aram Ramos. Fellows: Forum Shah, Sheila Aquino, Sadit Hussain, Min Thaw Zin, Abigail De Leon, Joshua Hu and Abdulfattah Ahmed.
