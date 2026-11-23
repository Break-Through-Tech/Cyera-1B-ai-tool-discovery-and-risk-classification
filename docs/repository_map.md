# Repository map

Start with [README.md](../README.md) for current capabilities and [dashboard/README.md](../dashboard/README.md) for running the application.

```text
dashboard/
  app.py                 Company views, upload controls, charts and report questions
  data.py                Team report loading, usage/event joins and investigation queue
  workflow.py            Saved Jev lookup, fresh-upload analysis and local evidence questions
  companies.json         Saved company/report/activity configuration

ai_discovery/
  ingest.py, catalog.py  Activity extraction and exact known-product matching
  enrich.py              Optional cited product research
  classify.py, jev.py     Provider selection and Jev AI/non-AI decisions
  jev_catalog.py         Resumable full-catalog Jev classification
  categorize.py          Catalog/review/local-model purpose assignment
  run_pipeline.py        Operational discovery, category and governance workflow
  dashboard_data.py      Offline adapter for uploaded activity
  retrieval.py           Local embeddings and ChromaDB report evidence storage/search
  ask.py                 Company-scoped evidence answers and optional generated drafts
  evaluate_*.py          Retrieval and answer-contract evaluation commands

governance/              Explainable scoring, evidence and fictional company scenarios
category_training/       Current Kaggle-14 category training and team's Jev category checks
modeling/                Earlier model experiments and optional dependency lists
preprocessing_scripts/   Dataset preparation and correction utilities

data/
  raw/                   Input catalogs and synthetic activity
  processed/jev_catalog/ Full Jev predictions, input manifest and run summary
  processed/dashboard_runs/ Team company reports and reviewed categories
  reviews/               Review records and evaluation fixtures
  cache/                 Reusable caches; local Chroma indexes are ignored by Git

artifacts/               Experiment metrics and validation artifacts
models/                  Locally supplied/trained model bundles, excluded from Git
tests/                   Automated behavior and consistency checks
docs/                    Current methods and clearly historical releases
notebooks/               Exploratory analyses
```

The dashboard's saved company views preserve the team's reviewed outputs. Uploads reuse exact product knowledge while recalculating company usage and governance. Local report questions use ChromaDB; no cloud database or paid call is needed to view or query saved evidence.

Model weights, local encoders, provider credentials and derived vector databases are not committed. Current category bundles are expected under `models/categories/`; older local encoder paths are documented in [the report assistant guide](governance_assistant.md).

Earlier backend experiments remain documented in [backend_release.md](backend_release.md), [project_history.md](project_history.md) and their method guides. They are historical context, not the current dashboard setup instructions.
