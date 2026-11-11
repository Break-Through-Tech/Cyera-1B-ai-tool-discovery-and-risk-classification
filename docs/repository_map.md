# Current backend location

See [backend_release.md](backend_release.md) for the active data/models, combined reporting and current results. `ai_discovery/report.py` is now integrated and `pipeline.json` names its releases. All training code stays in `modeling/`, all saved models in `artifacts/`. The obsolete `discovery/` wrapper is removed. The following records the earlier cleanup-only stage and is retained as history.

# Repository map — cleanup on main

```text
ai_discovery/           Activity extraction, catalog matching, optional research
  ingest.py            Reads activity; applications, accepted/rejected rows, source-row map
  catalog.py           Known AI/conventional/ambiguous/unknown matches
  enrich.py            Optional Gemini/search evidence collection

governance/            Risk engine, point policy and explicit fictional company scenarios
modeling/              All training, model comparison and prediction scripts
preprocessing_scripts/ Data cleaning, label correction, preparation and enrichment

data/                  Sources, prepared data, reviews and research cache
artifacts/             All saved models and experiment results
  category_tfidf/      Previously models/category/
  category_embeddings/ Previously models/category_embeddings/
  category_embeddings_tuned/ Previously models/category_embeddings_tuned/
  baseline/            Earlier baseline experiment
  embeddings/          Earlier embedding/similarity experiment
  governance/          Risk reports

tests/                 Automated behavior/consistency checks
notebooks/             Exploratory analysis and experiments
docs/                  Methods, limitations, historical records
README.md              Current starting point
```

## Changes merged by this cleanup

Three category training scripts moved from `preprocessing_scripts/` to `modeling/`, preserving filenames. Their imports support direct script and module execution. Saved category output directories moved from `models/` to `artifacts/`; executable defaults/examples now use those new paths. All 40 category artifact files are byte-for-byte unchanged.

The old long README, including historical results and EDA findings, is preserved in `docs/project_history.md`. The new root README describes current main, corrected sample paths, known limitations and explicitly unmerged features. Original experiment metadata retains historical path strings as provenance.

No source data, research evidence, trained model contents or results were deleted. No model retraining, label changes, dependency upgrades or risk-rule changes were performed.

## Why not merge the original simplification branch?

That branch also contains pending binary training and combined reporting changes. To respect the request to merge only simplification, this cleanup was built directly on main and includes only the applicable moves/path/documentation changes. PR #19 and the reporting branches remain separate.

The duplicate `discovery/` package, combined-report test consolidation, and relocation of earlier combined reports belong to those unmerged branches; they do not exist on current main and were not pulled in just to remove them. The paused dashboard draft remains outside the repository at `/Users/forumshah/Documents/Codex/2026-11-02/can-yo/outputs/paused-dashboard-backup-2026-11-11/`.
