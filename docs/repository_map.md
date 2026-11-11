# Repository map and structural cleanup

## One place for each responsibility

```text
ai_discovery/                 APPLICATION WORKFLOW
  ingest.py                   Sadit's activity extraction; accepted/rejected rows and mappings
  catalog.py                  Sadit's catalog matching and supplied categories
  enrich.py                   Optional cited research; not automatic classification
  report.py                   Combined report command; calls ingestion, matching, optional model and risk

governance/                   GOVERNANCE ASSESSMENT
  risk.py                     Point calculations, reasons, missing evidence
  policy.json                 Rule values and thresholds
  synthetic_scenarios.json    Explicit fictional company approvals/controls

modeling/                     ALL MODEL TRAINING/PREDICTION CODE
  train_corrected_binary.py   Current frozen binary experiment and prediction
  train_category_model.py     Sadit's 12-category TF-IDF experiment
  train_category_embeddings.py Sadit's embedding comparison
  tune_category_embeddings.py Sadit's embedding settings search
  train_baseline.py           Earlier baseline experiment
  train_embeddings.py        Earlier embedding experiment
  predict.py                 Earlier baseline prediction
  predict_embeddings.py      Earlier embedding prediction
  embedding_utils.py         Earlier shared encoder helpers
  requirements*.txt           Experiment dependency lists

preprocessing_scripts/        DATA PREPARATION, NOT MODEL TRAINING
  label_source_membership.py  Assign original dataset-origin labels
  apply_conflict_review.py    Apply reviewed label/scope/description changes
  reconcile_vendor_enrichment.py Reuse exact matching cached vendor research
  prepare_corrected_classification.py Prepare frozen grouped binary splits
  fix_final_ai_labels.py      Sadit's six additional subset resolutions
  map_kaggle_categories.py    Sadit's original-Kaggle category mapping
  ...                         Earlier cleaning, sampling and enrichment utilities

data/                         INPUTS AND PREPARED TABLES
artifacts/                    ALL SAVED MODELS AND RESULT FILES
  corrected_full_binary/      Current binary model, settings and reports
  category_tfidf/             Sadit's TF-IDF category model/results
  category_embeddings/        Sadit's default category embeddings/results
  category_embeddings_tuned/  Sadit's tuned category embeddings/results
  baseline/, embeddings/      Earlier reproducible experiments
  discovery_shared/           Current sample report summaries
  historical/discovery/       Superseded combined reports
  governance/                 Standalone risk reports

tests/                        AUTOMATED CHECKS
  test_discovery.py           Combined matching/reporting integration checks
  ...                         Dataset, model, governance and provenance checks

docs/                         EXPLANATIONS AND HISTORY
notebooks/                    EXPLORATORY ANALYSIS
README.md                     Current entry point and commands
```

## What changed

| Before | After | Reason |
|---|---|---|
| `discovery/` wrapper and old matching helpers | Removed; use `ai_discovery.report` | One reporting entry point and one active matching implementation |
| Three category-training scripts under `preprocessing_scripts/` | Same filenames under `modeling/` | Training belongs beside other model code |
| `models/category/` | `artifacts/category_tfidf/` | Saved outputs have one parent directory |
| `models/category_embeddings/` | `artifacts/category_embeddings/` | Same consolidation; model bytes unchanged |
| `models/category_embeddings_tuned/` | `artifacts/category_embeddings_tuned/` | Same consolidation; model bytes unchanged |
| `artifacts/discovery/` | `artifacts/historical/discovery/` | Prevent old reports being mistaken for current results |
| `test_discovery.py` and `test_shared_discovery.py` | One `test_discovery.py` testing the active implementation | Remove tests of retired matching helpers; preserve current integration checks |
| Long template/old-status README | Short current README; old content in `docs/project_history.md` | Current instructions separate from historical experiments |
| Paused untracked dashboard/upload draft | Recovery copy outside the repository | No unfinished duplicate pipeline in the active project |

The paused draft is recoverable at `/Users/forumshah/Documents/Codex/2026-11-02/can-yo/outputs/paused-dashboard-backup-2026-11-11/`. It was not committed or pushed. Resuming it requires adapting to the active shared backend; the old helpers are removed.

## What did not change

Source datasets, research citations/cache, frozen training splits, trained model bytes, evaluation results, label decisions and scoring rules were preserved. Moving saved models does not retrain them or improve their performance. Historical model metadata retains original experiment paths where recorded; those are provenance, not current executable defaults.

The `models/` directory is retired. Root `requirements.txt` reflects the team's broader environment; `modeling/requirements.txt` pins the current binary experiment. Dependency environments were not silently merged or upgraded.

## Run the moved training commands

Run from the repository root. Use fresh output directories when reproducing experiments.

```bash
python -m modeling.train_category_model --help
python -m modeling.train_category_embeddings --help
python -m modeling.tune_category_embeddings --help
```

These category scripts still target their original 12-category experiment. Moving them does not make them compatible with the 14-category lookup dataset. The tuning defaults now reference `artifacts/category_tfidf`, `artifacts/category_embeddings`, and `artifacts/category_embeddings_tuned`.

## Cleanup verification

All 40 relocated category artifact files were verified byte-for-byte against the previous commit. Category command entry points open successfully, and a fresh NovaTech report retains 30,000 accepted events, 21 identities and the same 6/7/8 AI/conventional/unresolved matching counts. Retired helper tests were replaced with checks of the active catalog matcher. The complete maintained suite contains 74 tests.
