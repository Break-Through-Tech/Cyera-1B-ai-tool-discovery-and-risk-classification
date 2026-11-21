# Component 5: AI Governance Console

A Streamlit dashboard over the saved Components 1–4 output. It does not re-run
detection, categorization or risk scoring; it only reads and presents them.

- **Investigate**: AI tools (and unresolved apps) that are high risk or not on
  the approved list, with a detail view per application. The view shows readable
  risk reasons and their points, missing evidence, purpose category and source,
  users by department, daily activity, and upload size compared with the
  company's typical app.
- **Overview**: AI tools by category, risk levels (incomplete assessments
  marked), department × AI tool usage heatmap, daily AI activity.
- **All applications**: full table with CSV download.

An AI tool with **no approval record** is listed as needing investigation:
missing evidence is not evidence of safety, and VISION.md's shadow-AI case is
a tool missing from the approved list.

## Run

From the project root, with the project virtual environment:

```bash
.venv/bin/pip install streamlit plotly        # once
.venv/bin/streamlit run dashboard/app.py
```

## Data it reads

`dashboard/companies.json` lists each company's pipeline run directory and the
activity CSV it was built from. The activity file's SHA-256 is checked against
the run, and every event is joined through the run's own `event_map`.

To regenerate a run (needs the trained category model in
`models/categories/kaggle14_run1`, see `ai_discovery/PIPELINE_GUIDE.md` §8E):

```bash
O=data/processed/pipeline_runs/novatech_online1
.venv/bin/python -m ai_discovery.run_pipeline \
  --activity "data/raw/summit_financial_usage (1).csv" \
  --discovery $O/categorized_applications.json \
  --decision-cache $O/discovery_decision_cache.json \
  --model-dir models/categories/kaggle14_run1 \
  --company "Summit Financial" --activity-provenance synthetic \
  --governance-catalog data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv \
  --governance-catalog data/processed/final/dataset_b_final_kaggle_categories.csv \
  --governance-catalog-provenance synthetic \
  --catalog-approval-company "Summit Financial" \
  --governance-scenarios governance/synthetic_scenarios.json \
  --out data/processed/dashboard_runs/summit_new
```

Add `--online` (and a `GEMINI_API_KEY` in the project-root `.env`) to research
apps without a saved decision, such as DocuMind AI. Then point
`companies.json` at the new directory.

## Tests

```bash
.venv/bin/python -m unittest tests.test_dashboard_data -v
```
