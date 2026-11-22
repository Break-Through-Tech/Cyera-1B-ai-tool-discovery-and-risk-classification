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

## Quick start (no API keys or trained model needed)

The saved pipeline results are in the repository, so viewing needs only three
packages. From the repository root, on `main`:

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install streamlit plotly pandas
streamlit run dashboard/app.py
```

It opens at http://localhost:8501. Things to look at:

1. **Summit Financial → Investigate**: ChatGPT and **DocuMind AI** are high risk.
   DocuMind AI was not in any catalog: online discovery researched it and classified
   it as an AI document-analysis tool. It is used only by Finance (84 employees),
   is unapproved, and uploads about 2.6× more per event than the company's typical app.
2. **Why this risk level** lists every rule that added points, in plain language,
   and which evidence is missing.
3. AI tools with **no approval record** (Gemini, Copilot, Jasper, Midjourney) are
   flagged, because a low score built on missing evidence is not a safe score.
4. **Overview**: AI tools by category, risk levels, and a department × tool heatmap.
5. **Jev check**: every AI tool's category also has a second opinion from the Jev
   API, with its probability. Jev agrees with 9 of 10 reviewed categories and
   questions Claude's. On 154 held-out tools Jev beat the local category model
   (macro-F1 0.59 vs 0.48); see `docs/jev_category_evaluation.md`. The saved
   opinions are in `data/processed/dashboard_runs/jev_second_opinion.json`, so
   viewing needs no Jev key. Regenerate with
   `python category_training/jev_second_opinion.py`.

All AI-tool categories in these runs were set by human review
(`data/reviews/categories_2026-11-21/`), correcting catalog labels such as
ChatGPT → "Communication And Support" and Grammarly → "Security And Privacy".

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
