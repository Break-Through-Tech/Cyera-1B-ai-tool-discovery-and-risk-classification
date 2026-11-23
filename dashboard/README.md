# Component 5: AI Governance Console

A Streamlit dashboard over the saved Components 1–4 output. Saved company views read and present those results. The **Analyze activity CSV** mode reuses exact saved product decisions and reviewed purposes, then recalculates usage and governance for the uploaded company.

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

The saved pipeline results are in the repository. Use the dependency file below; the core dashboard does not need provider keys, trained category weights, or the optional report-search encoder.

To review **PR #22 before it is merged**, use branch `recording-dashboard`. After merge, use the merged `main` version. Run from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r dashboard/requirements.txt
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

Add `--online` (with `GEMINI_API_KEY` and `TYPESAFE_API_KEY` configured locally) to research
apps without a saved decision, such as DocuMind AI. Then point
`companies.json` at the new directory.

## Tests

```bash
.venv/bin/python -m unittest tests.test_dashboard_data -v
```


## Integrated uploads, Jev catalog, and evidence questions

Choose **Analyze activity CSV**, enter a fictional company, and upload activity or select a sample. Required identity columns are `application_name` and `domain`; `employee_id`, `department`, `timestamp`, and transfer-volume fields support usage charts and governance. Missing fields stay missing evidence. Uploads are limited to 25 MB and 100,000 rows. Click **Analyze activity** to replace the current upload result.

The lookup combines the full saved Jev catalog (17,107 identities), the richer 21-product sample decisions, and the team's reviewed categories and saved unseen-product evidence. Matching is by normalized product name plus exact endpoint, never hostname alone. The sample's richer evidence has precedence over catalog-only predictions; reviewed categories are retained for compatible AI decisions. A Jev/catalog binary disagreement remains unresolved for review. No company approvals or usage counts are transferred across companies.

The default upload does not adopt catalog approvals. **Adopt fictional catalog controls and approvals** explicitly scopes these synthetic values to the selected fictional company. Optional control metadata remains synthetic. Product classification probabilities, optional category second opinions, and governance scores are displayed separately.

Sidebar filters select applications; usage/risk figures for each selected application retain their whole-company scope. Investigation, heatmaps, daily activity and company switching are preserved. The full Jev catalog remains a team artifact in the repository; the dashboard exports only company findings. The All applications tab exports filtered findings; Evidence & exports offers the complete company report and ingestion details.

**Evidence & exports** also supports local report questions with citations using the report assistant. Install `modeling/requirements-retrieval.txt` and supply the existing local encoder at its configured path. No encoder is downloaded and no Gemini generation is enabled by this interface. Questions use the complete current-company report, independently of display filters. Missing encoders produce a visible setup message.

For new products, the operational pipeline's explicit `--online` mode uses Gemini for cited public-product research and defaults to Jev for AI/non-AI detection. It needs both provider credentials and a local category-model bundle when inference is required. `--detection-provider gemini` retains the legacy classifier option. Viewing saved results or analyzing uploads never activates this online mode automatically. The new Jev provider adapter was tested with simulated responses, without another paid run.


## Artifact labels and acceptance walkthrough

See [the artifact guide](../docs/artifacts.md) for each deliverable's purpose, provenance, location and limitations. The All applications export includes **AI label source**, **Purpose label source**, **Usage data provenance** and **Contains synthetic evidence**; the Jev category opinion is distinct from the AI/non-AI prediction.

A reviewer can verify the core prototype without API keys:

1. Open Summit Financial and investigate DocuMind AI: the saved report records 84 Finance users and high priority (55), with missing-control evidence clearly identified.
2. Open Overview and inspect the department heatmap and daily activity. These represent synthetic events.
3. Switch to NovaTech, then choose Analyze activity CSV and use its sample for a different fictional company. Keep fictional-control adoption unchecked. Confirm usage is recomputed and old company approvals are not inherited.
4. Export company findings and check the source/provenance columns. Unknown labels are not non-AI labels.
5. Optional: install retrieval dependencies and the local encoder, then ask a report question under Evidence & exports. This is local ChromaDB evidence retrieval, not live Gemini prose generation.

For the optional report-search setup, first install `modeling/requirements-retrieval.txt`. If you do not already have the local MiniLM bundle, this explicit one-time command downloads the pinned public encoder (internet required; no paid API key):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='artifacts/category_embeddings/encoder')"
```

Model files remain ignored by Git. Normal dashboard startup and report questions do not download models automatically. Fresh local category-model inference and live online research/classification have separate setup requirements; they are not needed to view the checked-in company demonstrations.
