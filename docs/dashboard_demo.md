# Dashboard demo

Run from the repository root in your Python environment:

```bash
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

The dashboard works offline without API keys. It opens the saved Jev-classified NovaTech report when available, falling back to the earlier Component 4 report: 21 applications, 9 AI-focused, 12 conventional, and 3 high governance priorities. These are demonstration findings, not classifier accuracy measurements.

## Recording walkthrough

1. Explain the problem: an analyst needs to discover AI usage and understand governance concerns across departments.
2. Show **Overview** for counts, priority distribution, and departmental coverage. Counts at the top cover the entire analyzed file; charts and tables follow the sidebar filters. User counts belong to individual applications and cannot be added to obtain unique enterprise users.
3. Show **Applications**, filter by department or AI status, and download the filtered findings CSV.
4. Show **Investigate**, select an application, and explain its score using the displayed rules, points, evidence gaps, and control provenance. High priority can also apply to conventional tools. Synthetic controls are fictional; scores are priorities rather than breach probabilities.
5. Select **Analyze activity CSV** in the sidebar. Start with the sample, or upload a CSV. Click **Analyze / load data**. Usage and governance are recalculated from the input. Product decisions may be reused only for exact saved identities; previous company approvals and usage are not carried over. Fictional controls and approvals require the explicit checkbox.
6. Export the complete analysis JSON to preserve findings and explanations. **Open saved report** accepts a single-company Component 4 report.

## Input contract

Activity requires `application_name` and `domain`. Use the downloadable sample for the complete operational schema, including `user_id`, `department`, and `timestamp`. Missing usage fields remain missing evidence. Malformed identities are rejected and their ingestion details are available in Workflow & evidence. The prototype accepts at most 100,000 rows and 25 MB of activity data.

Optional control CSVs use `tool_name` and `domain` plus the operational governance fields. Uploaded controls are treated as synthetic. Uploading metadata by itself does not adopt approval values.

## What runs and what remains

```mermaid
flowchart LR
    A[Saved report or activity CSV] --> B[Dashboard adapter]
    B --> C[Operational ingestion and exact catalog matching]
    C --> D[Saved product decisions or unknown]
    D --> E[Company evidence and rule-based governance]
    E --> F[Overview / applications / explanations / exports]
```

- `dashboard/app.py`: interface, filters, charts, downloads, and investigation view.
- `ai_discovery/dashboard_data.py`: validates input, calls the existing ingestion/catalog/governance code, scopes evidence to the company, and prepares display/export rows.
- `tests/test_dashboard.py`: six automated checks covering sample totals, new activity, unknown products, company isolation, input validation, exports, and interface startup.

This interface does not run live Jev, Gemini, web research, model training, or fresh category-model inference. It can reuse saved product classifications/categories and exact catalog entries. Unresolved applications remain unknown. Jev sample classification has completed through the separate API command; see [the Jev run guide](jev_classification.md). New identities require another explicit classification run. No key is required to view saved results.

The optional report question feature needs the separately installed `modeling/requirements-retrieval.txt` dependencies and existing local encoder at the project's default encoder path. It retrieves evidence from the current report and does not make paid calls. Counts and explanations remain usable without that feature.

Synthetic activity, source-assigned labels, saved predictions, and fictional security controls limit what this demonstration establishes. Evaluating Jev against the local classifier on a shared held-out set and validating on representative real data remain future work.
