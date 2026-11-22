# Unified dashboard recording guide

Run `python -m pip install -r dashboard/requirements.txt`, then `python -m streamlit run dashboard/app.py` from the repository root. The interface uses the team's Component 5 dashboard, with integrated upload analysis, Jev catalog coverage, reviewed purposes, and evidence questions. Full setup and input details are in [dashboard/README.md](../dashboard/README.md).

## Recording sequence

1. Explain the problem: companies need visibility into AI usage and explainable governance concerns.
2. Open **Summit Financial → Investigate**. Show DocuMind AI, an application originally absent from the catalog, its saved discovery source, reviewed purpose, employee departments, and governance reasons. Its saved discovery used the earlier pipeline; do not call it a live Jev discovery result.
3. Show **Overview**: categories, risk distribution including missing evidence, department heatmap, and daily activity. These are synthetic event counts, not measured real-world adoption.
4. Switch to **NovaTech** to demonstrate company-scoped reports. Department filters select applications; the figures for each selected tool remain whole-company figures.
5. Select **Analyze activity CSV**, enter a different fictional company, and choose a sample or upload the operational schema. Click **Analyze activity**. Explain that saved product knowledge is reusable, while usage and governance are recalculated. Approvals are not borrowed from the saved company's report. Fictional controls require explicit adoption.
6. Investigate a matched product: show Jev AI/non-AI probabilities separately from the purpose-category opinion and the governance rule total. Unknown and conflicting identities remain under review.
7. Download the filtered findings, complete company report, or full catalog. Catalog coverage is 17,107 identities; each company's activity uses only a small subset.
8. If the local encoder is installed, ask **Why was DocuMind AI flagged?** under **Evidence & exports** and show the cited report facts. This is local evidence retrieval, not a live Gemini response.

## Explain the limits

The original labels and Jev predictions are preserved separately. Of 17,107 catalog identities, 9,814 have AI predictions, 1,412 conventional predictions, and 5,881 unknown/review results. There are 1,052 resolved disagreements with source labels. Neither agreement with noisy labels nor passing code checks establishes real-world accuracy.

The team's category comparison covers 154 held-out tools using noisy reference labels. It measures purpose categorization, not AI/non-AI detection. A reviewed binary Jev/local-model comparison remains separate evaluation work.

No paid calls occur while viewing reports, asking local evidence questions or analyzing uploaded activity. Unfamiliar products without saved evidence remain unresolved in the interface. The explicit online pipeline can research products with Gemini and classify them with Jev; live validation of the newly connected provider path and stronger independent evaluation remain necessary.
