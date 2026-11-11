# Integrated discovery and governance reports

`python -m discovery.report` joins enterprise usage aggregates, corrected catalog AI labels and explainable governance assessments. Outputs contain application names, departments, distinct user counts, transfer volumes, detection evidence, risk reasons and an ordered review queue. Employee identifiers are not exported.

This is a backend integration, not the final dashboard or conversational assistant. The corrected binary model and its test results are preserved without retraining or threshold changes. This branch builds on the training work in PR #19.

## Detection rules

Exact normalized hostname plus path is required. Parent domains never automatically inherit an AI feature's label. A catalog identity is accepted only when all matching records agree on product name and a resolved binary label. Multiple product scopes, conflicting labels and unresolved labels stay unknown. Even catalog detections request review because most labels are inherited source assignments.

Unknown tools can optionally use the description-only classifier when a separate CSV with `domain,description` is supplied and `--model artifacts/corrected_full_binary` is enabled. Descriptions must come from supplied evidence; the pipeline does not invent them or infer AI from application names. Model predictions remain experimental and uncalibrated. Empty or unrecognized vocabulary does not establish a conventional classification. Ambiguous catalog identities require scope resolution rather than an automatic model override.

Governance is calculated independently for every observed application, including unknown and conventional ones. The baseline Dataset B policy metadata has missing security controls; these remain unknown. Corrected classification labels and vendor names are not treated as security evidence. Company approval applies only through an explicitly supplied fictional scenario overlay. Reports hash source inputs and, when enabled, verify and identify the saved model.

## Generated reports

| Report | Application identities | Catalog AI | Catalog conventional | Unknown or ambiguous |
|---|---:|---:|---:|---:|
| `artifacts/discovery/summit-observed.json` | 22 | 5 | 3 | 14 |
| `artifacts/discovery/novatech-observed.json` | 21 | 5 | 3 | 13 |
| `artifacts/discovery/summit-fictional-policy.json` | 22 | 5 | 3 | 14 |

These counts measure conservative identity coverage, not classifier accuracy or the actual number of AI applications. No descriptions were invented for the included reports, so no model predictions were needed. Known real-world tools such as Cursor may remain unknown due to missing exact catalog identities. Slack/Zoom-style shared-domain ambiguity is deliberately visible.

DocuMind appears in 120 Summit events, with 84 Finance users. Its observed governance score is 30 (medium); AI detection and vendor controls remain unknown. The explicitly fictional unapproved overlay adds 25 points, yielding 55 (high). Neither output asserts a breach or verified vendor policy.

## Run

Install `modeling/requirements.txt` before enabling the optional classifier. Catalog-only reporting uses Python's standard library.

```bash
python -m discovery.report --company 'Summit Financial' \
  --activity 'data/raw/summit_financial_usage (1).csv' \
  --out /tmp/summit-discovery.json

# Optional supplied descriptions and trusted local model:
python -m discovery.report --company 'Summit Financial' \
  --activity 'data/raw/summit_financial_usage (1).csv' \
  --descriptions /path/to/product_descriptions.csv \
  --model artifacts/corrected_full_binary \
  --out /tmp/summit-with-descriptions.json

python -m unittest discover -s tests
```

Existing outputs are not overwritten. Invalid events and duplicate supplied-description identities fail explicitly. Tests cover shared-domain ambiguity, path separation, unseen-description routing, unknown retention, company policy scope, and the existing preparation/model/governance behavior: 72 tests pass.

Next: build an evidence-backed alias/description table for unresolved activity identities, then expose these reports in a dashboard. Category refresh, retrieval/RAG and Jev comparison remain separate unfinished work.
