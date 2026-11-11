# Shared enterprise discovery reports

## Active foundation

`ai_discovery.report` calls Sadit's `discover_known`, which uses `extract_applications` and catalog matching. No second ingestion or matching implementation is used by the reporting entry point. The duplicate `discovery/` package has been removed; use `python -m ai_discovery.report` as the single reporting entry point.

The integration preserves normalized name plus endpoint identity, paths, query/fragment distinctions during ingestion, all accepted events, rejected rows with reasons, invalid/missing metric counts, observed name variants, catalog candidates and source-row-to-application mappings. Repeated event IDs are counted as accepted rows, as in Sadit's ingestion; a separate repeated-ID count makes that explicit. Applications are not collapsed by hostname. Endpoints differing only in query/fragment can therefore remain separate in activity.

Governance receives only each application's accepted source rows. Valid transfer observations determine totals/means/maxima; missing/invalid observations remain gaps. Missing employee, department and timestamp observations also mark evidence incomplete. No employee identifiers are exported, although source-row mappings can be joined back to the input. Duplicate events can affect totals; input quality still matters.

## Classification and categories

Defaults are the raw Kaggle catalog, preserving product paths, and `dataset_b_final_kaggle_categories.csv` as Sadit's finalized known-tool input. His matching rules and per-product overrides are retained. Raw catalog entries carry source-assigned AI labels; these are not independently verified. Unknown and ambiguous records stay visible.

The accepted catalog category is displayed with `category_source=catalog_assignment`. The older 12-category models are not invoked against the newer 14-category scheme. Category-model retraining and evaluation remain future work. The six newly resolved labels are reflected in the finalized lookup subset; the full corrected training dataset and frozen binary model have not been relabeled or retrained in this integration.

Optionally, a supplied `domain,description` CSV and `--model artifacts/corrected_full_binary` allow the existing description-only model to assess unmatched applications. A description endpoint must identify exactly one observed product scope. Ambiguous catalog matches are not automatically overridden. Predictions remain experimental, uncalibrated and marked for review. The frozen model/hash and original evaluation remain unchanged.

## Governance evidence

Scores use accepted activity and, optionally, the explicitly synthetic company overlay in `governance/synthetic_scenarios.json`. The generated catalog governance columns and stored risk labels are not imported. An endpoint shared by multiple observed product scopes does not receive a broad endpoint-only scenario override. Unknown controls remain unknown. Medium/high priorities are policy demonstration choices, not breach probabilities.

Sadit's `ai_discovery.enrich` remains available as a separate evidence-collection stage. No enrichment, Gemini, Jev or other network/API calls are made by this reporting integration. Cited enrichment reports are not automatically converted into verified identities or descriptions.

## Sample results

| Company | Accepted events | Rejected events | Application identities | Catalog AI | Conventional | Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| Summit Financial | 25,120 | 0 | 22 | 6 | 7 | 9 |
| NovaTech | 30,000 | 0 | 21 | 6 | 7 | 8 |

These counts measure catalog-matching coverage, not model accuracy or actual AI use in every event. Checked-in `artifacts/discovery_shared/*-summary.json` files omit repetitive event mappings; full CLI reports retain them. Historical reports in `artifacts/historical/discovery/` used earlier matching rules and are preserved rather than silently rewritten.

DocuMind remains unknown in the catalog, with 84 Finance users, observed priority 30 (medium), and unknown vendor controls. Explicit fictional unapproved status produces 55 (high). Unknown identity does not suppress behavioral investigation.

## Run

```bash
python -m ai_discovery.report --company 'Summit Financial' \
  --activity 'data/raw/summit_financial_usage (1).csv' \
  --out /tmp/summit-shared.json

python -m ai_discovery.report --company NovaTech \
  --activity data/raw/novatech_usage.csv \
  --out /tmp/novatech-shared.json

# Optional: --descriptions /path/to/descriptions.csv
#           --model artifacts/corrected_full_binary
#           --scenarios governance/synthetic_scenarios.json
```

New output paths are required; reports are not overwritten. Catalog-only reporting uses the standard library. Optional model inference requires `modeling/requirements.txt`; the newly expanded root requirements are not needed for this backend.

Validation: 74 active-backend tests pass, including accepted/rejected accounting, repeated events, missing/invalid metrics, shared endpoints, description-model routing and the existing preparation/model/governance checks. The paused dashboard/upload draft is stored outside the repository and is not part of this backend.
