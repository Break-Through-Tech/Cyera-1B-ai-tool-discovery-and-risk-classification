# Jev product classification

Jev classifies AI-focused versus conventional products. Governance scores remain the separate explainable rules in `governance/policy.json`.

## Completed sample run

All **21 unique name/endpoint identities** in the 30,000-event NovaTech activity file were sent once to TypeSafe's official System One endpoint. Returned model: **jev-1.14.0**. Results: **9 AI, 12 conventional, 0 unknown**. There were 21 successful requests, consuming 31,655 reported input tokens. Dollar cost was not independently retrieved from billing. No employee IDs, department activity, approvals, existing binary target labels, or risk scores were sent.

Inputs were product names, exact observed endpoints, exact-match catalog descriptions, and available saved product research. Some research was previously generated using Gemini, so this is **not an independent Jev-versus-Gemini benchmark**. Matching previous labels is also not an accuracy measurement. Probability values describe the product-label choice, not breach probability or verified correctness.

## Files

- `ai_discovery/jev.py`: builds product-only evidence, sends authenticated HTTPS requests, validates typed choices/probabilities, caches responses, and writes a resumable classification report.
- `data/processed/pipeline_runs/novatech_jev_classification.json`: all 21 model decisions, evidence inputs, probabilities, model version, token usage, and input fingerprints.
- The dashboard reconstructs the Jev company demo locally from the compact classification JSON and existing activity file. The duplicate `novatech_jev_component4.json` is no longer tracked. This reconstruction makes no API calls; fictional controls are explicit, and existing purpose categories are retained only where the binary decision agrees.
- `data/cache/jev_classification.json`: local ignored response cache. Credentials are not stored in this cache.
- `.env`: local ignored credentials. Never commit this file or share its contents.

The dashboard defaults to the saved Jev report when available. Select **Saved Jev results** and click **Analyze / load data** to return to it. **Saved NovaTech report** still opens the earlier catalog/research report for reference. Upload analysis can reuse exact saved Jev product decisions while recalculating usage and company-scoped risk. Unknown applications require a separate classification run; the dashboard never makes paid requests automatically.

## Run against an activity file

Install `modeling/requirements-jev.txt`. Configure `TYPESAFE_API_KEY` in your local environment or ignored project-root `.env`, then run:

```bash
python -m ai_discovery.jev \
  --activity data/raw/novatech_usage.csv \
  --output data/processed/pipeline_runs/novatech_jev_classification.json
```

One request is made for each unique uncached identity/evidence/policy/model combination. Successful responses are saved immediately. Repeating the same run reuses those responses without paid calls. It stops at the first HTTP, network or invalid-response error and does not automatically retry or substitute a binary label. A stopped run is explicitly marked incomplete. `jev-latest` is an alias: cached responses preserve their actual returned version; a fresh model comparison should use an explicit model name and separate cache/output paths.

The classification asks for `ai`, `non_ai`, or `unknown`. AI/ML must be central to primary functionality; optional AI features are insufficient. Selected binary probabilities below 0.70 are marked unknown/review. This is a prototype threshold, not a validated cutoff. Model results are not human-reviewed labels; source labels are not rewritten.

To produce a dashboard-compatible governance report from a completed run, call `dashboard_report(report, activity_path, company, demo_controls)` in `ai_discovery.jev` and write its returned JSON. Explicit `demo_controls=True` adopts the invented catalog controls/approvals for that fictional scenario; it is not suitable as verified enterprise policy.

API contract: [TypeSafe quick start](https://docs.typesafe.ai/introduction/quickstart) and [System One reference](https://docs.typesafe.ai/api/system-one).


## Full current catalog

`python -m ai_discovery.jev_catalog` processes the union of the two operational catalogs, deduplicated by normalized name and endpoint. This is 17,107 identities from 17,440 source rows. It sends one product-isolated Choice question per identity, packing up to 100 questions into a request with a conservative 48,000-byte payload limit. The prepared run has 226 HTTP batches. Common policy is shared; each question contains only its own product evidence. Original labels and governance metadata are excluded from requests.

This full-catalog run uses existing names, endpoints and descriptions, without the sample run's additional Gemini research. Different evidence/policy means the earlier 21 responses cannot be silently reused as equivalent full-catalog responses. They remain separately available for the enterprise demo. Full-catalog results are not an independent held-out benchmark and do not replace training labels.

Outputs live in `data/processed/jev_catalog/`:

- `all_classified.csv`: one row per processed identity, Jev prediction, three probabilities, unknown/review status, preserved source labels and disagreement indicators. Unknown is not encoded as zero.
- `summary.json`: completion status, counts, reported input tokens and public-price estimate. This is not an account billing statement.
- `input_manifest.json`: exact source hashes, model version, policy and deduplication method.

Immutable successful batch responses are stored under ignored `data/cache/jev_catalog/`. Restarting the same command reuses checkpoints and sends only missing products. The process stops on an HTTP/network/validation failure rather than entering a retry loop. `--prepare-only` computes coverage and batching without paid calls. The complete catalog CSV remains a team artifact in this repository. Dashboard downloads are scoped to company findings; enterprise usage remains a separate, much smaller dataset.


### Completed full-catalog results

The completed Jev 1.14.0 run processed **17,107 / 17,107 identities** in **226 successful HTTP requests**:

| Result | Applications |
|---|---:|
| AI-focused | 9,814 |
| Conventional | 1,412 |
| Unknown / below review threshold | 5,881 |

There are **1,052 resolved predictions that disagree with the source labels** (including conflicting source-label sets). These are review candidates, not automatic corrections. Unknowns and disagreements are flagged in the export. The original catalogs are unchanged.

Reported input usage is **2,942,542 tokens**, corresponding to **$0.123587** at the documented $0.042/M input-token rate; account-specific billing was not independently retrieved. A cache-only regeneration verified all 17,107 rows without additional requests. Source file hashes remained unchanged. This full-catalog cost excludes the earlier 21-product demonstration run.
