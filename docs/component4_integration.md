# Component 4 v2.1: prototype governance assessment

## Online wait fix (v2.1)

Online research prints the app, search/extraction stage and a heartbeat every
15 seconds. --governance-timeout-seconds defaults to 90 seconds per API call.
SDK automatic retries are disabled to avoid multiplying the wait. A timeout or
request failure is recorded per app; missing fields remain unknown.
--governance-max-apps 1 limits a smoke test to one uncached app (up to two calls).

A checkpoint is saved next to the requested output, named
<output-name>.governance-checkpoint.json, after each cited search and extraction.
Ctrl+C exits cleanly. Rerun the same command to resume if the final report was not
written. A saved search resumes extraction without repeating the search. If the
final output exists, choose a new --out and supply --governance-research-cache
pointing to the previous checkpoint or completed report. Checkpoints are single-
writer files; do not run concurrent jobs targeting the same output path.

The checkpoint's actual location is printed at startup. Failures are not cached
as completed research. Existing unrelated files are not overwritten. This replaces
the previous end-of-run-only cache behavior. Successful cache entries still expire
after 30 days. Output report paths remain protected from overwrite.

Validation: 57 tests (15 Component 3, 13 legacy governance, 16 Component 4,
13 governance-evidence tests). Timeout and interrupt paths use simulated clients.
The installed SDK accepted timeout=90000ms and retry attempts=1 (zero retries)
in a no-network configuration check. No live Gemini call was made here.


This revision replaces the verified-only primary score. Synthetic activity and
synthetic controls are valid scoring inputs for this project, as VISION.md intends.
The main `governance.risk_score` is newly calculated from rule inputs; an existing
catalog `risk_label` is never copied or used as a feature.

## End-to-end flow

1. Component 1 extracts applications, counts usage and preserves accepted/rejected
   source-row mappings. Its normalized name plus full endpoint identifies an app.
2. Component 2 reuses exact catalog detections; optionally enriches and classifies
   unresolved products with the existing online workflow. Unknown is not non-AI.
3. Component 3 assigns purpose categories only to resolved AI tools, using catalog
   categories or provisional model predictions. Non-AI tools remain uncategorized.
4. Component 4 loads explicitly supplied governance catalogs for ALL applications.
   It matches normalized name plus full endpoint, never just hostname or a fuzzy
   name. Valid nonconflicting attributes are reused field by field. Missing values
   remain unknown. Conflicting catalog values are flagged and are not automatically
   overwritten by online research.
5. A supplied fictional-company scenario overrides catalog fields. Its legacy
   endpoint keys are applied only if exactly one observed app has that endpoint.
   Explicit application_id-keyed evidence overrides both, including explicit unknown.
6. With --governance-online, Gemini researches missing vendor controls and extracts
   cited attributes with plan context. No approval is researched. Only resolved AI
   records get the model-training check. Query/fragment endpoints need review and
   are not sent to research. Failed research leaves fields unknown and records an
   error outcome; it does not abort otherwise usable local assessments.
7. Rules combine available controls and activity into a score, risk level, reasons,
   provenance, completeness and review flags. Categories are not risk inputs.
8. A new JSON report preserves upstream records and ingestion; each record receives
   a governance object. The dashboard/investigation layer remains future work.

## Evidence policy

- synthetic: fictional inputs explicitly allowed for prototype scoring.
- researched: citation-backed model extraction, NOT independent verification.
- verified: an explicit reviewed-evidence assertion supplied by the caller.
- unknown: unavailable/untrusted evidence; adds no points, remains a gap.

The main score accepts the first three. Activity defaults to unknown until labeled.
A low score with incomplete evidence is not an assurance of safety. Complete means
all applicable checks have usable evidence, not that the app is secure.

Each triggered reason identifies evidence provenance. Source CSV hashes/row numbers
and online citations/plan context remain available. Vendor plan availability does
not establish that the company bought or enabled that plan. Thresholds remain the
existing demo-1.0 policy and are not calibrated breach probabilities.

Catalog approval values apply only with --catalog-approval-company matching
--company and --governance-catalog-provenance synthetic. This deliberately adopts
the catalog's fictional approvals for the named company. Scenario overrides take
precedence. Without this adoption flag or explicit company evidence, approval is
unknown. under_review is preserved, adds no unapproved points, and needs review.
An app absent from the catalog is not automatically unapproved: no exhaustive
approved-app list is assumed. Public research cannot settle company approval.

## Offline validation (project root, active virtual environment)

```powershell
python -m unittest discover -s tests -p test_component3_pipeline.py
python -m unittest discover -s tests -p test_governance.py
python -m unittest discover -s tests -p test_component4_pipeline.py
python -m unittest discover -s tests -p test_governance_evidence.py
```

Expected: 15 + 13 + 16 + 13 = 57 passing tests.

Run the existing NovaTech report through Component 4 without rerunning discovery:

```powershell
python -m governance.assess_pipeline `
  --pipeline-report data/processed/pipeline_runs/novatech_full1/categorized_applications.json `
  --activity data/raw/novatech_usage.csv `
  --company NovaTech `
  --activity-provenance synthetic `
  --governance-catalog data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv `
  --governance-catalog data/processed/final/dataset_b_final_kaggle_categories.csv `
  --governance-catalog-provenance synthetic `
  --catalog-approval-company NovaTech `
  --governance-scenarios governance/synthetic_scenarios.json `
  --out data/processed/pipeline_runs/novatech_component4_v2.json
```

On the source files reviewed in this session: 9 complete, 12 incomplete; 13 low,
5 medium, 3 high. These are fictional scenario results, not statements about real
vendors. All 30,000 accepted events, 21 applications, 9 AI/12 non-AI decisions and
4 catalog/5 provisional categories were preserved. Grammarly remains flagged for
category review without editing the catalog. under_review approvals explain some
gaps; exact identity misses explain others. Changed local inputs can change counts.

## Optional live governance research

First validate the offline run. Repeat the same command with --governance-online
and a NEW --out filename (e.g. novatech_component4_v2_online.json). It uses the
existing GEMINI_API_KEY / google-genai installation. --governance-api-model can
select a model. At most two calls are made per targeted app: cited search followed
by structured extraction. Results remain researched, not verified. Source support
and correct plan interpretation should be spot-checked before relying on results.
Live provider compatibility was not tested here; API behavior was tested with fakes.

The new report includes governance_research_cache and research_outcomes. To reuse
successful research, supply --governance-research-cache pointing to that previous
report. Entries expire after 30 days and are keyed by prompt version, model, exact
identity and requested fields. The input cache is read-only; new snapshots are
embedded in new output reports. No catalog is automatically changed. A changed
field request causes fresh research. Catalog conflicts require explicit review.

For future full runs, `python -m ai_discovery.run_pipeline` supports the same
--governance-* flags, --catalog-approval-company, --company and
--activity-provenance, alongside your usual model/catalog/discovery arguments.
Governance catalogs are explicit; omitting them assesses only supplied evidence
and usage. --online enables Component 2 discovery calls; --governance-online
separately enables Component 4 vendor research. Neither is enabled by default.
The full runner uses --governance-policy / --governance-evidence; the standalone
assessment command uses --policy / --evidence for those two inputs.

## Deliverable and compatibility

Install the project-relative files from the archive over the previous delivery.
The Desktop repository was not edited by this session. Review the included patch
and baseline hashes if you changed these files locally. Existing output paths are
refused. No catalogs, secrets, models, caches or activity data are included.

Evidence JSON retains version 1: {"version":1,"applications":{"app_ID":{
"sso_available":{"value":false,"provenance":"synthetic","source":"fixture"}}}}.
Approval evidence additionally requires company. Output governance schema is now
version 2: the primary score includes permitted synthetic evidence and the former
synthetic_demonstration object is removed. contains_synthetic_evidence and
assessment_basis make the intended scope explicit. Legacy risk.py CLI/report.py
still use the older workflow; use governance.assess_pipeline or run_pipeline.

Validation covered the 57 targeted tests and an offline NovaTech reassessment;
it did not include live API calls or the full training/model test suite.
