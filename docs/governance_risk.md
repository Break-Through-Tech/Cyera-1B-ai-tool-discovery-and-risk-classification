# Explainable governance risk engine — issue #8

`governance/risk.py` implements a configurable rule-based priority assessment using Dataset B and the two synthetic enterprise activity CSVs. It returns a 0–100 score, low/medium/high band, per-rule point contributions and evidence, unknown fields and a completeness status. It has no package dependencies beyond Python's standard library.

This is a demonstration governance policy, not a validated security model or breach probability. It does not learn from or overwrite the AI classifiers, interpret model probabilities as risk, or use Dataset B's existing `risk_label` as an input. All observed applications are assessed; upstream discovery can filter these outputs to AI-related applications later. The rule engine does not itself identify AI tools or scrape vendor policies.

## Default policy

| Signal | Points | Default trigger |
| --- | ---: | --- |
| Explicitly unapproved | 25 | Company-scoped scenario says `unapproved` |
| SSO unavailable | 10 | Explicit `false` |
| Customer content used for training | 20 | Explicit `true` |
| Long retention | 10 | More than 90 days or `indefinite` |
| Administrative controls unavailable | 10 | Explicit `false` |
| Enterprise plan unavailable | 5 | Explicit `false` |
| Potentially sensitive department | 10 | Finance, HR or Legal, case-insensitive |
| Large uploads | 15 | Mean at least 5,000,000 bytes OR any event at least 10,000,000 bytes |
| Large downloads | 5 | Mean at least 10,000,000 bytes |
| Broad adoption | 5 | At least 50 distinct users |

Contributions add; the final score is capped at 100. The uncapped total is also returned. Low is below 20, medium is 20–49 and high is 50–100. Thresholds and weights live in `governance/policy.json` and can be changed before assessing a new scenario. Upload mean and maximum trigger one rule, not two stacked penalties. Decimal byte thresholds are explicit; upload volume does not prove data leakage, department does not prove sensitive content, and download volume may reflect ordinary work.

Approvals do not subtract points or override behavioral evidence. Unknown or unsupported policy values add no points and are listed as unknown. **A low score with incomplete evidence is not assurance of safety**: it always requests review. `assessment_status=complete` means the six policy fields and transfer observations were supplied, not that the vendor is verified or the application safe. All rules still have the limitations of a simplified demonstration policy.

## Data handling and scope

Activity is aggregated within one company, by exact normalized hostname plus application path. Query strings and trailing slashes are removed; subdomains and paths remain distinct. In particular `github.com` and `github.com/copilot` never share an assessment, nor do Google Docs and Drive. This is a deliberate conservative join; aliases require explicit upstream normalization. Duplicate Dataset B identities are treated as ambiguous and not joined rather than picking an arbitrary catalog record. Application names are retained as observed aliases, not used to merge records.

Distinct users, department lists, event counts, observation dates, total/mean uploads and downloads, maximum upload and transfer coverage are included. Individual employee IDs are used to count users but are not exported. Duplicate or missing event IDs, missing required fields, invalid timestamps and negative/non-integer transfer counts are rejected. Missing transfer values stay unknown; partially observed transfer fields mark the assessment incomplete. Timestamps without a timezone in the supplied synthetic logs are interpreted as UTC; explicit offsets are normalized to UTC.

Dataset B's approval column is ignored because it is not scoped to a company. The optional `--scenarios` overlay supplies explicitly company-scoped synthetic approval and hypothetical control metadata. It requires `synthetic: true`, a matching company and only recognized governance fields. No unavailable approval, SSO or retention value is inferred from a missing catalog match. The catalog's populated vendor-policy fields, if present, are accepted as supplied metadata, not independently verified facts.

The current 16,762-row Dataset B has blank governance policy fields. Consequently observed-data reports are incomplete. `governance/synthetic_scenarios.json` deliberately supplies a few fictional approval/control configurations to exercise low/high outcomes. **Its real product names do not make the hypothetical settings real vendor claims.** It is not a maintained enterprise approval registry. Each output records whether scenarios were applied, company, metadata origin, policy version and input file SHA-256 hashes. No network calls or LLM decisions occur during scoring.

## Included scenario results

Checked-in `artifacts/governance/` reports cover both activity files with and without the synthetic overlay: 25,120 Summit Financial events across 22 application identities, and 30,000 NovaTech events across 21 identities.

| Case | Score | Band | Evidence status |
| --- | ---: | --- | --- |
| Summit DocuMind AI, observed inputs | 30 | medium | incomplete |
| Summit DocuMind AI, hypothetical unapproved status | 55 | high | incomplete |
| Summit ChatGPT, hypothetical unapproved/control settings | 95 | high | complete supplied fields |
| NovaTech ChatGPT, hypothetical approved/control settings | 15 | low | complete supplied fields |

DocuMind AI is absent from the catalog and appears in 120 Summit events from 84 distinct Finance users. Observed signals contribute 10 for Finance, 15 for upload volume and 5 for adoption. Its approval and vendor controls are unknown in the observed report. The synthetic overlay adds only an explicit unapproved status (+25), leaving SSO, retention and other controls unknown. The engine therefore demonstrates the intended shadow-AI review scenario without inventing vendor-policy evidence.

## Run and integrate

```bash
python3 -m governance.risk --company 'Summit Financial' \
  --activity 'data/raw/summit_financial_usage (1).csv' \
  --out /tmp/summit-observed.json

python3 -m governance.risk --company 'Summit Financial' \
  --activity 'data/raw/summit_financial_usage (1).csv' \
  --scenarios governance/synthetic_scenarios.json \
  --out /tmp/summit-synthetic-scenario.json

python3 -m governance.risk --company NovaTech \
  --activity 'data/raw/novatech_usage (1).csv' \
  --scenarios governance/synthetic_scenarios.json \
  --out /tmp/novatech-synthetic-scenario.json

python3 -m unittest discover -s tests -p 'test_governance.py'
```

The CLI refuses to overwrite existing results. For programmatic use, import `run(company, activity, metadata_path, policy_path, scenarios=None)` or `assess(aggregated_usage, metadata, policy)`. The JSON's `applications` list is ordered by descending score and then application identity. A dashboard should show `risk_level`, `assessment_status`, `unknown_fields`, `metadata_source` and `reasons` together, and visibly mark synthetic scenarios. Reporting only a low/medium/high badge would hide material uncertainty.

Tests cover policy thresholds, caps, known versus missing controls, missing transfer coverage, duplicate/invalid events, application-path separation, ambiguous metadata, company-scoped approvals and the actual synthetic DocuMind scenario. They verify rule consistency, not real-world risk accuracy. Future work is a shared discovery-to-assessment adapter and an investigation interface; vendor facts and organization-specific approval decisions require their own evidence and scope before production use.
