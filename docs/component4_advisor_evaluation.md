# Component 4 policy evaluation

## Purpose

These fictional cases evaluate whether the governance priorities are useful. They do not assert vendor facts. Synthetic data is valid for testing; independently assigned judgments are needed to evaluate usefulness.

Ask the advisor to complete the cases BEFORE viewing engine scores or point weights. For every case record priority (low/medium/high, or insufficient information), whether more evidence is needed, and a short rationale. A priority and an evidence gap can coexist. Do not force unknown cases into low. Keep these first judgments unchanged when comparing results later.

## Shared baseline

Unless a case overrides it: a fictional application is approved for the company, supports SSO and administrative controls, has an enterprise plan available, retains data for 30 days, and does not train on customer content. There are 10 Engineering users, mean upload 100,000 bytes, maximum single upload 500,000 bytes, and mean download 100,000 bytes. All evidence is complete synthetic scenario evidence. Capabilities being available does not establish that the company enabled them. AI status is explicitly listed.

The priority concerns governance review, not a probability of breach. Department and data volumes are proxies: content sensitivity is not observed.

## Cases (only listed facts differ from baseline)

| ID | AI status | Scenario changes | Advisor priority | More evidence? | Rationale |
|---|---|---|---|---|---|
| A01 | AI | No changes | | | |
| A02 | Non-AI | No changes | | | |
| A03 | AI | Retention 90 days | | | |
| A04 | AI | Retention 91 days | | | |
| A05 | AI | Retention indefinite | | | |
| A06 | AI | Retention unknown | | | |
| A07 | Non-AI | Explicitly unapproved | | | |
| A08 | AI | Customer content used for model training | | | |
| A09 | Non-AI | SSO unavailable | | | |
| A10 | Non-AI | No enterprise plan available; SSO and admin controls still available | | | |
| A11 | AI | 50 Finance users; other baseline facts unchanged | | | |
| A12 | AI | Unapproved, SSO unavailable, 50 Finance users | | | |
| A13 | Non-AI | Mean upload 4,000,000 bytes; maximum single upload 10,000,000 bytes | | | |
| A14 | AI | Approval under review; other baseline facts unchanged | | | |
| A15 | Unresolved | Approval, controls, retention, training policy and usage all unknown | | | |
| A16 | AI | Unapproved; no SSO/admin controls/enterprise plan; training true; indefinite retention; 50 Finance users; mean upload 12,000,000 bytes and maximum 20,000,000; mean download 10,000,000 | | | |

## Compare after review

1. Freeze the policy version and record each engine score, level, completeness and triggered rules separately from the advisor columns.
2. For cases given low/medium/high judgments, report exact agreement as a count and fraction, and a 3-by-3 confusion table. Report insufficient-information cases separately; also compare evidence-gap judgments.
3. Inspect every disagreement, especially advisor-high/engine-low cases. Do not assume that either side is automatically correct.
4. Discuss the retention step at 90 days, enterprise-plan availability, broad adoption, and the difference between available controls and enabled company controls.
5. If the policy changes, record why. Judge a fresh set of scenarios before scoring them to avoid evaluating only on cases used to tune the policy.

This small set supports a prototype policy review, not a general accuracy or security claim. External evidence accuracy needs a separate source audit; these tests do not perform one.

## Code review and test coverage

The uploaded files contain 54 tests: legacy governance 13, integrated Component 4 18, evidence assembly/research 13, extraction repair 10. Existing coverage includes missing evidence, preserved identities, provenance, score boundaries/cap, AI applicability, company scope, plan gates, caching, timeouts and interrupted extraction.

The new tests/test_governance_evaluation.py adds 12 offline tests, with multiple subcases: retention 0/30/90/91/indefinite/unknown; mean and single upload boundaries; download and adoption boundaries; isolated controls; non-AI training exclusion; approval under review; natural score combinations; integrated score cap and upload deduplication; and provenance equivalence without losing labels.

Expected point totals are written explicitly rather than computed by the implementation under test. Tests intentionally target demo-1.0; if the policy changes, review the expectations rather than automatically replacing them with engine outputs.

## Installation and command

Merge the ZIP's tests and docs folders into the project. No existing source files need replacement or deletion. No PowerShell scripts or network calls are involved.

From the project root, run:

```text
python -m unittest discover -s tests -p test_governance_evaluation.py
```

Expected: 12 tests, OK. Share the terminal output if anything fails. For usefulness evaluation, share the advisor's filled case table; no fresh online pipeline run is required.
