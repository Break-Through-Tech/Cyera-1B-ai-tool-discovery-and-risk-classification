# Source-reviewed classification dataset v2

This release adds **200 distinct conventional applications** and a balanced category sample of **96 AI products/features (8 in each of 12 categories)**. It preserves v1 datasets and model results. These labels were checked by an assistant against retrieved official product pages on 2026-11-04; they are **not independently human-verified ground truth**.

## Label contract

- `is_ai_tool=1`: the named product or specifically scoped feature centers on AI assistance, generation, prediction, or model training/inference.
- `is_ai_tool=0`: the named product centers on a conventional application workflow. This does **not** assert that its vendor or product has no AI capabilities.
- `has_ai_features=yes`: the cited source explicitly supports AI functionality. `unknown` means that availability was not established in this review. There are no inferred `no` labels.
- `category`: primary AI use case, populated only for positive examples. Conventional business functions are stored separately.
- These fields do not classify governance risk, security, data retention or vendor compliance.

For example, ClickUp is a conventional work management application (`0`, with AI features); ClickUp Brain is a specifically named AI assistant (`1`). GitHub Enterprise and GitHub Copilot follow the same scope distinction. This is a task-specific operational definition, rather than a universal definition of an “AI company.” Enterprise discovery should report the feature flag alongside the binary prediction.

The 12 named AI categories include `Audio / Speech Tools`. This replaces a forced catch-all for this curated sample; it does not rewrite v1's `Other` labels. Model category vocabularies must be fitted afresh for v2.

## Evidence and selection

`data/labels_v2/source_labels.json` is the editable source for a **future** dataset version and the immutable input for this frozen v2 release. Every entry includes the product scope, an assistant-written description and rationale, requested and resolved official URLs, page title, check date, extracted source-text SHA-256, label method and vendor/product family. Full downloaded web pages are not committed. A hash records which extracted text was reviewed, but does not let a later reader reconstruct a changed page; follow the source and reassess stale claims.

Candidate collection covered 244 conventional and 155 AI records. Retrieval was only a first filter: parked, inaccessible, identity-mismatched or changed-purpose candidates were omitted. For example, Rytr's current page describes relationship management rather than the proposed writing-generator label; Narrato redirects to Typeface; the Tabnine URL redirects to Tricentis. The two Copilot candidates that redirected to the same Microsoft page count as one product. Descriptions are paraphrases, not copied marketing text or scraped usage events.

The conventional sample spans 22 business functions, including CRM, HR, finance, support, storage, scheduling, identity, monitoring and development. Selection used a round-robin across functions to limit concentration. Products with optional AI features are retained as harder conventional cases. Ambiguous broad repositionings were omitted. AI examples were selected for current functional support and category coverage; this is not a random sample of enterprise traffic.

Official pages can overstate capabilities. Category boundaries and scope choices remain subjective. A second independent reviewer should resolve disagreements before making high-confidence benchmark claims. AI availability is a conservative partial annotation, not an exhaustive feature inventory.

## Frozen splits

| Split | Conventional | AI | Total | AI examples per category |
| --- | ---: | ---: | ---: | ---: |
| Train | 152 | 48 | 200 | 4 |
| Validation | 12 | 12 | 24 | 1 |
| Test | 36 | 36 | 72 | 3 |

The test has 36 conventional examples instead of v1's four. It is still small: three test positives per category cannot support precise category-level conclusions. Training is intentionally more conventional-heavy; use training-only class weighting or sampling when comparing models, and choose settings on validation only.

Related vendor/product families and identical normalized descriptions stay in one split. Explicit group mappings cover Microsoft/GitHub, Atlassian, Salesforce/Slack/Tableau, SAP/Concur and other selected parent/feature relationships. Family metadata is assistant-curated and may need extension when new sources are added. Never infer vendor ownership solely from a documentation hosting domain.

`manifest.json` records the source-label and output hashes. `protected_evaluation_groups.json` lists held-out families and description hashes. This release was frozen before v2 model training; no new model metrics are claimed.

## Reproduce and use

```bash
python3 preprocessing_scripts/prepare_classification_v2.py
python3 -m unittest discover -s tests -p 'test_classification_v2.py'
```

Normal execution checks byte-for-byte reproduction. `--freeze` writes an initial release, but refuses a changed existing manifest. Change labels in a **new version**, document the reason and create a newly versioned manifest rather than silently relabeling this test set.

The existing v1 trainers are **not a drop-in v2 training path**: their source-review overlay and category filtering were designed for v1. The next modeling change must load these splits explicitly, fit preprocessing/heads on train only, choose settings on validation, then evaluate the test once after selection. Use only product name/description as predictor inputs; evidence URLs, family IDs, review metadata, feature flags, labels and categories must not leak into the binary model.

Do not evaluate existing v1 models on this test and describe it as independent: those models may already have seen the same products in the catalog. If augmenting v2 train with v1/catalog data, first map all aliases and families and exclude every held-out related application and duplicate description. The protected list is a starting point; exact string filtering alone is insufficient. Pretrained encoders may also already know these public products. Results from synthetic assistant-authored descriptions do not establish performance on real enterprise descriptions or telemetry.
