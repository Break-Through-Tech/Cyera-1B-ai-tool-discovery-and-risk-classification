# Corrected labels after official-source conflict review

The approved recommendations from the November 5 review are applied in a **separate corrected export** at `data/processed/corrected_labels_2026-11-05/all_corrected.csv`. The original `source_membership/all_labeled.csv` remains the immutable source-assignment record.

| Corrected label | Records |
| --- | ---: |
| AI-focused (`1`) | 16,740 |
| Conventional primary purpose (`0`) | 516 |
| Unresolved (blank label) | 36 |
| Total retained | 17,292 |

Thirty binary labels changed: 13 conventional-source records become 1 and 17 AI-source records become 0. The 36 unresolved records have a blank corrected label and are also listed in `unresolved.csv`; they are not treated as conventional and are excluded from binary preparation.

Forty-two supported feature/product scopes have corrected names. Forty-three descriptions were replaced with capability-focused paraphrases: the 42 scoped records plus the missing AdRoll description. The paraphrases and reviewed scope names are research-derived annotations, not verbatim vendor statements. Examples include HubSpot Campaign Assistant, ClickUp Brain, Fotor AI Image Generator and Xero JAX. This avoids attaching a feature-specific positive label to an unchanged broad platform name or unrelated description. Source-domain fields are retained; a shared domain still does not imply the same application.

Each row records `source_is_ai_tool`, original name, review status, evidence URL and whether scope/description changed. The original description is stored on changed rows; unchanged descriptions remain directly available in both exports. `label_method` distinguishes source assignments from review against official sources; existing provenance codes are retained for reproducibility. The original `potential_label_conflict` flag is historical provenance, not a newly computed conflict status.

`eligible_for_binary_preparation` is false for unresolved records or missing descriptions. There are **16,728** records passing this basic gate. The remaining labeled records with missing descriptions retain their assigned labels but need description repair before text modeling. Passing this gate does not establish training/test readiness: duplicate and related identities still need grouping, unresolved relationships must be handled, and fresh train/validation/test splits must be created. No model has been retrained or evaluated here.

Most unflagged labels remain source-assigned and unverified. The reviewed labels are judgments based on official product sources, not independently human-verified ground truth. This export applies the user's approved recommendations; it does not claim that every catalog record was researched.

```bash
python3 preprocessing_scripts/apply_conflict_review.py
python3 preprocessing_scripts/apply_conflict_review.py --check
python3 -m unittest discover -s tests -p 'test_apply_conflict_review.py'
```

The application checks source/recommendation hashes before generating output. `summary.json` captures counts and input/output hashes for reproduction. The review findings and original evidence remain under `data/reviews/conflicts_2026-11-05/` and `docs/conflict_internet_review.md`. Use the corrected export for the next preparation stage; use the source-membership export only when the experiment explicitly calls for untouched dataset-origin labels.

## November 9 vendor-enrichment reconciliation

The team's vendor-enriched file is a **697-row reviewed subset**, not a replacement for the full 17,292-row corrected dataset. The updated pipeline reconciles that subset with the approved labels and fills vendors from exact matching cached research records, without additional API calls.

- `all_corrected.csv` now includes `vendor`, `vendor_status` and `vendor_evidence_url`. It retains the full 16,740/516/36 class counts. Only identities matching the cached product name, domain and original description receive a vendor; evidence is not transferred to a newly scoped feature.
- `data/processed/final/dataset_b_final.csv` remains a 14-column, 697-row subset. It has 305 AI, 386 conventional and six unresolved records with blank AI labels. Auphonic and Akismet are corrected to AI. Freshworks, Semrush Outline Builder, Notion, Airtable, Podium and Bubble remain excluded from binary preparation.
- Vendor coverage is **676 cached provider suggestions, 11 explicit UNKNOWN results and 10 pending records**, in both matching views. The previous export had 598 blank cells; 582 were recovered from existing matching cached `found` results. Cached suggestions remain model-interpreted research, not independently verified provider identities.
- `dataset_b_final.provenance.json` records per-record label/vendor status, citations, cache fingerprints, input/output hashes and explicit synthetic-governance provenance. This sidecar must travel with the subset. The synthetic approval values are not company-scoped.

The supplied approval, SSO, retention, training-on-customer-data and `risk_label` fields are **fictional demonstration data**, preserved from the existing governance generator. They were not established by the vendor research. The stored illustrative `risk_label` uses a different policy from the governance engine and must not be treated as that engine's output or as ground truth.

The risk CLI rejects a provenance-marked synthetic metadata file unless `--synthetic-metadata` is passed. With that explicit demo option, matched rows are marked `synthetic_dataset_b`. Company-unscoped approval values remain ignored; use a company scenario to supply approvals. The engine recalculates scores and never consumes the stored `risk_label` as an input. This protection depends on retaining the provenance sidecar; do not strip it when distributing this dataset.

```bash
# Rebuild corrected full data AND reconciled final subset from tracked source/cache files.
python3 preprocessing_scripts/apply_conflict_review.py
python3 preprocessing_scripts/apply_conflict_review.py --check

# Optional future vendor research writes an intermediate, preserving corrected final labels.
python3 preprocessing_scripts/fill_dataset_b_vendors.py --dry-run

# Explicit synthetic-policy demonstration only:
python3 -m governance.risk --company 'Summit Financial' \
  --activity 'data/raw/summit_financial_usage (1).csv' \
  --metadata data/processed/final/dataset_b_final.csv --synthetic-metadata \
  --out /tmp/summit-vendor-subset-demo.json
```

Vendor research now defaults to `data/processed/vendor_research/dataset_b_vendor_enriched.csv`, so rerunning it does not overwrite corrected labels in the final export. Its cited cache is the reconciliation input; run the application command afterward. Blank/unknown vendors need further research, not guesses. Vendor names, synthetic governance fields and prior risk labels are enrichment for investigation, **not predictor inputs for the binary classifier**. Training splits, retraining and the Jev benchmark are still pending.


**Training update:** [Grouped full-data preparation and binary retraining](corrected_full_binary.md) now use this export. Earlier mentions of pending splitting/training describe the enrichment stage; the original corrected export itself remains unsplit. Jev comparison remains pending.
