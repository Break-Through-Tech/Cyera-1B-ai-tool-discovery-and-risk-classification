# Corrected labels after official-source conflict review

The approved recommendations from the November 5 review are applied in a **separate corrected export** at `data/processed/corrected_labels_2026-11-05/all_corrected.csv`. The original `source_membership/all_labeled.csv` remains the immutable source-assignment record.

| Corrected label | Records |
| --- | ---: |
| AI-focused (`1`) | 16,740 |
| Conventional primary purpose (`0`) | 516 |
| Unresolved (blank label) | 36 |
| Total retained | 17,292 |

Thirty binary labels changed: 13 conventional-source records become 1 and 17 AI-source records become 0. The 36 unresolved records have a blank corrected label and are also listed in `unresolved.csv`; they are not treated as conventional and are excluded from binary preparation.

Forty-two supported feature/product scopes have corrected names. Forty-three descriptions were replaced with capability-focused paraphrases: the 42 scoped records plus the missing AdRoll description. The paraphrases and reviewed scope names are explicitly assistant-authored. Examples include HubSpot Campaign Assistant, ClickUp Brain, Fotor AI Image Generator and Xero JAX. This avoids attaching a feature-specific positive label to an unchanged broad platform name or unrelated description. Source-domain fields are retained; a shared domain still does not imply the same application.

Each row records `source_is_ai_tool`, original name, review status, evidence URL and whether scope/description changed. The original description is stored on changed rows; unchanged descriptions remain directly available in both exports. `label_method` distinguishes source assignments from assistant source review. The original `potential_label_conflict` flag is historical provenance, not a newly computed conflict status.

`eligible_for_binary_preparation` is false for unresolved records or missing descriptions. There are **16,728** records passing this basic gate. The remaining labeled records with missing descriptions retain their assigned labels but need description repair before text modeling. Passing this gate does not establish training/test readiness: duplicate and related identities still need grouping, unresolved relationships must be handled, and fresh train/validation/test splits must be created. No model has been retrained or evaluated here.

Most unflagged labels remain source-assigned and unverified. The reviewed labels are assistant judgments against official product sources, not independently human-verified ground truth. This export applies the user's approved recommendations; it does not claim that every catalog record was researched.

```bash
python3 preprocessing_scripts/apply_conflict_review.py
python3 preprocessing_scripts/apply_conflict_review.py --check
python3 -m unittest discover -s tests -p 'test_apply_conflict_review.py'
```

The application checks source/recommendation hashes before generating output. `summary.json` captures counts and input/output hashes for reproduction. The review findings and original evidence remain under `data/reviews/conflicts_2026-11-05/` and `docs/conflict_internet_review.md`. Use the corrected export for the next preparation stage; use the source-membership export only when the experiment explicitly calls for untouched dataset-origin labels.
