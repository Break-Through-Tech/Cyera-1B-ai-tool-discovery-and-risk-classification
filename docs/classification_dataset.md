# Issue #5: Prepare Classification Dataset

## Result

The dataset preparation implementation combines the team's 305 model-accepted AI records with its 30 historical non-AI enterprise records. It standardizes the features and binary labels, validates required fields, removes repeated application identities, withholds conflicting identity labels, and writes reproducible application-separated splits.

This completes dataset preparation for an initial baseline using the available team labels. It does not complete independent label verification, risk classification, or model training. All labels are explicitly marked `weak_supervision`.

| Split | AI (`1`) | Non-AI (`0`) | Total |
|---|---:|---:|---:|
| Train | 213 | 22 | 235 |
| Validation | 46 | 4 | 50 |
| Test | 46 | 4 | 50 |
| Combined | 305 | 30 | 335 |

The roughly 70/15/15 allocation is by connected application groups. Jira and Confluence share atlassian.com and remain together, so exact class proportions are not forced by splitting related applications apart. No current inputs were excluded; missing values and duplicates are nevertheless checked and reported by the builder.

## Label meaning

- `is_ai_tool = 1`: an AI-focused application according to the accepted catalog review.
- `is_ai_tool = 0`: a conventional enterprise application's primary purpose, according to the team's negative seed list.
- `category`: a business-function label for AI examples, inherited from Gemini. It is blank for non-AI rows because their old enterprise taxonomy differs from the AI taxonomy.
- Risk is a separate future target. No `risk_label`, approval status or vendor security claim is included in the classification output.

A negative label does not mean that a vendor has no AI features. The scope follows VISION.md's distinction between primarily AI applications and ordinary SaaS platforms. For instance, generic Slack usage and use of an AI feature within Slack are not interchangeable observations. Name/domain-only telemetry cannot always distinguish them. Scope-specific label review is still needed before claiming reliable discovery accuracy.

## Inputs and provenance

1. `data/processed/dataset_b/sample/accepted_dataset_b.csv`: 305 examples from the existing 500-row Gemini review. Accepted means model-accepted, not human-verified. Their descriptions and categories are retained.
2. `data/raw/non_ai_enterprise_tools.csv`: exactly recovered from Git commit `87f760769e39e89ba3bcb1691d8f0eb32c60e4b2`, original path `data/non_ai_enterprise_tools.csv`. This is the team's 30-example negative dataset referenced in issue #3. It was absent from current main's layout, not lost from history. Its historical governance fields are preserved in the raw source for provenance only and must not be treated as verified vendor policies.

No new external API calls or vendor-policy research were used. Uncertain and rejected catalog records are not silently converted into negatives. The 22 model-suggested negatives from the previous smoke-test package are not used here.

## Reproduce

Python 3.9+; no third-party dependencies.

From the repository root:

```sh
python3 preprocessing_scripts/prepare_classification.py
python3 -m unittest discover -s tests -v
```

The script finds repository inputs relative to its own location, so it can also run from another working directory. Use `--out /path/to/new-output` to write elsewhere, or `--overwrite` to deliberately replace generated outputs. `--ai`, `--non-ai` and `--seed` are configurable. The original source files are never modified.

## Outputs

`data/processed/classification/` contains:

- `classification.csv`: all 335 applications, annotated with split membership.
- `train.csv`, `validation.csv`, `test.csv`: mutually disjoint partitions.
- `excluded.csv`: exclusion reasons; currently headers only.
- `summary.json`: label mapping, class and category distributions, checks and limitations.
- `manifest.json`: input SHA-256 hashes, seed, split method and feature policy.

## Data dictionary

| Field | Meaning |
|---|---|
| application_id | Stable hash of normalized name and hostname |
| group_id | Connected identity group assigned to a single split |
| tool_name | Whitespace-normalized application name |
| domain | Lowercase hostname, without scheme, `www.`, or path |
| description | Whitespace-normalized source description |
| text_for_model | Tool name plus description; same composition for both classes |
| is_ai_tool | Binary detection target: `1` AI, `0` conventional primary purpose |
| category | Model-suggested AI business function; blank for negatives |
| category_source | Provenance for the category, blank when absent |
| label_source | `gemini_text_review` or `team_non_ai_seed` |
| label_quality | Always `weak_supervision`; never misrepresented as gold labels |
| source_record_id | Original tool ID from the corresponding input |
| source_file | Input filename for traceability |
| split | `train`, `validation`, or `test` |

## Cleaning and leakage controls

- Normalize whitespace, names used for matching, and URL hostnames.
- Require nonempty name/description, valid hostname, and the expected source label. Missing descriptions are excluded rather than invented.
- Deduplicate normalized name + hostname identities. Opposite labels for that identity withhold all conflicting rows.
- Link records sharing a normalized name, hostname or exact normalized description; keep their transitive connected group in one split. Different legitimate applications on a shared host can have opposite labels while remaining in the same group.
- Use a fixed seed of 42, stable ordering and approximate class-stratified group allocation.
- Require both classes in each split and verify no cross-split IDs, groups, hostnames, normalized names or identical normalized descriptions.
- Use **only `text_for_model` as the initial binary model input**. Do not use category, label provenance, filenames, risk/approval metadata or split membership as predictors. The shared name/description feature composition avoids handing the model source-specific extra fields such as keywords available only for positives.
- Fit TF-IDF and any other learned preprocessing on training rows only. Tune using validation; preserve test for final evaluation.

These controls do not automatically resolve all aliases, rebrands, different subdomains or near-duplicate wording. Manual identity and semantic-duplicate review may find additional related applications.

## Validation performed

Eight passing unittest checks cover the real dataset counts, deterministic results even when input order changes, both classes per split, transitive/mixed-label shared-host grouping, duplicate removal, identity label conflicts, missing/invalid fields, insufficient class groups, and separation of labels/governance fields from model text. The builder also runs its overlap and class checks on every execution.

## Limitations and next task

AI examples make up 91.0% of this dataset. An always-AI baseline gets 91.0% overall accuracy and 92% test accuracy without learning useful detection. Only four non-AI examples appear in each evaluation split; a single error changes non-AI recall by 25 percentage points. The class counts support baseline development, not strong claims about generalization.

Both source label sets need an independent audit. There may also be writing-style differences between catalog and team-authored descriptions. Do not treat models trained on these data as detecting all AI capabilities in ordinary SaaS platforms.

All 12 proposed AI categories appear in training; validation has no Research / Search example. The splits target binary classification; a future multiclass evaluation should add reviewed examples and consider category-stratified grouping. The existing category labels are useful starting annotations, not independently verified targets.

The next implementation task is issue #6: TF-IDF + Logistic Regression (and a majority-class baseline), with accuracy, per-class precision/recall, macro F1 and confusion matrices. Keep a separately reviewed unseen-tool evaluation set when expanding the corpus. Risk scoring belongs to issue #8 and must combine metadata with usage context, not reinterpret `is_ai_tool` as risk.

## Issue #5 acceptance checklist

- Combined AI/non-AI data: yes, 335 rows.
- Consistent binary labels and common features: yes.
- Duplicate and missing-description handling: implemented; no current exclusions.
- Class distribution examined: yes, imbalanced and documented.
- Application-separated training/validation/test splits: yes, checked.
- Prepared for binary modeling and later categorization: yes, with provenance and category limitations.
- Human-verified gold labels: no; not claimed.
