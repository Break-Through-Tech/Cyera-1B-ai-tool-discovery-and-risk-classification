# Full source-membership binary labels

This dataset implements the requested rule directly: every record from the full AI catalog is `1`, and every record from the full conventional source is `0`. There is no sample-size limit, Gemini acceptance filter, model inference or API call. This is a new export; existing source files, reviewed subsets, governance data, and prior model artifacts are unchanged.

| Source | Assigned label | Rows |
| --- | ---: | ---: |
| `data/processed/dataset_b/dataset_b_master.csv` | 1 | 16,762 |
| `data/processed/non_ai/dataset_b_non_ai_master.csv` | 0 | 530 |
| Combined | | 17,292 |

The conventional master contains the team's new 500-record SaaS source plus the existing 30 conventional examples. All records are retained, including rows previously rejected or marked uncertain, duplicates and missing descriptions. Labels are **source-assigned, not independently verified judgments about functionality**. They also do not describe governance risk.

## Outputs

`data/processed/source_membership/all_labeled.csv` contains every labeled source record. `ai_labeled.csv` and `non_ai_labeled.csv` provide the two individual labeled sources. Each row preserves the name, domain, description, source record ID and original label. `record_id` namespaces IDs by source. `label_method` records `source_dataset_membership`.

Cross-source matches by normalized name, domain/application path, or nonempty description flag 201 records in 144 potential conflict groups. Matching a shared host alone does not prove identical applications. Conflicts **do not change or remove the requested labels**. The row flags identify affected records, and `potential_conflicts.csv` lists matching groups without repeating full descriptions. The same pair can match multiple signals, so group counts are not counts of distinct applications.

`summary.json` records complete row/class counts and source/output SHA-256 hashes.

```bash
python3 preprocessing_scripts/label_source_membership.py
python3 preprocessing_scripts/label_source_membership.py --check
python3 -m unittest discover -s tests -p 'test_source_membership.py'
```

This change assigns labels only: it does not sample, create new evaluation splits, retrain models or call Jev. Before a later comparison, conflicting/related identities should be handled explicitly, missing descriptions checked, and splits grouped to avoid duplicate leakage. The full data is class-imbalanced (16,762 versus 530); an unbalanced all-AI predictor would appear highly accurate, so per-class recall/precision and macro F1 will matter. Any benchmark against these targets measures agreement with source assignments rather than independently verified real-world accuracy. Dataset-origin writing style can also be a confounder.
