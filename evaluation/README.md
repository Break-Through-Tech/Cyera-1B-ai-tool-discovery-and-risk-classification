# Trusted test set

Every score in this project so far is measured against labels from Kaggle or
Gemini, and nobody has checked those labels. This folder holds 150 tools for
**people** to label, so the AI detection and category results can be checked
against human judgment.

## What's in it

| Group | Count | Purpose |
|---|---:|---|
| Every app in the usage logs (AI and non-AI) | 22 | The apps the dashboard actually shows |
| Kaggle AI tools from the category model's held-out test split, 8 per category | 112 | Category accuracy across all 14; excludes the 154 tools used in `docs/jev_category_evaluation.md` |
| Conventional enterprise tools | 16 | AI detection: some have AI features but are not AI products |

The order is shuffled, so labellers cannot tell the groups apart.

| File | Who opens it |
|---|---|
| `trusted_test_set/labeling_sheet.xlsx` | **Labellers.** Name, website and description only |
| `trusted_test_set/items.csv` | **Not labellers.** Source, original labels, text |
| `trusted_test_set/predictions.csv` | **Not labellers.** Local model and Jev answers |

## How to label (about 1–2 hours)

1. Make your own copy, e.g. `labels_min.xlsx`. Read the **Guide** tab first.
2. On **Label here**, fill the blue columns using the drop-downs:
   - `is_ai`: `ai`, `non_ai` or `unsure`
   - `category`: one of the 14 categories, `Not applicable (non-AI)`, or `None of these fit`
   - `confidence`: high, medium or low
3. Judge from the description and your own knowledge of the product. **Don't open
   `items.csv` or `predictions.csv`, and don't discuss items with the other labeller
   until both of you are done.** Otherwise the agreement score means nothing.

A second labeller is strongly recommended. Their agreement with you shows how clear
the categories are, and only items you both agree on count as ground truth.

## Score

```bash
python evaluation/score_trusted_set.py --labels labels_min.xlsx labels_teammate.xlsx
```

This writes `trusted_test_set/report.md` and `results.json`:
- labeller agreement (Cohen's kappa)
- AI detection: source labels vs. people
- category accuracy and macro-F1 for **Jev**, the **local model** and **Kaggle's own label**,
  measured against people
- a McNemar test of Jev vs. the local model
- Jev's accuracy at each probability level, which checks the 0.5 review threshold

Disagreements between labellers go to `disagreements.csv`. Settle them together in
another copy of the sheet and pass it with `--adjudicated settled.xlsx`.

## Rebuild

```bash
python evaluation/build_trusted_test_set.py   # refuses to overwrite an existing set
```

Seed 42. Jev answers come from `data/cache/jev_category_cache.json`; new ones need
`TYPESAFE_API_KEY`. `--no-jev` skips Jev.
