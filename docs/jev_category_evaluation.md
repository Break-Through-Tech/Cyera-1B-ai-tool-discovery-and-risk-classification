# Jev vs. local model for AI-tool categories

**Question:** can the Jev API (Typesafe `systemone`, `jev-latest`) assign the 14
Kaggle purpose categories better than the project's local TF-IDF model, on the kind
of text a newly discovered app actually has?

## Result

On the same 154 held-out tools (11 per category), using the product **description
only**:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| **Jev** | **59.7%** | **0.591** |
| Local TF-IDF + logistic regression | 46.8% | 0.477 |

- Jev was right where the local model was wrong 32 times; the reverse happened 12
  times (exact McNemar test, p = 0.004).
- Macro-F1 gain: +0.114, 95% bootstrap interval +0.031 to +0.196.
- Jev scored higher in 12 of 14 categories, tied in one, and was lower only in
  Personal And Lifestyle.
- **Jev's probability is informative.** When it was at least 0.7 (66% of tools),
  Jev was right 70% of the time. Below 0.5 (16% of tools), it was right only 29% of
  the time. A low Jev probability is a useful "needs human review" signal.

## Why description only

The local model's reported held-out macro-F1 of 0.735 uses description **plus**
Kaggle's `Primary Task` phrase. In this catalog that phrase maps to a category
almost one-to-one (3,370 of 3,385 task values have exactly one category), and 86%
of test tools reuse a task phrase seen in training. A plain lookup of the phrase
reaches 85.8% accuracy. A newly discovered app has no Kaggle task phrase, so
description-only is the realistic test. The local model's 0.477 here is consistent
with its own description-only diagnostic (0.449).

## Method

- Rows: the local run's held-out **test** partition only
  (`models/categories/kaggle14_run1/split_manifest.csv`), so the local model never
  trained on them. Stratified sample, seed 42, rows without a description excluded.
- Jev received a `choice` question whose 14 options were each described by that
  category's most frequent primary tasks in **development rows only**
  (`artifacts/jev_category_eval/criteria.json`), so no test label leaked in.
- Both models saw identical text. Only public product descriptions were sent; no
  employee, usage or company data.
- Cost: 154 calls, about 161k input and 25k output tokens in total.

## Limits

- The reference labels are Kaggle's categories, which are noisy. Several Jev
  "errors" are arguably better than the label (e.g. an e-commerce copywriting tool
  labelled Specialized Technologies, predicted Marketing And Sales). Neither score
  is true accuracy.
- 154 tools; per-category scores rest on about 11 rows each.
- Jev probabilities are provider outputs, not calibrated confidence.
- Not yet wired into the operational pipeline. The pipeline guide anticipates an
  API category source (`category_source: api`); that integration is future work.

## Reproduce

```bash
python category_training/jev_category_eval.py --limit 10   # smoke test
python category_training/jev_category_eval.py              # full sample
```

Needs `TYPESAFE_API_KEY` in the project-root `.env` and the local model in
`models/categories/kaggle14_run1` (see `ai_discovery/PIPELINE_GUIDE.md` §8E).
Responses are cached in `data/cache/jev_category_cache.json`, so a re-run with the
same settings makes no new API calls.

Outputs: `artifacts/jev_category_eval/summary.json`, `predictions.csv`
(per tool: actual, Jev choice, probability, top three, local prediction),
`criteria.json`.
