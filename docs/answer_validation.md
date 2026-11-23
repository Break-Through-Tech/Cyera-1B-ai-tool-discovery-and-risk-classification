# Analyst answer validation and grounded fact selection

This extends the report assistant with harder question contracts and stricter generated-fact validation. No discovery/category models are retrained, no governance scores are changed, and no paid calls or dashboard work occur.

## Problems addressed

- User-count questions previously fell into the application-count route. They now abstain explicitly: application summaries cannot establish distinct users across tools or department-specific totals. The evidence view still displays recorded whole-application user counts.
- An absent application in a narrow named question could otherwise retrieve unrelated semantic neighbors. Supported named forms now use exact application matching; absent identities return no matching evidence.
- Unknown risk scores were rejected during indexing. A null score with `risk_level=unknown` now remains unknown, is indexed and is sorted after known scores. Missing user/event counts remain null, never fabricated zero.
- Evidence conflicts, category warnings, checks and user-count incompleteness are preserved. A conflicting control stays unknown. `not_applicable` is excluded from both true/false and unknown-control predicates.
- Exact control queries distinguish false from unknown. For example, “without SSO” requires explicit false; an absent SSO field is not sufficient.
- Questions asking for safety, compliance, breaches/leaks, probabilities, temporal trends or cross-company comparisons abstain within the limited snapshot interface. “Not approved” requires clarification because unapproved, under review and unknown are separate states.

The wording parser is deliberately narrow, not general natural-language understanding. Every applied filter is returned. Unsupported wording may require explicit flags or a reformulated question. Explicit scope does not override unsupported safety or aggregation conclusions.

## Current index schema 3 (ChromaDB)

Rebuild existing indexes with the current `ai_discovery.retrieval` code, using a **new output directory**. Legacy schema-1/2 bundles are refused rather than silently treating an old FAISS index as a Chroma collection. Evidence facts and answer validation retain the same contract.

```bash
python -m ai_discovery.retrieval \
  --report data/processed/pipeline_runs/novatech_component4_v2.json \
  --encoder /path/to/local/MiniLM/encoder \
  --out data/cache/governance_retrieval/novatech-grounded-chroma-v3

python -m ai_discovery.ask \
  --index data/cache/governance_retrieval/novatech-grounded-chroma-v3 \
  --company NovaTech --question 'Which applications do not support SSO?'

# Explicit predicates avoid relying on wording inference:
python -m ai_discovery.ask \
  --index data/cache/governance_retrieval/novatech-grounded-chroma-v3 \
  --company NovaTech --question 'Show matching report evidence' --no-infer \
  --control sso_available=false
```

`--control` can be repeated. Allowed fields are SSO, customer-data training, admin controls, enterprise-plan availability and retention. Boolean values use `true` or `false`, unknown uses `unknown`, retention uses its exact supplied value. Filters test source values, not related semantic text. Unknown risk also has an explicit `--risk-level unknown` option.

## Optional generation: exact facts rather than free prose

Citation IDs alone do not prevent a sentence such as “approved” from citing an unapproved application. Optional generation now returns this structured selection:

```json
{
  "claims": [
    {"evidence_id": "ev_...", "field": "/approval", "value": "unapproved"}
  ],
  "insufficient_evidence": false
}
```

The response is accepted only when the evidence ID belongs to this query's retrieved results, the field exists, and the typed JSON value exactly matches the corresponding indexed fact. Invented text/extra properties, cross-company IDs, nonexistent fields and changed values are rejected. False is not accepted as numeric zero. Rendering uses validated source values and includes evidence context: company, synthetic/activity provenance, assessment completeness, category provenance/status, user-count scope and quality warnings.

This changes the experimental `--generate` response contract. Unrestricted generated prose is no longer accepted. The model selects report facts; application code renders them. Unsupported questions, absent identities and counts do not make provider calls.

These checks establish only exact consistency with the supplied report projection. They do not establish vendor truth, prove that a fact is relevant to the question, or protect against incorrect source data. Live provider behavior and useful fact selection remain unvalidated; simulated clients exercise acceptance/rejection contracts.

## Functional evaluation

`data/reviews/assistant_validation_2026-11-18/` contains two small fictional report fixtures and 26 developer-authored cases. These are labeled QA fixtures, not real enterprises/vendors or risk-rule correctness examples. They exercise:

- AI/conventional/unknown and approval distinctions;
- unknown risk scores and missing usage counts;
- exact true/false/unknown control predicates, including not-applicable controls;
- numeric priority ranking and absent application identities;
- company scope, unsupported safety/trend/user-total questions and abstention.

All **26/26 functional cases pass** against a local MiniLM/ChromaDB index. This is a functional contract result, **not** 100% general question-answering accuracy or an independent benchmark. The cases were authored to specify desired behavior; fixes addressed implementation behavior, not model tuning. No model weights or relevance threshold were trained on them.

Reproduce using a local encoder:

```bash
python -m ai_discovery.retrieval \
  --report data/reviews/assistant_validation_2026-11-18/alpha-report.json \
  --report data/reviews/assistant_validation_2026-11-18/beta-report.json \
  --encoder /path/to/local/MiniLM/encoder \
  --out data/cache/governance_retrieval/qa-chroma-v3

python -m ai_discovery.evaluate_answers \
  --index data/cache/governance_retrieval/qa-chroma-v3 \
  --cases data/reviews/assistant_validation_2026-11-18/cases.json \
  --out /tmp/assistant-functional-evaluation.json
```

The original results remain in `artifacts/governance_answer_validation_2026-11-18/functional_cases.json`, with historical FAISS fixture/index hashes, statuses, counts, names and failures per case. The ChromaDB migration reran all 26 cases successfully with the same fixtures and local encoder; the original artifact is preserved rather than relabeled as a new Chroma run. The earlier six-query named-application semantic smoke evaluation is retained separately and is not replaced by this functional score.

Next: independent analyst-reviewed questions on real project snapshots, including paraphrases and difficult negative cases, then an explicitly desired live-provider smoke test and answer-usefulness review. Dashboard work remains paused.

## Runtime and final checks

Indexing/search defaults `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before lazy native imports, preserving explicit user settings. The small local indexes do not require nested numerical thread pools. These defaults were originally introduced after a macOS crash in the former FAISS backend. They remain conservative defaults for the ChromaDB/local-encoder implementation. The complete suite is validated with these defaults; an explicitly overridden threaded environment has not been validated here.

The final suite contains 160 tests, including 9 additional grounding/QA checks. The 26 functional cases pass separately using the local MiniLM encoder. No live provider calls were made. Overlapping suite/product names are resolved by full, non-contained mentions (GitHub Copilot does not become GitHub); questions mentioning both still require explicit scope.
