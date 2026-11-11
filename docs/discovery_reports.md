# Integrated discovery reports

The current release is documented in [backend_release.md](backend_release.md). Start there for active paths, commands, evaluation and known errors.

`ai_discovery/report.py` is the single reporting entry point, configured by `ai_discovery/pipeline.json`. It uses Sadit's `ingest.py` and `catalog.py` and our governance assessment. The obsolete `discovery/` wrapper is removed.

Reports preserve accepted/rejected rows, source-row mappings, observed application identities, departments and user counts. Every accepted input row counts as an event, including repeated event IDs; a separate repeated-ID count makes that explicit. Invalid metrics are excluded from aggregates and retained as evidence gaps. Missing user/department/timestamp information marks the assessment incomplete.

Catalog matching applies the reconciled full corrections, preserving product paths and unresolved scope. Optional description-based binary predictions can feed the new 14-category purpose model. Matched catalog categories are supplied assignments; model-derived categories are marked experimental. Unknown/ambiguous identities stay visible. Governance is independent of AI status and uses explicit evidence; generated catalog security values are not imported as vendor facts.

Current summaries: `artifacts/backend_release_2026-11-11/`. The older shared summaries and `artifacts/historical/discovery/` are preserved historical outputs with earlier versions of the lookup rules. CLI reports retain full event mappings; checked-in summaries omit repetitive mapping entries and say so.

The known handwritten diagnostics include incorrect predictions; passing integration tests means the pieces connect correctly, not that model answers are reliable. No enrichment, Gemini or Jev call runs automatically. Dashboard work remains paused.
