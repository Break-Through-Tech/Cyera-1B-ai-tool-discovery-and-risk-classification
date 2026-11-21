import json
import unittest
from pathlib import Path

import pandas as pd

from dashboard.data import (
    attach_events, build_apps, investigation_queue, load_company, load_config, plain_reason,
)

ROOT = Path(__file__).resolve().parents[1]


def record(name, is_ai, risk, score, approval, reasons=(), users=10, events=10, upload=1000):
    return {
        "application_id": f"app_{name}",
        "application_name": name.lower(),
        "endpoint": f"{name.lower()}.com",
        "is_ai_tool": is_ai,
        "observed_identity": {
            "name_variants": json.dumps([name]), "unique_employee_count": str(users),
            "event_count": str(events), "departments": json.dumps(["Finance"]),
            "total_bytes_uploaded": str(upload), "total_bytes_downloaded": "0",
        },
        "categorization": {"category": "Data And Analytics" if is_ai else None,
                           "category_status": "assigned" if is_ai else "not_applicable",
                           "category_source": "catalog" if is_ai else None, "top_three": []},
        "governance": {"risk_level": risk, "risk_score": str(score), "assessment_status": "complete",
                       "evidence": {"approved_status": {"value": approval}},
                       "reasons": list(reasons), "missing_evidence_rules": []},
    }


class PlainReasonTest(unittest.TestCase):
    def test_rules_become_sentences_with_evidence(self):
        self.assertEqual(plain_reason({"rule": "unapproved", "evidence": {}}, "Acme"),
                         "Not on Acme's approved application list.")
        self.assertEqual(
            plain_reason({"rule": "sensitive_department", "evidence": {"departments": ["finance", "hr"]}}),
            "Used by employees in sensitive departments: Finance, HR.")
        self.assertIn("indefinitely", plain_reason(
            {"rule": "long_retention", "evidence": {"data_retention": "indefinite", "threshold_days": 90}}))
        self.assertIn("7.9 MB", plain_reason(
            {"rule": "large_upload", "evidence": {"mean_bytes_uploaded": 7_948_364, "mean_threshold": 5_000_000}}))

    def test_unknown_rule_falls_back_to_engine_explanation(self):
        self.assertEqual(plain_reason({"rule": "new_rule", "explanation": "Engine text."}), "Engine text.")


class QueueTest(unittest.TestCase):
    def setUp(self):
        report = {"results": [
            record("SafeAI", True, "low", 0, "approved"),
            record("ShadowAI", True, "low", 15, "unknown"),
            record("RiskyAI", True, "high", 90, "approved"),
            record("Mystery", None, "medium", 30, "unapproved"),
            record("Spreadsheet", False, "high", 80, "unapproved"),
        ]}
        self.apps = build_apps(report, "Acme")

    def test_queue_flags_unapproved_unknown_and_high_risk_ai_only(self):
        q = investigation_queue(self.apps)
        self.assertEqual(set(q["name"]), {"ShadowAI", "RiskyAI", "Mystery"})
        self.assertNotIn("Spreadsheet", set(q["name"]))   # non-AI is out of scope
        self.assertNotIn("SafeAI", set(q["name"]))        # approved and low risk

    def test_queue_orders_by_risk_then_score(self):
        self.assertEqual(list(investigation_queue(self.apps)["name"]), ["RiskyAI", "Mystery", "ShadowAI"])

    def test_display_name_uses_observed_spelling(self):
        self.assertIn("ShadowAI", set(self.apps["name"]))


class EventJoinTest(unittest.TestCase):
    def test_join_uses_one_based_source_rows(self):
        report = {"ingestion": {"event_map": [
            {"source_row": 1, "event_id": "e1", "application_id": "a"},
            {"source_row": 2, "event_id": "e2", "application_id": "b"}]}}
        act = pd.DataFrame({"event_id": ["e1", "e2"], "timestamp": ["2026-09-01 08:00", "2026-09-02 08:00"]})
        out = attach_events(report, act)
        self.assertEqual(dict(zip(out["event_id"], out["application_id"])), {"e1": "a", "e2": "b"})

    def test_mismatched_activity_file_is_rejected(self):
        report = {"ingestion": {"event_map": [{"source_row": 1, "event_id": "e1", "application_id": "a"}]}}
        act = pd.DataFrame({"event_id": ["other"], "timestamp": ["2026-09-01"]})
        with self.assertRaises(ValueError):
            attach_events(report, act)


class SavedRunTest(unittest.TestCase):
    """Uses the saved Summit Financial run when it exists locally."""

    def test_documind_is_flagged_high_risk(self):
        entry = next(c for c in load_config() if c["company"] == "Summit Financial")
        if not (ROOT / entry["run_dir"] / "categorized_applications.json").exists():
            self.skipTest("Summit pipeline run not generated")
        run = load_company(entry)
        self.assertEqual(run.warnings, [])
        self.assertEqual(len(run.events), 25120)
        q = investigation_queue(run.apps)
        self.assertIn("DocuMind AI", list(q["name"])[:2])
        documind = run.apps[run.apps["name"] == "DocuMind AI"].iloc[0]
        self.assertEqual(documind["risk_level"], "high")
        self.assertGreater(documind["upload_vs_typical"], 2.5)


if __name__ == "__main__":
    unittest.main()
