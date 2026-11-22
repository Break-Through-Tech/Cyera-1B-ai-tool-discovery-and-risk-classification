"""Load Components 1-4 pipeline output for the governance dashboard (Component 5).

Reads, never modifies:
  * <run_dir>/categorized_applications.json written by ai_discovery.run_pipeline
  * the activity CSV that run was built from (joined through ingestion.event_map)

Nothing here recomputes detection, categories or risk. It only reshapes the saved
results and turns the governance rule evidence into plain-language sentences.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "companies.json"

RISK_ORDER = {"high": 0, "medium": 1, "low": 2}

AI_STATUS = {True: "AI tool", False: "Not AI", None: "Unresolved"}

CATEGORY_STATUS_TEXT = {
    "assigned": "",
    "provisional_prediction": "provisional",
    "not_applicable": "Not applicable (not an AI tool)",
    "pending_ai_detection": "Pending AI detection",
    "pending_identity_review": "Pending identity review",
    "pending_detection_conflict": "Pending: catalog and discovery disagree",
    "pending_catalog_category_review": "Pending catalog category review",
}


@dataclass
class CompanyRun:
    company: str
    apps: pd.DataFrame          # one row per observed application
    events: pd.DataFrame        # activity rows with application_id attached
    raw: dict                   # full pipeline report
    warnings: list = field(default_factory=list)


def load_config(path: Path = CONFIG) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["companies"]


def load_jev(path: Path = CONFIG) -> dict:
    """Jev second-opinion categories keyed by endpoint; {} when not generated.

    Written by category_training/jev_second_opinion.py. Advisory only: it never
    replaces the assigned category.
    """
    rel = json.loads(Path(path).read_text(encoding="utf-8")).get("jev_second_opinion")
    if not rel or not _path(rel).exists():
        return {}
    data = json.loads(_path(rel).read_text(encoding="utf-8"))
    return {k: v for k, v in data.get("apps", {}).items() if v.get("status") == "ok"}


def jev_check(category, opinion) -> str:
    """Short table label: does Jev's category agree with the assigned one?"""
    if not opinion or not isinstance(category, str) or not category:
        return "—"
    pct = f"{opinion['probability']:.0%}"
    if opinion["category"] == category:
        return f"✓ agrees ({pct})"
    return f"≠ {opinion['category']} ({pct})"


def _path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def _num(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def display_name(record: dict) -> str:
    """Prefer the spelling seen in the activity log (e.g. 'Cursor' not 'cursor')."""
    variants = _json_list(record.get("observed_identity", {}).get("name_variants"))
    return variants[0] if variants else record.get("application_name", "?")


def _mb(n) -> str:
    return f"{_num(n) / 1_000_000:.1f} MB"


def _retention_text(value) -> str:
    if value in (None, ""):
        return "an unknown period"
    v = str(value).lower()
    if v == "indefinite":
        return "indefinitely"
    if v.endswith("_days"):
        return f"for {v.split('_')[0]} days"
    return v.replace("_", " ")


def plain_reason(reason: dict, company: str = "the company") -> str:
    """Turn one governance reason (rule + evidence) into a sentence an analyst can read."""
    rule = reason.get("rule")
    ev = reason.get("evidence") or {}
    if rule == "unapproved":
        return f"Not on {company}'s approved application list."
    if rule == "no_sso":
        return "No enterprise SSO/SAML login was identified."
    if rule == "no_admin_controls":
        return "No administrative controls are available to the company."
    if rule == "no_enterprise_plan":
        return "No enterprise plan is available."
    if rule == "customer_data_training":
        return "Vendor terms indicate submitted content may be used to train its models."
    if rule == "long_retention":
        limit = ev.get("threshold_days")
        tail = f" (review threshold: {limit} days)" if limit else ""
        return f"Submitted data is retained {_retention_text(ev.get('data_retention'))}{tail}."
    if rule == "sensitive_department":
        depts = [str(d).title() if str(d).lower() not in ("hr", "it") else str(d).upper()
                 for d in ev.get("departments", [])]
        if depts:
            return f"Used by employees in sensitive departments: {', '.join(depts)}."
        return "Used by employees in sensitive departments."
    if rule == "broad_adoption":
        return (f"Used by {ev.get('unique_users', '?')} employees "
                f"(review threshold: {ev.get('threshold', '?')}).")
    if rule == "large_upload":
        parts = [f"Average upload of {_mb(ev.get('mean_bytes_uploaded'))} per event"]
        if ev.get("mean_threshold"):
            parts.append(f"(threshold {_mb(ev['mean_threshold'])})")
        text = " ".join(parts)
        if ev.get("max_bytes_uploaded"):
            text += f"; largest single upload {_mb(ev['max_bytes_uploaded'])}"
        return text + "."
    if rule == "large_download":
        return f"Average download of {_mb(ev.get('mean_bytes_downloaded'))} per event exceeds the threshold."
    return reason.get("explanation") or str(rule)


def _category_label(cat: dict) -> str:
    status = cat.get("category_status")
    if cat.get("category"):
        suffix = " (provisional)" if status == "provisional_prediction" else ""
        return f"{cat['category']}{suffix}"
    return CATEGORY_STATUS_TEXT.get(status, status or "—")


def _app_row(rec: dict, company: str) -> dict:
    oi = rec.get("observed_identity", {})
    gov = rec.get("governance") or {}
    cat = rec.get("categorization") or {}
    ev = gov.get("evidence") or {}
    approval = (ev.get("approved_status") or {}).get("value") or "unknown"
    reasons = gov.get("reasons") or []
    return {
        "application_id": rec.get("application_id") or oi.get("application_id"),
        "name": display_name(rec),
        "endpoint": rec.get("endpoint") or oi.get("observed_endpoint"),
        "is_ai_tool": rec.get("is_ai_tool"),
        "ai_status": AI_STATUS.get(rec.get("is_ai_tool"), "Unresolved"),
        "detection_reason": rec.get("reason") or "",
        "category": cat.get("category"),
        "category_status": cat.get("category_status"),
        "category_source": cat.get("category_source"),
        "category_label": _category_label(cat),
        "top_three": cat.get("top_three") or [],
        "risk_level": gov.get("risk_level") or "unknown",
        "risk_score": _num(gov.get("risk_score")),
        "assessment_status": gov.get("assessment_status") or "unknown",
        "needs_review": bool(gov.get("needs_review")),
        "approval": approval,
        "missing_evidence": gov.get("missing_evidence_rules") or [],
        "reasons": [
            {"rule": r.get("rule"), "points": _num(r.get("points")),
             "text": plain_reason(r, company), "provenance": r.get("evidence_provenance")}
            for r in sorted(reasons, key=lambda r: -_num(r.get("points")))
        ],
        "users": _num(oi.get("unique_employee_count")),
        "events": _num(oi.get("event_count")),
        "departments": _json_list(oi.get("departments")),
        "bytes_uploaded": _num(oi.get("total_bytes_uploaded")),
        "bytes_downloaded": _num(oi.get("total_bytes_downloaded")),
    }


def build_apps(report: dict, company: str) -> pd.DataFrame:
    rows = [_app_row(r, company) for r in report.get("results", [])]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mean_upload"] = df["bytes_uploaded"] / df["events"].where(df["events"] > 0)
    typical = df["mean_upload"].median()
    df["upload_vs_typical"] = df["mean_upload"] / typical if typical else float("nan")
    df["_risk_rank"] = df["risk_level"].map(RISK_ORDER).fillna(3)
    return df.sort_values(["_risk_rank", "risk_score", "users"],
                          ascending=[True, False, False]).reset_index(drop=True)


def attach_events(report: dict, activity: pd.DataFrame) -> pd.DataFrame:
    """Join activity rows to application_id using the pipeline's own event map."""
    emap = pd.DataFrame(report.get("ingestion", {}).get("event_map", []))
    if emap.empty:
        return activity.iloc[0:0].assign(application_id=pd.Series(dtype=str))
    act = activity.copy()
    act["source_row"] = range(1, len(act) + 1)   # 1-based data-record index
    merged = act.merge(emap[["source_row", "application_id", "event_id"]],
                       on="source_row", how="inner", suffixes=("", "_map"))
    if "event_id" in act.columns:
        mismatched = (merged["event_id"].astype(str) != merged["event_id_map"].astype(str)).sum()
        if mismatched:
            raise ValueError(f"{mismatched} activity rows do not line up with the pipeline event map; "
                             "the activity CSV is not the one this run was built from.")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    return merged.drop(columns=["event_id_map"])


MISSING_EVIDENCE_TEXT = {
    "unapproved": "approval status",
    "no_sso": "SSO availability",
    "customer_data_training": "whether content is used for model training",
    "long_retention": "data retention policy",
    "no_admin_controls": "admin controls",
    "no_enterprise_plan": "enterprise plan",
}


def missing_evidence_text(rules: list) -> list[str]:
    return [MISSING_EVIDENCE_TEXT.get(r, r.replace("_", " ")) for r in rules]


def flag_reason(row) -> str:
    """One short line saying why an app is in the investigation queue."""
    if pd.isna(row["is_ai_tool"]):
        return "Unknown app: AI status not yet determined"
    if row["risk_level"] == "high":
        return "High governance risk"
    if row["approval"] == "unapproved":
        return "AI tool explicitly not approved"
    if row["approval"] == "under_review":
        return "AI tool still under approval review"
    return "AI tool not on the approved list (approval unknown)"


def investigation_queue(apps: pd.DataFrame) -> pd.DataFrame:
    """AI or unresolved apps that are high risk, or not on the approved list.

    An AI tool whose approval is unknown is treated as *not approved*: VISION.md's
    shadow-AI case is exactly a tool missing from the approved list. A low score
    with missing evidence is not evidence of safety.
    """
    if apps.empty:
        return apps
    candidate = apps["is_ai_tool"].isin([True]) | apps["is_ai_tool"].isna()
    flagged = apps["risk_level"].eq("high") | apps["approval"].ne("approved") | apps["is_ai_tool"].isna()
    q = apps[candidate & flagged].copy()
    q["flag"] = q.apply(flag_reason, axis=1)
    q["_unresolved"] = q["is_ai_tool"].isna()
    return q.sort_values(["_risk_rank", "_unresolved", "risk_score", "users"],
                         ascending=[True, False, False, False]).drop(columns="_unresolved")


def load_company(entry: dict) -> CompanyRun:
    run_dir = _path(entry["run_dir"])
    report = json.loads((run_dir / "categorized_applications.json").read_text(encoding="utf-8"))
    company = entry["company"]
    warnings = []
    activity_path = _path(entry["activity"])
    expected = report.get("ingestion", {}).get("summary", {}).get("input_sha256")
    if expected and _sha256(activity_path) != expected:
        warnings.append("Activity CSV differs from the file this run was built from; "
                        "usage charts may not match the assessment.")
    activity = pd.read_csv(activity_path, dtype={"event_id": str, "employee_id": str})
    events = attach_events(report, activity)
    return CompanyRun(company, build_apps(report, company), events, report, warnings)
