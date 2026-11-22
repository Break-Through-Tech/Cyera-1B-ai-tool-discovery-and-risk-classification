"""AI Governance Console - Component 5 (VISION.md).

Run from the project root:
    .venv/bin/streamlit run dashboard/app.py

Reads saved Components 1-4 output listed in dashboard/companies.json. It does not
re-run detection, categorization or risk scoring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.data import (  # noqa: E402
    investigation_queue, jev_check, load_company, load_config, load_jev, missing_evidence_text,
)

# Reference palette (dataviz skill): status colours are reserved for risk and
# always shown with an icon + text label; one blue for single-series marks.
RISK_COLOR = {"high": "#d03b3b", "medium": "#fab219", "low": "#0ca30c", "unknown": "#8a8984"}
RISK_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢", "unknown": "⚪"}
SERIES = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SOURCE_TEXT = {"catalog": "AI-tool catalog", "model": "category model (provisional)",
               "human_review": "human review", None: "—"}

st.set_page_config(page_title="AI Governance Console", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Loading pipeline results…")
def get_run(entry: dict):
    run = load_company(entry)
    return run.apps, run.events, run.warnings


def risk_badge(level: str, score=None) -> str:
    s = f" ({score})" if score is not None else ""
    return f"{RISK_ICON.get(level, '⚪')} **{level.title()} risk{s}**"


def style_fig(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), bargap=0.4,
                      showlegend=False, hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", title=None)
    return fig


# ---------------------------------------------------------------- sidebar
companies = load_config()
with st.sidebar:
    st.header("AI Governance Console")
    names = [c["company"] for c in companies]
    choice = st.radio("Company", names)
    entry = next(c for c in companies if c["company"] == choice)
    st.divider()
    st.caption(
        "Prototype on **synthetic** activity and fictional company approvals. "
        "Risk scores are explainable rule totals, not security guarantees. "
        "Missing evidence is not evidence of safety."
    )

apps, events, warnings = get_run(entry)
for w in warnings:
    st.warning(w)

jev = load_jev()
apps = apps.assign(
    jev_opinion=apps["endpoint"].map(lambda e: jev.get(e)),
)
apps["jev_check"] = [jev_check(c, o) for c, o in zip(apps["category"], apps["jev_opinion"])]

ai_ids = set(apps.loc[apps["is_ai_tool"].isin([True]), "application_id"])
ai_apps = apps[apps["application_id"].isin(ai_ids)]
unresolved = apps[apps["is_ai_tool"].isna()]
queue = investigation_queue(apps)
ai_events = events[events["application_id"].isin(ai_ids)]

period = ""
if not events.empty:
    period = f"{events['timestamp'].min():%b %d} – {events['timestamp'].max():%b %d, %Y}"

st.title(f"{choice}")
st.caption(f"Activity {period} · {len(events):,} events · {events['employee_id'].nunique():,} employees · "
           f"{len(apps)} applications observed")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("AI tools discovered", len(ai_apps))
k2.metric("AI tools not approved", int((ai_apps["approval"] != "approved").sum()),
          help="Unapproved, under review, or with no approval record.")
k3.metric("High-risk AI tools", int((ai_apps["risk_level"] == "high").sum()))
k4.metric("Unknown apps to resolve", len(unresolved),
          help="Apps whose AI status could not be determined yet.")
k5.metric("Employees using AI", f"{ai_events['employee_id'].nunique():,}",
          help=f"of {events['employee_id'].nunique():,} employees in the activity log")

tab_inv, tab_over, tab_all = st.tabs(["🔎 Investigate", "📊 Overview", "📋 All applications"])

# ---------------------------------------------------------------- investigate
with tab_inv:
    st.subheader("Needs investigation")
    if queue.empty:
        st.success("No AI tools are high risk or missing from the approved list.")
    else:
        st.caption("AI tools (and unknown apps) that are high risk or not on the approved list.")
        qv = queue.assign(
            Risk=queue["risk_level"].map(lambda l: f"{RISK_ICON.get(l, '⚪')} {l.title()}"),
            Departments=queue["departments"].map(", ".join),
        )[["name", "flag", "Risk", "risk_score", "category_label", "jev_check", "users", "Departments"]]
        st.dataframe(
            qv.rename(columns={"name": "Application", "flag": "Why it's here", "risk_score": "Score",
                               "category_label": "Category", "jev_check": "Jev check", "users": "Users"}),
            hide_index=True, width="stretch",
        )

    st.divider()
    order = list(queue["name"]) + [n for n in apps["name"] if n not in set(queue["name"])]
    selected = st.selectbox("Investigate an application", order, index=0)
    app = apps[apps["name"] == selected].iloc[0]
    app_events = events[events["application_id"] == app["application_id"]]

    st.markdown(f"### {app['name']}  \n`{app['endpoint']}`")
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(risk_badge(app["risk_level"], app["risk_score"]))
    b2.markdown(f"**AI status:** {app['ai_status']}")
    b3.markdown(f"**Approval:** {app['approval'].replace('_', ' ')}")
    b4.markdown(f"**Assessment:** {app['assessment_status']}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Why this risk level")
        if app["reasons"]:
            for r in app["reasons"]:
                st.markdown(f"- **+{r['points']}** {r['text']}")
        else:
            st.markdown("_No risk rules were triggered._")
        if app["missing_evidence"]:
            st.warning("**Incomplete assessment.** No evidence was available for: "
                       + ", ".join(missing_evidence_text(app["missing_evidence"]))
                       + ". The score could be higher once these are known.")
        if pd.isna(app["is_ai_tool"]):
            st.info("**AI status unresolved.** " + (app["detection_reason"] or "")
                    + " Run the pipeline with `--online` (needs a Gemini key) to research it.")

        st.markdown("#### Purpose category")
        source = app["category_source"] if isinstance(app["category_source"], str) else None
        st.markdown(f"**{app['category_label']}**"
                    + (f"  \nSource: {SOURCE_TEXT.get(source, source)}" if source else ""))
        if app["category_source"] == "model" and app["top_three"]:
            st.caption("Model's top guesses (scores uncalibrated): " + " · ".join(
                f"{t['category']} {t.get('score_uncalibrated', 0):.0%}" for t in app["top_three"]))

        op = app["jev_opinion"]
        if isinstance(op, dict):
            agrees = op["category"] == app["category"]
            msg = (f"**Jev second opinion:** {op['category']} ({op['probability']:.0%})  \n"
                   + " · ".join(f"{t['category']} {t['probability']:.0%}" for t in op["top_three"])
                   + f"  \n_Based on the {op['evidence_source']}. "
                   "Jev beat the local model on held-out tools (macro-F1 0.59 vs 0.48); "
                   "below 50% probability it is right only about 30% of the time._")
            (st.success if agrees else st.warning)(
                ("✓ Agrees with the assigned category. " if agrees
                 else "≠ Disagrees with the assigned category; worth a second look. ") + msg)

    with right:
        st.markdown("#### Usage")
        u1, u2 = st.columns(2)
        u1.metric("Employees", f"{app['users']:,}")
        u2.metric("Events", f"{app['events']:,}")
        u3, u4 = st.columns(2)
        u3.metric("Avg upload / event", f"{(app['mean_upload'] or 0) / 1e6:.1f} MB")
        ratio = app["upload_vs_typical"]
        u4.metric("vs. typical app", f"{ratio:.1f}×" if pd.notna(ratio) else "—",
                  help="Average upload per event compared with the median application at this company.")
        if not app_events.empty:
            st.caption(f"First seen {app_events['timestamp'].min():%b %d} · last seen {app_events['timestamp'].max():%b %d}")

    if not app_events.empty:
        c1, c2 = st.columns(2)
        by_dept = (app_events.groupby("department")["employee_id"].nunique()
                   .sort_values().reset_index(name="employees"))
        fig = px.bar(by_dept, x="employees", y="department", orientation="h",
                     color_discrete_sequence=[SERIES])
        fig.update_traces(hovertemplate="%{y}: %{x} employees<extra></extra>", marker_line_width=0)
        c1.markdown("**Employees by department**")
        c1.plotly_chart(style_fig(fig, 260), width="stretch")

        daily = (app_events.set_index("timestamp").resample("D")["event_id"].count()
                 .reset_index(name="events"))
        fig = px.line(daily, x="timestamp", y="events", color_discrete_sequence=[SERIES])
        fig.update_traces(line_width=2, hovertemplate="%{x|%b %d}: %{y} events<extra></extra>")
        c2.markdown("**Daily activity**")
        c2.plotly_chart(style_fig(fig, 260), width="stretch")

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**AI tools by category**")
        cats = ai_apps.assign(category=ai_apps["category"].fillna("Uncategorized"))
        by_cat = (cats.groupby("category")
                  .agg(tools=("name", "count"), names=("name", ", ".join), users=("users", "sum"))
                  .sort_values("tools").reset_index())
        fig = px.bar(by_cat, x="tools", y="category", orientation="h",
                     color_discrete_sequence=[SERIES], custom_data=["names"])
        fig.update_traces(hovertemplate="%{y}<br>%{x} tool(s): %{customdata[0]}<extra></extra>",
                          marker_line_width=0)
        fig.update_xaxes(dtick=1)
        st.plotly_chart(style_fig(fig), width="stretch")
        st.caption("Categories use the 14 Kaggle labels; '(provisional)' model guesses are counted too.")

    with c2:
        st.markdown("**Risk level of AI tools**")
        levels = ["high", "medium", "low"]
        counts = ai_apps["risk_level"].value_counts().reindex(levels, fill_value=0)
        incomplete = (ai_apps[ai_apps["assessment_status"] != "complete"]["risk_level"]
                      .value_counts().reindex(levels, fill_value=0))
        fig = go.Figure(go.Bar(
            x=counts.values, y=[f"{RISK_ICON[l]} {l.title()}" for l in levels], orientation="h",
            marker_color=[RISK_COLOR[l] for l in levels], marker_line_width=0,
            text=[f"{c}  ({i} incomplete)" if i else str(c) for c, i in zip(counts, incomplete)],
            textposition="outside", hovertemplate="%{y}: %{x} AI tools<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(cliponaxis=False)
        fig.update_xaxes(range=[0, max(counts.max(), 1) * 1.6], dtick=1)
        st.plotly_chart(style_fig(fig), width="stretch")
        st.caption("'Incomplete' = some governance evidence missing, so the score may be understated.")

    st.markdown("**Who uses which AI tools** — unique employees by department")
    if not ai_events.empty:
        names = apps.set_index("application_id")["name"]
        heat = (ai_events.assign(app=ai_events["application_id"].map(names))
                .pivot_table(index="app", columns="department", values="employee_id",
                             aggfunc="nunique", fill_value=0))
        heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(heat, color_continuous_scale=SEQUENTIAL, aspect="auto", text_auto=True)
        fig.update_traces(hovertemplate="%{y} · %{x}: %{z} employees<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 40 + 34 * len(heat)), width="stretch")

    st.markdown("**AI tool activity over time** (events per day, all AI tools)")
    if not ai_events.empty:
        daily = (ai_events.set_index("timestamp").resample("D")["event_id"]
                 .count().reset_index(name="events"))
        fig = px.line(daily, x="timestamp", y="events", color_discrete_sequence=[SERIES])
        fig.update_traces(line_width=2, hovertemplate="%{x|%a %b %d}: %{y:,} events<extra></extra>")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_fig(fig, 260), width="stretch")

# ---------------------------------------------------------------- all apps
with tab_all:
    only_ai = st.toggle("AI tools and unknown apps only", value=False)
    view = apps[apps["is_ai_tool"].isin([True]) | apps["is_ai_tool"].isna()] if only_ai else apps
    table = view.assign(
        Risk=view["risk_level"].map(lambda l: f"{RISK_ICON.get(l, '⚪')} {l.title()}"),
        Departments=view["departments"].map(", ".join),
        Reasons=view["reasons"].map(lambda rs: " · ".join(r["text"] for r in rs)),
        upload_mb=(view["mean_upload"] / 1e6).round(2),
    )[["name", "endpoint", "ai_status", "category_label", "jev_check", "Risk", "risk_score", "approval",
       "assessment_status", "users", "events", "upload_mb", "Departments", "Reasons"]]
    table = table.rename(columns={
        "name": "Application", "endpoint": "Endpoint", "ai_status": "AI status",
        "category_label": "Category", "jev_check": "Jev check", "risk_score": "Score", "approval": "Approval",
        "assessment_status": "Assessment", "users": "Users", "events": "Events",
        "upload_mb": "Avg upload (MB)"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download as CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{choice.lower().replace(' ', '_')}_applications.csv", mime="text/csv")
