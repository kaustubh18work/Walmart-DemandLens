from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandlens.core.config import settings
from demandlens.walmart.loader import loader
from demandlens.walmart.profiler import data_quality, profile_walmart
from demandlens.forecasting.features.pipeline import aggregate
from demandlens.forecasting.backtesting.backtester import evaluate_models, MODELS
from demandlens.insights.engine import generate_insights
from demandlens.reports.csv_report import create_csv_report
from demandlens.reports.pdf_report import create_pdf_report


# ============================================================
# Application configuration
# ============================================================
st.set_page_config(
    page_title="Walmart DemandLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# No theme switch exists in the product. Custom surfaces follow the
# operating system/browser preference via prefers-color-scheme.
st.markdown(
    """
<style>
:root {
  --dl-blue:#2563eb; --dl-blue2:#60a5fa; --dl-green:#16a34a;
  --dl-purple:#7c3aed; --dl-orange:#d97706; --dl-red:#dc2626;
  --dl-text:#172033; --dl-muted:#64748b; --dl-border:rgba(100,116,139,.18);
  --dl-card:rgba(255,255,255,.90); --dl-card2:rgba(248,250,252,.96);
  --dl-shadow:0 8px 30px rgba(15,23,42,.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --dl-text:#e8eef9; --dl-muted:#9aa9bf; --dl-border:rgba(148,163,184,.18);
    --dl-card:rgba(15,25,42,.90); --dl-card2:rgba(17,31,52,.96);
    --dl-shadow:0 10px 32px rgba(0,0,0,.24);
  }
  .stApp {
    background:
      radial-gradient(circle at 80% 0%, rgba(37,99,235,.10), transparent 28%),
      radial-gradient(circle at 15% 20%, rgba(124,58,237,.06), transparent 22%);
  }
}
.block-container {padding-top:1rem; padding-bottom:2rem; max-width:1540px;}
[data-testid="stSidebar"] {border-right:1px solid var(--dl-border);}
[data-testid="stSidebar"] .stRadio > div {gap:4px;}
[data-testid="stSidebar"] label {border-radius:10px; padding:6px 8px;}
.dl-brand{display:flex;align-items:center;gap:11px;margin:3px 0 22px 2px;}
.dl-logo{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,#2563eb,#60a5fa);color:#fff;font-size:22px;
 box-shadow:0 8px 20px rgba(37,99,235,.22);}
.dl-brand-name{font-size:1.18rem;font-weight:850;color:var(--dl-text);}
.dl-brand-sub{font-size:.68rem;color:var(--dl-muted);}
.dl-kicker{color:var(--dl-muted);font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;}
.dl-title{color:var(--dl-text);font-size:1.9rem;line-height:1.1;font-weight:850;margin-bottom:.25rem;}
.dl-subtitle{color:var(--dl-muted);font-size:.92rem;}
.dl-card{background:var(--dl-card);border:1px solid var(--dl-border);border-radius:16px;
 padding:16px 17px;box-shadow:var(--dl-shadow);backdrop-filter:blur(10px);}
.dl-tight{padding:13px 15px;}
.dl-card-title{font-size:.9rem;font-weight:800;color:var(--dl-text);}
.dl-card-value{font-size:1.48rem;font-weight:850;color:var(--dl-text);margin-top:3px;}
.dl-card-meta{font-size:.74rem;color:var(--dl-muted);margin-top:4px;line-height:1.4;}
.dl-positive{color:var(--dl-green);font-weight:800;font-size:.75rem;}
.dl-warning{color:var(--dl-orange);font-weight:800;font-size:.75rem;}
.dl-negative{color:var(--dl-red);font-weight:800;font-size:.75rem;}
.dl-hero{border:1px solid var(--dl-border);border-radius:20px;padding:20px 22px;
 background:linear-gradient(135deg,rgba(37,99,235,.10),rgba(124,58,237,.05) 55%,var(--dl-card));
 margin-bottom:15px;}
.dl-live{display:inline-flex;align-items:center;gap:7px;color:var(--dl-green);font-size:.75rem;font-weight:800;}
.dl-dot{width:8px;height:8px;border-radius:50%;background:var(--dl-green);
 box-shadow:0 0 0 4px rgba(22,163,74,.12);}
.dl-section{margin:20px 0 10px;display:flex;align-items:end;justify-content:space-between;}
.dl-section h3{margin:0;color:var(--dl-text);font-size:1.04rem;}
.dl-section span{color:var(--dl-muted);font-size:.74rem;}
.dl-pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 9px;font-size:.69rem;font-weight:800;}
.dl-blue{background:rgba(37,99,235,.12);color:var(--dl-blue);}
.dl-green{background:rgba(22,163,74,.12);color:var(--dl-green);}
.dl-orange{background:rgba(217,119,6,.13);color:var(--dl-orange);}
.dl-purple{background:rgba(124,58,237,.13);color:var(--dl-purple);}
.dl-insight{border-left:3px solid var(--dl-blue);padding:10px 12px;margin:8px 0;background:var(--dl-card2);border-radius:0 10px 10px 0;}
.dl-action{border:1px solid var(--dl-border);border-radius:12px;padding:11px 12px;background:var(--dl-card2);margin:7px 0;}
.dl-action-title{font-size:.79rem;font-weight:800;color:var(--dl-text);}
.dl-action-text{font-size:.73rem;color:var(--dl-muted);margin-top:3px;line-height:1.4;}
.dl-empty{border:1px dashed var(--dl-border);border-radius:16px;padding:32px;text-align:center;background:var(--dl-card);}
.dl-footer{color:var(--dl-muted);font-size:.68rem;text-align:center;padding:24px 0 4px;}
.dl-run-row{display:grid;grid-template-columns:22px 1fr auto;gap:9px;align-items:center;padding:7px 0;border-bottom:1px solid var(--dl-border);}
.dl-run-row:last-child{border-bottom:0;}
.dl-run-icon{font-size:.8rem;text-align:center;}
.dl-run-main{font-size:.75rem;font-weight:750;color:var(--dl-text);}
.dl-run-sub{font-size:.66rem;color:var(--dl-muted);}
.dl-run-time{font-size:.65rem;color:var(--dl-muted);white-space:nowrap;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Utilities
# ============================================================
def now_local() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def greeting(hour: int) -> str:
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 22:
        return "Good evening"
    return "Good night"


def fmt_dt(value) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%d %b %Y, %I:%M:%S %p %Z")
    except Exception:
        return str(value)


def fmt_seconds(value) -> str:
    return f"{float(value or 0):.2f}s"


def money_like(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"


def esc(value) -> str:
    return html.escape(str(value))


def quality_status(score: float) -> str:
    if score >= 95:
        return "Excellent"
    if score >= 85:
        return "Healthy"
    if score >= 70:
        return "Review"
    return "Attention"


# ============================================================
# Cached Walmart dataset
# ============================================================
@st.cache_data(show_spinner=False)
def get_profile():
    return profile_walmart()


@st.cache_data(show_spinner=False)
def get_quality():
    return data_quality()


@st.cache_data(show_spinner=False)
def get_store_options():
    return sorted(loader.train["Store"].dropna().astype(int).unique().tolist())


@st.cache_data(show_spinner=False)
def get_department_options():
    return sorted(loader.train["Dept"].dropna().astype(int).unique().tolist())


profile = get_profile()
quality = get_quality()


# ============================================================
# Session state
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_run" not in st.session_state:
    st.session_state.last_run = None


# ============================================================
# Payload + forecast execution
# ============================================================
def build_payload(req, history, forecast, metrics, best, insights, started, completed):
    return {
        "selection": {
            "store": req["store"],
            "department": req["department"],
            "store_label": req["store_label"],
            "department_label": req["department_label"],
        },
        "horizon": int(len(forecast)),
        "confidence_level": float(req["confidence"]),
        "run_started_at": started,
        "run_completed_at": completed,
        "run_duration_seconds": round(
            (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds(), 3
        ),
        "dataset": {
            "date_start": profile["date_start"],
            "date_end": profile["date_end"],
            "records": profile["records"],
            "stores": profile["stores"],
            "departments": profile["departments"],
            "frequency": profile["frequency"],
            "target": profile["target"],
        },
        "history": [
            {
                "Date": r.Date.date().isoformat(),
                "Weekly_Sales": float(r.Weekly_Sales),
                "IsHoliday": bool(r.IsHoliday),
            }
            for r in history.itertuples()
        ],
        "forecast": [
            {
                "Date": r.Date.date().isoformat(),
                "forecast": float(r.forecast),
                "lower": float(r.lower),
                "upper": float(r.upper),
                "IsHoliday": bool(getattr(r, "IsHoliday", False)),
            }
            for r in forecast.itertuples()
        ],
        "metrics": metrics,
        "best_model": best,
        "insights": insights,
    }


def _model_predict(model, horizon, alpha, future_calendar=None):
    if model.name == "LightGBM":
        return model.predict(horizon, alpha=alpha, future_calendar=future_calendar)
    return model.predict(horizon, alpha=alpha)


def run_forecast(req, progress_bar, status_box, detail_box, timeline_box):
    started = now_iso()
    event_log: list[dict] = []

    def update(pct: int, label: str, **meta):
        pct = int(max(0, min(100, pct)))
        progress_bar.progress(pct)
        status_box.markdown(f"### {pct}% · {esc(label)}")

        detail_bits = []
        if meta.get("current_model"):
            detail_bits.append(f"Model: **{esc(meta['current_model'])}**")
        if meta.get("model_stage"):
            detail_bits.append(f"Stage: **{esc(meta['model_stage'])}**")
        if meta.get("model_started_at"):
            detail_bits.append(f"Started: `{fmt_dt(meta['model_started_at'])}`")
        if meta.get("model_fit_completed_at"):
            detail_bits.append(f"Fit completed: `{fmt_dt(meta['model_fit_completed_at'])}`")
        if meta.get("model_prediction_completed_at"):
            detail_bits.append(f"Prediction completed: `{fmt_dt(meta['model_prediction_completed_at'])}`")
        if meta.get("model_duration_seconds") is not None:
            detail_bits.append(f"Duration: `{fmt_seconds(meta['model_duration_seconds'])}`")
        detail_box.markdown(" · ".join(detail_bits) or "Running locally against the bundled Walmart dataset.")

        event = {"label": label, "pct": pct, "time": now_iso(), **meta}
        event_log.append(event)
        recent = event_log[-8:]
        rows = []
        for e in recent:
            icon = "✓" if e["pct"] >= pct and e is not recent[-1] else "●"
            rows.append(
                f"<div class='dl-run-row'><div class='dl-run-icon'>{icon}</div>"
                f"<div><div class='dl-run-main'>{esc(e['label'])}</div>"
                f"<div class='dl-run-sub'>{esc(e.get('model_stage',''))}</div></div>"
                f"<div class='dl-run-time'>{fmt_dt(e['time'])}</div></div>"
            )
        timeline_box.markdown(
            "<div class='dl-card dl-tight'><div class='dl-kicker'>Execution timeline</div>"
            + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    update(3, "Loading bundled Walmart dataset")
    loader.validate()
    update(7, "Validating Walmart dataset schema")
    ts = aggregate(loader.train, req["store"], req["department"])

    if len(ts) < 60:
        raise ValueError(
            "This selection has fewer than 60 weekly observations. Choose All Stores or a broader store/department selection."
        )

    update(10, f"Preparing {len(req['models'])} model backtests")
    names = req["models"]
    total_models = len(names)

    def progress_callback(name, _pct, stage="running", **extra):
        # Percentages are stage boundaries, not simulated time. They move only
        # when the model reports a real lifecycle event.
        idx = names.index(name) if name in names else 0
        block = 52 / max(total_models, 1)
        base = 10 + idx * block
        stage_offset = {
            "started": 0,
            "fitted": block * 0.55,
            "predicted": block * 0.82,
            "completed": block,
            "failed": block,
        }.get(stage, block * 0.1)
        update(
            round(base + stage_offset),
            f"{name.upper()} — {stage}",
            current_model=name.upper(),
            model_stage=stage,
            **extra,
        )

    metrics, _, _ = evaluate_models(ts, names, progress_callback)
    successful = [m for m in metrics if m["status"] == "ok"]
    if not successful:
        details = "; ".join(f"{m['model']}: {m.get('error','unknown error')}" for m in metrics)
        raise RuntimeError(f"All selected models failed. {details}")

    best = min(successful, key=lambda x: x["wmae"])
    update(66, f"Selected {best['model']} from validation results", current_model=best["model"])

    full_fit_started = now_iso()
    perf_start = time.perf_counter()
    update(
        70,
        f"Training {best['model']} on full history",
        current_model=best["model"],
        model_started_at=full_fit_started,
        model_stage="full-history fit",
    )

    model = MODELS[best["key"]]()
    model.fit(ts)

    full_fit_completed = now_iso()
    full_fit_seconds = round(time.perf_counter() - perf_start, 3)
    best["full_fit_started_at"] = full_fit_started
    best["full_fit_completed_at"] = full_fit_completed
    best["full_fit_duration_seconds"] = full_fit_seconds

    update(
        78,
        f"Full-history training completed for {best['model']}",
        current_model=best["model"],
        model_stage="full-history fit completed",
        model_started_at=full_fit_started,
        model_fit_completed_at=full_fit_completed,
        model_duration_seconds=full_fit_seconds,
    )

    future_calendar = None
    try:
        future_calendar = loader.future_calendar(req["store"], req["department"])
    except Exception:
        future_calendar = None

    prediction_started = now_iso()
    update(
        82,
        f"Generating {req['horizon']}-week forecast",
        current_model=best["model"],
        model_stage="forecast generation",
        model_started_at=prediction_started,
    )
    forecast = _model_predict(
        model,
        req["horizon"],
        1 - req["confidence"],
        future_calendar=future_calendar,
    )
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    prediction_completed = now_iso()
    prediction_seconds = round(
        (datetime.fromisoformat(prediction_completed) - datetime.fromisoformat(prediction_started)).total_seconds(), 3
    )
    update(
        88,
        "Forecast generation completed",
        current_model=best["model"],
        model_stage="forecast complete",
        model_started_at=prediction_started,
        model_prediction_completed_at=prediction_completed,
        model_duration_seconds=prediction_seconds,
    )

    history = ts.tail(104).copy()
    update(91, "Generating AI decision intelligence")
    insights = generate_insights(history, forecast, metrics)

    completed = now_iso()
    payload = build_payload(req, history, forecast, metrics, best, insights, started, completed)
    payload["prediction_started_at"] = prediction_started
    payload["prediction_completed_at"] = prediction_completed
    payload["prediction_duration_seconds"] = prediction_seconds

    update(94, "Building CSV forecast report")
    csv_path = create_csv_report(forecast, req, best)

    update(97, "Building visual AI PDF report")
    pdf_path = create_pdf_report(payload)
    payload["reports"] = {"csv": str(csv_path), "pdf": str(pdf_path)}

    update(100, "Forecast complete — reports ready", current_model=best["model"], model_stage="complete")
    return payload


# ============================================================
# Charts
# ============================================================
def transparent_layout(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(color="#64748b", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,.16)", zeroline=False),
    )
    return fig


def forecast_chart(payload):
    hist = pd.DataFrame(payload["history"])
    fc = pd.DataFrame(payload["forecast"])
    hist["Date"] = pd.to_datetime(hist["Date"])
    fc["Date"] = pd.to_datetime(fc["Date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Weekly_Sales"], mode="lines",
        name="Actual", line=dict(color="#2563eb", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=fc["Date"], y=fc["upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc["Date"], y=fc["lower"], mode="lines",
        fill="tonexty", fillcolor="rgba(37,99,235,.12)",
        line=dict(width=0), name="Confidence interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc["Date"], y=fc["forecast"], mode="lines+markers",
        name=f"Forecast · {payload['best_model']['model']}",
        line=dict(color="#16a34a", width=2.5), marker=dict(size=5),
    ))
    fig.add_vline(
        x=hist["Date"].max(), line_dash="dash", line_color="#94a3b8",
        annotation_text="Forecast start", annotation_position="top left",
    )
    fig.update_layout(yaxis_title="Weekly Sales")
    return transparent_layout(fig, 460)


def model_bar_chart(metrics):
    ok = metrics[metrics["status"] == "ok"].sort_values("wmae", ascending=True)
    fig = go.Figure(go.Bar(
        x=ok["wmae"], y=ok["model"], orientation="h",
        marker=dict(color="#2563eb"),
        text=ok["wmae"].map(lambda x: f"{x:.2f}"),
        textposition="outside",
        hovertemplate="%{y}: WMAE %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="WMAE · lower is better", yaxis_title=None, showlegend=False)
    return transparent_layout(fig, 270)


def dataset_trend_chart():
    df = loader.train.groupby("Date", as_index=False)["Weekly_Sales"].sum()
    fig = go.Figure(go.Scatter(
        x=df["Date"], y=df["Weekly_Sales"], mode="lines",
        line=dict(color="#2563eb", width=2), name="Weekly sales",
    ))
    fig.update_layout(yaxis_title="Weekly Sales", showlegend=False)
    return transparent_layout(fig, 350)


# ============================================================
# Sidebar
# ============================================================
current = now_local()
page_options = [
    "Overview", "Dataset", "Forecast Studio", "Model Comparison",
    "Backtesting", "AI Insights", "Reports", "System Info",
]

with st.sidebar:
    st.markdown(
        "<div class='dl-brand'><div class='dl-logo'>↗</div>"
        "<div><div class='dl-brand-name'>DemandLens</div>"
        "<div class='dl-brand-sub'>Walmart Demand Intelligence</div></div></div>",
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        page_options,
        index=page_options.index(st.session_state.page),
        key="navigation",
        label_visibility="collapsed",
    )
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='dl-card dl-tight'><div class='dl-kicker'>Bundled dataset</div>"
        f"<div class='dl-card-title' style='margin-top:5px'>Walmart Sales Data</div>"
        f"<div class='dl-card-meta'>{profile['stores']} stores · {profile['departments']} departments</div>"
        f"<div class='dl-card-meta'>{profile['records']:,} records · {profile['frequency'].lower()}</div>"
        f"<div style='margin-top:8px'><span class='dl-pill dl-green'>Validated</span></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:9px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='dl-card dl-tight'><div class='dl-kicker'>Data quality</div>"
        f"<div class='dl-card-value'>{quality['quality_score']:.1f}/100</div>"
        f"<div class='dl-card-meta'>{quality_status(quality['quality_score'])} · "
        f"{quality['duplicate_keys']} duplicate keys · {quality['missing_cells']} missing cells</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:9px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='dl-card dl-tight'><div class='dl-kicker'>Local application time</div>"
        f"<div class='dl-card-title' style='margin-top:5px'>{current.strftime('%I:%M:%S %p')}</div>"
        f"<div class='dl-card-meta'>{current.strftime('%d %b %Y')} · {current.tzname() or 'Local time'}</div></div>",
        unsafe_allow_html=True,
    )

    st.caption("System appearance · no theme switch · Walmart dataset only")


# ============================================================
# Header
# ============================================================
last_result = st.session_state.last_result
last_run = st.session_state.last_run

h1, h2 = st.columns([3.3, 1.7])
with h1:
    st.markdown(
        f"<div class='dl-title'>{greeting(current.hour)}, Kaustubh 👋</div>"
        "<div class='dl-subtitle'>Walmart demand forecasting, model comparison and AI decision intelligence in one local workspace.</div>",
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f"<div class='dl-card dl-tight'><div style='display:flex;justify-content:space-between;gap:12px'>"
        f"<div><div class='dl-kicker'>System</div><div class='dl-live' style='margin-top:6px'><span class='dl-dot'></span>Online</div></div>"
        f"<div><div class='dl-kicker'>Last forecast</div><div class='dl-card-meta' style='margin-top:6px'>{fmt_dt(last_run) if last_run else 'No run yet'}</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# Overview
# ============================================================
def render_overview():
    forecast_total = peak = 0.0
    confidence = 0.0
    winner = "—"

    if last_result:
        fc = pd.DataFrame(last_result["forecast"])
        ins = last_result["insights"]
        forecast_total = float(fc["forecast"].sum())
        peak = float(fc["forecast"].max())
        confidence = float(ins["confidence"])
        winner = last_result["best_model"]["model"]

    cards = [
        ("Historical Sales", money_like(profile["total_sales"]), "All bundled Walmart records"),
        ("Average Weekly Sales", money_like(loader.train["Weekly_Sales"].mean()), "Historical average"),
        ("Forecast Total", money_like(forecast_total) if last_result else "—", f"Next {last_result['horizon']} weeks" if last_result else "Generate a forecast"),
        ("Winning Model", winner, "Lowest validation WMAE" if last_result else "Awaiting backtest"),
        ("Decision Confidence", f"{confidence:.0f}%" if last_result else "—", "Derived from variability" if last_result else "Awaiting forecast"),
        ("Forecast Peak", f"{peak:,.0f}" if last_result else "—", "Projected weekly sales"),
    ]
    cols = st.columns(6)
    for col, (title, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='dl-card dl-tight'><div class='dl-kicker'>{esc(title)}</div>"
                f"<div class='dl-card-value'>{esc(value)}</div><div class='dl-card-meta'>{esc(sub)}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='dl-hero'><div class='dl-kicker'>Walmart Demand Intelligence</div>"
        "<div style='font-size:1.25rem;font-weight:850;color:var(--dl-text);margin-top:5px'>"
        "From historical demand to an explainable forecast.</div>"
        "<div class='dl-card-meta' style='max-width:800px'>"
        "The engine uses the bundled Walmart dataset, time-ordered backtesting and automatic WMAE-based model selection. "
        "No external CSV upload or data connector is used.</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='dl-section'><h3>Demand overview</h3><span>All Walmart stores and departments</span></div>", unsafe_allow_html=True)
    st.plotly_chart(dataset_trend_chart(), use_container_width=True, config={"displayModeBar": False})

    if last_result:
        st.markdown("<div class='dl-section'><h3>Latest forecast signal</h3><span>Historical demand and projected horizon</span></div>", unsafe_allow_html=True)
        st.plotly_chart(forecast_chart(last_result), use_container_width=True, config={"displayModeBar": False})

        a, b = st.columns(2)
        with a:
            st.markdown("<div class='dl-card'><div class='dl-card-title'>Key AI insights</div>", unsafe_allow_html=True)
            for item in last_result["insights"]["insights"][:4]:
                st.markdown(
                    f"<div class='dl-insight'><b>{esc(item['title'])}</b><div class='dl-card-meta'>{esc(item['text'])}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("<div class='dl-card'><div class='dl-card-title'>Next action plan</div>", unsafe_allow_html=True)
            for item in last_result["insights"]["actions"]:
                st.markdown(
                    f"<div class='dl-action'><div class='dl-action-title'>{esc(item['priority'])} · {esc(item['title'])}</div>"
                    f"<div class='dl-action-text'>{esc(item['text'])}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='dl-empty'><div style='font-size:2rem'>✦</div>"
            "<div class='dl-card-title' style='font-size:1rem;margin-top:8px'>Forecast engine ready</div>"
            "<div class='dl-card-meta' style='max-width:700px;margin:6px auto'>"
            "Open Forecast Studio to compare SARIMA, Prophet and LightGBM, select the best model automatically, "
            "generate a forecast and create the complete AI PDF report.</div></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# Dataset
# ============================================================
def render_dataset():
    st.markdown("<div class='dl-section'><h3>Dataset Explorer</h3><span>Internal Walmart project data only</span></div>", unsafe_allow_html=True)

    cards = [
        ("Records", f"{profile['records']:,}", "train.csv"),
        ("Stores", profile["stores"], "unique stores"),
        ("Departments", profile["departments"], "unique departments"),
        ("Date range", f"{profile['date_start']} → {profile['date_end']}", "historical target"),
        ("Holiday weeks", profile["holiday_weeks"], "distinct holiday dates"),
        ("Quality", f"{quality['quality_score']:.1f}/100", quality_status(quality["quality_score"])),
    ]
    cols = st.columns(6)
    for col, (title, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='dl-card dl-tight'><div class='dl-kicker'>{esc(title)}</div>"
                f"<div class='dl-card-value' style='font-size:1.15rem'>{esc(value)}</div>"
                f"<div class='dl-card-meta'>{esc(sub)}</div></div>",
                unsafe_allow_html=True,
            )

    t1, t2, t3 = st.tabs(["Demand Trend", "Store Performance", "Data Preview"])
    with t1:
        st.plotly_chart(dataset_trend_chart(), use_container_width=True, config={"displayModeBar": False})
    with t2:
        store_perf = (
            loader.train.groupby("Store")["Weekly_Sales"]
            .agg(["sum", "mean", "std"])
            .sort_values("sum", ascending=False)
            .reset_index()
            .head(25)
        )
        store_perf.columns = ["Store", "Total Sales", "Average Weekly Sales", "Volatility"]
        st.dataframe(store_perf, use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(loader.train.head(100), use_container_width=True, hide_index=True)

    st.markdown("<div class='dl-section'><h3>Data quality checks</h3><span>Executed against the bundled train dataset</span></div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    for col, title, value in [
        (q1, "Duplicate keys", quality["duplicate_keys"]),
        (q2, "Negative sales", quality["negative_sales"]),
        (q3, "Missing cells", quality["missing_cells"]),
        (q4, "Quality score", f"{quality['quality_score']:.1f}/100"),
    ]:
        col.markdown(
            f"<div class='dl-card dl-tight'><div class='dl-kicker'>{title}</div><div class='dl-card-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# Forecast Studio
# ============================================================
def forecast_controls():
    stores = ["All Stores"] + [f"Store {x}" for x in get_store_options()]
    depts = ["All Departments"] + [f"Department {x}" for x in get_department_options()]

    store_label = st.selectbox("Store scope", stores, key="fc_store")
    dept_label = st.selectbox("Department scope", depts, key="fc_dept")
    horizon = st.select_slider(
        "Forecast horizon · weeks",
        options=[4, 8, 12, 16, 26, 39],
        value=12,
        key="fc_horizon",
    )
    confidence = st.select_slider(
        "Prediction interval",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        key="fc_confidence",
    )
    selected = st.multiselect(
        "Models to compare",
        ["SARIMA", "Prophet", "LightGBM"],
        default=["SARIMA", "Prophet", "LightGBM"],
        key="fc_models",
    )
    keys = {"SARIMA": "sarima", "Prophet": "prophet", "LightGBM": "lightgbm"}

    return {
        "store": None if store_label == "All Stores" else int(store_label.split()[-1]),
        "department": None if dept_label == "All Departments" else int(dept_label.split()[-1]),
        "horizon": int(horizon),
        "confidence": float(confidence),
        "models": [keys[x] for x in selected],
        "store_label": store_label,
        "department_label": dept_label,
    }


def render_forecast_studio():
    st.markdown(
        "<div class='dl-section'><h3>Forecast Studio</h3>"
        "<span>Real execution against the bundled Walmart dataset</span></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2.05])
    with left:
        st.markdown(
            "<div class='dl-card'><div class='dl-card-title'>Forecast configuration</div>"
            "<div class='dl-card-meta'>No upload, drag-and-drop or external data source is supported.</div>",
            unsafe_allow_html=True,
        )
        req = forecast_controls()
        st.markdown(
            "<div style='margin-top:8px'><span class='dl-pill dl-blue'>Walmart-only</span> "
            "<span class='dl-pill dl-purple'>Local ML</span> "
            "<span class='dl-pill dl-green'>Reproducible</span></div>",
            unsafe_allow_html=True,
        )
        generate = st.button(
            "🚀 Generate Forecast",
            type="primary",
            use_container_width=True,
            disabled=not req["models"],
            key="generate_forecast",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if last_result:
            best = last_result["best_model"]
            st.markdown(
                f"<div style='height:10px'></div><div class='dl-card'><div class='dl-kicker'>Latest run</div>"
                f"<div class='dl-card-title' style='margin-top:5px'>{esc(best['model'])}</div>"
                f"<div class='dl-positive' style='margin-top:6px'>WMAE {best['wmae']:.2f}</div>"
                f"<div class='dl-card-meta'>Completed {fmt_dt(last_result['run_completed_at'])}</div>"
                f"<div class='dl-card-meta'>Runtime {fmt_seconds(last_result['run_duration_seconds'])}</div></div>",
                unsafe_allow_html=True,
            )

    with right:
        if generate:
            st.markdown(
                "<div class='dl-card'><div class='dl-card-title'>Live forecast execution</div>"
                "<div class='dl-card-meta'>Progress advances at real pipeline boundaries. Model timestamps are captured from actual execution.</div>",
                unsafe_allow_html=True,
            )
            progress = st.progress(0)
            status = st.empty()
            detail = st.empty()
            timeline = st.empty()

            try:
                result = run_forecast(req, progress, status, detail, timeline)
                st.session_state.last_result = result
                st.session_state.last_run = result["run_completed_at"]
                st.success(
                    f"Forecast completed · {fmt_dt(result['run_completed_at'])} · "
                    f"runtime {fmt_seconds(result['run_duration_seconds'])}"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='dl-card' style='min-height:320px'><div class='dl-card-title'>Pipeline readiness</div>"
                "<div class='dl-card-meta'>A production run executes these real stages:</div>",
                unsafe_allow_html=True,
            )
            stages = [
                ("01", "Validate Walmart dataset"),
                ("02", "Prepare selected weekly time series"),
                ("03", "Backtest SARIMA / Prophet / LightGBM"),
                ("04", "Select lowest-WMAE model"),
                ("05", "Train winner on full history"),
                ("06", "Generate forecast and intervals"),
                ("07", "Generate AI insights and action plan"),
                ("08", "Generate PDF + CSV reports"),
            ]
            for number, label in stages:
                st.markdown(
                    f"<div class='dl-action'><div class='dl-action-title'>{number} · {esc(label)}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    if last_result:
        st.markdown(
            "<div class='dl-section'><h3>Latest forecast</h3><span>Actual demand, forecast and uncertainty interval</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(forecast_chart(last_result), use_container_width=True, config={"displayModeBar": False})

        fc = pd.DataFrame(last_result["forecast"])
        display = fc.copy()
        display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%d %b %Y")
        display = display.rename(columns={
            "Date": "Date",
            "forecast": "Forecast",
            "lower": "Lower Bound",
            "upper": "Upper Bound",
            "IsHoliday": "Holiday",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)


# ============================================================
# Model comparison / Backtesting
# ============================================================
def render_model_comparison(backtesting=False):
    title = "Backtesting" if backtesting else "Model Comparison"
    subtitle = (
        "Time-ordered validation, real fit/prediction timestamps and runtime"
        if backtesting else
        "SARIMA vs Prophet vs LightGBM from the latest forecast run"
    )
    st.markdown(
        f"<div class='dl-section'><h3>{title}</h3><span>{subtitle}</span></div>",
        unsafe_allow_html=True,
    )

    if not last_result:
        st.markdown(
            "<div class='dl-empty'><div style='font-size:2rem'>◒</div>"
            "<div class='dl-card-title'>No model run yet</div>"
            "<div class='dl-card-meta'>Generate a forecast in Forecast Studio to populate real validation metrics and timestamps.</div></div>",
            unsafe_allow_html=True,
        )
        return

    metrics = pd.DataFrame(last_result["metrics"])
    st.plotly_chart(model_bar_chart(metrics), use_container_width=True, config={"displayModeBar": False})

    best = last_result["best_model"]
    cards = [
        ("Winner", best["model"]),
        ("WMAE", f"{best['wmae']:.2f}"),
        ("MAE", f"{best['mae']:.2f}"),
        ("RMSE", f"{best['rmse']:.2f}"),
        ("MAPE", f"{best['mape']:.2f}%"),
    ]
    cols = st.columns(5)
    for col, (title, value) in zip(cols, cards):
        col.markdown(
            f"<div class='dl-card dl-tight'><div class='dl-kicker'>{title}</div>"
            f"<div class='dl-card-value'>{esc(value)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='dl-section'><h3>Model execution timeline</h3>"
        "<span>Actual timestamps recorded during this run</span></div>",
        unsafe_allow_html=True,
    )

    display = metrics.copy()
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "seconds": st.column_config.NumberColumn("Runtime (s)", format="%.3f"),
            "started_at": st.column_config.TextColumn("Fit started"),
            "fit_completed_at": st.column_config.TextColumn("Fit completed"),
            "prediction_started_at": st.column_config.TextColumn("Prediction started"),
            "prediction_completed_at": st.column_config.TextColumn("Prediction completed"),
            "completed_at": st.column_config.TextColumn("Model completed"),
        },
    )

    st.markdown(
        f"<div class='dl-card'><div class='dl-card-title'>🏆 Why {esc(best['model'])} won</div>"
        f"<div class='dl-card-meta'>The winner is selected automatically from successful models using the lowest validation WMAE. "
        f"No model is hard-coded as the preferred production model.</div>"
        f"<div style='margin-top:9px'><span class='dl-pill dl-green'>Lowest WMAE</span> "
        f"<span class='dl-pill dl-blue'>Time-ordered validation</span> "
        f"<span class='dl-pill dl-purple'>Uncertainty included</span></div></div>",
        unsafe_allow_html=True,
    )

    if backtesting:
        st.markdown(
            "<div class='dl-section'><h3>Backtesting methodology</h3><span>How the validation result should be interpreted</span></div>",
            unsafe_allow_html=True,
        )
        st.info(
            "The dataset is split chronologically: earlier observations train the model and the later holdout "
            "is used for validation. This avoids random shuffling and better reflects real forecasting."
        )


# ============================================================
# AI Insights
# ============================================================
def render_insights():
    st.markdown(
        "<div class='dl-section'><h3>AI Decision Intelligence</h3>"
        "<span>Data-grounded interpretation of the latest forecast</span></div>",
        unsafe_allow_html=True,
    )

    if not last_result:
        st.markdown(
            "<div class='dl-empty'><div style='font-size:2rem'>✦</div>"
            "<div class='dl-card-title'>Generate a forecast first</div>"
            "<div class='dl-card-meta'>Insights are calculated from historical demand, forecast trajectory, variability, holiday behavior and model performance.</div></div>",
            unsafe_allow_html=True,
        )
        return

    ins = last_result["insights"]
    cards = [
        ("Trend", ins["trend"].title(), "Recent 26-week direction"),
        ("Forecast change", f"{ins['forecast_change_pct']:+.1f}%", "Vs recent 12-week average"),
        ("Confidence", f"{ins['confidence']:.0f}%", "Derived decision indicator"),
        ("Peak", f"{ins['peak_demand']:,.0f}", ins["peak_date"]),
    ]
    cols = st.columns(4)
    for col, (title, value, sub) in zip(cols, cards):
        col.markdown(
            f"<div class='dl-card dl-tight'><div class='dl-kicker'>{title}</div>"
            f"<div class='dl-card-value'>{esc(value)}</div><div class='dl-card-meta'>{esc(sub)}</div></div>",
            unsafe_allow_html=True,
        )

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='dl-card'><div class='dl-card-title'>What the data says</div>", unsafe_allow_html=True)
        for item in ins["insights"]:
            st.markdown(
                f"<div class='dl-insight'><b>{esc(item['title'])}</b>"
                f"<div class='dl-card-meta'>{esc(item['text'])}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='dl-card'><div class='dl-card-title'>Next action plan</div>", unsafe_allow_html=True)
        for item in ins["actions"]:
            st.markdown(
                f"<div class='dl-action'><div class='dl-action-title'>{esc(item['priority'])} · {esc(item['title'])}</div>"
                f"<div class='dl-action-text'>{esc(item['text'])}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='dl-section'><h3>Forecast interpretation</h3><span>Visual context for the AI recommendations</span></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(forecast_chart(last_result), use_container_width=True, config={"displayModeBar": False})


# ============================================================
# Reports
# ============================================================
def render_reports():
    st.markdown(
        "<div class='dl-section'><h3>Report Center</h3>"
        "<span>Visual PDF + machine-readable CSV generated from the same forecast run</span></div>",
        unsafe_allow_html=True,
    )

    if not last_result:
        st.markdown(
            "<div class='dl-empty'><div style='font-size:2rem'>▤</div>"
            "<div class='dl-card-title'>No report available yet</div>"
            "<div class='dl-card-meta'>Generate a forecast. Both reports are created automatically.</div></div>",
            unsafe_allow_html=True,
        )
        return

    csv_path = Path(last_result["reports"]["csv"])
    pdf_path = Path(last_result["reports"]["pdf"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div class='dl-card'><div class='dl-kicker'>CSV report</div>"
            "<div class='dl-card-value'>Forecast Dataset</div>"
            "<div class='dl-card-meta'>Forecast dates, confidence bounds, holiday calendar, selection and winning model.</div></div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download Forecast CSV",
            csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.markdown(
            "<div class='dl-card'><div class='dl-kicker'>PDF report</div>"
            "<div class='dl-card-value'>AI Decision Report</div>"
            "<div class='dl-card-meta'>Visual forecast, model comparison, execution timestamps, AI insights, recommendations and action plan.</div></div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download Full AI PDF",
            pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown(
        "<div class='dl-section'><h3>Report contents</h3><span>Included in every production report</span></div>",
        unsafe_allow_html=True,
    )
    contents = [
        "Report generation date and time",
        "Forecast run start / completion / duration",
        "Historical vs forecast visualization",
        "Forecast dates and uncertainty bounds",
        "SARIMA / Prophet / LightGBM validation metrics",
        "Model fit, prediction and completion timestamps",
        "Full-history winner training timestamp",
        "AI insights and decision confidence",
        "Recommended next action plan",
        "Walmart dataset metadata",
    ]
    for i in range(0, len(contents), 2):
        cols = st.columns(2)
        for j, item in enumerate(contents[i:i + 2]):
            cols[j].markdown(
                f"<div class='dl-action'><div class='dl-action-title'>✓ {esc(item)}</div></div>",
                unsafe_allow_html=True,
            )


# ============================================================
# System
# ============================================================
def render_system_info():
    st.markdown(
        "<div class='dl-section'><h3>System Information</h3>"
        "<span>Production execution contract</span></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, title, value in [
        (cols[0], "Application", settings.app_name),
        (cols[1], "Version", settings.version),
        (cols[2], "Engine", "Streamlit"),
        (cols[3], "Dataset", "Bundled Walmart"),
    ]:
        col.markdown(
            f"<div class='dl-card dl-tight'><div class='dl-kicker'>{esc(title)}</div>"
            f"<div class='dl-card-value' style='font-size:1.05rem'>{esc(value)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='dl-section'><h3>Production guarantees</h3><span>Intentional product behavior</span></div>",
        unsafe_allow_html=True,
    )

    guarantees = [
        ("Walmart-only data", "The application reads the dataset packaged under data/. There is no upload widget, drag-and-drop area or external CSV ingestion feature."),
        ("Real progress", "Progress advances at real pipeline boundaries: validation, model lifecycle events, full-history training, forecast generation, AI analysis and report generation."),
        ("Real timestamps", "Model start, fit completion, prediction start/completion, full-history training and report creation are recorded from actual execution."),
        ("Automatic model selection", "Successful models are ranked by time-ordered validation WMAE; the lowest WMAE becomes the forecast winner."),
        ("System appearance", "There is no in-app theme selector. Custom UI surfaces use prefers-color-scheme so the interface follows system/browser appearance."),
        ("Reproducible reports", "The PDF and CSV are generated from the same payload used to render the forecast dashboard."),
    ]
    for title, text in guarantees:
        st.markdown(
            f"<div class='dl-action'><div class='dl-action-title'>{esc(title)}</div>"
            f"<div class='dl-action-text'>{esc(text)}</div></div>",
            unsafe_allow_html=True,
        )

    if last_result:
        st.markdown(
            "<div class='dl-section'><h3>Latest execution metadata</h3><span>Captured from the latest run</span></div>",
            unsafe_allow_html=True,
        )
        meta = [
            ("Run started", fmt_dt(last_result["run_started_at"])),
            ("Run completed", fmt_dt(last_result["run_completed_at"])),
            ("Run duration", fmt_seconds(last_result["run_duration_seconds"])),
            ("Winner full fit", fmt_dt(last_result["best_model"].get("full_fit_completed_at"))),
            ("Forecast generation", fmt_dt(last_result.get("prediction_completed_at"))),
            ("PDF", Path(last_result["reports"]["pdf"]).name),
        ]
        cols = st.columns(3)
        for col, (title, value) in zip(cols * 2, meta):
            col.markdown(
                f"<div class='dl-card dl-tight'><div class='dl-kicker'>{esc(title)}</div>"
                f"<div class='dl-card-meta' style='margin-top:7px;color:var(--dl-text);font-weight:750'>{esc(value)}</div></div>",
                unsafe_allow_html=True,
            )


# ============================================================
# Router
# ============================================================
if st.session_state.page == "Overview":
    render_overview()
elif st.session_state.page == "Dataset":
    render_dataset()
elif st.session_state.page == "Forecast Studio":
    render_forecast_studio()
elif st.session_state.page == "Model Comparison":
    render_model_comparison(False)
elif st.session_state.page == "Backtesting":
    render_model_comparison(True)
elif st.session_state.page == "AI Insights":
    render_insights()
elif st.session_state.page == "Reports":
    render_reports()
elif st.session_state.page == "System Info":
    render_system_info()

st.markdown(
    "<div class='dl-footer'>© 2026 Kaustubh Suryawanshi · Walmart Demand Intelligence · "
    "Data Science / AI-ML portfolio application · Local Walmart dataset only</div>",
    unsafe_allow_html=True,
)
