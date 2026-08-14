from __future__ import annotations

import numpy as np
import pandas as pd


def generate_insights(history, forecast, metrics):
    y = history["Weekly_Sales"].astype(float)
    f = forecast["forecast"].astype(float)

    recent = float(y.tail(12).mean())
    future = float(f.mean())
    change = (future / recent - 1) * 100 if recent else 0.0

    tail26 = y.tail(26)
    slope = (
        float(np.polyfit(np.arange(len(tail26)), tail26, 1)[0])
        if len(tail26) >= 2 else 0.0
    )
    cv = float(tail26.std() / max(1.0, tail26.mean()))

    holiday_mean = float(history.loc[history["IsHoliday"], "Weekly_Sales"].mean()) if history["IsHoliday"].any() else 0.0
    nonholiday = ~history["IsHoliday"]
    nonholiday_mean = float(history.loc[nonholiday, "Weekly_Sales"].mean()) if nonholiday.any() else 0.0
    holiday_lift = (holiday_mean / nonholiday_mean - 1) * 100 if nonholiday_mean else 0.0

    trend = "upward" if slope > 0 else "downward" if slope < 0 else "stable"
    confidence = max(55.0, min(96.0, 90.0 - cv * 45.0))

    peak = forecast.loc[forecast["forecast"].idxmax()]
    best = next((m for m in metrics if m.get("status") == "ok"), None)

    model_quality = (
        f"{best['model']} leads validation with WMAE {best['wmae']:.2f}."
        if best else "No successful model metric is available."
    )

    insight_rows = [
        {
            "type": "trend",
            "title": "Demand trend",
            "text": (
                f"Recent demand shows a {trend} direction. The forecast average is "
                f"{change:+.1f}% versus the most recent 12-week baseline."
            ),
        },
        {
            "type": "seasonality",
            "title": "Seasonal structure",
            "text": (
                "The application forecasts at Walmart's weekly grain and compares "
                "SARIMA, Prophet and LightGBM against a time-ordered validation window."
            ),
        },
        {
            "type": "holiday",
            "title": "Holiday effect",
            "text": (
                f"Historical holiday weeks differ by approximately {holiday_lift:+.1f}% "
                "from non-holiday weeks in the selected history."
            ),
        },
        {
            "type": "risk",
            "title": "Forecast variability",
            "text": (
                f"Recent coefficient of variation is {cv:.1%}; the derived decision "
                f"confidence indicator is {confidence:.0f}%."
            ),
        },
        {
            "type": "model",
            "title": "Model decision",
            "text": model_quality,
        },
    ]

    actions = []
    if change > 5:
        actions.append({
            "priority": "High",
            "title": "Prepare replenishment capacity",
            "text": f"Forecast demand is {change:.1f}% above the recent baseline; review inventory and replenishment capacity before the projected increase.",
        })
    elif change < -5:
        actions.append({
            "priority": "High",
            "title": "Review inventory exposure",
            "text": f"Forecast demand is {abs(change):.1f}% below the recent baseline; review replenishment quantities and overstock exposure.",
        })
    else:
        actions.append({
            "priority": "Medium",
            "title": "Maintain current demand plan",
            "text": "The forecast remains close to the recent baseline; maintain the current plan while monitoring new observations.",
        })

    actions.extend([
        {
            "priority": "High" if confidence < 70 else "Medium",
            "title": "Monitor forecast uncertainty",
            "text": f"The projected peak is {pd.to_datetime(peak['Date']).date().isoformat()} at {peak['forecast']:,.0f} weekly sales. Review the confidence interval before committing inventory.",
        },
        {
            "priority": "Medium",
            "title": "Review holiday weeks",
            "text": "Use the holiday effect and the forecast calendar together when planning replenishment around peak seasonal periods.",
        },
        {
            "priority": "Low",
            "title": "Re-run model comparison with new data",
            "text": "When additional Walmart observations become available, repeat time-ordered backtesting and allow the lowest-WMAE model to win again.",
        },
    ])

    return {
        "forecast_change_pct": float(change),
        "confidence": float(confidence),
        "trend": trend,
        "volatility_cv": cv,
        "holiday_lift_pct": float(holiday_lift),
        "peak_date": pd.to_datetime(peak["Date"]).date().isoformat(),
        "peak_demand": float(peak["forecast"]),
        "insights": insight_rows,
        "actions": actions,
    }
