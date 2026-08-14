import pandas as pd

from demandlens.insights.engine import generate_insights


def test_insights_contract():
    dates = pd.date_range("2024-01-05", periods=80, freq="7D")
    history = pd.DataFrame({
        "Date": dates,
        "Weekly_Sales": [1000 + i * 2 for i in range(80)],
        "IsHoliday": [False] * 78 + [True, False],
    })
    forecast_dates = pd.date_range(dates[-1] + pd.Timedelta(days=7), periods=4, freq="7D")
    forecast = pd.DataFrame({
        "Date": forecast_dates,
        "forecast": [1200, 1210, 1220, 1230],
        "lower": [1100, 1110, 1120, 1130],
        "upper": [1300, 1310, 1320, 1330],
    })
    metrics = [{
        "model": "TestModel", "status": "ok",
        "mae": 10, "rmse": 12, "mape": 1.2, "wmae": 9
    }]
    result = generate_insights(history, forecast, metrics)
    assert "insights" in result
    assert "actions" in result
    assert result["peak_demand"] == 1230
