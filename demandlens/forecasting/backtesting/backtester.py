import time
from datetime import datetime

from ..metrics.metrics import mae, rmse, mape, wmae
from ..models.sarima import SARIMAForecaster
from ..models.prophet_model import ProphetForecaster
from ..models.lightgbm_model import LightGBMForecaster

MODELS = {
    "sarima": SARIMAForecaster,
    "prophet": ProphetForecaster,
    "lightgbm": LightGBMForecaster,
}


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluate_models(ts, names, progress=lambda *a, **k: None):
    """Run a deterministic time-series holdout comparison.

    Progress callbacks are emitted only at real pipeline boundaries. Each model
    records wall-clock timestamps for start, fit completion, prediction
    completion and final completion so the UI/report can show actual execution
    timing instead of simulated progress.
    """
    if len(ts) < 60:
        raise ValueError("At least 60 weekly observations are required for reliable model comparison.")

    split = max(24, int(len(ts) * 0.8))
    train = ts.iloc[:split].copy()
    valid = ts.iloc[split:].copy()
    results = []
    total = len(names)

    for i, name in enumerate(names):
        if name not in MODELS:
            continue

        model_cls = MODELS[name]
        model_name = model_cls.name
        started_at = _now()
        started = time.perf_counter()
        base = 12 + (i / max(total, 1)) * 56

        progress(
            name,
            round(base),
            "started",
            model_started_at=started_at,
        )

        try:
            model = model_cls().fit(train)
            fit_completed_at = _now()
            fit_seconds = round(time.perf_counter() - started, 3)
            progress(
                name,
                round(base + 0.45 * (56 / max(total, 1))),
                "fitted",
                model_started_at=started_at,
                model_fit_completed_at=fit_completed_at,
                model_duration_seconds=fit_seconds,
            )

            prediction_started_at = _now()
            pred = model.predict(len(valid))
            prediction_completed_at = _now()
            progress(
                name,
                round(base + 0.75 * (56 / max(total, 1))),
                "predicted",
                model_started_at=started_at,
                model_fit_completed_at=fit_completed_at,
                model_prediction_started_at=prediction_started_at,
                model_prediction_completed_at=prediction_completed_at,
                model_duration_seconds=round(time.perf_counter() - started, 3),
            )

            y = valid.Weekly_Sales.values
            yh = pred.forecast.values[: len(y)]
            completed_at = _now()
            total_seconds = round(time.perf_counter() - started, 3)
            r = {
                "model": model_name,
                "key": name,
                "mae": mae(y, yh),
                "rmse": rmse(y, yh),
                "mape": mape(y, yh),
                "wmae": wmae(y, yh, valid.IsHoliday.values),
                "seconds": total_seconds,
                "started_at": started_at,
                "fit_completed_at": fit_completed_at,
                "prediction_started_at": prediction_started_at,
                "prediction_completed_at": prediction_completed_at,
                "completed_at": completed_at,
                "status": "ok",
            }
            results.append(r)
            progress(
                name,
                round(12 + ((i + 1) / max(total, 1)) * 56),
                "completed",
                model_started_at=started_at,
                model_fit_completed_at=fit_completed_at,
                model_prediction_completed_at=prediction_completed_at,
                model_completed_at=completed_at,
                model_duration_seconds=total_seconds,
            )
        except Exception as e:
            completed_at = _now()
            total_seconds = round(time.perf_counter() - started, 3)
            r = {
                "model": model_name,
                "key": name,
                "mae": None,
                "rmse": None,
                "mape": None,
                "wmae": None,
                "seconds": total_seconds,
                "started_at": started_at,
                "fit_completed_at": None,
                "prediction_started_at": None,
                "prediction_completed_at": None,
                "completed_at": completed_at,
                "status": "error",
                "error": str(e),
            }
            results.append(r)
            progress(
                name,
                round(12 + ((i + 1) / max(total, 1)) * 56),
                "failed",
                model_started_at=started_at,
                model_completed_at=completed_at,
                model_duration_seconds=total_seconds,
            )

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: r["wmae"])
    for i, r in enumerate(ok, 1):
        r["rank"] = i

    return results, train, valid
