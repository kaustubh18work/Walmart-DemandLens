from demandlens.forecasting.features.pipeline import (
    aggregate,
    make_ml_features,
)


def test_aggregate_selection(test_loader):
    ts = aggregate(
        test_loader.train,
        store=1,
        department=1,
    )

    assert len(ts) >= 60
    assert ts["Date"].is_monotonic_increasing
    assert ts["Weekly_Sales"].notna().all()


def test_feature_pipeline_has_expected_lags(test_loader):
    ts = aggregate(
        test_loader.train,
        store=1,
        department=1,
    )

    features = make_ml_features(ts)

    for lag in (1, 2, 4, 8, 12, 26, 52):
        assert f"lag_{lag}" in features.columns

    assert features["trend_idx"].iloc[-1] == len(features) - 1
