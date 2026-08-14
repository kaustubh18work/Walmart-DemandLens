def test_walmart_dataset_loads(test_loader):
    df = test_loader.train

    assert not df.empty
    assert {
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday",
    } <= set(df.columns)


def test_walmart_profile(test_loader, monkeypatch):
    from demandlens.walmart import profiler

    monkeypatch.setattr(profiler, "loader", test_loader)

    profile = profiler.profile_walmart()

    assert profile["records"] > 0
    assert profile["stores"] > 0
    assert profile["departments"] > 0
    assert profile["frequency"] == "Weekly"


def test_data_quality(test_loader, monkeypatch):
    from demandlens.walmart import profiler

    monkeypatch.setattr(profiler, "loader", test_loader)

    quality = profiler.data_quality()

    assert 0 <= quality["quality_score"] <= 100
    assert quality["duplicate_keys"] >= 0
