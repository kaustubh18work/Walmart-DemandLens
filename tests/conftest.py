from pathlib import Path

import pytest

from demandlens.walmart.loader import WalmartDataLoader


@pytest.fixture
def test_loader():
    data_dir = Path(__file__).parent / "fixtures" / "walmart"
    return WalmartDataLoader(data_dir=data_dir)
