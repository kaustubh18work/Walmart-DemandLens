from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WalmartFiles:
    train: Path
    test: Path
    features: Path
    stores: Path
    raw_calendar: Path
    raw_sell_prices: Path
    raw_validation: Path
    raw_evaluation: Path


def locate_walmart_files(data_dir: Path) -> WalmartFiles:
    raw = data_dir / "raw"
    required = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "features": data_dir / "features.csv",
        "stores": data_dir / "stores.csv",
        "raw_calendar": raw / "calendar.csv",
        "raw_sell_prices": raw / "sell_prices.csv",
        "raw_validation": raw / "sales_train_validation.csv",
        "raw_evaluation": raw / "sales_train_evaluation.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Walmart dataset files: {', '.join(missing)}")
    return WalmartFiles(**required)
