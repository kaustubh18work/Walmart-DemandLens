from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.config import settings
from .schema import locate_walmart_files


class WalmartDataLoader:
    """Loads the bundled Walmart dataset."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or settings.data_dir)
        self._train: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._stores: pd.DataFrame | None = None

    def _require(self, filename: str) -> Path:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Walmart dataset file: {path}")
        return path

    def validate(self) -> None:
        """Validate that the primary training dataset exists."""
        self._require("train.csv")

    def validate_full_dataset(self) -> None:
        """Validate the complete bundled Walmart dataset."""
        locate_walmart_files(self.data_dir)

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            path = self._require("train.csv")
            df = pd.read_csv(path, parse_dates=["Date"])

            required = {
                "Store",
                "Dept",
                "Date",
                "Weekly_Sales",
                "IsHoliday",
            }
            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"train.csv is missing columns: {sorted(missing)}"
                )

            df["IsHoliday"] = df["IsHoliday"].astype(bool)
            df = df.sort_values(
                ["Store", "Dept", "Date"]
            ).reset_index(drop=True)

            self._train = df

        return self._train

    @property
    def test(self) -> pd.DataFrame:
        if self._test is None:
            path = self._require("test.csv")
            df = pd.read_csv(path, parse_dates=["Date"])

            if "IsHoliday" in df:
                df["IsHoliday"] = df["IsHoliday"].astype(bool)

            self._test = df.sort_values(
                ["Store", "Dept", "Date"]
            ).reset_index(drop=True)

        return self._test

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            path = self._require("features.csv")
            df = pd.read_csv(path, parse_dates=["Date"])

            if "IsHoliday" in df:
                df["IsHoliday"] = df["IsHoliday"].astype(bool)

            self._features = df

        return self._features

    @property
    def stores(self) -> pd.DataFrame:
        if self._stores is None:
            path = self._require("stores.csv")
            self._stores = pd.read_csv(path)

        return self._stores

    def merged(self) -> pd.DataFrame:
        return (
            self.train
            .merge(
                self.features,
                on=["Store", "Date", "IsHoliday"],
                how="left",
                suffixes=("", "_feature"),
            )
            .merge(self.stores, on="Store", how="left")
        )

    def future_calendar(
        self,
        store: int | None = None,
        department: int | None = None,
    ) -> pd.DataFrame:
        """Return bundled Walmart test-calendar dates where available."""
        df = self.test.copy()

        if store is not None:
            df = df[df["Store"] == store]

        if department is not None:
            df = df[df["Dept"] == department]

        return (
            df[["Date", "IsHoliday"]]
            .drop_duplicates()
            .sort_values("Date")
        )

    def raw_status(self) -> dict[str, bool]:
        raw = self.data_dir / "raw"

        names = [
            "calendar.csv",
            "sell_prices.csv",
            "sales_train_validation.csv",
            "sales_train_evaluation.csv",
            "sample_submission.csv",
        ]

        return {
            name: (raw / name).exists()
            for name in names
        }


loader = WalmartDataLoader()
