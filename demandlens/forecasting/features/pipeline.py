import numpy as np


def aggregate(df, store=None, department=None):
    x = df.copy()
    if store is not None:
        x = x[x.Store == store]
    if department is not None:
        x = x[x.Dept == department]

    # Weekly Walmart observations; aggregate to a single time series.
    g = x.groupby("Date", as_index=False).agg(
        Weekly_Sales=("Weekly_Sales", "sum"),
        IsHoliday=("IsHoliday", "max"),
    )
    return g.sort_values("Date").reset_index(drop=True)


def make_ml_features(ts):
    x = ts.copy().sort_values("Date")
    d = x.Date
    x["year"] = d.dt.year
    x["month"] = d.dt.month
    x["weekofyear"] = d.dt.isocalendar().week.astype(int)
    x["trend_idx"] = np.arange(len(x))

    for lag in (1, 2, 4, 8, 12, 26, 52):
        x[f"lag_{lag}"] = x.Weekly_Sales.shift(lag)

    for win in (4, 8, 12, 26):
        x[f"roll_mean_{win}"] = x.Weekly_Sales.shift(1).rolling(win).mean()
        x[f"roll_std_{win}"] = x.Weekly_Sales.shift(1).rolling(win).std()

    return x
