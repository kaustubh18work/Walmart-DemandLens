from .loader import loader

def profile_walmart():
    df = loader.train
    missing = float(df.isna().mean().mean()*100)
    dates = df["Date"].sort_values().drop_duplicates()
    diffs = dates.diff().dropna().dt.days
    freq = "Weekly" if not diffs.empty and abs(diffs.median()-7) <= 1 else "Daily"
    return {
        "records": int(len(df)), "stores": int(df.Store.nunique()), "departments": int(df.Dept.nunique()),
        "date_start": df.Date.min().date().isoformat(), "date_end": df.Date.max().date().isoformat(),
        "frequency": freq, "missing_values": missing, "target": "Weekly_Sales",
        "columns": list(df.columns), "raw_files": loader.raw_status(),
        "holiday_weeks": int(df.loc[df.IsHoliday, "Date"].nunique()),
        "total_sales": float(df.Weekly_Sales.sum()),
    }

def data_quality():
    df=loader.train
    dup=int(df.duplicated(["Store","Dept","Date"]).sum())
    neg=int((df.Weekly_Sales<0).sum())
    return {"duplicate_keys":dup,"negative_sales":neg,"missing_cells":int(df.isna().sum().sum()),"quality_score":max(0,100-min(100,dup/len(df)*100)-min(20,neg/len(df)*100))}
