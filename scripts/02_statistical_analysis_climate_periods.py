import pandas as pd

# ============================================================
# PARAMETERS
# ============================================================
HIST_START = "1990-01-01"
HIST_END   = "2019-12-31 23:00"

FUT_START  = "2040-01-01"
FUT_END    = "2069-12-31 23:00"

INPUT_FILE = "results/climate_3h_all_zones.csv"
OUTPUT_DIR = "results/statistics/"

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_FILE, parse_dates=["time"])

# Convert tas from Kelvin to Celsius and huss to percentage
df["tas"] = df["tas"] - 273.15
df["huss"] = df["huss"] * 100

df["month"] = df["time"].dt.month
df["hour"]  = df["time"].dt.hour

# ============================================================
# SPLIT PERIODS
# ============================================================
df_hist = df[
    (df["period"] == "historical") &
    (df["time"] >= HIST_START) &
    (df["time"] <= HIST_END)
]

df_fut = df[
    (df["period"] == "ssp585") &
    (df["time"] >= FUT_START) &
    (df["time"] <= FUT_END)
]

# ============================================================
# FUNCTION: CLIMATOLOGICAL STATISTICS
# ============================================================
def compute_statistics(df, label):

    # --- Annual statistics ---
    annual = df.groupby("zone").agg(
        tas_mean=("tas", "mean"),
        tas_max=("tas", "max"),
        tas_min=("tas", "min"),
        huss_mean=("huss", "mean"),
        vas_mean=("vas", "mean"),
        rsds_mean=("rsds", "mean")
    ).reset_index()

    annual.to_csv(f"{OUTPUT_DIR}annual_stats_{label}.csv", index=False)

    # --- Monthly climatology ---
    monthly = df.groupby(["zone", "month"]).agg(
        tas_mean=("tas", "mean"),
        huss_mean=("huss", "mean"),
        vas_mean=("vas", "mean"),
        rsds_mean=("rsds", "mean")
    ).reset_index()

    monthly.to_csv(f"{OUTPUT_DIR}monthly_climatology_{label}.csv", index=False)



# ============================================================
# FUNCTION: THERMAL EXTREMES (BUILDING-ORIENTED)
# ============================================================
def compute_extremes(df, label):

    extremes = df.groupby("zone").agg(
        hours_above_35=("tas", lambda x: (x > 35).sum()),
        hours_above_40=("tas", lambda x: (x > 40).sum()),
        warm_nights=("tas", lambda x:
            ((x > 25) & (df.loc[x.index, "hour"].isin([21, 0, 3, 6]))).sum())
    ).reset_index()

    extremes.to_csv(f"{OUTPUT_DIR}thermal_extremes_{label}.csv", index=False)

    return extremes


# ============================================================
# HOT DAYS & HEATWAVES (percentile-based)
# hot day: daily maximum > historical 90th percentile
# heatwave: >= 3 consecutive hot days
# ============================================================
def compute_hotdays_and_heatwaves(df_period, df_hist, label, percentile=95, min_duration=3):
    # daily max per zone for historical baseline
    hist_daily = (
        df_hist.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"]
        .max()
        .reset_index()
        .rename(columns={"tas": "tas_daily_max", "time": "date"})
    )

    # threshold per zone
    thresholds = (
        hist_daily.groupby("zone").tas_daily_max.quantile(percentile / 100.0).reset_index()
    )
    thresholds.columns = ["zone", "threshold"]

    # daily max for the target period
    period_daily = (
        df_period.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"]
        .max()
        .reset_index()
        .rename(columns={"tas": "tas_daily_max", "time": "date"})
    )

    records = []
    per_year_rows = []
    zones = sorted(df_period["zone"].unique())
    from itertools import groupby

    for zone in zones:
        thr_row = thresholds[thresholds["zone"] == zone]
        if thr_row.empty:
            continue
        threshold = float(thr_row["threshold"].iloc[0])

        pd_zone = period_daily[period_daily["zone"] == zone].set_index("date").sort_index()
        if pd_zone.index.size == 0:
            continue

        idx = pd.date_range(pd_zone.index.min(), pd_zone.index.max(), freq="D")
        tas_series = pd_zone.reindex(idx)["tas_daily_max"].fillna(-9999)

        is_hot = tas_series > threshold
        hot_days = int(is_hot.sum())

        # annual counts (use calendar year)
        ser = pd.Series(is_hot.astype(int), index=tas_series.index)
        annual_counts = ser.groupby(ser.index.year).sum()

        lengths = [sum(1 for _ in group) for val, group in groupby(is_hot) if val]
        heatwave_events = sum(1 for L in lengths if L >= min_duration)
        heatwave_days = sum(L for L in lengths if L >= min_duration)
        max_hw_duration = max(lengths) if lengths else 0

        records.append({
            "zone": zone,
            f"threshold_{percentile}pct": threshold,
            "hot_days": hot_days,
            "hot_days_mean_per_year": float(annual_counts.mean()) if len(annual_counts) else 0.0,
            "heatwave_events": heatwave_events,
            "heatwave_days": heatwave_days,
            "heatwave_max_duration": max_hw_duration,
        })

        # store per-year rows for this zone
        for yr, cnt in annual_counts.items():
            per_year_rows.append({"zone": zone, "year": int(yr), "hot_days": int(cnt)})

    out = pd.DataFrame.from_records(records)
    per_year_df = pd.DataFrame.from_records(per_year_rows)
    per_year_df = per_year_df.merge(thresholds, on="zone", how="left")
    per_year_df.rename(columns={"threshold": f"threshold_{percentile}pct"}, inplace=True)
    per_year_df.to_csv(f"{OUTPUT_DIR}thermal_hotdays_by_year_{label}.csv", index=False)


    return out


# ============================================================
# RUN ANALYSIS
# ============================================================
print("Running historical climate statistics...")
compute_statistics(df_hist, "historical_1990_2019")
compute_extremes(df_hist, "historical_1990_2019")
# hot days & heatwaves using historical baseline thresholds
compute_hotdays_and_heatwaves(df_hist, df_hist, "historical_1990_2019", percentile=90, min_duration=3)

print("Running future climate statistics...")
compute_statistics(df_fut, "future_2040_2069_ssp585")
compute_extremes(df_fut, "future_2040_2069_ssp585")
compute_hotdays_and_heatwaves(df_fut, df_hist, "future_2040_2069_ssp585", percentile=90, min_duration=3)

print("✅ Statistical analysis completed successfully")
