import pandas as pd

# ============================================================
# PARAMETERS
# ============================================================
from heatwaves_utils import (
    HIST_START, HIST_END, FUT_START, FUT_END, INPUT_FILE,
    calculate_calendar_thresholds, thresholds_for_dates,
)
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
    #(df["period"] == "historical") &
    (df["time"] >= HIST_START) &
    (df["time"] <= HIST_END)
]

df_fut = df[
    #(df["period"] == "ssp585") &
    (df["time"] >= FUT_START) &
    (df["time"] <= FUT_END)
]

# ============================================================
# HOT DAYS & HEATWAVES (percentile-based)
# hot day: daily maximum > historical moving-window percentile for that calendar day
# heatwave: >= 3 consecutive hot days
# ============================================================

def compute_hotdays_and_heatwaves(
    df_period,
    df_hist,
    label,
    summary_rows=None,
    percentile=95,
    min_duration=3,
    threshold_output_path=None,
):
    # daily max per zone for historical baseline
    hist_daily = (
        df_hist.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"]
        .max()
        .reset_index()
        .rename(columns={"tas": "tas_daily_max", "time": "date"})
    )

    # daily max for the target period
    period_daily = (
        df_period.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"]
        .max()
        .reset_index()
        .rename(columns={"tas": "tas_daily_max", "time": "date"})
    )

    records = []
    per_year_rows = []
    threshold_rows = []
    daily_threshold_rows = []
    zones = sorted(df_period["zone"].unique())
    from itertools import groupby

    for zone in zones:
        pd_zone = period_daily[period_daily["zone"] == zone].set_index("date").sort_index()
        if pd_zone.index.size == 0:
            continue

        hist_zone = hist_daily[hist_daily["zone"] == zone].set_index("date").sort_index()
        calendar_thresholds = calculate_calendar_thresholds(
            hist_zone["tas_daily_max"], percentile=percentile
        )
        if calendar_thresholds.isna().all():
            continue
        threshold_by_day = (
            calendar_thresholds.rename(f"threshold_{percentile}pct")
            .rename_axis("day_of_year")
            .reset_index()
        )
        threshold_by_day["month_day"] = pd.date_range(
            "2000-01-01", periods=366, freq="D"
        ).strftime("%m-%d")
        threshold_by_day.insert(0, "zone", zone)
        daily_threshold_rows.append(threshold_by_day)

        # Keep the existing summary field as the mean of the calendar-day thresholds.
        threshold = float(calendar_thresholds.mean())
        threshold_rows.append({"zone": zone, "threshold": threshold})

        idx = pd.date_range(pd_zone.index.min(), pd_zone.index.max(), freq="D")
        tas_series = pd_zone.reindex(idx)["tas_daily_max"].fillna(-9999)
        daily_thresholds = thresholds_for_dates(idx, calendar_thresholds)

        is_hot = tas_series > daily_thresholds
        hot_days = int(is_hot.sum())

        # annual counts (use calendar year)
        ser = pd.Series(is_hot.astype(int), index=tas_series.index)
        annual_counts = ser.groupby(ser.index.year).sum()

        lengths = [sum(1 for _ in group) for val, group in groupby(is_hot) if val]
        heatwave_events = sum(1 for L in lengths if L >= min_duration)
        heatwave_days = sum(L for L in lengths if L >= min_duration)
        max_hw_duration = max(lengths) if lengths else 0

        result_row = {
            "zone": zone,
            "period": label,
          #  f"threshold_{percentile}pct": threshold,
          # "hot_days": hot_days,
          # "hot_days_mean_per_year": float(annual_counts.mean()) if len(annual_counts) else 0.0,
          # "heatwave_days": heatwave_days,
            "heatwave_events": heatwave_events,
            "heatwave_max_duration": max_hw_duration,
        }
        records.append(result_row)
        if summary_rows is not None:
            summary_rows.append(result_row)

        # store per-year rows for this zone
        for yr, cnt in annual_counts.items():
            per_year_rows.append({"zone": zone, "year": int(yr), "hot_days": int(cnt)})

    if threshold_output_path is not None:
        threshold_columns = [
            "zone",
            "day_of_year",
            "month_day",
            f"threshold_{percentile}pct",
        ]
        daily_thresholds = (
            pd.concat(daily_threshold_rows, ignore_index=True)[threshold_columns]
            if daily_threshold_rows
            else pd.DataFrame(columns=threshold_columns)
        )
        daily_thresholds.to_csv(threshold_output_path, index=False)

    thresholds = pd.DataFrame.from_records(threshold_rows, columns=["zone", "threshold"])
    out = pd.DataFrame.from_records(records)
    per_year_df = pd.DataFrame.from_records(per_year_rows)
    per_year_df = per_year_df.merge(thresholds, on="zone", how="left")
    per_year_df.rename(columns={"threshold": f"threshold_{percentile}pct"}, inplace=True)
    per_year_df.to_csv(f"{OUTPUT_DIR}thermal_hotdays_by_year_{label}.csv", index=False)


    return out


# ============================================================
# RUN ANALYSIS
# ============================================================
summary_rows = []

print("Running historical climate statistics...")
# hot days & heatwaves using historical baseline thresholds
compute_hotdays_and_heatwaves(
    df_hist,
    df_hist,
    "historical_1990_2019",
    summary_rows=summary_rows,
    percentile=95,
    min_duration=3,
    threshold_output_path=f"{OUTPUT_DIR}daily_heatwave_thresholds_by_zone.csv",
)

print("Running future climate statistics...")
compute_hotdays_and_heatwaves(df_fut, df_hist, "future_2040_2069_ssp585", summary_rows=summary_rows, percentile=95, min_duration=3)

summary_df = pd.DataFrame.from_records(summary_rows)
if not summary_df.empty:
    summary_df.to_csv(f"{OUTPUT_DIR}heatwave_events_by_zone_period.csv", index=False)
    print(f"Saved heatwave summary to {OUTPUT_DIR}heatwave_events_by_zone_period.csv")
else:
    print("No heatwave summary rows were generated")

print("✅ Statistical analysis completed successfully")
