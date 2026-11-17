import os
import re

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = "results/statistics/daily_heatwave_thresholds_by_zone.csv"
OUTPUT_DIR = "results/figures/daily_thresholds"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Daily threshold file not found: {INPUT_FILE}")

thresholds = pd.read_csv(INPUT_FILE)
required_columns = {"zone", "day_of_year"}
missing_columns = required_columns.difference(thresholds.columns)
if missing_columns:
    raise ValueError(f"Daily threshold file is missing columns: {sorted(missing_columns)}")

threshold_columns = [column for column in thresholds.columns if column.startswith("threshold_")]
if len(threshold_columns) != 1:
    raise ValueError("Expected exactly one threshold percentile column")
threshold_column = threshold_columns[0]

thresholds["day_of_year"] = pd.to_numeric(thresholds["day_of_year"], errors="raise")
thresholds[threshold_column] = pd.to_numeric(thresholds[threshold_column], errors="raise")
if thresholds.duplicated(["zone", "day_of_year"]).any():
    raise ValueError("Daily threshold file contains duplicate zone/day entries")
if thresholds.empty:
    raise ValueError("Daily threshold file contains no data")

month_starts = pd.date_range("2000-01-01", periods=12, freq="MS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for zone, zone_thresholds in thresholds.groupby("zone", sort=True):
    zone_thresholds = zone_thresholds.sort_values("day_of_year")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        zone_thresholds["day_of_year"],
        zone_thresholds[threshold_column],
        color="tab:red",
        linewidth=1.5,
    )
    ax.set_title(f"Local Moving Threshold for a Hot Day Definition : {zone}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Calendar day")
    ax.set_ylabel("Threshold (°C)")
    ax.set_xticks(month_starts.dayofyear, month_starts.strftime("%b"))
    ax.set_xlim(1, 366)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    zone_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(zone).strip()).strip("._")
    if not zone_slug:
        raise ValueError(f"Cannot create an output filename for zone: {zone!r}")
    output_file = os.path.join(OUTPUT_DIR, f"daily_threshold_{zone_slug}.png")
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_file}")
