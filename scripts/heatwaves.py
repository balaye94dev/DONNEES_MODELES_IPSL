from datetime import timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parameters and periods (match scripts/02_statistical_analysis_climate_periods.py)
HIST_START = "1990-01-01"
HIST_END = "2019-12-31 23:00"

FUT_START = "2040-01-01"
FUT_END = "2069-12-31 23:00"

INPUT_FILE = "results/climate_3h_all_zones.csv"
OUTPUT_DIR = "results/figures/heatwaves"
os.makedirs(OUTPUT_DIR, exist_ok=True)

min_duration = 3  # minimum consecutive hot days to count as heatwave
percentile = 90   # percentile threshold based on historical daily maxima


def find_heatwave_periods(daily_max, threshold, min_duration=3):
	"""Return list of (start_date, duration_days) for runs where daily_max > threshold."""
	# daily_max: pandas Series indexed by date
	is_hot = (daily_max > threshold).astype(bool)
	hw_periods = []
	current_start = None
	current_len = 0
	for dt, val in is_hot.items():
		if val:
			if current_start is None:
				current_start = dt
				current_len = 1
			else:
				current_len += 1
		else:
			if current_start is not None and current_len >= min_duration:
				hw_periods.append((current_start, current_len))
			current_start = None
			current_len = 0
	# tail
	if current_start is not None and current_len >= min_duration:
		hw_periods.append((current_start, current_len))
	return hw_periods


def plot_zone_heatwaves(zone, hist_daily, fut_daily, threshold, min_duration=3):
	# hist_daily and fut_daily are Series indexed by date with daily max tas
	hw_hist = find_heatwave_periods(hist_daily, threshold, min_duration)
	hw_fut = find_heatwave_periods(fut_daily, threshold, min_duration)

	fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
	for ax, hw_periods, title in [(axes[0], hw_hist, f"Historical Period ({HIST_START[:4]}-{HIST_END[:4]})"),
								   (axes[1], hw_fut, f"Future Period ({FUT_START[:4]}-{FUT_END[:4]})")]:
		if hw_periods:
			starts, durations = zip(*hw_periods)
			durations_days = list(durations)
			sizes = [max(40, d * 40) for d in durations_days]
			scatter = ax.scatter(
				starts,
				durations_days,
				c=durations_days,
				cmap="plasma",
				s=sizes,
				edgecolor="black",
				alpha=0.8,
				linewidth=0.7,
			)
			cbar = fig.colorbar(scatter, ax=ax)
			cbar.set_label("Duration (days)")
		else:
			ax.text(0.5, 0.5, "No heatwaves detected", ha="center", va="center", transform=ax.transAxes)

		ax.set_title(title)
		ax.set_xlabel("Year")
		ax.grid(True, linestyle="--", alpha=0.6)

	axes[0].set_ylabel("Duration (days)")
	fig.suptitle(f"Heatwaves in : {zone}", fontsize=16)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	out_png = os.path.join(OUTPUT_DIR, f"heatwaves_{zone}.png")
	fig.savefig(out_png, dpi=200)
	plt.close(fig)
	print(f"Wrote {out_png}")


def main():
	df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
	# convert Kelvin to Celsius if necessary (assume tas in K)
	if df["tas"].mean() > 100:  # heuristic
		df["tas"] = df["tas"] - 273.15

	# split periods
	df_hist = df[(df["period"] == "historical") & (df["time"] >= HIST_START) & (df["time"] <= HIST_END)].copy()
	df_fut = df[(df["period"] == "ssp585") & (df["time"] >= FUT_START) & (df["time"] <= FUT_END)].copy()

	# compute daily maxima per zone
	hist_daily_all = (
		df_hist.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"].max().reset_index()
		.rename(columns={"tas": "tas_daily_max", "time": "date"})
	)

	# thresholds per zone from historical baseline
	thresholds = hist_daily_all.groupby("zone").tas_daily_max.quantile(percentile / 100.0).reset_index()
	thresholds.columns = ["zone", "threshold"]

	# daily maxima for each period per zone
	period_hist_daily = hist_daily_all.copy()
	period_fut_daily = (
		df_fut.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"].max().reset_index()
		.rename(columns={"tas": "tas_daily_max", "time": "date"})
	)

	zones = sorted(df["zone"].unique())
	for zone in zones:
		thr_row = thresholds[thresholds["zone"] == zone]
		if thr_row.empty or np.isnan(thr_row["threshold"].iloc[0]):
			print(f"Skipping zone {zone}: no threshold available")
			continue
		threshold = float(thr_row["threshold"].iloc[0])

		pd_hist_zone = period_hist_daily[period_hist_daily["zone"] == zone].set_index("date").sort_index()
		pd_fut_zone = period_fut_daily[period_fut_daily["zone"] == zone].set_index("date").sort_index()

		# create continuous date ranges
		if not pd_hist_zone.empty:
			idx_hist = pd.date_range(pd_hist_zone.index.min(), pd_hist_zone.index.max(), freq="D")
			hist_series = pd_hist_zone.reindex(idx_hist)["tas_daily_max"]
		else:
			hist_series = pd.Series(dtype=float)

		if not pd_fut_zone.empty:
			idx_fut = pd.date_range(pd_fut_zone.index.min(), pd_fut_zone.index.max(), freq="D")
			fut_series = pd_fut_zone.reindex(idx_fut)["tas_daily_max"]
		else:
			fut_series = pd.Series(dtype=float)

		# replace NaN with very low value so they are not counted as hot
		hist_series = hist_series.fillna(-9999)
		fut_series = fut_series.fillna(-9999)

		plot_zone_heatwaves(zone, hist_series, fut_series, threshold, min_duration=min_duration)


if __name__ == "__main__":
	main()