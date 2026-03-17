
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# slightly larger fonts for readability
plt.rcParams.update({
	'font.size': 10,
	'axes.titlesize': 12,
	'axes.labelsize': 12,
	'xtick.labelsize': 9,
	'ytick.labelsize': 9,
	'legend.fontsize': 9,
})


ROOT = Path(__file__).resolve().parents[1]
HIST_CSV = ROOT / "results" / "statistics" / "monthly_climatology_historical_1990_2019.csv"
FUT_CSV = ROOT / "results" / "statistics" / "monthly_climatology_future_2040_2069_ssp585.csv"
OUTDIR = ROOT / "results" / "figures" / "monthly_profiles"


def load_monthly(csv_path):
	if not csv_path.exists():
		raise FileNotFoundError(f"Missing file: {csv_path}")
	df = pd.read_csv(csv_path)
	# expect columns: zone, month, tas_mean
	if 'tas_mean' not in df.columns:
		raise ValueError(f"Expected 'tas_mean' column in {csv_path}")
	return df


def plot_zone(zone, df_hist, df_fut, outpath):
	# prepare monthly series 1..12
	months = np.arange(1, 13)
	h = df_hist.set_index('month').reindex(months)['tas_mean']
	f = df_fut.set_index('month').reindex(months)['tas_mean']

	plt.figure(figsize=(8.5, 4.5))
	plt.plot(months, h, marker='o', linestyle='--', color='tab:blue', label='Historical (1990-2019)')
	plt.plot(months, f, marker='s', linestyle='-', color='tab:red', label='Future (2040-2069, SSP5-8.5)')

	plt.xticks(months, ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
	plt.xlim(0.5, 12.5)
	plt.xlabel('Month')
	plt.ylabel('Temperature (°C)')
	plt.title(f'Monthly Mean Temperature Profiles : {zone}')
	plt.grid(alpha=0.25)
	plt.legend()
	outpath.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(outpath, dpi=300)
	plt.close()

    
def plot_all_zones_subplots(hist, fut, outpath, ncols=None):
	months = np.arange(1, 13)
	zones = sorted(set(hist['zone'].unique()).union(set(fut['zone'].unique())))
	# exclude 'Linguere' from the combined subplots
	zones = [z for z in zones if z != 'Linguere']
	n = len(zones)
	# auto-select columns to form near-square grid, capped at 4 columns
	if ncols is None:
		ncols = min(4, int(np.ceil(np.sqrt(n)))) if n > 0 else 1
	else:
		ncols = int(ncols)
	nrows = int(np.ceil(n / ncols)) if ncols > 0 else 1

	# compute common y-limits for consistency
	all_vals = []
	for z in zones:
		all_vals.extend(hist[hist['zone'] == z]['tas_mean'].dropna().tolist())
		all_vals.extend(fut[fut['zone'] == z]['tas_mean'].dropna().tolist())
	if all_vals:
		ymin = min(all_vals)
		ymax = max(all_vals)
		yrange = ymax - ymin
		pad = max(0.2, 0.05 * yrange) if yrange > 0 else 0.5
		ylims = (ymin - pad, ymax + pad)
	else:
		ylims = None

	# create a figure and add exactly one subplot per zone using GridSpec
	fig = plt.figure(figsize=(4.5 * ncols, 3.2 * nrows))
	gs = fig.add_gridspec(nrows, ncols)

	for idx, zone in enumerate(zones):
		r = idx // ncols
		c = idx % ncols
		ax = fig.add_subplot(gs[r, c])
		dh = hist[hist['zone'] == zone].set_index('month').reindex(months)['tas_mean']
		dfu = fut[fut['zone'] == zone].set_index('month').reindex(months)['tas_mean']
		ax.plot(months, dh, linestyle='--', color='tab:blue', alpha=0.6, linewidth=0.9)
		ax.plot(months, dfu, linestyle='-', color='tab:red', alpha=1.0, linewidth=1.2)
		ax.set_title(zone, fontsize=11)
		ax.set_xticks(months)
		ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45, fontsize=9)
		if ylims is not None:
			ax.set_ylim(ylims)
		ax.grid(alpha=0.15)

	# global labels and legend: leave room on the left for a centered y-label
	fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.08, hspace=0.35)
	fig.text(0.5, 0.0, 'Month', ha='center', fontsize=12)
	# y-label placed in the left margin, vertically centered
	fig.text(0.03, 0.5, 'Temperature (°C)', va='center', rotation='vertical', fontsize=12)

	# build common legend above the subplots, centered
	handles = [plt.Line2D([0], [0], color='tab:blue', linestyle='--'), plt.Line2D([0], [0], color='tab:red', linestyle='-')]
	labels = ['Historical (1990-2019)', 'Future (2040-2069, SSP5-8.5)']
	# place legend slightly above subplots with extra vertical space
	fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=12)

	outpath.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(outpath, dpi=300, bbox_inches='tight')
	plt.close()


def main():
	hist = load_monthly(HIST_CSV)
	fut = load_monthly(FUT_CSV)

	zones = sorted(set(hist['zone'].unique()).union(set(fut['zone'].unique())))
	OUTDIR.mkdir(parents=True, exist_ok=True)

	for zone in zones:
		dh = hist[hist['zone'] == zone].sort_values('month')
		df = fut[fut['zone'] == zone].sort_values('month')
		if dh.empty and df.empty:
			continue
		outpath = OUTDIR / f"{zone.replace(' ', '_')}_monthly_hist_vs_fut.png"
		print(f"Plotting {zone} -> {outpath}")
		plot_zone(zone, dh, df, outpath)

	# also create a combined figure with all zones
	combined_out = OUTDIR / "monthly_all_zones_hist_vs_fut.png"
	print(f"Plotting combined all-zones -> {combined_out}")
	plot_all_zones_subplots(hist, fut, combined_out)

	print(f"Done. Figures written to {OUTDIR}")


if __name__ == '__main__':
	main()

