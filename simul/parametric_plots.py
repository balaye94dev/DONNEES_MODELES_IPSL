"""Publication-style parametric plots for All_Parametric_Results.xlsx

Creates line plots (publication style) of Energy (kWh) and Discomfort (hrs = 8760 - Comfort)
for each parameter (Wall Conductivity, Roof Absorptance, Infiltration) in every sheet.

Usage:
	python simul/parametric_plots.py simul/All_Parametric_Results.xlsx --out results/figures/parametric

Saves PNG and PDF for each parameter per sheet.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_column(cols, keywords):
	low = [str(c).lower() for c in cols]
	for kw in keywords:
		for i, c in enumerate(low):
			if kw in c:
				return cols[i]
	return None


def _safe_name(s: str) -> str:
	return (
		str(s).replace("/", "_").replace("\\", "_")
		.replace(" ", "_").replace("(", "").replace(")", "")
	)


def publication_style():
	sns.set_style("ticks")
	sns.set_context("paper", font_scale=1.2)
	mpl.rcParams.update({
		"font.family": "serif",
		"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
		"axes.linewidth": 0.8,
		"axes.labelsize": 10,
		"xtick.labelsize": 9,
		"ytick.labelsize": 9,
		"legend.fontsize": 9,
	})


def plot_line(ax, x, y, label=None, color=None, marker="o"):
	ax.plot(x, y, marker=marker, markersize=5, linewidth=1.6, label=label, color=color)



def process_workbook(xlsx_path: str, out_dir: str):
	if not os.path.exists(xlsx_path):
		raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
	os.makedirs(out_dir, exist_ok=True)

	xls = pd.ExcelFile(xlsx_path)
	sheet_names = list(xls.sheet_names)

	# parameter keywords to search for
	param_keywords = {
		"Wall Conductivity": ["wall", "conductivity", "w/m-k"],
		"Roof Absorptance": ["roof", "absorptance", "absoptance"],
		"Infiltration": ["infiltr", "infiltration", "ach"]
	}

	colors = sns.color_palette("tab10", n_colors=max(3, len(sheet_names)))

	# For each parameter, create one figure containing all sheets' curves
	for pname, kws in param_keywords.items():
		# energy figure
		fig_e, ax_e = plt.subplots(figsize=(7, 4))
		# discomfort figure
		fig_d, ax_d = plt.subplots(figsize=(7, 4))

		xlabel_label = None

		for i, sheet in enumerate(sheet_names):
			try:
				df = xls.parse(sheet)
			except Exception as e:
				print(f"Skipping sheet {sheet}: failed to read ({e})")
				continue

			if df.empty:
				print(f"Skipping empty sheet: {sheet}")
				continue

			cols = list(df.columns)
			pcol = find_column(cols, kws)
			if pcol is None:
				print(f"Parameter '{pname}' not found in sheet {sheet}; skipping")
				continue

			if xlabel_label is None:
				xlabel_label = pcol

			energy_col = find_column(cols, ["energy", "kwh", "consumption"])
			comfort_col = find_column(cols, ["comfort", "comfort (hrs)", "comfort_hrs"])

			# ensure discomfort column exists
			if "discomfort_hrs" not in df.columns and comfort_col and comfort_col in df.columns:
				ser = df[comfort_col]
				if ser.max() <= 1.0:
					comfort_hours = ser * 8760
				else:
					comfort_hours = ser
				df = df.copy()
				df["discomfort_hrs"] = 8760 - comfort_hours

			# group
			try:
				g = df.groupby(pcol)
			except Exception:
				print(f"Unable to group by {pcol} in sheet {sheet}; skipping")
				continue

			color = colors[i % len(colors)]

			# energy
			if energy_col and energy_col in df.columns:
				s_e = g[energy_col].mean().sort_index()
				if not s_e.empty:
					ax_e.plot(s_e.index.values, s_e.values, marker="o", label=sheet, color=color, linewidth=1.6)
			else:
				print(f"Energy column not found in sheet {sheet}; energy skipped for {sheet}")

			# discomfort
			if "discomfort_hrs" in df.columns:
				s_d = g["discomfort_hrs"].mean().sort_index()
				if not s_d.empty:
					ax_d.plot(s_d.index.values, s_d.values, marker="s", label=sheet, color=color, linewidth=1.6)
			else:
				print(f"Discomfort not available in sheet {sheet}; discomfort skipped for {sheet}")

		# finalize energy figure
		if xlabel_label is None:
			xlabel_label = pname
		ax_e.set_xlabel(xlabel_label)
		ax_e.set_ylabel("Energy (kWh)")
		ax_e.set_title(f"{pname} and Energy")
		ax_e.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
		ax_e.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
		fig_e.tight_layout()
		fig_e.subplots_adjust(right=0.75)
		out_e = os.path.join(out_dir, f"{_safe_name(pname)}_energy_all_sheets.png")
		fig_e.savefig(out_e, dpi=300, bbox_inches="tight")
		plt.close(fig_e)
		print(f"Saved: {out_e}")

		# finalize discomfort figure
		ax_d.set_xlabel(xlabel_label)
		ax_d.set_ylabel("Discomfort (hrs)")
		ax_d.set_title(f"{pname} and Discomfort")
		ax_d.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
		ax_d.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
		fig_d.tight_layout()
		fig_d.subplots_adjust(right=0.75)
		out_d = os.path.join(out_dir, f"{_safe_name(pname)}_discomfort_all_sheets.png")
		fig_d.savefig(out_d, dpi=300, bbox_inches="tight")
		plt.close(fig_d)
		print(f"Saved: {out_d}")


def main():
	p = argparse.ArgumentParser(description="Create publication-style parametric plots from workbook")
	p.add_argument("excel", help="Path to All_Parametric_Results.xlsx")
	p.add_argument("--out", help="Output directory for figures", default=os.path.join("simul", "figures", "parametric"))
	args = p.parse_args()

	publication_style()
	process_workbook(args.excel, args.out)


if __name__ == "__main__":
	main()

