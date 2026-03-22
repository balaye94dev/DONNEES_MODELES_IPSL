import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Publication style (match existing plotting scripts)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'axes.grid': True,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'legend.frameon': True,
    'legend.framealpha': 0.98,
    'legend.edgecolor': 'black',
})

# File paths
hist_file = "results/statistics/annual_stats_historical_1990_2019.csv"
fut_file = "results/statistics/annual_stats_future_2040_2069_ssp585.csv"

# Load data
df_hist = pd.read_csv(hist_file)
df_fut = pd.read_csv(fut_file)

# Determine numeric columns to compare (intersection, excluding 'zone')
common_cols = [c for c in df_hist.columns if c != 'zone' and c in df_fut.columns]

# Coerce numeric, keep NaNs where missing
for c in common_cols:
    df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
    df_fut[c] = pd.to_numeric(df_fut[c], errors='coerce')

# Create one grouped bar chart per column with zones on the x-axis
colors = {'hist':'#1f77b4', 'fut':'#d62728'}

for col in common_cols:
    # Merge historical and future values by zone to align x-axis
    df_h = df_hist[['zone', col]].rename(columns={col: 'value_hist'})
    df_f = df_fut[['zone', col]].rename(columns={col: 'value_fut'})
    merged = pd.merge(df_h, df_f, on='zone', how='outer')

    # Drop rows where both values are missing
    merged = merged[~(merged['value_hist'].isna() & merged['value_fut'].isna())].reset_index(drop=True)
    if merged.empty:
        print(f"Skipping {col}: no data")
        continue

    # Optionally drop zones where both values are zero (empty)
    merged = merged[~((merged['value_hist'].fillna(0) == 0) & (merged['value_fut'].fillna(0) == 0))].reset_index(drop=True)

    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=100)

        ax.bar(x - width/2, merged['value_hist'], width, label='Historical (1990-2019)',
            color=colors['hist'], alpha=0.85, edgecolor='black', linewidth=0.7)
        ax.bar(x + width/2, merged['value_fut'], width, label='Future (2040-2069 SSP585)',
            color=colors['fut'], alpha=0.85, edgecolor='black', linewidth=0.7)

    # Labels and title
    pretty = col.replace('_', ' ').title()
    ax.set_title(f'{pretty}: Historical vs Future (Annual)', fontsize=13, pad=12)
    ax.set_xlabel('Zone', fontsize=11)
    ax.set_ylabel(pretty, fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(merged['zone'], rotation=45, ha='right')

    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    plt.tight_layout()
    out = f"results/figures/annual_by_zone_{col}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {out}")
    plt.close()

print('All done.')
