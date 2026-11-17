import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "results/statistics"
HIST_FILE = os.path.join(INPUT_DIR, "thermal_hotdays_by_year_historical_1990_2019.csv")
FUT_FILE = os.path.join(INPUT_DIR, "thermal_hotdays_by_year_future_2040_2069_ssp585.csv")
OUT_DIR = "results/figures/hotdays"
DPI = 200
os.makedirs(OUT_DIR, exist_ok=True)

# load
if not os.path.exists(HIST_FILE):
    raise FileNotFoundError(f"Historical per-year file not found: {HIST_FILE}")
if not os.path.exists(FUT_FILE):
    raise FileNotFoundError(f"Future per-year file not found: {FUT_FILE}")

hist = pd.read_csv(HIST_FILE)
fut = pd.read_csv(FUT_FILE)

# ensure types
hist['year'] = hist['year'].astype(int)
hist['hot_days'] = pd.to_numeric(hist['hot_days'], errors='coerce').fillna(0).astype(int)

fut['year'] = fut['year'].astype(int)
fut['hot_days'] = pd.to_numeric(fut['hot_days'], errors='coerce').fillna(0).astype(int)

zones = sorted(set(hist['zone'].unique()).union(fut['zone'].unique()))

for zone in zones:
    h_zone = hist[hist['zone'] == zone].sort_values('year')
    f_zone = fut[fut['zone'] == zone].sort_values('year')

    plt.figure(figsize=(8, 4))
    for data, color, name in [(h_zone, 'tab:blue', 'historical'),
                              (f_zone, 'tab:red', 'future')]:
        if data.empty:
            continue
        plt.plot(data['year'], data['hot_days'], '-o', color=color, label=name)

        # period mean: dashed line over the years of the period
        mean_val = data['hot_days'].mean()
        plt.hlines(mean_val, data['year'].min(), data['year'].max(),
                   colors=color, linestyles='--', linewidth=2,
                   label=f'{name} mean: {mean_val:.1f} d/yr')

    plt.title(f'Annual hot days per year : {zone}')
    plt.xlabel('Year')
    plt.ylabel('Hot days (per year)')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f'hotdays_evolution_{zone}.png')
    plt.savefig(out_png, dpi=DPI)
    plt.close()
    print(f'Wrote {out_png}')

# also produce a combined CSV with both periods labelled
hist2 = hist.copy(); hist2['period'] = 'historical'
fut2 = fut.copy(); fut2['period'] = 'future'
combined = pd.concat([hist2, fut2], ignore_index=True)
combined.to_csv(os.path.join(OUT_DIR, 'thermal_hotdays_by_year_combined.csv'), index=False)
print('Wrote combined per-year CSV')