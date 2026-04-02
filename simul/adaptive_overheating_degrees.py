import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Charger Excel
# ==============================

file = "./simul/compare.xlsx"
sheets = pd.read_excel(file, sheet_name=None)

# ==============================
# 2. Fonction degrés-heures
# ==============================

def compute_degree_hours(T_int, T_ext):
    
    T_rm = T_ext.rolling(window=720, min_periods=1).mean()
    T_comf = 0.31 * T_rm + 17.8
    T_lim = T_comf + 3.5
    
    overheating = (T_int - T_lim).clip(lower=0)
    
    return overheating

# ==============================
# 3. Collecte données
# ==============================

scenarios = [
    "Actual + Vent",
    "Actual + NoVent",
    "Future + Vent",
    "Future + NoVent"
]

# helper to choose operative temperature with fallback when a column is missing or all NaN
def get_operative_series(df, period, case):
    key = f"operative_temp_{period}_{case}"
    if key in df.columns and not df[key].isna().all():
        return df[key]
    # try alternative case
    alt = 'no_vent' if case == 'vent' else 'vent'
    alt_key = f"operative_temp_{period}_{alt}"
    if alt_key in df.columns and not df[alt_key].isna().all():
        return df[alt_key]
    # fallback to NaN series
    return pd.Series(np.nan, index=df.index)

# compute total degree-hours per city and scenario
rows = []
for city, df in sheets.items():
    for scen in scenarios:
        period = 'actual' if 'Actual' in scen else 'future'
        case = 'vent' if 'Vent' in scen else 'no_vent'
        T_ext = df[f'outdoor_temp_{period}'] if f'outdoor_temp_{period}' in df.columns else pd.Series(np.nan, index=df.index)
        T_int = get_operative_series(df, period, case)
        overheating = compute_degree_hours(T_int, T_ext)
        total_dh = overheating.sum() if overheating.notna().any() else np.nan
        rows.append({'city': city, 'scenario': scen, 'total_dh': float(total_dh)})

df_summary = pd.DataFrame(rows)

# pivot to wide format: rows=cities, cols=scenarios
df_wide = df_summary.pivot(index='city', columns='scenario', values='total_dh')
# ==============================================================
# 4. FIGURE: boxplot per city (cities on x-axis)
# ==============================================================

# build long-form per-hour overheating values for each city/scenario
rows_long = []
for city, df in sheets.items():
    for scen in scenarios:
        period = 'actual' if 'Actual' in scen else 'future'
        case = 'vent' if 'Vent' in scen else 'no_vent'
        T_ext = df[f'outdoor_temp_{period}'] if f'outdoor_temp_{period}' in df.columns else pd.Series(np.nan, index=df.index)
        T_int = get_operative_series(df, period, case)
        overheating = compute_degree_hours(T_int, T_ext)
        for v in overheating.values:
            rows_long.append({'city': city, 'scenario': scen, 'overheating': float(v) if not np.isnan(v) else np.nan})

df_long = pd.DataFrame(rows_long)
df_long = df_long.dropna(subset=['overheating'])

# plotting
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

# preferred: seaborn boxplot (nicer grouping). Fallback to matplotlib if seaborn not available.
try:
    import seaborn as sns
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    palette = {'Actual + Vent': '#0072B2', 'Actual + NoVent': '#D55E00', 'Future + Vent': '#56B4E9', 'Future + NoVent': '#E69F00'}
    # use whis to set whiskers at 2.5th and 97.5th percentiles
    # hide fliers (outliers) beyond whiskers
    sns.boxplot(x='city', y='overheating', hue='scenario', data=df_long,
                order=df_wide.index.tolist(), hue_order=scenarios, palette=palette,
                showmeans=True, whis=[2.5, 97.5], showfliers=False, ax=ax)
    ax.set_ylabel('Degrés-heures de surchauffe (°C·h)')
    ax.set_title('Distribution horaire de la surchauffe par ville et scénario')
    ax.legend(title='Scénarios', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig('./simul/figures/boxplot_degree_hours_by_city.png', dpi=300)
    plt.show()
except Exception:
    # matplotlib fallback: grouped boxplots using positions
    fig, ax = plt.subplots(figsize=(14, 7))
    cities = df_wide.index.tolist()
    n_scen = len(scenarios)
    width = 0.7 / n_scen
    positions = []
    data = []
    pos = []
    for i, city in enumerate(cities):
        for k, scen in enumerate(scenarios):
            vals = df_long[(df_long['city'] == city) & (df_long['scenario'] == scen)]['overheating'].values
            data.append(vals)
            positions.append(i - 0.35 + k * width + width / 2)
    # set whiskers at 2.5th and 97.5th percentiles
    bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, manage_ticks=False, whis=[2.5, 97.5], showfliers=False)
    # color boxes
    colors = ['#0072B2', '#D55E00', '#56B4E9', '#E69F00']
    for patch, color in zip(bp['boxes'], colors * len(cities)):
        patch.set_facecolor(color)
        patch.set_edgecolor('k')
    # x-ticks at city centers
    centers = [i for i in range(len(cities))]
    ax.set_xticks(centers)
    ax.set_xticklabels(cities, rotation=45, ha='right')
    # create custom legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[i], edgecolor='k', label=scenarios[i]) for i in range(n_scen)]
    ax.legend(handles=legend_handles, title='Scénarios', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylabel('Degrés-heures de surchauffe (°C·h)')
    ax.set_title('Distribution horaire de la surchauffe par ville et scénario')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig('./simul/figures/boxplot_degree_hours_by_city.png', dpi=300)
    plt.show()