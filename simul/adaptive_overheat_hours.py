import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Charger Excel
# ==============================

file = "./simul/compare.xlsx"
sheets = pd.read_excel(file, sheet_name=None)

cities = list(sheets.keys())

# ==================================
# 2. Fonction Surchauffe Adaptative
# ==================================

def compute_dh(T_int, T_ext):
    
    T_rm = T_ext.rolling(window=720, min_periods=1).mean()
    T_comf = 0.31 * T_rm + 17.8
    T_lim = T_comf + 3.5
    
    return (T_int > T_lim).sum()


# ==============================
# 3. Calcul
# ==============================

scenarios = [
    "Actual + NoVent",
    "Actual + Vent",
    "Future + NoVent",
    "Future + Vent",
    
]

data = []

for scenario in scenarios:
    
    values = []
    
    for city, df in sheets.items():
        
        if "Actual" in scenario:
            T_ext = df["outdoor_temp_actual"]
        else:
            T_ext = df["outdoor_temp_future"]
        
        if "Vent" in scenario:
            case = "vent"
        else:
            case = "no_vent"
        
        # pick operative temperature column and use it as-is (no fallback)
        period = 'actual' if 'Actual' in scenario else 'future'
        key = f"operative_temp_{period}_{case}"

        if key in df.columns:
            T_int = df[key]
        else:
            # requested column missing entirely: create NaN series
            T_int = pd.Series(np.nan, index=df.index)
        
        dh = compute_dh(T_int, T_ext)
        values.append(dh)
    
    data.append(values)

data = np.array(data)


# ==============================
# 4. FIGURE BUBBLE
# ==============================

fig, ax = plt.subplots(figsize=(12,6))

for i in range(len(scenarios)):
    for j in range(len(cities)):
        
        # scale bubble size (matplotlib 's' is area in points2)
        val = data[i, j]
        if np.isnan(val) or val == 0:
            size = 20
        else:
            size = max(40, val * 2)
        
        # Couleurs intelligentes
        if "Actual" in scenarios[i]:
            color = "blue"
        else:
            color = "red"
        
        ax.scatter(j, i, s=size, color=color, alpha=0.9)
        
        # annotate only when value is not NaN
        if not np.isnan(data[i, j]):
            ax.text(j, i, int(data[i, j]), ha='center', va='center', fontsize=8)
        
# add padding so bubbles don't touch the axes
# use a small data-unit padding around the integer tick positions
x_pad = 0.6
y_pad = 0.6
ax.set_xlim(-x_pad, len(cities) - 1 + x_pad)
ax.set_ylim(-y_pad, len(scenarios) - 1 + y_pad)

# ==============================
# 5. Mise en forme
# ==============================

ax.set_xticks(range(len(cities)))
ax.set_xticklabels(cities, rotation=45)

ax.set_yticks(range(len(scenarios)))
ax.set_yticklabels(scenarios)

ax.set_title("Overheating Hours", fontsize=18, pad=10, fontweight='bold')
ax.set_xlabel("Cities")

plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("./simul/figures/adaptive_overheat_hours.png", dpi=300)
plt.show()

# ==============================
# Per-city full-year profiles
# ==============================
out_dir = "./simul/figures/T_profiles"
os.makedirs(out_dir, exist_ok=True)

for city, df in sheets.items():
    # x axis: integer index to guarantee full-year (8760) plotting
    x = np.arange(len(df))

    for period_label, period_key in [('Actual', 'actual'), ('Future', 'future')]:
        t_ext_col = f'outdoor_temp_{period_key}'
        if t_ext_col not in df.columns:
            continue

        T_ext = df[t_ext_col]
        T_rm = T_ext.rolling(window=720, min_periods=1).mean()
        T_comf = 0.31 * T_rm + 17.8
        T_lim_Upper = T_comf + 3.5
        T_lim_Lower = T_comf - 3.5

        # operative temps if available
        op_no_key = f'operative_temp_{period_key}_no_vent'
        op_vent_key = f'operative_temp_{period_key}_vent'
        op_no = df[op_no_key] if op_no_key in df.columns else None
        op_vent = df[op_vent_key] if op_vent_key in df.columns else None

        plt.figure(figsize=(14,6))
        plt.plot(x, T_ext, label='Outdoor temp', alpha=0.6, color='gray')
        if op_no is not None:
            plt.plot(x, op_no, label='Operative without ventilation', color='green', alpha=0.7)
        if op_vent is not None:
            plt.plot(x, op_vent, label='Operative with ventilation', color='blue', alpha=0.7)
        plt.plot(x, T_comf, label='T_comf', color='orange', linestyle='--', alpha=0.7, linewidth=2)
        plt.plot(x, T_lim_Upper, label='T_lim_Upper_80%', color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.plot(x, T_lim_Lower, label='T_lim_Lower_80%', color='blue', linestyle='--', alpha=0.7, linewidth=2)
        # highlight comfort zone (between lower and upper adaptive limits)
        plt.fill_between(x, T_lim_Lower, T_lim_Upper, color='lightblue', alpha=0.7, label='Comfort zone')

        plt.title(f"{city}  {period_label} Yearly Temperature Profiles", fontsize=18, fontweight='bold')
        plt.ylabel('Temperature (°C)')
        plt.xlabel('Hours of the year')
        plt.legend(ncol=1, fontsize=8)

        # full span
        plt.xlim(0, len(df))

        # xticks: show month names. If the DataFrame index is not datetime,
        # assume an hourly series starting on a non-leap year Jan 1 and synthesize
        # a DatetimeIndex for labeling purposes.
        def _month_ticks_from_index(idx):
            month_positions = []
            month_labels = []
            for m in range(1, 13):
                matches = np.where(idx.month == m)[0]
                if matches.size > 0:
                    pos = int(matches[0])
                    month_positions.append(pos)
                    month_labels.append(idx[pos].strftime('%b'))
            return month_positions, month_labels

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            positions, labels = _month_ticks_from_index(df.index)
        else:
            # synthesize hourly datetime index starting on a non-leap year
            synth_idx = pd.date_range('2001-01-01', periods=len(df), freq='H')
            positions, labels = _month_ticks_from_index(synth_idx)

        # fallback to hourly numeric ticks if month extraction failed
        if len(positions) == 0:
            positions = list(np.arange(0, len(df), 1000))
            if len(positions) == 0 or positions[-1] != len(df):
                positions.append(len(df) - 1)
            labels = [str(p) for p in positions]

        plt.xticks(positions, labels, rotation=45)

        plt.tight_layout()
        fname = os.path.join(out_dir, f"{city}_{period_key}_full_year_profiles.png")
        plt.savefig(fname, dpi=200)
        plt.close()