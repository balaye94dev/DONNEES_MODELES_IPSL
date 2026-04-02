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
    "Actual + Vent",
    "Actual + NoVent",
    "Future + Vent",
    "Future + NoVent"
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
        
        # pick operative temperature column, with fallback if one case is all-NaN
        period = 'actual' if 'Actual' in scenario else 'future'
        key = f"operative_temp_{period}_{case}"

        if key in df.columns and not df[key].isna().all():
            T_int = df[key]
        else:
            # fallback to the other ventilation case if present
            alt_case = 'no_vent' if case == 'vent' else 'vent'
            alt_key = f"operative_temp_{period}_{alt_case}"
            if alt_key in df.columns and not df[alt_key].isna().all():
                T_int = df[alt_key]
            else:
                # column missing or entirely NaN: create NaN series so compute_dh returns 0
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
        
        # scale bubble size (matplotlib 's' is area in points^2)
        val = data[i, j]
        if np.isnan(val) or val == 0:
            size = 20
        else:
            size = max(30, val * 3)
        
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
