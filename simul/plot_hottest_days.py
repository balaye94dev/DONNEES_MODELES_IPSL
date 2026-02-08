import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_hottest_days(xls_path="simul/compare.xlsx", out_dir="simul/figures"):
    try:
        sheets = pd.read_excel(xls_path, sheet_name=None)
    except Exception as e:
        raise SystemExit(f"Erreur lecture fichier {xls_path}: {e}")

    if not sheets:
        raise SystemExit(f"Aucune feuille trouvée dans {xls_path}")

    sheet_names = list(sheets.keys())
    n = len(sheet_names)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    os.makedirs(out_dir, exist_ok=True)

    for idx, sheet_name in enumerate(sheet_names):
        ax = axes_flat[idx]
        df = sheets[sheet_name].copy()

        # create datetime index assuming hourly data starting from 2025-01-01
        df["datetime"] = pd.date_range(start="2025-01-01 00:00:00", periods=len(df), freq="h")

        # Required columns
        missing_col = False
        for col in ["op_temp_actual", "op_temp_future"]:
            if col not in df.columns:
                ax.set_visible(False)
                print(f"Feuille '{sheet_name}': colonne manquante '{col}', on saute.")
                missing_col = True
                break
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if missing_col:
            continue

        # drop rows without both temps
        df = df.dropna(subset=["op_temp_actual", "op_temp_future"]) 
        if df.empty:
            ax.set_visible(False)
            print(f"Feuille '{sheet_name}': données vides après suppression des NaN, on saute.")
            continue

        # determine hottest day as the day when the absolute maximum of op_temp_actual occurs
        df["date"] = df["datetime"].dt.date
        if df["op_temp_actual"].dropna().empty:
            ax.set_visible(False)
            print(f"Feuille '{sheet_name}': aucune valeur valide pour 'op_temp_actual', on saute.")
            continue

        max_idx = df["op_temp_actual"].idxmax()
        try:
            hottest_dt = pd.to_datetime(df.loc[max_idx, "datetime"]) if max_idx in df.index else None
            hottest_day = hottest_dt.date() if hottest_dt is not None else None
        except Exception:
            hottest_day = None

        if hottest_day is None:
            ax.set_visible(False)
            print(f"Feuille '{sheet_name}': impossible de déterminer le jour le plus chaud, on saute.")
            continue

        day_mask = df["datetime"].dt.date == hottest_day
        day_data = df.loc[day_mask].copy()
        if day_data.empty:
            ax.set_visible(False)
            print(f"Feuille '{sheet_name}': aucune donnée pour le jour {hottest_day}, on saute.")
            continue

        # plot raw hourly values (no smoothing)
        ax.plot(day_data["datetime"], day_data["op_temp_actual"], label="Actuel", color="C0", linewidth=1.5, marker='o', markersize=3)
        ax.plot(day_data["datetime"], day_data["op_temp_future"], label="Futur", color="C1", linewidth=1.5, marker='o', markersize=3)

        # set x-axis to show hours of the day
        try:
            # derive midnight from the actual day's datetimes to preserve tz/dtype
            day_start = day_data["datetime"].dt.normalize().iloc[0]
            day_end = day_start + pd.Timedelta(days=1)
            ax.set_xlim(day_start, day_end)
        except Exception:
            pass

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # format hottest day as 'MonthName day' (e.g. July 17)
        try:
            _dt = pd.to_datetime(hottest_day)
            display_day = f"{_dt.strftime('%B')} {_dt.day}"
        except Exception:
            display_day = str(hottest_day)
        ax.set_title(f"{sheet_name}  {display_day}", fontsize=12)
        ax.set_ylabel("Operative Temperature (°C)")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    # hide any unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "hottest_days_compare.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Figure saved to {out_path}")


if __name__ == '__main__':
    plot_hottest_days()
