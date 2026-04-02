import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def read_and_sum_energy(xls_path="simul/compare.xlsx"):
    try:
        sheets = pd.read_excel(xls_path, sheet_name=None)
    except Exception as e:
        raise SystemExit(f"Error reading {xls_path}: {e}")

    labels = []
    actual_kwh = []
    future_kwh = []

    for name, df in sheets.items():
        # ensure DataFrame
        if df is None or df.empty:
            print(f"Sheet '{name}' is empty, skipping.")
            continue

        # required columns
        if 'energy_use_actual' not in df.columns and 'energy_use_future' not in df.columns:
            print(f"Sheet '{name}': missing both energy_use_actual and energy_use_future, skipping.")
            continue

        # coerce to numeric
        a = pd.to_numeric(df.get('energy_use_actual'), errors='coerce')
        f = pd.to_numeric(df.get('energy_use_future'), errors='coerce')

        total_a_J = a.sum(skipna=True) if a is not None else 0.0
        total_f_J = f.sum(skipna=True) if f is not None else 0.0

        # convert Joules to kWh: 1 kWh = 3.6e6 J
        total_a_kwh = total_a_J / 3.6e6
        total_f_kwh = total_f_J / 3.6e6

        labels.append(str(name))
        actual_kwh.append(total_a_kwh)
        future_kwh.append(total_f_kwh)

    return labels, np.array(actual_kwh), np.array(future_kwh)


def plot_energy_histogram(labels, actual_kwh, future_kwh, out_path=None):
    if out_path is None:
        out_dir = os.path.join('simul', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'energy_compare.png')
    else:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    bars_a = ax.bar(x - width/2, actual_kwh, width, label='Actual', color='C0', alpha=0.9)
    bars_f = ax.bar(x + width/2, future_kwh, width, label='Future', color='#d62728', alpha=0.9)

    # Display axis and labels in thousands (e.g., 140 represents 140,000 kWh)
    ax.set_ylabel('Total energy (kWh/m²/an)', font = 'Century Gothic', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1000:,.0f}"))
    ax.set_title('Total Annual energy consumption', font = 'Century Gothic', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', font = 'Century Gothic', fontsize=12)
    ax.legend()

    # annotate bars with kWh values on top
    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            label = f"{h/1000:,.0f}"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    _label_bars(bars_a)
    _label_bars(bars_f)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved histogram to {out_path}")


if __name__ == '__main__':
    xls = os.path.join('simul', 'compare.xlsx')
    labels, a_kwh, f_kwh = read_and_sum_energy(xls)
    if not labels:
        print('No data to plot.')
    else:
        plot_energy_histogram(labels, a_kwh, f_kwh)
