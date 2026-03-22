import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


THRESHOLD_C = 26.0


def read_and_compute_overheating(xls_path="simul/compare.xlsx", threshold=THRESHOLD_C):
    try:
        sheets = pd.read_excel(xls_path, sheet_name=None)
    except Exception as e:
        raise SystemExit(f"Error reading {xls_path}: {e}")

    labels = []
    actual_oh = []
    future_oh = []

    for name, df in sheets.items():
        if df is None or df.empty:
            print(f"Sheet '{name}' is empty, skipping.")
            continue

        # expected columns similar to other scripts
        a_col = 'op_temp_actual'
        f_col = 'op_temp_future'
        if a_col not in df.columns or f_col not in df.columns:
            print(f"Sheet '{name}': missing required columns '{a_col}' or '{f_col}', skipping.")
            continue

        # coerce to numeric
        a = pd.to_numeric(df[a_col], errors='coerce')
        f = pd.to_numeric(df[f_col], errors='coerce')

        # compute mean exceedance (°C) above threshold
        exceed_a = (a - threshold).clip(lower=0)
        exceed_f = (f - threshold).clip(lower=0)

        mean_a = exceed_a.mean(skipna=True)
        mean_f = exceed_f.mean(skipna=True)

        labels.append(str(name))
        actual_oh.append(float(mean_a) if not np.isnan(mean_a) else 0.0)
        future_oh.append(float(mean_f) if not np.isnan(mean_f) else 0.0)

    return labels, np.array(actual_oh), np.array(future_oh)


def plot_overheating(labels, actual_oh, future_oh, out_path=None):
    if out_path is None:
        out_dir = os.path.join('simul', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'overheating_compare.png')
    else:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    x = np.arange(len(labels))
    width = 0.35

    # make figure wider for many labels and a bit taller so annotations are visible
    fig_w = max(6, len(labels) * 0.8)
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars_a = ax.bar(x - width/2, actual_oh, width, label='Actual', color='C0', alpha=0.75)
    bars_f = ax.bar(x + width/2, future_oh, width, label='Future', color="#eb0707", alpha=0.75)

    ax.set_ylabel('Overheating degree (°C)', font = 'Century', fontsize=11)
    ax.set_title('Indoor overheating Degrees — Actual vs Future (threshold 26°C)', font = 'Century')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', font = 'Century')
    ax.legend()


    # add some headroom so labels above bars aren't cut off
    max_h = 0
    if len(actual_oh) > 0:
        try:
            max_h = max(max_h, float(np.nanmax(actual_oh)))
        except Exception:
            pass
    if len(future_oh) > 0:
        try:
            max_h = max(max_h, float(np.nanmax(future_oh)))
        except Exception:
            pass
    ax.set_ylim(0, max_h * 1.2 if max_h > 0 else 1)

    # annotate bars: prefer bar_label if available, otherwise fallback to annotate
    try:
        ax.bar_label(bars_a, labels=[f"{v:.2f} °C" for v in actual_oh], padding=4, fontsize=5)
        ax.bar_label(bars_f, labels=[f"{v:.2f} °C" for v in future_oh], padding=4, fontsize=5)
    except Exception:
        for bar in list(bars_a) + list(bars_f):
            h = bar.get_height()
            ax.annotate(f"{h:.2f} °C", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved overheating comparison to {out_path}")


if __name__ == '__main__':
    xls = os.path.join('simul', 'compare.xlsx')
    labels, a_oh, f_oh = read_and_compute_overheating(xls)
    if not labels:
        print('No valid sheets found to compute overheating.')
    else:
        plot_overheating(labels, a_oh, f_oh)
