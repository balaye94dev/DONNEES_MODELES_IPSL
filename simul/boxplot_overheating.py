"""Create boxplots of overheating degree distributions per sheet in compare.xlsx.

Each sheet must contain `op_temp_actual` and `op_temp_future`. The script
computes exceedance above a threshold (default 26°C) and draws side-by-side
boxplots per sheet showing the 5th - 95th percentile whiskers for Actual and
Future periods.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def apply_rcparams():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
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
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })


def collect_distributions(xls_path, threshold=26.0):
    try:
        sheets = pd.read_excel(xls_path, sheet_name=None)
    except Exception as e:
        raise SystemExit(f"Error reading {xls_path}: {e}")

    labels = []
    actual_list = []
    future_list = []

    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        if 'op_temp_actual' not in df.columns or 'op_temp_future' not in df.columns:
            continue

        a = pd.to_numeric(df['op_temp_actual'], errors='coerce')
        f = pd.to_numeric(df['op_temp_future'], errors='coerce')

        exceed_a = (a - threshold).clip(lower=0).dropna().values
        exceed_f = (f - threshold).clip(lower=0).dropna().values

        if exceed_a.size == 0 and exceed_f.size == 0:
            continue

        labels.append(str(name))
        actual_list.append(np.asarray(exceed_a))
        future_list.append(np.asarray(exceed_f))

    return labels, actual_list, future_list


def plot_box(labels, actual_list, future_list, out_path=None, show_means=True):
    if out_path is None:
        out_dir = Path('simul') / 'figures'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'overheating_boxplot.png'
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_rcparams()

    # Filter empty
    filtered = [(lab, a, f) for lab, a, f in zip(labels, actual_list, future_list)
                if (a is not None and len(a) > 0) or (f is not None and len(f) > 0)]

    if not filtered:
        print('No valid data to plot.')
        return

    labels_f, actual_f, future_f = zip(*filtered)
    n = len(labels_f)
    x = np.arange(n)
    fig_w = max(8, n * 0.6)
    fig_h = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    width = 0.35
    pos_a = x - width / 2
    pos_f = x + width / 2

    bp_a = ax.boxplot(actual_f, positions=pos_a, widths=width * 0.9, whis=(5, 95), patch_artist=True,
                      showfliers=False)
    bp_f = ax.boxplot(future_f, positions=pos_f, widths=width * 0.9, whis=(5, 95), patch_artist=True,
                      showfliers=False)

    for box in bp_a['boxes']:
        box.set(facecolor='#1f77b4', alpha=0.85, linewidth=0.7)
    for median in bp_a['medians']:
        median.set(color='black', linewidth=1.0)

    for box in bp_f['boxes']:
        box.set(facecolor='#d62728', alpha=0.85, linewidth=0.7)
    for median in bp_f['medians']:
        median.set(color='black', linewidth=1.0)

    for whisk in bp_a['whiskers'] + bp_a.get('caps', []):
        whisk.set(color='black', linewidth=0.7)
    for whisk in bp_f['whiskers'] + bp_f.get('caps', []):
        whisk.set(color='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_f, rotation=45, ha='right')
    ax.set_ylabel('Overheating degree (°C)')
    ax.set_title('Indoor overheating Degrees Distribution')

    import matplotlib.patches as mpatches
    p_act = mpatches.Patch(facecolor='#1f77b4', alpha=0.85, label='Actual')
    p_fut = mpatches.Patch(facecolor='#d62728', alpha=0.85, label='Future')
    ax.legend(handles=[p_act, p_fut], frameon=True)

    if show_means:
        means_a = [np.nan if (a is None or len(a) == 0) else float(np.nanmean(a)) for a in actual_f]
        means_f = [np.nan if (f is None or len(f) == 0) else float(np.nanmean(f)) for f in future_f]
        for xi, m in zip(pos_a, means_a):
            if not np.isnan(m):
                ax.scatter(xi, m, marker='D', s=40, facecolor='white', edgecolor='black', zorder=6)
        for xi, m in zip(pos_f, means_f):
            if not np.isnan(m):
                ax.scatter(xi, m, marker='D', s=40, facecolor='#d62728', edgecolor='black', zorder=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved boxplot to {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=os.path.join('simul', 'compare.xlsx'))
    parser.add_argument('--threshold', '-t', type=float, default=26.0)
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--no-mean', action='store_true', help='Do not overlay mean markers')
    args = parser.parse_args()

    labels, actual_list, future_list = collect_distributions(args.input, threshold=args.threshold)
    plot_box(labels, actual_list, future_list, out_path=args.out, show_means=not args.no_mean)
