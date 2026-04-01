"""Plot actual vs future overheat hours per city from CSV.

Usage:
    python simul/plot_overheat_scatter.py --input simul/overheat_hours_by_sheet.csv --out results
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(input_csv: Path, out_dir: Path):
    df = pd.read_csv(input_csv)
    if 'overheat_hours_actual' not in df.columns or 'overheat_hours_future' not in df.columns:
        raise SystemExit('CSV must contain overheat_hours_actual and overheat_hours_future columns')

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
    })
    sns.set_style('white')

    cities = df['sheet'] if 'sheet' in df.columns else df.index.astype(str)
    actual = df['overheat_hours_actual']
    future = df['overheat_hours_future']

    # marker sizes: scale hours to reasonable marker area
    min_size = 40
    max_size = 800
    all_hours = pd.concat([actual, future])
    if all_hours.max() > 0:
        sizes = min_size + (all_hours - all_hours.min()) / (all_hours.max() - all_hours.min()) * (max_size - min_size)
    else:
        sizes = pd.Series(min_size, index=all_hours.index)

    # sizes for actual and future correspondingly
    sizes_actual = sizes.iloc[:len(actual)].values
    sizes_future = sizes.iloc[len(actual):].values

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(cities))
    ax.scatter(x_pos, actual, s=sizes_actual, color='tab:blue', alpha=0.8, label='Actual')
    ax.scatter(x_pos, future, s=sizes_future, color='tab:orange', alpha=0.8, label='Future')

    for i, city in enumerate(cities):
        ax.text(i, max(actual.iloc[i], future.iloc[i]) + 5, city, ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cities, rotation=45, ha='right')
    ax.set_ylabel('Total overheat hours')
    ax.set_title('Actual and Future Overheat Hours per City (marker size ∝ hours)')
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / 'overheat_hours_scatter.png'
    fig.savefig(out_path, bbox_inches='tight')
    print('Saved', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='simul/overheat_hours_by_sheet.csv')
    p.add_argument('--out', '-o', default='results')
    args = p.parse_args()
    main(Path(args.input), Path(args.out))
