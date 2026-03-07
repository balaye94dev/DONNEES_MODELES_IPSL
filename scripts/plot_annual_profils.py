
"""
Plot annual temperature profiles per zone for a future period and highlight a reference year.

This script aggregates monthly means per year from a (possibly large) climate
CSV (`results/climate_3h_all_zones.csv`) and produces one PNG per zone in
`results/figures/annual_profiles/`.

Usage:
    python scripts/plot_annual_profils.py

Options can be adjusted at top of file or via CLI arguments.
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = ROOT / "results" / "climate_3h_all_zones.csv"
OUTDIR = ROOT / "results" / "figures" / "annual_profiles"


def aggregate_monthly_by_year(input_csv, var='tas', start_year=2040, end_year=2069, chunksize=200000, kelvin=False):
    # kept for backward compatibility; simple wrapper that reuses hourly aggregator
    df = aggregate_hourly_by_year(input_csv, var=var, start_year=start_year, end_year=end_year, chunksize=chunksize, kelvin=kelvin)
    # convert hour_of_year back to month by mapping hours to month using a representative non-leap year
    if df.empty:
        return pd.DataFrame(columns=['zone', 'year', 'month', 'mean'])

    # create mapping hour_of_year -> month
    ref = pd.date_range('2001-01-01', periods=8760, freq='H')
    h2m = pd.Series(ref.month, index=np.arange(1, len(ref) + 1))
    df['month'] = df['hour_of_year'].map(h2m)
    # aggregate means per month
    out = df.groupby(['zone', 'year', 'month'])['mean'].mean().reset_index()
    return out


def aggregate_hourly_by_year(input_csv, var='tas', start_year=2040, end_year=2069, chunksize=200000, kelvin=False):
    """
    Aggregate hourly values into mean per (zone, year, hour_of_year).
    Returns DataFrame with columns ['zone','year','hour_of_year','mean']
    """
    sums = {}
    reader = pd.read_csv(input_csv, chunksize=chunksize)
    dt_col = None
    var_col = var

    for chunk in reader:
        if dt_col is None:
            candidates = [c for c in chunk.columns if c.lower() in ('time', 'date', 'datetime', 'timestamp')]
            if not candidates:
                candidates = [c for c in chunk.columns if 'time' in c.lower() or 'date' in c.lower()]
            dt_col = candidates[0] if candidates else None
        if dt_col is None:
            raise ValueError('Could not detect a date/time column in the input CSV')

        chunk[dt_col] = pd.to_datetime(chunk[dt_col], errors='coerce')
        chunk = chunk.dropna(subset=[dt_col])
        chunk['year'] = chunk[dt_col].dt.year
        # compute hour of year 1..8760 (non-leap mapping)
        # anchor to non-leap year 2001 to get consistent mapping
        doy = (chunk[dt_col] - pd.to_datetime(chunk[dt_col].dt.year.astype(str) + '-01-01'))
        hour_of_year = (doy.dt.total_seconds() // 3600).astype(int) + 1
        chunk['hour_of_year'] = hour_of_year

        if var_col not in chunk.columns:
            alt = next((c for c in chunk.columns if c.lower().startswith('tas') or 'air' in c.lower()), None)
            if alt:
                var_col = alt
            else:
                raise ValueError(f"Variable column '{var}' not found in CSV and no alternative detected.")

        chunk = chunk[(chunk['year'] >= start_year) & (chunk['year'] <= end_year)]
        if chunk.empty:
            continue

        grp = chunk.groupby(['zone', 'year', 'hour_of_year'])[var_col].agg(['sum', 'count']).reset_index()
        for _, row in grp.iterrows():
            key = (row['zone'], int(row['year']), int(row['hour_of_year']))
            s = float(row['sum'])
            c = int(row['count'])
            if c == 0:
                # no valid samples for this group
                continue
            if key in sums:
                sums[key] = (sums[key][0] + s, sums[key][1] + c)
            else:
                sums[key] = (s, c)

    records = []
    for (zone, year, hour), (s, c) in sums.items():
        m = s / c
        if kelvin:
            m = m - 273.15
        records.append({'zone': zone, 'year': year, 'hour_of_year': hour, 'mean': m})

    if not records:
        return pd.DataFrame(columns=['zone', 'year', 'hour_of_year', 'mean'])

    df = pd.DataFrame.from_records(records)
    df.to_csv('./results/epw_future_2040_2069/hourly_profils.csv', index=False)  # for debugging
    return df


def plot_zone(df_zone, zone, ref_year, outpath):
    # Detect whether hourly or monthly aggregation
    if 'hour_of_year' in df_zone.columns:
        # pivot by hour_of_year
        p = df_zone.pivot(index='hour_of_year', columns='year', values='mean')
        x = p.index.values
        xticks = None
        xlabel = 'Hour of year'
    else:
        p = df_zone.pivot(index='month', columns='year', values='mean')
        x = np.arange(1, 13)
        xticks = x
        xlabel = 'Month'

    plt.figure(figsize=(10, 5))
    years = sorted([c for c in p.columns if not pd.isna(c)])
    # plot each year faintly
    for y in years:
        vals = p[y].reindex(x).values
        plt.plot(x, vals, color='gray', alpha=0.2, linewidth=0.6)

    # plot period mean
    period_mean = p.mean(axis=1).reindex(x)
    plt.plot(x, period_mean.values, color='black', linewidth=1.8, label='Future Period Mean Profile')

    # highlight reference year if present
    if ref_year in p.columns:
        ry = p[ref_year].reindex(x).values
        plt.plot(x, ry, color='tab:red', linewidth=2.2, label=f'Reference Year {ref_year}')

    if xticks is not None:
        plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel('Temperature (°C)')
    plt.title(f'Annual temperature profiles — {zone}')
    plt.grid(alpha=0.2)
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=str(INPUT_DEFAULT))
    parser.add_argument('--var', default='tas', help='Variable column to use (default: tas)')
    parser.add_argument('--start', type=int, default=2040)
    parser.add_argument('--end', type=int, default=2069)
    parser.add_argument('--ref', type=int, default=2045, help='Reference year to highlight')
    parser.add_argument('--outdir', default=str(OUTDIR))
    parser.add_argument('--no-hourly', action='store_true', help='Use monthly aggregation instead of hourly')
    parser.add_argument('--kelvin', action='store_true', help='Convert Kelvin to Celsius when True')
    parser.add_argument('--auto-ref', action='store_true', help='Auto-select reference year closest to period mean')
    args = parser.parse_args()

    hourly = not args.no_hourly

    inp = Path(args.input)
    outdir = Path(args.outdir)
    if not inp.exists():
        print(f"Input file not found: {inp}")
        return

    # Auto-detect units (Kelvin vs Celsius) by sampling the variable column,
    # unless user explicitly set --kelvin. If median sample value > 100, assume Kelvin.
    sample_kelvin = False
    if args.kelvin:
        sample_kelvin = True
    else:
        try:
            sample = pd.read_csv(inp, nrows=2000)
            var_col = args.var
            if var_col not in sample.columns:
                alt = next((c for c in sample.columns if c.lower().startswith('tas') or 'air' in c.lower()), None)
                if alt:
                    var_col = alt
            if var_col in sample.columns:
                med = pd.to_numeric(sample[var_col], errors='coerce').median()
                if pd.notna(med) and med > 100:
                    sample_kelvin = True
                    print(f"Detected large sample values for '{var_col}' (median={med:.1f}) — assuming Kelvin and converting to °C.")
        except Exception:
            # if sampling fails, fall back to user flag only
            sample_kelvin = args.kelvin

    if hourly:
        print("Aggregating hourly means by year (hour-of-year). This may take a while...")
        agg = aggregate_hourly_by_year(inp, var=args.var, start_year=args.start, end_year=args.end, kelvin=sample_kelvin)
    else:
        print("Aggregating monthly means by year (this may take a while for large files)...")
        agg = aggregate_monthly_by_year(inp, var=args.var, start_year=args.start, end_year=args.end, kelvin=sample_kelvin)
    if agg.empty:
        print("No data found for the specified period and variable.")
        return

    zones = agg['zone'].unique()
    outdir.mkdir(parents=True, exist_ok=True)
    for zone in zones:
        df_zone = agg[agg['zone'] == zone]
        outpath = outdir / f"{zone.replace(' ', '_')}_annual_profiles_{args.start}-{args.end}.png"

        # determine reference year (auto or provided)
        if args.auto_ref:
            # build pivot to compare years to period mean
            if 'hour_of_year' in df_zone.columns:
                p = df_zone.pivot(index='hour_of_year', columns='year', values='mean')
                x = p.index.values
            else:
                p = df_zone.pivot(index='month', columns='year', values='mean')
                x = np.arange(1, 13)

            period_mean = p.mean(axis=1).reindex(x)
            years = sorted([c for c in p.columns if not pd.isna(c)])
            best_year = None
            best_dist = float('inf')
            for y in years:
                series = p[y].reindex(x)
                if series.isna().all() or period_mean.isna().all():
                    continue
                mask = (~series.isna()) & (~period_mean.isna())
                if not mask.any():
                    continue
                diff = series[mask].values - period_mean[mask].values
                dist = np.sqrt((diff ** 2).mean())
                if dist < best_dist:
                    best_dist = dist
                    best_year = y

            if best_year is None:
                chosen_ref = args.ref
            else:
                chosen_ref = int(best_year)
                print(f"Auto-selected reference year for zone {zone}: {chosen_ref} (RMSE={best_dist:.3f})")
        else:
            chosen_ref = args.ref

        print(f"Plotting {zone} -> {outpath}")
        plot_zone(df_zone, zone, chosen_ref, outpath)

    print(f"Done. Figures written to {outdir}")


if __name__ == '__main__':
    main()
