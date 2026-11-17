"""Fonctions communes aux scripts 02_statistical_analysis_climate_periods.py et heatwaves.py.

Un seul endroit pour les périodes, les seuils et la détection des vagues de chaleur :
les deux scripts restent ainsi toujours cohérents.
"""
import pandas as pd

# ------------------------------------------------------------
# Paramètres communs
# ------------------------------------------------------------
HIST_START = "1990-01-01"
HIST_END = "2019-12-31 23:00"
FUT_START = "2040-01-01"
FUT_END = "2069-12-31 23:00"

INPUT_FILE = "results/climate_3h_all_zones.csv"

PERCENTILE = 95
WINDOW_DAYS = 15
MIN_DURATION = 3


# ------------------------------------------------------------
# Chargement
# ------------------------------------------------------------
def load_climate(path=INPUT_FILE):
    """Charge le CSV, convertit tas en °C si besoin, renvoie (df, df_hist, df_fut)."""
    df = pd.read_csv(path, parse_dates=["time"])
    if df["tas"].mean() > 100:  # heuristique : Kelvin
        df["tas"] = df["tas"] - 273.15
    df_hist = df[(df["time"] >= HIST_START) & (df["time"] <= HIST_END)].copy()
    df_fut = df[(df["time"] >= FUT_START) & (df["time"] <= FUT_END)].copy()
    return df, df_hist, df_fut


def daily_max_by_zone(df):
    """Maximum journalier de tas par zone -> colonnes zone, date, tas_daily_max."""
    return (
        df.groupby(["zone", pd.Grouper(key="time", freq="D")])["tas"]
        .max()
        .reset_index()
        .rename(columns={"tas": "tas_daily_max", "time": "date"})
    )


def continuous_zone_series(daily, zone):
    """Série journalière continue pour une zone. Jours manquants = -9999 (jamais chauds)."""
    sub = daily[daily["zone"] == zone].set_index("date").sort_index()["tas_daily_max"]
    if sub.empty:
        return sub
    idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
    return sub.reindex(idx).fillna(-9999)


# ------------------------------------------------------------
# Seuils calendaires (percentile sur fenêtre glissante)
# ------------------------------------------------------------
def calculate_calendar_thresholds(hist_daily, percentile=PERCENTILE, window_days=WINDOW_DAYS):
    """Seuil de percentile pour chacun des 366 jours calendaires (fenêtre glissante circulaire)."""
    if window_days <= 0 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd number")

    half_window = window_days // 2
    calendar_days = hist_daily.index.map(lambda d: d.replace(year=2000).dayofyear).to_numpy()
    values = hist_daily.to_numpy()

    window_samples = pd.concat(
        [
            pd.DataFrame({
                "calendar_day": ((calendar_days - offset - 1) % 366) + 1,
                "tas_daily_max": values,
            })
            for offset in range(-half_window, half_window + 1)
        ],
        ignore_index=True,
    )
    return (
        window_samples.groupby("calendar_day")["tas_daily_max"]
        .quantile(percentile / 100.0)
        .reindex(range(1, 367))
    )


def thresholds_for_dates(dates, calendar_thresholds):
    """Associe à chaque date son seuil calendaire."""
    calendar_days = dates.map(lambda d: d.replace(year=2000).dayofyear)
    return pd.Series(calendar_thresholds.reindex(calendar_days).to_numpy(), index=dates)


# ------------------------------------------------------------
# Détection des vagues de chaleur
# ------------------------------------------------------------
def find_heatwave_periods(daily_max, threshold, min_duration=MIN_DURATION):
    """Liste de (date_début, durée_en_jours) pour les séquences de jours chauds >= min_duration."""
    is_hot = (daily_max > threshold).astype(bool)
    periods = []
    start, length = None, 0
    for dt, val in is_hot.items():
        if val:
            if start is None:
                start = dt
            length += 1
        else:
            if start is not None and length >= min_duration:
                periods.append((start, length))
            start, length = None, 0
    if start is not None and length >= min_duration:
        periods.append((start, length))
    return periods


def heatwave_days_per_year(periods, years):
    """Nombre de jours en vague de chaleur par année civile (une vague à cheval sur 2 années est répartie)."""
    counts = {y: 0 for y in years}
    for start, length in periods:
        for d in pd.date_range(start, periods=length, freq="D"):
            if d.year in counts:
                counts[d.year] += 1
    return pd.Series(counts)


def analyse_zone(hist_series, fut_series, min_duration=MIN_DURATION, percentile=PERCENTILE):
    """Seuils (calculés sur l'historique) + vagues de chaleur des deux périodes pour une zone.

    Renvoie None si la zone n'a pas de données historiques exploitables.
    """
    hist_valid = hist_series[hist_series > -9999]
    if hist_valid.empty:
        return None
    thresholds = calculate_calendar_thresholds(hist_valid, percentile=percentile)
    if thresholds.isna().all():
        return None

    hw_hist = find_heatwave_periods(hist_series, thresholds_for_dates(hist_series.index, thresholds), min_duration)
    hw_fut = find_heatwave_periods(fut_series, thresholds_for_dates(fut_series.index, thresholds), min_duration)
    return {"thresholds": thresholds, "hw_hist": hw_hist, "hw_fut": hw_fut}