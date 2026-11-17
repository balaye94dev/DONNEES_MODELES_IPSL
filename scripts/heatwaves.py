import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from heatwaves_utils import (
    FUT_END, FUT_START, HIST_END, HIST_START, MIN_DURATION,
    analyse_zone, continuous_zone_series, daily_max_by_zone,
    heatwave_days_per_year, load_climate,
)

OUTPUT_DIR = "results/figures/heatwaves"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIST_YEARS = range(int(HIST_START[:4]), int(HIST_END[:4]) + 1)
FUT_YEARS = range(int(FUT_START[:4]), int(FUT_END[:4]) + 1)


def plot_zone_heatwaves(zone, hw_hist, hw_fut):
    """Figure par zone : en haut chaque vague (début, durée), en bas les jours de vague par année.

    Mise en page à 3 colonnes (2 panneaux + colonne étroite réservée à la barre de couleur) :
    les 4 graphiques ont toujours exactement la même taille, avec ou sans vagues détectées.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], wspace=0.12, hspace=0.3)

    ax_top_h = fig.add_subplot(gs[0, 0])
    ax_top_f = fig.add_subplot(gs[0, 1], sharey=ax_top_h)
    ax_bot_h = fig.add_subplot(gs[1, 0])
    ax_bot_f = fig.add_subplot(gs[1, 1], sharey=ax_bot_h)
    cax = fig.add_subplot(gs[0, 2])

    # échelle de couleur commune aux deux périodes (couleurs comparables)
    all_dur = [d for _, d in hw_hist + hw_fut]
    vmax = max(all_dur) if all_dur else MIN_DURATION + 1
    norm = Normalize(vmin=MIN_DURATION, vmax=max(vmax, MIN_DURATION + 1))

    panels = [
        (ax_top_h, ax_bot_h, hw_hist, HIST_YEARS, f"Historical ({HIST_START[:4]}-{HIST_END[:4]})", "tab:blue"),
        (ax_top_f, ax_bot_f, hw_fut, FUT_YEARS, f"Future ({FUT_START[:4]}-{FUT_END[:4]})", "tab:red"),
    ]

    for ax, ax2, periods, years, title, color in panels:
        if periods:
            starts, durations = zip(*periods)
            ax.scatter(
                starts, durations, c=durations, cmap="plasma", norm=norm,
                s=[max(40, d * 40) for d in durations],
                edgecolor="black", alpha=0.8, linewidth=0.7,
            )
        else:
            ax.text(0.5, 0.5, "No heatwaves detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(pd.Timestamp(f"{years[0]}-01-01"), pd.Timestamp(f"{years[-1]}-12-31"))
        ax.set_title(f"{title}\n{len(periods)} heatwave events", fontsize=14)
        ax.set_xlabel("Start date")
        ax.grid(True, linestyle="--", alpha=0.6)

        per_year = heatwave_days_per_year(periods, years)
        ax2.bar(per_year.index, per_year.values, color=color, alpha=0.8)
        ax2.axhline(per_year.mean(), color="black", linestyle="--", linewidth=1,
                    label=f"Mean: {per_year.mean():.1f} d/yr")
        ax2.set_xlim(years[0] - 1, years[-1] + 1)
        ax2.set_xlabel("Year")
        ax2.legend(loc="upper left")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    # une seule barre de couleur, toujours présente
    fig.colorbar(ScalarMappable(norm=norm, cmap="plasma"), cax=cax).set_label("Duration (days)")

    ax_top_h.set_ylabel("Duration (days)")
    ax_bot_h.set_ylabel("Days in heatwave per year")
    plt.setp(ax_top_f.get_yticklabels(), visible=False)
    plt.setp(ax_bot_f.get_yticklabels(), visible=False)
    fig.suptitle(zone, fontsize=28, fontweight="bold")

    out_png = os.path.join(OUTPUT_DIR, f"heatwaves_{zone}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def plot_summary(summary):
    """Figure de synthèse : toutes les zones, historique vs futur."""
    summary = summary.sort_values("days_per_year_fut", ascending=False)
    x = np.arange(len(summary))
    w = 0.4

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.35 * len(summary) + 3)))
    metrics = [
        ("events_per_year", "Heatwave events per year"),
		("days_per_year", "Heatwave days per year"),
        ("max_duration", "Longest heatwave (days)"),
    ]
    for ax, (key, label) in zip(axes, metrics):
        ax.bar(x - w / 2, summary[f"{key}_hist"], w, label="Historical", color="tab:blue")
        ax.bar(x + w / 2, summary[f"{key}_fut"], w, label="Future", color="tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["zone"], rotation=90)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.legend()
    fig.suptitle("Heatwaves Statistics", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out_png = os.path.join(OUTPUT_DIR, "heatwaves_summary_all_zones.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_png}")

    summary.to_csv(os.path.join(OUTPUT_DIR, "heatwaves_summary_all_zones.csv"), index=False)


def main():
    _, df_hist, df_fut = load_climate()
    hist_daily = daily_max_by_zone(df_hist)
    fut_daily = daily_max_by_zone(df_fut)

    rows = []
    for zone in sorted(hist_daily["zone"].unique()):
        hist_series = continuous_zone_series(hist_daily, zone)
        fut_series = continuous_zone_series(fut_daily, zone)

        result = analyse_zone(hist_series, fut_series, min_duration=MIN_DURATION)
        if result is None:
            print(f"Skipping zone {zone}: no threshold available")
            continue

        hw_hist, hw_fut = result["hw_hist"], result["hw_fut"]
        plot_zone_heatwaves(zone, hw_hist, hw_fut)

        rows.append({
            "zone": zone,
            "days_per_year_hist": heatwave_days_per_year(hw_hist, HIST_YEARS).mean(),
            "days_per_year_fut": heatwave_days_per_year(hw_fut, FUT_YEARS).mean(),
            "events_per_year_hist": len(hw_hist) / len(HIST_YEARS),
            "events_per_year_fut": len(hw_fut) / len(FUT_YEARS),
            "max_duration_hist": max((d for _, d in hw_hist), default=0),
            "max_duration_fut": max((d for _, d in hw_fut), default=0),
        })

    if rows:
        plot_summary(pd.DataFrame(rows))


if __name__ == "__main__":
    main()