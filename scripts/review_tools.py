"""Shared helpers: the model's features and the evidence card. Used by every script from Step 3.5 on."""
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_BINS = 15
START_DATE, BIN_DAYS = date(2025, 6, 1), 12
BIN_STARTS = [START_DATE + timedelta(days=BIN_DAYS * i) for i in range(N_BINS)]
X_DATES = pd.to_datetime(BIN_STARTS) + pd.Timedelta(days=BIN_DAYS / 2)
CLASSES = ["Paddy", "Other kharif crop", "Perennial"]


def cols(prefix):
    return [f"{prefix}_{i:02d}" for i in range(N_BINS)]


def build_features(frame):
    """Exactly the features of scripts/11_rf_baseline.py: gap-filled curves plus 7 summaries."""
    filled = [frame[cols(p)].astype(float).interpolate(axis=1, limit_direction="both")
              for p in ["VH", "VV", "NDVI", "LSWI"]]
    X = pd.concat(filled, axis=1)
    ndvi, lswi, vh = X[cols("NDVI")].to_numpy(), X[cols("LSWI")].to_numpy(), X[cols("VH")].to_numpy()
    X["ndvi_june"] = ndvi[:, 0:3].mean(axis=1)
    X["ndvi_min_jul_aug"] = ndvi[:, 3:8].min(axis=1)
    X["ndvi_max"] = ndvi.max(axis=1)
    X["flood_index"] = (lswi - ndvi)[:, 0:10].max(axis=1)
    X["vh_min"] = vh.min(axis=1)
    X["vh_amplitude"] = vh.max(axis=1) - vh.min(axis=1)
    X["vh_rise_after_min"] = np.array([row[np.argmin(row):].max() - row.min() for row in vh])
    return X.fillna(X.median())


BLUE, ORANGE, AQUA, REF_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def make_card(row, review_id, out_path):
    """Radar (this point vs 10 neighbours) above, NDVI and LSWI below. No label on the card."""
    vh = row[cols("VH")].to_numpy(dtype=float)
    ref = vh - row[cols("ANOM")].to_numpy(dtype=float)
    ndvi = row[cols("NDVI")].to_numpy(dtype=float)
    lswi = row[cols("LSWI")].to_numpy(dtype=float)
    clear = ~np.isnan(ndvi)

    fig, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True, facecolor=SURFACE,
                                     gridspec_kw={"hspace": 0.25})
    for ax in (ax_r, ax_o):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    ax_r.spines["bottom"].set_visible(False)
    ax_o.spines["bottom"].set_color(GRID)

    dot = dict(marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax_r.plot(X_DATES, ref, color=REF_GREY, linewidth=2, label="10 nearest neighbours (median)")
    ax_r.plot(X_DATES, vh, color=BLUE, linewidth=2, label="this point", **dot)
    ax_r.set_ylabel("Radar VH, γ⁰ (dB)", color=INK_2, fontsize=8)
    ax_r.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK)

    ax_o.axhline(0, color=GRID, linewidth=1)
    ax_o.plot(X_DATES[clear], ndvi[clear], color=AQUA, linewidth=2, label="NDVI (greenness)", **dot)
    ax_o.plot(X_DATES[clear], lswi[clear], color=ORANGE, linewidth=2, label="LSWI (water)", **dot)
    ax_o.set_ylim(-0.4, 1.0)
    ax_o.set_ylabel("Sentinel-2 index", color=INK_2, fontsize=8)
    ax_o.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5, labelcolor=INK)
    ax_o.xaxis.set_major_locator(mdates.MonthLocator())
    ax_o.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle(f"{review_id}   ({row['lat']:.5f}, {row['lon']:.5f})   orbit {row['orbit']}",
                 x=0.125, ha="left", y=1.02, color=INK, fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
