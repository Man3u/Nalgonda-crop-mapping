"""Step 3.2a: Draw a blind, stratified review sample and make one evidence card per point."""
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from config import PROCESSED, ROOT

SEED = 42
PER_STRATUM = 10
START_DATE, BIN_DAYS, N_BINS = date(2025, 6, 1), 12, 15
BIN_STARTS = [START_DATE + timedelta(days=BIN_DAYS * i) for i in range(N_BINS)]
X = pd.to_datetime(BIN_STARTS) + pd.Timedelta(days=BIN_DAYS / 2)      # middle of each bin

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"                  # categorical slots 1-3
REF_GREY = "#a3a29d"                                                   # context line, not a series
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CARD_DIR = ROOT / "outputs" / "review_cards"


def cols(prefix):
    return [f"{prefix}_{i:02d}" for i in range(N_BINS)]


df = pd.read_csv(PROCESSED / "labels_v2_kharif2025.csv")

# ------------------------------------------------------------------
# 1. Stratified random sample: the same number from every label group (strata do not overlap)
# ------------------------------------------------------------------
STRATA = {
    "paddy_high": df["label"].eq("Paddy") & df["confidence"].eq("high"),
    "paddy_medium": df["label"].eq("Paddy") & df["confidence"].eq("medium"),
    "other_high": df["label"].eq("Other kharif crop") & df["confidence"].eq("high"),
    "other_medium": df["label"].eq("Other kharif crop") & df["confidence"].eq("medium"),
    "perennial": df["label"].eq("Perennial"),
    "review_moved_from_other": df["label"].eq("Review") & df["label_v1"].eq("Other kharif crop"),
    "review_other": df["label"].eq("Review") & ~df["label_v1"].eq("Other kharif crop"),
}

picked = []
print(f"{'stratum':26s} sampled of available")
for name, mask in STRATA.items():
    pool = df[mask]
    n = min(PER_STRATUM, len(pool))
    picked.append(pool.sample(n=n, random_state=SEED).assign(stratum=name, stratum_size=len(pool)))
    print(f"{name:26s} {n:7d} {len(pool):9d}")

sample = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle = blind order
sample.insert(0, "review_id", [f"R{i:03d}" for i in range(1, len(sample) + 1)])
print(f"{'TOTAL':26s} {len(sample):7d}")

# ------------------------------------------------------------------
# 2. Two files: an answer key (do not open while reviewing) and a blind layer for QGIS
# ------------------------------------------------------------------
key_cols = ["review_id", "point_id", "stratum", "stratum_size", "label", "confidence", "reason"]
sample[key_cols].to_csv(PROCESSED / "review_key_v2.csv", index=False)

blind = sample[["review_id", "orbit", "lon", "lat"]].copy()
blind["card"] = [str(CARD_DIR / f"{rid}.png") for rid in blind["review_id"]]
blind["review_label"] = ""          # you fill these three in QGIS
blind["review_confidence"] = ""
blind["review_notes"] = ""
gdf = gpd.GeoDataFrame(blind, geometry=gpd.points_from_xy(blind["lon"], blind["lat"]), crs="EPSG:4326")
gdf.to_file(PROCESSED / "review_blind_v2.gpkg", layer="review", driver="GPKG")


# ------------------------------------------------------------------
# 3. One evidence card per point: radar above, optical below; no label printed on it
# ------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


CARD_DIR.mkdir(parents=True, exist_ok=True)
for _, row in sample.iterrows():
    vh = row[cols("VH")].to_numpy(dtype=float)
    ref = vh - row[cols("ANOM")].to_numpy(dtype=float)          # neighbours' median VH
    ndvi = row[cols("NDVI")].to_numpy(dtype=float)
    lswi = row[cols("LSWI")].to_numpy(dtype=float)
    clear = ~np.isnan(ndvi)

    fig, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True, facecolor=SURFACE,
                                     gridspec_kw={"hspace": 0.25})
    for ax in (ax_r, ax_o):
        style_axes(ax)
    ax_r.spines["bottom"].set_visible(False)
    ax_o.spines["bottom"].set_color(GRID)

    dot = dict(marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax_r.plot(X, ref, color=REF_GREY, linewidth=2, label="10 nearest neighbours (median)")
    ax_r.plot(X, vh, color=BLUE, linewidth=2, label="this point", **dot)
    ax_r.set_ylabel("Radar VH, γ⁰ (dB)", color=INK_2, fontsize=8)
    ax_r.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5,
                labelcolor=INK)

    ax_o.axhline(0, color=GRID, linewidth=1)
    ax_o.plot(X[clear], ndvi[clear], color=AQUA, linewidth=2, label="NDVI (greenness)", **dot)
    ax_o.plot(X[clear], lswi[clear], color=ORANGE, linewidth=2, label="LSWI (water)", **dot)
    ax_o.set_ylim(-0.4, 1.0)
    ax_o.set_ylabel("Sentinel-2 index", color=INK_2, fontsize=8)
    ax_o.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5,
                labelcolor=INK)
    ax_o.xaxis.set_major_locator(mdates.MonthLocator())
    ax_o.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle(f"{row['review_id']}   ({row['lat']:.5f}, {row['lon']:.5f})   orbit {row['orbit']}",
                 x=0.125, ha="left", y=1.02, color=INK, fontsize=10, fontweight="bold")
    fig.savefig(CARD_DIR / f"{row['review_id']}.png", dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

print(f"\nSaved {len(sample)} cards to outputs/review_cards/")
print("Saved data/processed/review_blind_v2.gpkg (for QGIS) and review_key_v2.csv (answer key: do not open yet)")
