"""Step 3.1b: Error analysis. Why did 40% of candidates land in Review? Test the hypotheses with the data."""
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from config import PROCESSED, ROOT

START_DATE, BIN_DAYS, N_BINS = date(2025, 6, 1), 12, 15
BIN_STARTS = [START_DATE + timedelta(days=BIN_DAYS * i) for i in range(N_BINS)]
BIN_LABELS = [d.strftime("%m-%d") for d in BIN_STARTS]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5, fixed order
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

df = pd.read_csv(PROCESSED / "candidates_kharif2025.csv")
df[["radar_dips", "optical_flood"]] = df[["radar_dips", "optical_flood"]].fillna("")


def cols(prefix):
    return [f"{prefix}_{i:02d}" for i in range(N_BINS)]


GROUPS = {
    "Paddy (high)": df["label"].eq("Paddy"),
    "Other crop (high)": df["label"].eq("Other kharif crop") & df["confidence"].eq("high"),
    "Review: dip, then dry": df["reason"].eq("radar dip but optical shows dry soil soon after"),
    "Review: optical flood only": df["reason"].eq("optical flood signal without a radar dip-and-rise"),
    "Review: green in June": df["reason"].eq("already green in June: orchard or perennial?"),
}


def share_by_bin(text_column, mask):
    """Percent of the group's points that have an event (dip or flood) in each 12-day bin."""
    counts = pd.Series(0, index=BIN_LABELS)
    for text in text_column[mask]:
        for d in (x.strip() for x in text.split(",") if x.strip()):
            counts[d] += 1
    return (100 * counts / max(int(mask.sum()), 1)).round(0).astype(int)


pd.set_option("display.width", 200)

# ---- H1: is the "dip, then dry" a regional weather event (same date everywhere)?
print("H1  RADAR DIPS: % of each group's points with a dip in each 12-day bin")
dips = pd.DataFrame({name: share_by_bin(df["radar_dips"], m) for name, m in GROUPS.items()})
dips["ALL POINTS"] = share_by_bin(df["radar_dips"], df["label"].notna())
print(dips.to_string())

# ---- H2: are "optical flood only" points really wet bare soil (low NDVI all season)?
print("\nH2  OPTICAL FLOOD SIGNALS: % of each group's points flagged in each bin (Jun-Sep only)")
floods = pd.DataFrame({name: share_by_bin(df["optical_flood"], m) for name, m in GROUPS.items()})
print(floods.loc[[b for b in BIN_LABELS if b < "10-01"]].to_string())

# ---- Group profiles
vh = df[cols("VH")]
ndvi = df[cols("NDVI")]
profile = pd.DataFrame({
    name: {
        "points": int(m.sum()),
        "NDVI max (median)": ndvi[m].max(axis=1).median(),
        "NDVI in June (median)": ndvi[m].iloc[:, 0:3].median(axis=1).median(),
        "VH amplitude dB (median)": (vh[m].max(axis=1) - vh[m].min(axis=1)).median(),
        "clear optical bins (median)": df.loc[m, "clear_bins"].median(),
        "share in orbit 92 (%)": 100 * df.loc[m, "orbit"].eq(92).mean(),
    }
    for name, m in GROUPS.items()
}).round(2)
print("\nGROUP PROFILES")
print(profile.to_string())

# ---- Figure: median season curve of each group, radar above, optical below (no dual axis)
fig, (ax_vh, ax_ndvi) = plt.subplots(
    2, 1, figsize=(9, 6.5), sharex=True, facecolor=SURFACE, gridspec_kw={"hspace": 0.15}
)
for ax in (ax_vh, ax_ndvi):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
ax_vh.spines["bottom"].set_visible(False)
ax_ndvi.spines["bottom"].set_color(GRID)

x = pd.to_datetime(BIN_STARTS) + pd.Timedelta(days=BIN_DAYS / 2)    # plot at the middle of each bin
for (name, m), color in zip(GROUPS.items(), COLORS):
    style = dict(color=color, linewidth=2, marker="o", markersize=4,
                 markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax_vh.plot(x, vh[m].median().to_numpy(), label=f"{name} (n={int(m.sum())})", **style)
    ax_ndvi.plot(x, ndvi[m].median().to_numpy(), **style)

ax_vh.set_ylabel("Radar VH, γ⁰ (dB), median", color=INK_2, fontsize=9)
ax_ndvi.set_ylabel("Sentinel-2 NDVI, median", color=INK_2, fontsize=9)
ax_ndvi.set_ylim(0, 1)
ax_ndvi.xaxis.set_major_locator(mdates.MonthLocator())
ax_ndvi.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax_vh.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False,
             fontsize=8.5, labelcolor=INK, handlelength=1.8)
fig.suptitle("Candidate groups: median season curves, kharif 2025, Nalgonda",
             x=0.125, ha="left", y=1.06, color=INK, fontsize=12, fontweight="bold")
fig.text(0.125, 0.015, "12-day bins. Each line is the median of all candidate points in the group; "
         "the bottom panel uses clear Sentinel-2 observations only.", color=INK_2, fontsize=7.5)

out = ROOT / "outputs" / "figures" / "fig02_candidate_groups_kharif2025.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"\nFigure saved: {out.relative_to(ROOT)}")
