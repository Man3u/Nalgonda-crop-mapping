"""Step 2.6: Two sensors, one field. Sentinel-2 NDVI and LSWI beside Sentinel-1 VH for the test fields."""
import ee
import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no pop-up window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from config import BUFFER_M, EE_PROJECT, FIELDS, INTERIM, ROOT

ee.Initialize(project=EE_PROJECT)

START, END = "2025-06-01", "2025-12-01"
CLEAR_THRESHOLD = 0.60    # Cloud Score+ cs_cdf: a pixel counts as clear at or above this
MIN_CLEAR_SHARE = 0.8     # keep a date only if at least 80% of the field circle is clear
FLOOD_MARGIN = 0.05       # optical flood signal when LSWI + margin >= NDVI

# Fields to draw, and which radar orbit to show for each (one orbit per line, never mixed)
FIGURE = {
    "Paddy B": 92,
    "Dry": 165,
    "Dry 2 (Devarakonda)": 165,
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, fixed order
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

s2_base = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(START, END)
cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")


def optical_series(field):
    """NDVI and LSWI over the clear part of the field circle, for every Sentinel-2 date."""
    def one_date(img):
        clear = img.select("cs_cdf").gte(CLEAR_THRESHOLD).rename("clear")
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        lswi = img.normalizedDifference(["B8", "B11"]).rename("LSWI")
        v = ndvi.addBands(lswi).updateMask(clear).addBands(clear).reduceRegion(
            ee.Reducer.mean(), field, 10
        )
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "clear": v.get("clear"),
            "NDVI": v.get("NDVI"),
            "LSWI": v.get("LSWI"),
        })

    col = s2_base.filterBounds(field).linkCollection(cloud_score, ["cs_cdf"]).sort("system:time_start")
    rows = [f["properties"] for f in col.map(one_date).getInfo()["features"]]
    df = pd.DataFrame(rows, columns=["date", "clear", "NDVI", "LSWI"]).dropna(subset=["clear"])
    df = df.groupby("date", as_index=False).mean()   # two overlapping tiles -> one row per date
    n_dates = len(df)
    df = df[(df["clear"] >= MIN_CLEAR_SHARE) & df["NDVI"].notna()].reset_index(drop=True)
    df["flood_signal"] = df["LSWI"] + FLOOD_MARGIN >= df["NDVI"]
    return df, n_dates


# ------------------------------------------------------------------
# 1. Optical series for every field
# ------------------------------------------------------------------
optical = []
for label, (lat, lon) in FIELDS.items():
    field = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)   # Earth Engine wants [lon, lat]
    df, n_dates = optical_series(field)

    print(f"\n{label}: {n_dates} Sentinel-2 dates, {len(df)} clear enough to use")
    for r in df.itertuples():
        bar = "#" * max(0, int(r.NDVI * 30))
        flag = "  <- flood signal" if r.flood_signal else ""
        print(f"  {r.date} | NDVI {r.NDVI:5.2f} | LSWI {r.LSWI:5.2f} {bar}{flag}")

    df.insert(0, "field", label)
    optical.append(df)

s2 = pd.concat(optical)
s2.to_csv(INTERIM / "s2_field_indices_kharif2025.csv", index=False)
s2["date"] = pd.to_datetime(s2["date"])

# ------------------------------------------------------------------
# 2. Radar series, reused from Step 2.5 (no need to ask Earth Engine again)
# ------------------------------------------------------------------
s1 = pd.read_csv(INTERIM / "s1_field_series_kharif2025.csv", parse_dates=["date"])

# ------------------------------------------------------------------
# 3. Figure: two panels sharing one time axis (never two y-scales on one panel)
# ------------------------------------------------------------------
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

for (label, orbit), color in zip(FIGURE.items(), COLORS):
    radar = s1[(s1["field"] == label) & (s1["orbit"] == orbit)].sort_values("date")
    optic = s2[s2["field"] == label].sort_values("date")
    if radar.empty:
        print(f"(no radar rows for {label}, orbit {orbit}: run scripts/03_s1_explore.py first)")
    style = dict(color=color, linewidth=2, marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax_vh.plot(radar["date"], radar["VHg"], label=f"{label} (orbit {orbit})", **style)
    ax_ndvi.plot(optic["date"], optic["NDVI"], **style)

ax_vh.set_ylabel("Radar VH, γ⁰ (dB)", color=INK_2, fontsize=9)
ax_ndvi.set_ylabel("Sentinel-2 NDVI", color=INK_2, fontsize=9)
ax_ndvi.set_ylim(0, 1)
ax_ndvi.xaxis.set_major_locator(mdates.MonthLocator())
ax_ndvi.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax_vh.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False,
             fontsize=9, labelcolor=INK, handlelength=1.8)
fig.suptitle("Two sensors, one season: test fields, kharif 2025, Nalgonda",
             x=0.125, ha="left", y=0.99, color=INK, fontsize=12, fontweight="bold")
fig.text(0.125, 0.015,
         "Radar: Sentinel-1 every 12 days, one relative orbit per field. "
         "Optical: only Sentinel-2 dates with at least 80% of the field cloud-free (Cloud Score+ ≥ 0.60).",
         color=INK_2, fontsize=7.5)

out = ROOT / "outputs" / "figures" / "fig01_s1_s2_test_fields_kharif2025.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"\nFigure saved: {out.relative_to(ROOT)}")