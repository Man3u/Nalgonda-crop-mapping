"""Step 2.5b: Sentinel-1 radar time series for test fields and control points, kharif 2025."""
import ee
import geopandas as gpd
import pandas as pd
from config import EE_PROJECT, PROCESSED, INTERIM

ee.Initialize(project=EE_PROJECT)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
START, END = "2025-06-01", "2025-12-01"
PASS = "DESCENDING"          # Step 2.5 showed only descending scenes exist here
BUFFER_M = 30                # radius (m) of the circle averaged at each point
RUN_MONTHLY_TABLE = False    # district-wide monthly table from Step 2.5 (slow, already done)

WATER_VH, WATER_VV = -20.0, -16.0   # rule-of-thumb open-water thresholds (dB)

# label: (latitude, longitude), exactly as copied from Google Maps
FIELDS = {
    "Paddy A": (16.906201926363888, 79.51326660858167),
    "Paddy B": (16.903981848266753, 79.50420212162051),
    "Paddy C": (16.900684783885, 79.49293916943586),
    "Dry": (16.92075352173126, 79.38121341757046),
    "CONTROL water (Nagarjuna Sagar)": (16.57647824105806, 79.31269505232173),
    "CONTROL town (Nalgonda)": (17.051762780036384, 79.26482294730344),
}

# ------------------------------------------------------------------
# Data sources
# ------------------------------------------------------------------
aoi_gdf = gpd.read_file(PROCESSED / "study_area.gpkg", layer="nalgonda_district_2016")
aoi = ee.Geometry(aoi_gdf.geometry.union_all().simplify(0.001).__geo_interface__)

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterDate(START, END)
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("orbitProperties_pass", PASS))
    .select(["VV", "VH"])
)

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
water_history = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)

WC_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse", 70: "Snow/ice", 80: "Permanent water",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss/lichen",
}


def fmt(x, width=6):
    return f"{x:{width}.1f}" if x is not None and pd.notna(x) else "n/a".rjust(width)


# ------------------------------------------------------------------
# Optional: district-wide monthly table (Step 2.5)
# ------------------------------------------------------------------
if RUN_MONTHLY_TABLE:
    s1_aoi = s1.filterBounds(aoi)
    print("Sentinel-1 scenes over the district:", s1_aoi.size().getInfo())
    months = [("Jun", 6), ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11)]
    for name, m in months:
        start = ee.Date.fromYMD(2025, m, 1)
        month_col = s1_aoi.filterDate(start, start.advance(1, "month"))
        views = month_col.select("VH").count().unmask(0).rename("views")
        stats = views.addBands(month_col.median()).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e9, bestEffort=True
        ).getInfo()
        print(f"{name} 2025 | views: {fmt(stats.get('views'), 4)} "
              f"| VV: {fmt(stats.get('VV'))} dB | VH: {fmt(stats.get('VH'))} dB")


# ------------------------------------------------------------------
# Functions for one field
# ------------------------------------------------------------------
def land_check(field):
    """What do two independent global maps say is at this spot?"""
    info = ee.Dictionary({
        "wc": worldcover.reduceRegion(ee.Reducer.mode(), field, 10).get("Map"),
        "occ": water_history.reduceRegion(ee.Reducer.mean(), field, 30).get("occurrence"),
    }).getInfo()
    return WC_CLASSES.get(info.get("wc"), "unknown"), info.get("occ")


def radar_series(field):
    """Mean VV and VH inside the field circle, for every Sentinel-1 date."""
    def one_date(img):
        v = img.reduceRegion(ee.Reducer.mean(), field, 10)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "VV": v.get("VV"),
            "VH": v.get("VH"),
        })

    col = s1.filterBounds(field).sort("system:time_start")
    rows = [f["properties"] for f in col.map(one_date).getInfo()["features"]]
    df = pd.DataFrame(rows, columns=["date", "VV", "VH"]).dropna()
    return df.groupby("date", as_index=False).mean()   # merge duplicate scenes on one date


def summarise(df):
    """Turn a time series into a handful of numbers (hand-made ML features)."""
    rise = df["VH"].diff()
    i_rise = rise.idxmax()
    return {
        "dates": len(df),
        "VH_min": df["VH"].min(),
        "VH_max": df["VH"].max(),
        "amplitude": df["VH"].max() - df["VH"].min(),
        "peak_date": df.loc[df["VH"].idxmax(), "date"],
        "biggest_rise": rise.max(),
        "rise_ends": df.loc[i_rise, "date"],
        "water_dates": int(df["water"].sum()),
        "VV_mean": df["VV"].mean(),
    }


# ------------------------------------------------------------------
# Run every field
# ------------------------------------------------------------------
all_series, summary = [], []

for label, (lat, lon) in FIELDS.items():
    field = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)   # Earth Engine wants [lon, lat]
    lc_name, occ = land_check(field)
    df = radar_series(field)

    print(f"\n{label} at {lat:.4f}, {lon:.4f}")
    print(f"  WorldCover 2021: {lc_name} | water 1984-2021: {fmt(occ, 3)}% of the time")
    if df.empty:
        print("  no radar data here")
        continue

    df["water"] = (df["VH"] < WATER_VH) & (df["VV"] < WATER_VV)
    for r in df.itertuples():
        bar = "#" * max(0, int(r.VH + 30))
        flag = "  <- water?" if r.water else ""
        print(f"  {r.date} | VV {r.VV:6.1f} | VH {r.VH:6.1f} {bar}{flag}")

    summary.append({"field": label, "worldcover": lc_name, "water_%": occ, **summarise(df)})
    df.insert(0, "field", label)
    all_series.append(df)

# ------------------------------------------------------------------
# Summary table and CSV
# ------------------------------------------------------------------
pd.set_option("display.width", 220)
print("\nSUMMARY (VH in dB)")
print(pd.DataFrame(summary).set_index("field").round(1).to_string())

out = INTERIM / "s1_field_series_kharif2025.csv"
pd.concat(all_series).to_csv(out, index=False)
print(f"\nSaved {out.relative_to(INTERIM.parents[1])}")