"""Step 3.1: Sample candidate points across the district's cropland and propose labels (D5, D6)."""
import json
import math
from datetime import date, timedelta

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from config import EE_PROJECT, PROCESSED

ee.Initialize(project=EE_PROJECT)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
START_DATE = date(2025, 6, 1)
BIN_DAYS, N_BINS = 12, 15          # 15 bins of 12 days: 1 Jun to 27 Nov 2025
POINTS_PER_MANDAL = 6
SEED = 42                          # fixed seed: the same points on every run
EDGE_BUFFER_M = 30                 # points must sit at least this far inside cropland
SMOOTH_RADIUS_M = 15               # average every image over a ~30 m circle
CLEAR_THRESHOLD = 0.60             # Cloud Score+ cs_cdf
PREFERRED_ORBIT, OTHER_ORBIT = 165, 92
MIN_ORBIT_BINS = 10                # use orbit 165 where it has >= 10 of the 15 bins

# Evidence rules (D5, D6), the same logic we tested on the seven fields
DIP_DB, RISE_DB = 4.0, 5.0
RISE_BINS = 7                      # about 90 days
NEAR_BINS = 2                      # "within about 3 weeks" = 2 bins either side
FLOOD_MARGIN = 0.05
FALLOW_MAX_NDVI = 0.40
CROP_MAX_NDVI = 0.50
EARLY_GREEN_NDVI = 0.45            # already green in June: orchard or perennial?
MIN_CLEAR_BINS = 3

BIN_STARTS = [START_DATE + timedelta(days=BIN_DAYS * i) for i in range(N_BINS)]
END_DATE = START_DATE + timedelta(days=BIN_DAYS * N_BINS)
FLOOD_SEASON_BINS = [i for i, d in enumerate(BIN_STARTS) if d < date(2025, 10, 1)]


def cols(prefix):
    return [f"{prefix}_{i:02d}" for i in range(N_BINS)]


def bin_label(i):
    return BIN_STARTS[i].strftime("%m-%d")


# ------------------------------------------------------------------
# 1. Study area, mandals, and where points are allowed
# ------------------------------------------------------------------
gpkg = PROCESSED / "study_area.gpkg"
district = gpd.read_file(gpkg, layer="nalgonda_district_2016").to_crs(4326)
aoi = ee.Geometry(district.geometry.union_all().simplify(0.001).__geo_interface__)

mandals = gpd.read_file(gpkg, layer="nalgonda_mandals_2016").to_crs(4326).reset_index(drop=True)
mandals["mandal_id"] = mandals.index + 1
m_small = mandals[["mandal_id", "geometry"]].copy()
m_small["geometry"] = m_small.geometry.simplify(0.0005)
mandal_fc = ee.FeatureCollection(json.loads(m_small.to_json()))

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
cropland_core = worldcover.eq(40).focal_min(radius=EDGE_BUFFER_M, kernelType="circle", units="meters")

mandal_img = ee.Image().int().paint(mandal_fc, "mandal_id").rename("mandal").updateMask(cropland_core)
points = mandal_img.stratifiedSample(
    numPoints=POINTS_PER_MANDAL, classBand="mandal", region=aoi,
    scale=30, seed=SEED, geometries=True, tileScale=8,
)
print(f"Candidate points: {points.size().getInfo()} across {len(mandals)} mandals")


# ------------------------------------------------------------------
# 2. One common time grid for both sensors: 15 bins of 12 days
# ------------------------------------------------------------------
def binned(col, empty, bands):
    """Median of each 12-day bin; the empty placeholder keeps a bin with no images from breaking."""
    images = []
    for i, start in enumerate(BIN_STARTS):
        s = ee.Date(start.isoformat())
        one = col.filterDate(s, s.advance(BIN_DAYS, "day")).merge(ee.ImageCollection([empty])).median()
        images.append(one.rename([f"{b}_{i:02d}" for b in bands]))
    return ee.Image.cat(images)


# ---- Sentinel-1: gamma0, averaged in linear power, one orbit per pixel (D4)
def s1_prepare(img):
    cos_theta = img.select("angle").multiply(math.pi / 180).cos()
    g0_db = img.select(["VH", "VV"]).subtract(cos_theta.log10().multiply(10))
    linear = g0_db.multiply(math.log(10) / 10).exp()                  # dB -> linear power
    smooth = linear.focal_mean(radius=SMOOTH_RADIUS_M, kernelType="circle", units="meters")
    back = smooth.log10().multiply(10).rename(["VH", "VV"]).toFloat()  # linear -> dB
    return ee.Image(back.copyProperties(img, ["system:time_start"]))


s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate(START_DATE.isoformat(), END_DATE.isoformat())
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
)
EMPTY_S1 = ee.Image.constant([0, 0]).rename(["VH", "VV"]).toFloat().updateMask(ee.Image(0))


def s1_orbit_stack(orbit):
    col = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)).map(s1_prepare)
    return binned(col, EMPTY_S1, ["VH", "VV"])


stack_pref = s1_orbit_stack(PREFERRED_ORBIT)
stack_other = s1_orbit_stack(OTHER_ORBIT)
in_pref = stack_pref.select("VH_.*").reduce(ee.Reducer.count()).unmask(0).gte(MIN_ORBIT_BINS)
s1_stack = ee.ImageCollection([
    stack_other.updateMask(in_pref.Not()),
    stack_pref.updateMask(in_pref),
]).mosaic()
orbit_band = ee.Image.constant(OTHER_ORBIT).where(in_pref, PREFERRED_ORBIT).rename("orbit").toInt()


# ---- Sentinel-2: NDVI and LSWI from clear pixels only
def s2_prepare(img):
    clear = img.select("cs_cdf").gte(CLEAR_THRESHOLD)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    lswi = img.normalizedDifference(["B8", "B11"]).rename("LSWI")
    idx = ndvi.addBands(lswi).updateMask(clear)
    smooth = idx.focal_mean(radius=SMOOTH_RADIUS_M, kernelType="circle", units="meters")
    smooth = smooth.rename(["NDVI", "LSWI"]).toFloat()
    return ee.Image(smooth.copyProperties(img, ["system:time_start"]))


s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(START_DATE.isoformat(), END_DATE.isoformat())
    .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"])
    .map(s2_prepare)
)
EMPTY_S2 = ee.Image.constant([0, 0]).rename(["NDVI", "LSWI"]).toFloat().updateMask(ee.Image(0))
s2_stack = binned(s2, EMPTY_S2, ["NDVI", "LSWI"])

# ------------------------------------------------------------------
# 3. Sample every band at every point, in one request
# ------------------------------------------------------------------
stack = ee.Image.cat([s1_stack, s2_stack, orbit_band]).unmask(-9999)   # -9999 = "no data"
print("Sampling 61 bands at every point (this can take a few minutes)...")
samples = stack.sampleRegions(
    collection=points, properties=["mandal"], scale=10, geometries=True, tileScale=4
).getInfo()["features"]

records = []
for f in samples:
    row = dict(f["properties"])
    row["lon"], row["lat"] = f["geometry"]["coordinates"]
    records.append(row)
df = pd.DataFrame(records).replace(-9999, np.nan)
df.insert(0, "point_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])


# ------------------------------------------------------------------
# 4. Evidence and a proposed label for every point
# ------------------------------------------------------------------
def evidence(row):
    vh = pd.Series(row[cols("VH")].to_numpy(dtype=float))
    ndvi = pd.Series(row[cols("NDVI")].to_numpy(dtype=float))
    lswi = pd.Series(row[cols("LSWI")].to_numpy(dtype=float))

    # Radar: dip on the raw series, sustained rise on the smoothed series
    smooth = vh.rolling(3, center=True, min_periods=1).median()
    previous = vh.shift(1).rolling(2, min_periods=1).median()
    dips = [i for i in range(N_BINS) if vh[i] <= previous[i] - DIP_DB]
    radar_paddy = [
        i for i in dips
        if smooth.iloc[i + 1:i + 1 + RISE_BINS].notna().any()
        and smooth.iloc[i + 1:i + 1 + RISE_BINS].max() - vh[i] >= RISE_DB
    ]

    # Optical: flood signal (water) and dry signal, clear bins only
    clear = ndvi.notna()
    flood = [i for i in FLOOD_SEASON_BINS if clear[i] and lswi[i] + FLOOD_MARGIN >= ndvi[i]]
    dry = [i for i in range(N_BINS) if clear[i] and lswi[i] + FLOOD_MARGIN < ndvi[i]]

    ndvi_max = ndvi.max()
    early_ndvi = ndvi.iloc[0:3].median()
    n_clear = int(clear.sum())

    if n_clear < MIN_CLEAR_BINS:
        label, conf, why = "Unknown", "low", "too few clear optical bins"
    elif radar_paddy:
        near = {j for b in radar_paddy for j in range(b - NEAR_BINS, b + NEAR_BINS + 1)}
        after = {j for b in radar_paddy for j in range(b + 1, b + NEAR_BINS + 1)}
        if near & set(flood):
            label, conf, why = "Paddy", "high", "radar dip-and-rise + optical flood within 3 weeks"
        elif after & set(dry):
            label, conf, why = "Review", "low", "radar dip but optical shows dry soil soon after"
        elif not any(clear[j] for j in near if 0 <= j < N_BINS):
            label, conf, why = "Paddy", "medium", "radar dip-and-rise; no clear optical data near the dip"
        else:
            label, conf, why = "Review", "low", "radar dip-and-rise; optical inconclusive"
    elif flood:
        label, conf, why = "Review", "low", "optical flood signal without a radar dip-and-rise"
    elif pd.notna(early_ndvi) and early_ndvi >= EARLY_GREEN_NDVI:
        label, conf, why = "Review", "low", "already green in June: orchard or perennial?"
    elif ndvi_max < FALLOW_MAX_NDVI:
        label, conf, why = "Fallow", "high" if n_clear >= 6 else "medium", "no green-up all season"
    elif ndvi_max >= CROP_MAX_NDVI:
        label, conf, why = "Other kharif crop", "high", "clear green-up, no flooding"
    else:
        label, conf, why = "Other kharif crop", "medium", "moderate green-up, no flooding"

    return pd.Series({
        "label": label,
        "confidence": conf,
        "reason": why,
        "ndvi_max": round(float(ndvi_max), 2) if pd.notna(ndvi_max) else np.nan,
        "ndvi_peak": bin_label(int(ndvi.idxmax())) if clear.any() else "",
        "clear_bins": n_clear,
        "radar_dips": ", ".join(bin_label(i) for i in dips),
        "optical_flood": ", ".join(bin_label(i) for i in flood),
    })


ev = df.apply(evidence, axis=1)
series_cols = cols("VH") + cols("VV") + cols("NDVI") + cols("LSWI")
out = pd.concat([df[["point_id", "mandal", "orbit", "lon", "lat"]], ev, df[series_cols]], axis=1)

# ------------------------------------------------------------------
# 5. Save (CSV for Python, GeoPackage for QGIS) and summarise
# ------------------------------------------------------------------
out.to_csv(PROCESSED / "candidates_kharif2025.csv", index=False)
gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["lon"], out["lat"]), crs="EPSG:4326")
gdf.to_file(PROCESSED / "candidates_kharif2025.gpkg", layer="candidates", driver="GPKG")

pd.set_option("display.width", 200)
print("\nPROPOSED LABELS x CONFIDENCE")
print(pd.crosstab(out["label"], out["confidence"], margins=True, margins_name="total"))
print("\nPROPOSED LABELS x RADAR ORBIT")
print(pd.crosstab(out["label"], out["orbit"]))
print("\nREASONS")
print(out["reason"].value_counts().to_string())
print("\nSaved data/processed/candidates_kharif2025.csv and .gpkg")
