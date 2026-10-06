"""Step 4.1: Map the district. Train an Earth Engine Random Forest, verify it against
scikit-learn, classify all cropland, report mandal areas, and start the GeoTIFF export."""
import json

import ee
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from config import EE_PROJECT, PROCESSED, ROOT
from ee_stack import N_BINS, cropland_mask, gap_fill, season_stack
from review_tools import CLASSES, build_features

ee.Initialize(project=EE_PROJECT)
CODE = {"Paddy": 1, "Other kharif crop": 2, "Perennial": 3}
NAME = {v: k for k, v in CODE.items()}
pd.set_option("display.width", 200)

# ------------------------------------------------------------------
# 1. Training table (identical values to the scikit-learn model)
# ------------------------------------------------------------------
cand = pd.read_csv(PROCESSED / "labels_v2_kharif2025.csv")
truth = pd.read_csv(PROCESSED / "training_set_v2.csv")[["point_id", "truth", "is_gold"]]
df = cand.merge(truth, on="point_id")
X = build_features(df)
FEATURES = list(X.columns)
use = df["truth"].isin(CLASSES)

rows = X[use].round(4).to_dict("records")
codes = df.loc[use, "truth"].map(CODE).tolist()
fc = ee.FeatureCollection([ee.Feature(None, {**r, "cls": c}) for r, c in zip(rows, codes)])
clf = ee.Classifier.smileRandomForest(numberOfTrees=500, minLeafPopulation=2, seed=42) \
        .train(fc, "cls", FEATURES)
print(f"Earth Engine Random Forest trained on {int(use.sum())} points, {len(FEATURES)} features")

# ------------------------------------------------------------------
# 2. CHECK 1 - same features, two libraries: do they agree?
# ------------------------------------------------------------------
gold = df["is_gold"] & df["truth"].isin(CLASSES)
gold_rows = X[gold].round(4).to_dict("records")
ee_pred = ee.FeatureCollection([ee.Feature(None, r) for r in gold_rows]) \
            .classify(clf).aggregate_array("classification").getInfo()
ee_pred = pd.Series(ee_pred, index=X[gold].index).map(NAME)
sk = joblib.load(ROOT / "models" / "rf_v2.joblib")
sk_pred = pd.Series(sk["model"].predict(X[gold][sk["features"]]), index=X[gold].index)
y = df.loc[gold, "truth"]
print(f"\nCHECK 1  Earth Engine vs scikit-learn on {int(gold.sum())} reviewed points, identical features")
print(f"  the two classifiers agree with each other: {100 * (ee_pred == sk_pred).mean():.1f}%")
print(f"  accuracy vs reviewer - scikit-learn {100 * (sk_pred == y).mean():.1f}%, "
      f"Earth Engine {100 * (ee_pred == y).mean():.1f}%")
print("  (scikit-learn has seen these points in training, so both numbers are optimistic;"
      " only the agreement matters here)")

# ------------------------------------------------------------------
# 3. Build the image stack and CHECK 2 - does it reproduce the CSV values?
# ------------------------------------------------------------------
aoi_gdf = gpd.read_file(PROCESSED / "study_area.gpkg", layer="nalgonda_district_2016")
aoi = ee.Geometry(aoi_gdf.geometry.union_all().simplify(0.001).__geo_interface__)
stack, _ = season_stack(aoi, year=2025)
filled = ee.Image.cat([gap_fill(stack, p) for p in ["VH", "VV", "NDVI", "LSWI"]])


def band(prefix, i):
    return filled.select(f"{prefix}_{i:02d}")


ndvi = [band("NDVI", i) for i in range(N_BINS)]
lswi = [band("LSWI", i) for i in range(N_BINS)]
vh = [band("VH", i) for i in range(N_BINS)]
vh_min = ee.ImageCollection([b.rename("x") for b in vh]).min()
vh_max = ee.ImageCollection([b.rename("x") for b in vh]).max()
suffix, run = [None] * N_BINS, None
for i in reversed(range(N_BINS)):
    run = vh[i].rename("x") if run is None else run.max(vh[i].rename("x"))
    suffix[i] = run
after_min = ee.ImageCollection(
    [suffix[i].updateMask(vh[i].subtract(vh_min).abs().lt(1e-4)) for i in range(N_BINS)]).max()

extra = ee.Image.cat([
    ee.ImageCollection([b.rename("x") for b in ndvi[0:3]]).mean().rename("ndvi_june"),
    ee.ImageCollection([b.rename("x") for b in ndvi[3:8]]).min().rename("ndvi_min_jul_aug"),
    ee.ImageCollection([b.rename("x") for b in ndvi]).max().rename("ndvi_max"),
    ee.ImageCollection([lswi[i].subtract(ndvi[i]).rename("x") for i in range(10)]).max().rename("flood_index"),
    vh_min.rename("vh_min"),
    vh_max.subtract(vh_min).rename("vh_amplitude"),
    after_min.subtract(vh_min).rename("vh_rise_after_min"),
]).toFloat()
image = filled.addBands(extra).select(FEATURES)

pts = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"pid": r.point_id})
    for r in df[gold].itertuples()])
sampled = image.sampleRegions(collection=pts, properties=["pid"], scale=10, tileScale=4).getInfo()["features"]
img_vals = pd.DataFrame([f["properties"] for f in sampled]).set_index("pid")
csv_vals = X[gold].set_index(df.loc[gold, "point_id"])
common = img_vals.index.intersection(csv_vals.index)
diff = (img_vals.loc[common, FEATURES] - csv_vals.loc[common, FEATURES]).abs()
print(f"\nCHECK 2  image stack vs the CSV values, {len(common)} points")
print(f"  median absolute difference across all {len(FEATURES)} features: {diff.values.flatten().mean():.4f}")
print("  largest differences:")
print(diff.mean().sort_values(ascending=False).head(5).round(3).to_string())
img_pred = pd.Series(sk["model"].predict(img_vals.loc[common, sk["features"]]), index=common)
csv_pred = pd.Series(sk["model"].predict(csv_vals.loc[common, sk["features"]]), index=common)
print(f"  same prediction from image features and CSV features: {100 * (img_pred == csv_pred).mean():.1f}%")

# ------------------------------------------------------------------
# 4. Classify the district, and start the GeoTIFF export first
# ------------------------------------------------------------------
classified = image.classify(clf).rename("crop").updateMask(cropland_mask()).clip(aoi)

img_task = ee.batch.Export.image.toDrive(
    image=classified.toByte(), description="nalgonda_crop_2025_10m",
    folder="nalgonda_crop_mapping", fileNamePrefix="nalgonda_crop_2025_10m",
    region=aoi, scale=10, crs="EPSG:32644", maxPixels=1e13)
img_task.start()
print(f"\nMAP EXPORT started (task {img_task.id}) - 10 m GeoTIFF to Google Drive")

# ------------------------------------------------------------------
# 5. Quick approximate district shares (coarse scale, so it answers now)
# ------------------------------------------------------------------
hist = classified.reduceRegion(reducer=ee.Reducer.frequencyHistogram(), geometry=aoi,
                               scale=500, maxPixels=1e9, bestEffort=True).getInfo()["crop"]
counts = pd.Series({NAME[int(float(k))]: v for k, v in hist.items()})
share = (100 * counts / counts.sum()).round(1)
print("\nAPPROXIMATE DISTRICT SHARES (500 m sample of the 10 m map, mapped cropland only)")
print(pd.DataFrame({"share %": share}).to_string())

# ------------------------------------------------------------------
# 6. Mandal areas as a batch task (no five-minute limit)
# ------------------------------------------------------------------
mandals = gpd.read_file(PROCESSED / "study_area.gpkg", layer="nalgonda_mandals_2016").to_crs(4326)
print("\nMandal layer columns:", list(mandals.columns))
name_col = next((c for c in mandals.columns if mandals[c].dtype == object and c != "geometry"), None)
m_small = mandals[["geometry"]].copy()
m_small["mandal"] = mandals[name_col].astype(str) if name_col else (mandals.index + 1).astype(str)
m_small["geometry"] = m_small.geometry.simplify(0.0005)
mandal_fc = ee.FeatureCollection(json.loads(m_small.to_json()))

ha = ee.Image.pixelArea().divide(10000)
area_img = ee.Image.cat([classified.eq(c).multiply(ha).rename(n.split()[0].lower())
                         for n, c in CODE.items()])
stats_fc = area_img.reduceRegions(collection=mandal_fc, reducer=ee.Reducer.sum(),
                                  scale=20, tileScale=16)
tbl_task = ee.batch.Export.table.toDrive(
    collection=stats_fc.select([".*"], None, False), description="nalgonda_mandal_area_2025",
    folder="nalgonda_crop_mapping", fileNamePrefix="nalgonda_mandal_area_2025", fileFormat="CSV")
tbl_task.start()
print(f"MANDAL AREA EXPORT started (task {tbl_task.id}) - CSV to the same Drive folder")
print("\nWatch both at https://code.earthengine.google.com/tasks")
print("When the CSV finishes, download it to data/processed/ and we will read it from there.")
