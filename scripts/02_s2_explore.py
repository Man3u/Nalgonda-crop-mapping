"""
Step 2.4: Explore Sentinel-2 over Nalgonda (pre-2016 extent) for kharif 2025.

For each month (June to November 2025):
  - how many Sentinel-2 scenes exist
  - how many cloud-free views each pixel gets, on average
  - the district's mean NDVI from a cloud-masked median composite
It also prints a thumbnail link of each month's NDVI map.
"""
import ee
import geopandas as gpd

from config import EE_PROJECT, PROCESSED

ee.Initialize(project=EE_PROJECT)

# --- Study area: read our own boundary and send it to Earth Engine ---
aoi_gdf = gpd.read_file(PROCESSED / "study_area.gpkg", layer="nalgonda_district_2016")
outline = aoi_gdf.geometry.union_all().simplify(0.001)   # about 100 m tolerance, fewer vertices
aoi = ee.Geometry(outline.__geo_interface__)

# --- Sentinel-2 surface reflectance, linked to Cloud Score+ ---
CLEAR_THRESHOLD = 0.60   # keep pixels at least 60% likely to be clear

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(aoi)
      .filterDate("2025-06-01", "2025-12-01"))
cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
s2 = s2.linkCollection(cloud_score, ["cs_cdf"])

def clear_ndvi(img):
    """Compute NDVI and hide every pixel that is not clear."""
    clear = img.select("cs_cdf").gte(CLEAR_THRESHOLD)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi.updateMask(clear).copyProperties(img, ["system:time_start"])

ndvi_col = s2.map(clear_ndvi)
print("Sentinel-2 scenes over the district, Jun-Nov 2025:", s2.size().getInfo())
print()

# --- Month by month ---
months = [("Jun", 6), ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11)]
vis = {"min": 0, "max": 0.8, "palette": ["8c510a", "d8b365", "f6e8c3", "c7eae5", "5ab4ac", "01665e"]}

for name, m in months:
    start = ee.Date.fromYMD(2025, m, 1)
    month_col = ndvi_col.filterDate(start, start.advance(1, "month"))

    clear_views = month_col.count().unmask(0).rename("clear")   # clear looks per pixel
    median_ndvi = month_col.median().rename("NDVI")              # one clean image per month

    stats = clear_views.addBands(median_ndvi).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e9, bestEffort=True
    ).getInfo()

    ndvi_mean = stats.get("NDVI")
    ndvi_text = f"{ndvi_mean:.2f}" if ndvi_mean is not None else " n/a"
    print(f"{name} 2025 | scenes: {month_col.size().getInfo():3d} | "
          f"avg clear views per pixel: {stats['clear']:4.1f} | mean NDVI: {ndvi_text}")
    print("   map:", median_ndvi.clip(aoi).getThumbURL({**vis, "region": aoi, "dimensions": 600}))