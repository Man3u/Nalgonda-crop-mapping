"""
Step 2.2: Build the study-area layers for both boundary vintages.

Inputs : data/raw/boundaries/districts.json and mandals.json (pre-October-2016)
         geoBoundaries India ADM2 v6, via Earth Engine (represents 2021)
Output : data/processed/study_area.gpkg with three layers
"""
import ee
import geopandas as gpd

from config import EE_PROJECT, RAW, PROCESSED, CRS_WGS84, CRS_UTM

OUT = PROCESSED / "study_area.gpkg"

# 1. Old (pre-2016) boundaries from the downloaded files
districts = gpd.read_file(RAW / "boundaries" / "districts.json")
mandals = gpd.read_file(RAW / "boundaries" / "mandals.json")

old_district = districts[districts["D_N"] == "NALGONDA"].copy()
old_mandals = mandals[mandals["D_N"] == "NALGONDA"].copy()

# 2. Current (2021) boundary from geoBoundaries, filtered on Earth Engine's servers
ee.Initialize(project=EE_PROJECT)
adm2 = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM2")
matches = (adm2
           .filter(ee.Filter.eq("shapeGroup", "IND"))
           .filter(ee.Filter.eq("shapeName", "Nalgonda")))
print("Matches found in geoBoundaries:", matches.size().getInfo())

new_district = gpd.GeoDataFrame.from_features(matches.getInfo()["features"], crs=CRS_WGS84)

# 3. Repair any invalid geometry before measuring or overlaying
for gdf in (old_district, old_mandals, new_district):
    gdf["geometry"] = gdf.make_valid()

# 4. Area must be measured in metres, so reproject to UTM 44N first
def area_km2(gdf):
    return gdf.to_crs(CRS_UTM).area.sum() / 1e6

print(f"Old Nalgonda (pre-2016): {area_km2(old_district):,.0f} km2, {len(old_mandals)} mandals")
print(f"New Nalgonda (2021):     {area_km2(new_district):,.0f} km2")

# 5. How much of the new district lies inside the old one?
new_utm = new_district.to_crs(CRS_UTM)
old_utm = old_district.to_crs(CRS_UTM)
inside = gpd.overlay(new_utm, old_utm, how="intersection")
share = inside.area.sum() / new_utm.area.sum()
print(f"Share of new district inside the old district: {share:.1%}")

# 6. Save all three layers into one GeoPackage (kept in WGS 84)
old_district.to_file(OUT, layer="nalgonda_district_2016", driver="GPKG")
old_mandals.to_file(OUT, layer="nalgonda_mandals_2016", driver="GPKG")
new_district.to_file(OUT, layer="nalgonda_district_2021", driver="GPKG")
print("Saved:", OUT)