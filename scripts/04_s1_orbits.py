"""Which Sentinel-1 relative orbits cover the district, and how much of it?"""
import ee
import geopandas as gpd
from config import EE_PROJECT, PROCESSED

ee.Initialize(project=EE_PROJECT)

aoi_gdf = gpd.read_file(PROCESSED / "study_area.gpkg", layer="nalgonda_district_2016")
aoi = ee.Geometry(aoi_gdf.geometry.union_all().simplify(0.001).__geo_interface__)

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate("2025-06-01", "2025-12-01")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
)

orbits = s1.aggregate_histogram("relativeOrbitNumber_start").getInfo()
aoi_area = aoi.area(100)

for orb, n in sorted(orbits.items()):
    orbit_no = int(float(orb))
    track = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit_no)).sort("system:time_start")
    footprint = track.geometry().dissolve(100)
    share = footprint.intersection(aoi, 100).area(100).divide(aoi_area).multiply(100).getInfo()
    dates = (
        track.aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("MM-dd"))
        .distinct()
        .getInfo()
    )
    print(f"Relative orbit {orbit_no:3d} | scenes: {n:2d} | covers {share:5.1f}% of district "
          f"| {len(dates)} dates, first: {dates[:4]}")