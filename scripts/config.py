"""Project-wide settings. Every script imports from here, so paths and IDs live in one place."""
from pathlib import Path

EE_PROJECT = "nalgonda-crop-mapping"   # Google Cloud project used for Earth Engine

# Folder paths, worked out from where this file sits, so they work on any computer
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"   # geographic, degrees: for storing data and web maps
CRS_UTM = "EPSG:32644"    # UTM zone 44N, metres: for measuring area and distance
# Test fields and control points: label -> (latitude, longitude), as copied from Google Maps
FIELDS = {
    "Paddy A": (16.906201926363888, 79.51326660858167),
    "Paddy B": (16.903981848266753, 79.50420212162051),
    "Paddy C": (16.900684783885, 79.49293916943586),
    "Dry": (16.92075352173126, 79.38121341757046),
    "Dry 2 (Devarakonda)": (16.68857435715174, 78.93937817300532),
    "CONTROL dam wall": (16.57647824105806, 79.31269505232173),
    "CONTROL town (Nalgonda)": (17.051762780036384, 79.26482294730344),
}
BUFFER_M = 30   # radius (m) of the circle averaged at each point