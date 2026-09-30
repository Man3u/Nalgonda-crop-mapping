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