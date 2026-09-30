"""Step 2.5d: Sentinel-1 time series per field, one relative orbit at a time, kharif 2025."""
import math

import ee
import pandas as pd
from config import BUFFER_M, EE_PROJECT, FIELDS, INTERIM

ee.Initialize(project=EE_PROJECT)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
START, END = "2025-06-01", "2025-12-01"
PASS = "DESCENDING"
BUFFER_M = 30        # radius (m) of the circle averaged at each point
DIP_DB = 4.0         # flood dip: VH at least this far below the previous two dates
RISE_DB = 5.0        # ...followed by a smoothed rise of at least this much
RISE_DAYS = 90       # ...within this many days

# label: (latitude, longitude), exactly as copied from Google Maps
FIELDS = {
    "Paddy A": (16.906201926363888, 79.51326660858167),
    "Paddy B": (16.903981848266753, 79.50420212162051),
    "Paddy C": (16.900684783885, 79.49293916943586),
    "Dry": (16.92075352173126, 79.38121341757046),
    "Dry 2 (Devarakonda)": (16.68857435715174, 78.93937817300532),
    "CONTROL dam wall": (16.57647824105806, 79.31269505232173),
    "CONTROL town (Nalgonda)": (17.051762780036384, 79.26482294730344),
}

# ------------------------------------------------------------------
# Data sources
# ------------------------------------------------------------------
s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterDate(START, END)
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("orbitProperties_pass", PASS))
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
# Earth Engine side
# ------------------------------------------------------------------
def to_gamma0(img):
    """sigma0 -> gamma0 (dB): removes most of the brightness difference caused by viewing angle."""
    cos_theta = img.select("angle").multiply(math.pi / 180).cos()
    correction = cos_theta.log10().multiply(10)
    g0 = img.select(["VV", "VH"]).subtract(correction).rename(["VVg", "VHg"])
    out = img.select(["VV", "VH"]).addBands(g0)
    return ee.Image(out.copyProperties(img, ["system:time_start", "relativeOrbitNumber_start"]))


def land_check(field):
    """What do two independent global maps say is at this spot?"""
    info = ee.Dictionary({
        "wc": worldcover.reduceRegion(ee.Reducer.mode(), field, 10).get("Map"),
        "occ": water_history.reduceRegion(ee.Reducer.mean(), field, 30).get("occurrence"),
    }).getInfo()
    return WC_CLASSES.get(info.get("wc"), "unknown"), info.get("occ")


def radar_series(field):
    """Mean sigma0 and gamma0 in the field circle, for every date and orbit."""
    def one_date(img):
        v = img.reduceRegion(ee.Reducer.mean(), field, 10)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "orbit": img.get("relativeOrbitNumber_start"),
            "VV": v.get("VV"), "VH": v.get("VH"),
            "VVg": v.get("VVg"), "VHg": v.get("VHg"),
        })

    col = s1.filterBounds(field).map(to_gamma0).sort("system:time_start")
    rows = [f["properties"] for f in col.map(one_date).getInfo()["features"]]
    df = pd.DataFrame(rows, columns=["date", "orbit", "VV", "VH", "VVg", "VHg"]).dropna()
    df["orbit"] = df["orbit"].astype(int)
    return df.groupby(["orbit", "date"], as_index=False).mean()


# ------------------------------------------------------------------
# Python side: one clean single-orbit series -> smoothed curve + features
# ------------------------------------------------------------------
def analyse(track):
    t = track.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(t["date"])

    # Smoothed curve: the median of each date and its two neighbours kills one-date spikes
    t["VH_smooth"] = t["VHg"].rolling(3, center=True, min_periods=1).median()

    # Flood dip, on the RAW series: a sudden drop below the previous two dates
    previous = t["VHg"].shift(1).rolling(2, min_periods=1).median()
    t["dip"] = t["VHg"] <= previous - DIP_DB

    # Paddy-like: a dip followed by a sustained rise
    paddy_like = False
    for i in t.index[t["dip"]]:
        window = (dates > dates[i]) & (dates <= dates[i] + pd.Timedelta(days=RISE_DAYS))
        later = t.loc[window, "VH_smooth"]
        if not later.empty and later.max() - t.loc[i, "VHg"] >= RISE_DB:
            paddy_like = True
            break

    rise = t["VH_smooth"].diff()
    feats = {
        "dates": len(t),
        "VH_min": t["VH_smooth"].min(),
        "VH_max": t["VH_smooth"].max(),
        "amplitude": t["VH_smooth"].max() - t["VH_smooth"].min(),
        "peak_date": t.loc[t["VH_smooth"].idxmax(), "date"],
        "biggest_rise": rise.max(),
        "rise_ends": t.loc[rise.idxmax(), "date"] if rise.notna().any() else None,
        "dip_dates": ", ".join(t.loc[t["dip"], "date"].str[5:]) or "-",
        "paddy_like": paddy_like,
    }
    return t, feats


# ------------------------------------------------------------------
# Run every field
# ------------------------------------------------------------------
all_series, summary = [], []

for label, (lat, lon) in FIELDS.items():
    field = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)   # Earth Engine wants [lon, lat]
    lc_name, occ = land_check(field)
    df = radar_series(field)

    print(f"\n{'=' * 78}\n{label} at {lat:.4f}, {lon:.4f} | WorldCover: {lc_name} | water: {fmt(occ, 3)}%")
    if df.empty:
        print("  no radar data here")
        continue

    for orbit, track in df.groupby("orbit"):
        t, feats = analyse(track)
        print(f"  -- orbit {orbit}: {len(t)} dates (gamma0, dB) --")
        print("  date       | VV g0  | VH g0  | VH smooth")
        for r in t.itertuples():
            bar = "#" * max(0, int(r.VHg + 30))
            flag = "  <- dip" if r.dip else ""
            print(f"  {r.date} | {r.VVg:6.1f} | {r.VHg:6.1f} | {r.VH_smooth:6.1f} {bar}{flag}")
        summary.append({"field": label, "orbit": orbit, "worldcover": lc_name, **feats})
        t.insert(0, "field", label)
        all_series.append(t)

series = pd.concat(all_series)

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
pd.set_option("display.width", 250)
print(f"\n{'=' * 78}\nSUMMARY (features from smoothed gamma0 VH, dB)")
print(pd.DataFrame(summary).set_index(["field", "orbit"]).round(1).to_string())

# ------------------------------------------------------------------
# Track offset: same spot, two orbits. Does gamma0 close the gap?
# ------------------------------------------------------------------
print("\nTRACK OFFSET for points seen by both orbits (season mean, orbit A minus orbit B)")
means = series.groupby(["field", "orbit"])[["VH", "VHg"]].mean()
for name, grp in means.groupby(level="field"):
    if len(grp) == 2:
        (a, ra), (b, rb) = [(idx[1], row) for idx, row in grp.iterrows()]
        print(f"  {name:25s} | sigma0 VH gap: {ra.VH - rb.VH:+5.1f} dB "
              f"| gamma0 VH gap: {ra.VHg - rb.VHg:+5.1f} dB   (orbit {a} minus {b})")

out = INTERIM / "s1_field_series_kharif2025.csv"
series.to_csv(out, index=False)
print(f"\nSaved {out.relative_to(INTERIM.parents[1])}")