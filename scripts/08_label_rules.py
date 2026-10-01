"""Step 3.1c: Labelling rules v2 (D7). Local radar anomaly, symmetric two-sensor paddy rule, Perennial class."""
import warnings
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from config import PROCESSED

# ------------------------------------------------------------------
# Settings (the physical thresholds are unchanged from v1)
# ------------------------------------------------------------------
START_DATE, BIN_DAYS, N_BINS = date(2025, 6, 1), 12, 15
K_NEIGHBOURS = 10                  # local reference = the 10 nearest points on the same orbit
DIP_DB, RISE_DB = 4.0, 5.0
RISE_BINS, NEAR_BINS = 7, 2        # about 90 days; about 3 weeks
FLOOD_MARGIN = 0.05
FALLOW_MAX_NDVI, CROP_MAX_NDVI = 0.40, 0.50
EARLY_GREEN_NDVI = 0.45            # green before the monsoon crops are sown
PERENNIAL_MAX_AMP_DB = 6.0         # ...and a radar curve that barely changes
MIN_CLEAR_BINS = 3

BIN_STARTS = [START_DATE + timedelta(days=BIN_DAYS * i) for i in range(N_BINS)]
FLOOD_SEASON_BINS = [i for i, d in enumerate(BIN_STARTS) if d < date(2025, 10, 1)]
KM_PER_DEG_LON, KM_PER_DEG_LAT = 106.4, 110.6     # at about 17 degrees north


def cols(prefix):
    return [f"{prefix}_{i:02d}" for i in range(N_BINS)]


def bin_label(i):
    return BIN_STARTS[i].strftime("%m-%d")


df = pd.read_csv(PROCESSED / "candidates_kharif2025.csv")


# ------------------------------------------------------------------
# 1. Remove the shared (common-mode) radar signal with a LOCAL reference
# ------------------------------------------------------------------
def local_reference(group):
    """For each point: the median VH of its K nearest neighbours on the same orbit, bin by bin."""
    xy = np.c_[group["lon"].to_numpy() * KM_PER_DEG_LON, group["lat"].to_numpy() * KM_PER_DEG_LAT]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)                         # a point is not its own neighbour
    nearest = np.argsort(dist, axis=1)[:, :K_NEIGHBOURS]
    vh = group[cols("VH")].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN bins just stay NaN
        ref = np.nanmedian(vh[nearest], axis=1)            # (points, neighbours, bins) -> (points, bins)
    return pd.DataFrame(ref, index=group.index, columns=cols("REF"))


ref = pd.concat([local_reference(g) for _, g in df.groupby("orbit")]).loc[df.index]
anom = pd.DataFrame(df[cols("VH")].to_numpy() - ref.to_numpy(), index=df.index, columns=cols("ANOM"))
df = pd.concat([df, anom], axis=1)


def series(row, prefix):
    return pd.Series(row[cols(prefix)].to_numpy(dtype=float))


def dip_bins(s):
    """Bins where the value drops at least DIP_DB below the median of the two previous bins."""
    previous = s.shift(1).rolling(2, min_periods=1).median()
    return [i for i in range(N_BINS) if s[i] <= previous[i] - DIP_DB]


# ------------------------------------------------------------------
# 2. Evidence and label, rules v2
# ------------------------------------------------------------------
def evidence(row):
    vh, anom = series(row, "VH"), series(row, "ANOM")
    ndvi, lswi = series(row, "NDVI"), series(row, "LSWI")

    smooth = vh.rolling(3, center=True, min_periods=1).median()   # growth is measured on the real VH
    amplitude = smooth.max() - smooth.min()

    def rises_after(i, base):
        later = smooth.iloc[i + 1:i + 1 + RISE_BINS]
        return bool(later.notna().any() and later.max() - base >= RISE_DB)

    dips = dip_bins(anom)                                           # dips are found on the anomaly
    radar_paddy = [i for i in dips if rises_after(i, vh[i])]

    clear = ndvi.notna()
    flood = [i for i in FLOOD_SEASON_BINS if clear[i] and lswi[i] + FLOOD_MARGIN >= ndvi[i]]
    dry = [i for i in range(N_BINS) if clear[i] and lswi[i] + FLOOD_MARGIN < ndvi[i]]
    flood_then_growth = [i for i in flood if rises_after(i, vh.iloc[max(0, i - 1):i + 2].min())]

    ndvi_max = ndvi.max()
    early_ndvi = ndvi.iloc[0:3].median()
    early_green = bool(pd.notna(early_ndvi) and early_ndvi >= EARLY_GREEN_NDVI)
    n_clear = int(clear.sum())

    if n_clear < MIN_CLEAR_BINS:
        label, conf, why = "Unknown", "low", "too few clear optical bins"
    elif early_green and amplitude < PERENNIAL_MAX_AMP_DB:
        label, conf, why = "Perennial", "medium", "green from June and stable radar all season"
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
    elif flood_then_growth:
        label, conf, why = "Paddy", "medium", "optical flood + radar growth afterwards (shallow dip)"
    elif flood:
        label, conf, why = "Review", "low", "optical flood without radar growth afterwards"
    elif early_green:
        label, conf, why = "Review", "low", "green in June but strongly seasonal radar"
    elif ndvi_max < FALLOW_MAX_NDVI:
        label, conf, why = "Fallow", "medium", "no green-up all season"
    elif ndvi_max >= CROP_MAX_NDVI:
        label, conf, why = "Other kharif crop", "high", "clear green-up, no flooding"
    else:
        label, conf, why = "Other kharif crop", "medium", "moderate green-up, no flooding"

    return pd.Series({
        "label": label,
        "confidence": conf,
        "reason": why,
        "ndvi_max": round(float(ndvi_max), 2) if pd.notna(ndvi_max) else np.nan,
        "ndvi_june": round(float(early_ndvi), 2) if pd.notna(early_ndvi) else np.nan,
        "vh_amplitude": round(float(amplitude), 1),
        "clear_bins": n_clear,
        "radar_dips": ", ".join(bin_label(i) for i in dips),
        "optical_flood": ", ".join(bin_label(i) for i in flood),
    })


ev = df.apply(evidence, axis=1)
keep = ["point_id", "mandal", "orbit", "lon", "lat"]
v1 = df[["label", "confidence"]].add_suffix("_v1")
series_cols = cols("VH") + cols("ANOM") + cols("VV") + cols("NDVI") + cols("LSWI")
out = pd.concat([df[keep], ev, v1, df[series_cols]], axis=1)

# ------------------------------------------------------------------
# 3. Did the anomaly remove the shared events? Compare dip timing, raw vs anomaly
# ------------------------------------------------------------------
pd.set_option("display.width", 200)


def dip_share(prefix):
    counts = pd.Series(0, index=[bin_label(i) for i in range(N_BINS)])
    for _, row in df.iterrows():
        for i in dip_bins(series(row, prefix)):
            counts.iloc[i] += 1
    return (100 * counts / len(df)).round(0).astype(int)


print("RADAR DIPS, % of ALL points per bin: raw VH (v1) vs local anomaly (v2)")
print(pd.DataFrame({"raw VH": dip_share("VH"), "anomaly": dip_share("ANOM")}).T.to_string())

print("\nHOW LABELS MOVED: v1 (rows) -> v2 (columns)")
print(pd.crosstab(out["label_v1"], out["label"], margins=True, margins_name="total"))

print("\nV2 LABELS x CONFIDENCE")
print(pd.crosstab(out["label"], out["confidence"], margins=True, margins_name="total"))

print("\nV2 LABELS x RADAR ORBIT")
print(pd.crosstab(out["label"], out["orbit"]))

print("\nV2 REASONS")
print(out["reason"].value_counts().to_string())

# ------------------------------------------------------------------
# 4. Save as a new version; v1 stays untouched for comparison
# ------------------------------------------------------------------
out.to_csv(PROCESSED / "labels_v2_kharif2025.csv", index=False)
gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["lon"], out["lat"]), crs="EPSG:4326")
gdf.to_file(PROCESSED / "labels_v2_kharif2025.gpkg", layer="labels_v2", driver="GPKG")
print("\nSaved data/processed/labels_v2_kharif2025.csv and .gpkg")
