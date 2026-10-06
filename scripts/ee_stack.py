"""Shared Earth Engine recipe: the 15-bin season stack (VH, VV, NDVI, LSWI). Mirrors script 06."""
import math
from datetime import date, timedelta

import ee

BIN_DAYS, N_BINS = 12, 15
CLEAR_THRESHOLD = 0.60
SMOOTH_RADIUS_M, EDGE_BUFFER_M = 15, 30
PREFERRED_ORBIT, OTHER_ORBIT, MIN_ORBIT_BINS = 165, 92, 10


def bin_starts(year=2025):
    start = date(year, 6, 1)
    return [start + timedelta(days=BIN_DAYS * i) for i in range(N_BINS)]


def cropland_mask():
    """WorldCover 2021 cropland, shrunk by 30 m to avoid edges (D6)."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return wc.eq(40).focal_min(radius=EDGE_BUFFER_M, kernelType="circle", units="meters")


def _binned(col, empty, bands, starts):
    images = []
    for i, s in enumerate(starts):
        d = ee.Date(s.isoformat())
        one = col.filterDate(d, d.advance(BIN_DAYS, "day")).merge(ee.ImageCollection([empty])).median()
        images.append(one.rename([f"{b}_{i:02d}" for b in bands]))
    return ee.Image.cat(images)


def _s1_prepare(img):
    """sigma0 -> gamma0, averaged in linear power over a ~30 m circle (D4)."""
    cos_theta = img.select("angle").multiply(math.pi / 180).cos()
    g0_db = img.select(["VH", "VV"]).subtract(cos_theta.log10().multiply(10))
    linear = g0_db.multiply(math.log(10) / 10).exp()
    smooth = linear.focal_mean(radius=SMOOTH_RADIUS_M, kernelType="circle", units="meters")
    back = smooth.log10().multiply(10).rename(["VH", "VV"]).toFloat()
    return ee.Image(back.copyProperties(img, ["system:time_start"]))


def _s2_prepare(img):
    clear = img.select("cs_cdf").gte(CLEAR_THRESHOLD)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    lswi = img.normalizedDifference(["B8", "B11"]).rename("LSWI")
    idx = ndvi.addBands(lswi).updateMask(clear)
    smooth = idx.focal_mean(radius=SMOOTH_RADIUS_M, kernelType="circle", units="meters")
    return ee.Image(smooth.rename(["NDVI", "LSWI"]).toFloat().copyProperties(img, ["system:time_start"]))


def season_stack(aoi, year=2025):
    """Returns (60-band stack, orbit band). Change year to map another season (D3)."""
    starts = bin_starts(year)
    end = starts[-1] + timedelta(days=BIN_DAYS)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
          .filterDate(starts[0].isoformat(), end.isoformat())
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")))
    empty1 = ee.Image.constant([0, 0]).rename(["VH", "VV"]).toFloat().updateMask(ee.Image(0))

    def orbit_stack(orb):
        return _binned(s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orb)).map(_s1_prepare),
                       empty1, ["VH", "VV"], starts)

    pref, other = orbit_stack(PREFERRED_ORBIT), orbit_stack(OTHER_ORBIT)
    in_pref = pref.select("VH_.*").reduce(ee.Reducer.count()).unmask(0).gte(MIN_ORBIT_BINS)
    s1_stack = ee.ImageCollection([other.updateMask(in_pref.Not()),
                                   pref.updateMask(in_pref)]).mosaic()
    orbit_band = ee.Image.constant(OTHER_ORBIT).where(in_pref, PREFERRED_ORBIT).rename("orbit").toInt()

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
          .filterDate(starts[0].isoformat(), end.isoformat())
          .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"])
          .map(_s2_prepare))
    empty2 = ee.Image.constant([0, 0]).rename(["NDVI", "LSWI"]).toFloat().updateMask(ee.Image(0))
    return s1_stack.addBands(_binned(s2, empty2, ["NDVI", "LSWI"], starts)), orbit_band


def gap_fill(stack, prefix):
    """Linear interpolation along the 15 bins; ends take the nearest value (as pandas interpolate)."""
    bands = [stack.select(f"{prefix}_{i:02d}") for i in range(N_BINS)]
    big = ee.Image.constant(1e6).toFloat()
    fwd_v, fwd_d, cur_v, cur_d = [], [], ee.Image.constant(0).toFloat(), big
    for b in bands:                                   # forward: last valid value and its distance
        valid = b.mask()
        cur_v = cur_v.where(valid, b.unmask(0))
        cur_d = cur_d.add(1).where(valid, 0)
        fwd_v.append(cur_v); fwd_d.append(cur_d)
    bwd_v, bwd_d, cur_v, cur_d = [None] * N_BINS, [None] * N_BINS, ee.Image.constant(0).toFloat(), big
    for i in reversed(range(N_BINS)):                 # backward: next valid value and its distance
        valid = bands[i].mask()
        cur_v = cur_v.where(valid, bands[i].unmask(0))
        cur_d = cur_d.add(1).where(valid, 0)
        bwd_v[i], bwd_d[i] = cur_v, cur_d
    out = []
    for i, b in enumerate(bands):
        denom = fwd_d[i].add(bwd_d[i]).max(1)         # weighted by distance = linear interpolation
        interp = fwd_v[i].multiply(bwd_d[i]).add(bwd_v[i].multiply(fwd_d[i])).divide(denom)
        out.append(interp.where(b.mask(), b.unmask(0)).rename(f"{prefix}_{i:02d}").toFloat())
    return ee.Image.cat(out)
