# Decision log

## D1: Study area uses two boundary vintages (2026-09-30)

**Context.** The downloaded boundaries (Tank Information System, repo last updated April 2016)
show the undivided Nalgonda district: 14,234 km², 59 mandals. Telangana reorganised its
districts on 11 October 2016, splitting Nalgonda into Nalgonda, Suryapet and Yadadri
Bhuvanagiri. The current Nalgonda boundary (geoBoundaries, representing 2021) measures
7,180 km² (official figure: 7,122 km²).

**Decision.** Run all satellite processing once over the pre-2016 extent, which contains
99.2% of the 2021 district. Report results for both boundaries, and show both on the dashboard.

**Consequences.**
- Comparisons with current official crop statistics use the 2021 boundary.
- Pre-2016 results allow comparison with older statistics published for the undivided district.
- The 0.8% mismatch between the two sources (boundary slivers) is documented, not hidden.

**Why.** The dashboard deploys from this repository and needs its data; the 2016 source
repository could disappear; the files are well under GitHub's 100 MB limit.
<<<<<<< HEAD
=======
## D2: Small vector data is kept in the repository (2026-09-30)

**Decision.** Commit the raw boundary files (about 19 MB) and small processed vector outputs
(GeoPackage, CSV). Keep satellite imagery and large rasters out of Git; scripts regenerate them.

**Why.** The dashboard deploys from this repository and needs its data; the 2016 source
repository could disappear; the files are well under GitHub's 100 MB limit.

## D3: Train on kharif 2025, test on kharif 2026

- **Date:** 2026-09-30
- **Decision:** Reference labels and model training use kharif 2025 (Jun–Nov 2025). Kharif 2026 is held out entirely as a temporal test set.
- **Why:** A random train/test split inside one season overstates accuracy, because neighbouring pixels from the same year look alike. Testing on a different year measures whether the model survives changes in rainfall, sowing dates and crop mix (temporal drift).
- **Constraint:** On the decision date the 2026 season was still in progress. Phase 1 tests on Jun–Sep 2026 (early-season mapping). Phase 2 repeats the test on the full season once Nov 2026 imagery is available.
- **Evidence behind the season design (Step 2.4):** Sentinel-2 averaged only 2.8–3.6 cloud-free views per pixel per month during Jul–Oct 2025 (7.9 in Nov). Optical data alone is too sparse in the key growth months, so Sentinel-1 radar will be added.
>>>>>>> 1ef6a7a (Explore Sentinel-2 kharif 2025 cloud cover and NDVI; add decision D3)

## D4: Sentinel-1 geometry: one relative orbit per pixel, in gamma0

- **Date:** 2026-09-30
- **Evidence:** In kharif 2025 only descending IW scenes cover the district, from two relative orbits: 165 (west, 71.1% of the district) and 92 (east, 59.1%), which overlap by about 30%. Mixing them in one time series made values zigzag every 5 days, by up to 12 dB at the dam-wall control point.
- **Decision:** Every pixel's time series comes from a single relative orbit and orbits are never mixed. Backscatter is converted to gamma0 (sigma0 minus 10·log10 cos θ). The orbit number is kept with each sample. Training labels are drawn from both orbit zones, and accuracy is reported for each zone. The choice of orbit inside the overlap is made at the export step.
- **Check:** At a cropland point seen by both orbits, gamma0 cut the gap between the orbits from 1.9 to 1.1 dB. It did not help for structures (dam wall −8.5 → −7.8 dB; town +2.8 → +3.6 dB), which are not mapped. With only one cropland point this is limited evidence, to be retested on more points.
- **Consequence:** Prefer features measured relative to each pixel's own series (amplitude, dip depth, rise) over absolute backscatter levels.
