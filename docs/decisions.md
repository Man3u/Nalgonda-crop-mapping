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

## D4: Sentinel-1 geometry: one relative orbit per pixel, in gamma0

- **Date:** 2026-09-30
- **Evidence:** In kharif 2025 only descending IW scenes cover the district, from two relative orbits: 165 (west, 71.1% of the district) and 92 (east, 59.1%), which overlap by about 30%. Mixing them in one time series made values zigzag every 5 days, by up to 12 dB at the dam-wall control point.
- **Decision:** Every pixel's time series comes from a single relative orbit and orbits are never mixed. Backscatter is converted to gamma0 (sigma0 minus 10·log10 cos θ). The orbit number is kept with each sample. Training labels are drawn from both orbit zones, and accuracy is reported for each zone. The choice of orbit inside the overlap is made at the export step.
- **Check:** At a cropland point seen by both orbits, gamma0 cut the gap between the orbits from 1.9 to 1.1 dB. It did not help for structures (dam wall −8.5 → −7.8 dB; town +2.8 → +3.6 dB), which are not mapped. With only one cropland point this is limited evidence, to be retested on more points.
- **Consequence:** Prefer features measured relative to each pixel's own series (amplitude, dip depth, rise) over absolute backscatter levels.

## D5: Reference labels come from two-sensor verification, not visual picking alone

- **Date:** 2026-09-30
- **Evidence:** Of four points picked as paddy from Google Maps imagery, only two were confirmed by their 2025 time series. A radar-only dip-and-rise rule flagged a rain-fed field ("Dry") as paddy. Sentinel-2 showed no standing water three weeks later (LSWI about 0 on 2 and 5 Aug 2025), so it was a false positive. For the two confirmed paddy fields, the radar flood dip and the optical flood signal (LSWI + 0.05 >= NDVI) fell within 7 to 9 days of each other.
- **Decision:** A candidate point becomes a label only after its kharif time series is checked with both sensors. Paddy is high confidence when the radar dip-and-rise and the optical flood signal occur within about 3 weeks of each other; medium when one sensor shows the signal and the other has no clear data in that window; rejected when they conflict. Every label records the evidence behind it.
- **Why:** No field survey is possible. Two independent physical signals (radar surface scattering, optical water absorption in the SWIR) are unlikely to agree by chance.
- **Definition and limitation:** "Paddy" means flooded, transplanted rice. Direct-seeded rice without a flooded phase cannot be separated from other crops with these data and falls under "other kharif crop".
- **Sampling:** Candidate points are kept away from field edges and boundaries, to avoid mixed pixels (see the dam-wall control).

## D6: Class scheme: Paddy / Other kharif crop / Fallow, inside cropland only

- **Date:** 2026-09-30
- **Decision:** Mapping is restricted to ESA WorldCover 2021 cropland (class 40), shrunk by 30 m to avoid field and land-cover edges. Inside it, three classes: **Paddy** (flooded, transplanted rice; D5 two-sensor evidence), **Other kharif crop** (clear green-up without flooding, NDVI peak >= 0.40), **Fallow** (no real green-up, NDVI peak < 0.40). Water, built-up and forest are masked out, not classified.
- **Why:** Every class must be verifiable from the satellite evidence we have. Cotton, maize and red gram cannot yet be told apart reliably, so they are grouped; unsupervised clustering will test whether the data supports splitting them.
- **Known risks:** WorldCover is from 2021, so some cropland may have changed since. Orchards (e.g. citrus) sit inside "cropland" but are green from June; the rules send them to Review instead of guessing.

## D7: Labelling rules v2 and revised classes (Paddy / Other kharif crop / Perennial)

- **Date:** 2026-10-01
- **Evidence (error analysis of 354 candidates, Step 3.1b):** Rules v1 sent 40% of points to Review, for three reasons. (1) Radar dips shared by many fields at once: 38% of "dip, then dry" points dipped in 13–24 June against 6% of paddy, and every group, orchards included, dropped 1.5–2 dB in 7–18 July. That is a common-mode signal (weather or scene), not farming. (2) 37 "optical flood only" points matched paddy on every measure (NDVI peak 0.81; flood signal in 31 Jul–11 Aug for 78% of them vs 0% of other crops) but had shallow radar dips, likely from mixed pixels. (3) 31 points were green from June with a stable radar curve: perennial vegetation. No point had an NDVI peak below 0.40 (0 of 354; by the rule of three, fallow is under 1% of core cropland).
- **Decision:** (a) Radar flood dips are detected on a local anomaly: VH minus the median VH of the 10 nearest candidate points on the same orbit, bin by bin. Crop growth is still measured on real VH. A local reference is used because the June event affected only about 11% of points (patchy rain). (b) Paddy is also accepted, at medium confidence, when an optical flood signal is followed by at least 5 dB of radar growth, without a sudden dip. (c) New class **Perennial**: NDVI at least 0.45 in June and a radar amplitude under 6 dB. (d) **Fallow** is dropped as a training class (no examples); the rule stays as a flag. Training classes are now Paddy, Other kharif crop and Perennial.
- **Guard against self-deception:** All thresholds are unchanged from v1. Rules v2 are judged by the independent visual review (Step 3.2), not by how small the Review group becomes. Labels are versioned: v1 is kept unchanged.
