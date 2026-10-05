"""Step 3.2c: Write the agreed review verdicts (from the review session log) into the blind review layer."""
import geopandas as gpd
from config import PROCESSED

ALLOWED = {"Paddy", "Other kharif crop", "Perennial", "Not cropland", "Can't tell"}

# review_id | label | confidence | note   (img = imagery inside the ring; card = 2025 time series)
VERDICTS = """
R001|Paddy|medium|CARD ONLY: own 2-bin dip Jul, flood hidden by cloud, wet canopy, harvest Nov
R002|Other kharif crop|low|CARD ONLY: zigzag radar (mixed spot?), peak .61, LSWI <= .2
R003|Paddy|high|CARD ONLY: LSWI > NDVI early Aug, low radar Jul-Aug, NDVI .86, LSWI .39
R004|Other kharif crop|medium|CARD ONLY: short crop, peak .68 mid-Aug, LSWI low
R005|Paddy|high|CARD ONLY: own deep dips Jul-Aug, LSWI > NDVI twice, NDVI .82
R006|Paddy|medium|img paddy land near stream; own dip late Jul, +10 dB, wet canopy
R007|Paddy|high|img paddy land; dip Aug 3 bins, LSWI > NDVI early Aug
R008|Paddy|high|img paddy land; own dip mid-Jul to Aug, LSWI >= NDVI
R009|Can't tell|low|img mixed; jumpy radar, weak crop cycle
R010|Other kharif crop|low|img bare land; follows neighbours, NDVI about .5, LSWI low
R011|Not cropland|high|img sheds; jumpy radar, no crop cycle
R012|Paddy|high|img paddy land (brown photo); own dip -26 early Aug, LSWI = NDVI
R013|Paddy|high|img paddy land; LSWI .38 vs NDVI .18 early Aug; dip shared (canal belt)
R014|Paddy|high|img paddy land; LSWI > NDVI early Aug, wet canopy
R015|Not cropland|high|img real-estate layout; no crop cycle
R016|Other kharif crop|low|img mixed; follows neighbours, slow green-up, LSWI low
R017|Paddy|high|img paddy land; own dip late Jul, LSWI > NDVI early Aug
R018|Paddy|high|img paddy land; own dip Aug 3 bins, LSWI > NDVI
R019|Perennial|medium|img orchard; green in June, no flood
R020|Paddy|high|img paddy land; own dip to -24, LSWI > NDVI early Aug
R021|Paddy|high|img paddy land; own dip -25 late Jul, LSWI > NDVI
R022|Other kharif crop|low|img mixed; follows neighbours, LSWI <= .29
R023|Paddy|high|img paddy land; own dip -31 dB (open water)
R024|Other kharif crop|high|img dryland; no flood, bright radar, green late Nov
R025|Perennial|high|img orchard; green all season, stable radar
R026|Paddy|high|img paddy land; LSWI > NDVI early Jul, early harvest mid-Oct
R027|Other kharif crop|low|CONFLICT img paddy land; card no flood, LSWI <= .27
R028|Perennial|medium|img orchard; green from Jul, flat to Nov
R029|Paddy|high|img paddy land; own dip late Aug-early Sep, 2 bins
R030|Other kharif crop|high|img dryland; NDVI about .5, no flood
R031|Not cropland|high|img real-estate layout; jumpy radar, flat NDVI
R032|Perennial|high|img orchard; NDVI .50 in June, green all season
R033|Other kharif crop|medium|CONFLICT img paddy land (cloudy); card LSWI <= .14, rain-fed
R034|Paddy|low|img paddy land; odd calendar, flood signal mid-Aug, green-up only Oct
R035|Other kharif crop|low|img mixed; no dip when neighbours dip, late green-up
R036|Paddy|medium|img paddy land; dip shared (canal belt), LSWI .47 mid-Sep
R037|Other kharif crop|low|CONFLICT img paddy land; card no flood, green late Nov
R038|Paddy|high|img paddy land; own dip early Aug 3 bins, wet canopy
R039|Paddy|high|img paddy land; flooded Jun, harvest Oct, re-flooded Nov (rabi)
R040|Other kharif crop|medium|img dryland; short weak crop, gone by Oct
R041|Paddy|high|img paddy land; own dip Aug-Sep 3 bins, NDVI .85
R042|Paddy|medium|img paddy land; field cleared mid-Jul, wet canopy, no clear flood
R043|Perennial|high|img orchard; NDVI .70 in June, green all season
R044|Paddy|medium|img paddy land; own dip early Aug, LSWI only about .25
R045|Perennial|low|CONFLICT img dryland; card green all season, no flood
R046|Other kharif crop|low|img dryland; one own radar dip, no flood seen, green late Nov
R047|Paddy|high|img paddy land; own dip 3 bins, LSWI = NDVI early Aug
R048|Other kharif crop|high|img dryland; no flood, late green-up, green late Nov
R049|Paddy|high|img paddy land; own dip -28.7 (water), LSWI >= NDVI
R050|Other kharif crop|high|img dryland; follows neighbours, no flood
R051|Perennial|low|CONFLICT img dryland; card green all season, stable radar
R052|Other kharif crop|low|CONFLICT img paddy land; card no flood, LSWI <= .25
R053|Paddy|high|img paddy land; wet Jun-Jul, harvest mid-Oct, re-flooded late Nov
R054|Paddy|medium|img paddy land; own dips Aug-Sep, wet canopy, no direct flood
R055|Paddy|high|img paddy land; LSWI = NDVI early Aug, own dip Aug-Sep
R056|Other kharif crop|medium|CARD ONLY (imagery not recorded): no flood, LSWI <= .15
R057|Paddy|medium|img mixed; own dip Aug 2 bins, LSWI >= NDVI late Jul-mid Aug
R058|Other kharif crop|low|CONFLICT img paddy land; card no flood, green late Nov
R059|Perennial|low|CONFLICT img dryland; card green all season, stable radar
R060|Other kharif crop|high|img dryland; follows neighbours, LSWI <= .27
R061|Paddy|medium|img mixed; LSWI >= NDVI mid-Jul and early Aug, own dip
R062|Paddy|medium|img paddy land; paddy timing, wet canopy, no clear flood
R063|Paddy|low|CONFLICT img dryland; card strong paddy (LSWI >= NDVI Aug)
R064|Other kharif crop|low|CONFLICT img paddy land; card no flood, two NDVI peaks
R065|Paddy|low|CONFLICT img dryland; own dips Aug-Sep, wet canopy, no direct flood
"""

rows = [line.split("|") for line in VERDICTS.strip().splitlines()]
verdicts = {r[0]: r[1:] for r in rows}
assert len(verdicts) == 65, f"expected 65 verdicts, found {len(verdicts)}"
assert all(v[0] in ALLOWED for v in verdicts.values()), "a label is misspelt"
assert all(v[1] in {"high", "medium", "low"} for v in verdicts.values()), "a confidence is misspelt"

path = PROCESSED / "review_blind_v2.gpkg"
gdf = gpd.read_file(path, layer="review")
missing = set(gdf["review_id"]) - set(verdicts)
assert not missing, f"no verdict for {sorted(missing)}"

gdf["review_label"] = gdf["review_id"].map(lambda r: verdicts[r][0])
gdf["review_confidence"] = gdf["review_id"].map(lambda r: verdicts[r][1])
gdf["review_notes"] = gdf["review_id"].map(lambda r: verdicts[r][2])
gdf["verdict_source"] = gdf["review_notes"].map(
    lambda n: "card only" if n.startswith("CARD ONLY") else "imagery (reviewer) + card (assistant)")
gdf.to_file(path, layer="review", driver="GPKG", mode="w")

print(gdf["review_label"].value_counts().to_string())
print(f"\nWrote {len(gdf)} verdicts into {path.name}")
