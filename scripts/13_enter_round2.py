"""Step 3.5b: Write the agreed round-2 verdicts into the blind layer (same method as 09b)."""
import geopandas as gpd
from config import PROCESSED

ALLOWED = {"Paddy", "Other kharif crop", "Perennial", "Not cropland", "Can't tell"}

VERDICTS = """
S001|Other kharif crop|low|img mixed; radar below neighbours but no dip; smooth rise to .73, LSWI .25-.30, no flood
S002|Other kharif crop|low|CONFLICT img cloudy/orchard; June NDVI .23 too low for orchard, flat peak .55, LSWI <= .21
S003|Paddy|high|img paddy (cloudy); own dip -26.2 mid-Jul, field cleared .12, +13 dB rise, NDVI .86, LSWI .38
S004|Paddy|high|img paddy; own dip -24.3 early Aug; LSWI .33 > NDVI .30; peak .74, harvest Nov
S005|Paddy|medium|img paddy; radar follows neighbours, no flood; wet canopy .34-.38, NDVI .86, sharp Nov harvest
S006|Paddy|medium|img mixed; own dips Jun-Aug; LSWI .22 > NDVI .13 early Aug; rise to .76, green late Nov
S007|Paddy|low|img dryland/paddy; follows neighbours, no flood; wet canopy .40, peak .85 Sep, sharp harvest
S008|Paddy|high|img paddy; own dip -22.7 mid-Jul; LSWI .17 = NDVI .18; rise to .70, LSWI .35
S009|Paddy|high|img paddy; own dip -20.5 early Aug; LSWI .20 > NDVI .09; rise to .81, LSWI .38
S010|Paddy|medium|img paddy; own dip -21 early Aug; wet canopy .36-.39; sharp Oct-Nov harvest; one NDVI .98 looks like cloud edge
S011|Paddy|high|img paddy; own deep dips -22.3 Aug and -25.8 Sep; field cleared .13; rise to .81, LSWI .42
S012|Paddy|medium|img paddy; green in June .49 (previous crop), cleared .15 mid-Jul, own dips, rise to .75
S013|Other kharif crop|low|img dryland; flat NDVI .35-.60, no crop cycle, bright stable radar; possibly scrub
S014|Paddy|high|img paddy; own dip -23.6 mid-Jul; LSWI .13 near NDVI .19; rise to .79, LSWI .35
S015|Paddy|medium|img paddy; own dip -25 in June (early transplant?); rise to .76, LSWI .35, sharp harvest
S016|Paddy|high|img paddy; LSWI .19 = NDVI .19 early Aug; rise to .80, LSWI .42, sharp Nov harvest
S017|Other kharif crop|high|img dryland; slow rise to .66, green late Nov, LSWI <= .19, no flood
S018|Paddy|high|img paddy; own dips Jun, Jul, Sep; LSWI .30 > NDVI .23 early Aug; rise to .75, LSWI .40
S019|Other kharif crop|high|img dryland; follows neighbours, rise to .71, green late Nov, LSWI <= .25
S020|Paddy|high|img paddy; own dip -22.7 early Aug, 9 dB below neighbours, 3 bins; rise to .81, LSWI .33
S021|Paddy|medium|img paddy; own dips Jul; no flood crossing; wet canopy .38-.44; peak .75, sharp harvest
S022|Paddy|medium|img paddy; green June .45, cleared .29 mid-Jul; no flood; wet canopy .34-.38; peak .82
S023|Paddy|high|img paddy; bare Jun-Aug (.09-.13); own dip -20.5 early Aug; rise to .90, LSWI .40
S024|Not cropland|low|CONFLICT img not cropland; card shows crop cycle .20->.72, no dip, no flood; weeds/scrub?
S025|Paddy|high|img paddy; own deep dips Jun-Jul (-23); LSWI = NDVI early Jul; rise to .72, LSWI .42
"""

rows = [line.split("|") for line in VERDICTS.strip().splitlines()]
verdicts = {r[0]: r[1:] for r in rows}
assert len(verdicts) == 25, f"expected 25 verdicts, found {len(verdicts)}"
assert all(v[0] in ALLOWED for v in verdicts.values()), "a label is misspelt"

path = PROCESSED / "review_blind_round2.gpkg"
gdf = gpd.read_file(path, layer="review")
assert not set(gdf["review_id"]) - set(verdicts), "a point has no verdict"
gdf["review_label"] = gdf["review_id"].map(lambda r: verdicts[r][0])
gdf["review_confidence"] = gdf["review_id"].map(lambda r: verdicts[r][1])
gdf["review_notes"] = gdf["review_id"].map(lambda r: verdicts[r][2])
gdf["verdict_source"] = "imagery (reviewer) + card (assistant)"
gdf.to_file(path, layer="review", driver="GPKG", mode="w")
print(gdf["review_label"].value_counts().to_string())
print(f"\nWrote {len(gdf)} verdicts into {path.name}")
