"""Step 3.4: First model. Random Forest on the full season curves, trained on rule labels,
tested on the blind-reviewed points (never seen in training). Then: points for active learning."""
import joblib
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from config import PROCESSED, ROOT

SEED = 42
N_BINS = 15
CLASSES = ["Paddy", "Other kharif crop", "Perennial"]
BLOCK_KM = 10                                   # spatial blocks for cross-validation
pd.set_option("display.width", 200)


def cols(prefix):
    return [f"{prefix}_{i:02d}" for i in range(N_BINS)]


# ------------------------------------------------------------------
# 1. Data: all 354 candidates, plus the reviewed verdicts
# ------------------------------------------------------------------
cand = pd.read_csv(PROCESSED / "labels_v2_kharif2025.csv")
gold = pd.read_csv(PROCESSED / "review_results_v2.csv")[["point_id", "review_label", "weight", "stratum"]]
cand = cand.merge(gold, on="point_id", how="left")
cand["reviewed"] = cand["review_label"].notna()


# ------------------------------------------------------------------
# 2. Features: fill cloud gaps in time, then add a few physically meaningful summaries
# ------------------------------------------------------------------
def gap_fill(frame, prefix):
    """Linear interpolation along time; ends are filled with the nearest observed value."""
    block = frame[cols(prefix)].astype(float)
    return block.interpolate(axis=1, limit_direction="both")


X = pd.concat([gap_fill(cand, p) for p in ["VH", "VV", "NDVI", "LSWI"]], axis=1)
ndvi, lswi, vh = X[cols("NDVI")].to_numpy(), X[cols("LSWI")].to_numpy(), X[cols("VH")].to_numpy()
X["ndvi_june"] = ndvi[:, 0:3].mean(axis=1)            # green before sowing? (orchard OR previous crop)
X["ndvi_min_jul_aug"] = ndvi[:, 3:8].min(axis=1)      # orchards stay green; cleared fields drop
X["ndvi_max"] = ndvi.max(axis=1)
X["flood_index"] = (lswi - ndvi)[:, 0:10].max(axis=1) # > about -0.05 means water at some point Jun-Sep
X["vh_min"] = vh.min(axis=1)
X["vh_amplitude"] = vh.max(axis=1) - vh.min(axis=1)
X["vh_rise_after_min"] = np.array([row[np.argmin(row):].max() - row.min() for row in vh])
X = X.fillna(X.median())                              # any point with no data at all for a sensor

# ------------------------------------------------------------------
# 3. Training set: confident rule labels, EXCLUDING every reviewed point
# ------------------------------------------------------------------
train = cand["label"].isin(CLASSES) & ~cand["reviewed"]
y = cand.loc[train, "label"]
print("TRAINING LABELS (rule labels, reviewed points held out)")
print(y.value_counts().to_string())

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                            random_state=SEED, n_jobs=-1)

# Spatial cross-validation: whole 10 km blocks are held out together
km_x = cand["lon"] * 106.4
km_y = cand["lat"] * 110.6
blocks = (km_x // BLOCK_KM).astype(int).astype(str) + "_" + (km_y // BLOCK_KM).astype(int).astype(str)
cv_pred = cross_val_predict(rf, X[train], y, groups=blocks[train], cv=GroupKFold(n_splits=5))
print(f"\nSpatial 5-fold CV, agreement with the RULE labels: {100 * (cv_pred == y).mean():.1f}%")
print("  (this measures how well the model copies the rules, not truth; the real test is next)")

rf.fit(X[train], y)

# ------------------------------------------------------------------
# 4. The real test: reviewed points, judged by the reviewer, never seen in training
# ------------------------------------------------------------------
test = cand["reviewed"] & cand["review_label"].isin(CLASSES)
t = cand[test].copy()
t["model"] = rf.predict(X[test])
t["rules"] = t["label"]                               # may be "Review": that counts as no answer
w = t["weight"]


def scores(pred):
    correct = (pred == t["review_label"])
    paddy = t["review_label"] == "Paddy"
    said_paddy = pred == "Paddy"
    return {
        "accuracy % (weighted)": 100 * (w * correct).sum() / w.sum(),
        "accuracy % (unweighted)": 100 * correct.mean(),
        "paddy recall %": 100 * (w * (paddy & said_paddy)).sum() / (w * paddy).sum(),
        "paddy precision %": 100 * (w * (paddy & said_paddy)).sum() / max((w * said_paddy).sum(), 1e-9),
    }


print(f"\nTEST on {len(t)} reviewed points (Not cropland and Can't tell excluded)")
print(pd.DataFrame({"rules v2": scores(t["rules"]), "random forest": scores(t["model"])}).round(1).to_string())
print("\nRandom forest vs reviewer (rows = reviewer, columns = model), unweighted counts")
print(pd.crosstab(t["review_label"], t["model"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_string())

# ------------------------------------------------------------------
# 5. What did the model rely on?
# ------------------------------------------------------------------
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTOP 12 FEATURES (mean decrease in impurity)")
print((100 * imp.head(12)).round(1).to_string())

# ------------------------------------------------------------------
# 6. Active learning: unreviewed points where the model disagrees with the rules
# ------------------------------------------------------------------
proba = pd.DataFrame(rf.predict_proba(X), columns=rf.classes_, index=cand.index)
cand["model"] = proba.idxmax(axis=1)
cand["model_prob"] = proba.max(axis=1).round(2)
pool = cand[~cand["reviewed"] & (cand["model"] != cand["label"])
            & cand["label"].isin(CLASSES + ["Review"])]
print(f"\nACTIVE-LEARNING POOL: {len(pool)} unreviewed points where model and rules disagree")
print(pd.crosstab(pool["label"], pool["model"], margins=True).to_string())

keep = ["point_id", "orbit", "lon", "lat", "label", "reason", "model", "model_prob"]
pool[keep].sort_values("model_prob", ascending=False).to_csv(PROCESSED / "active_learning_pool_v1.csv", index=False)
gpd.GeoDataFrame(pool[keep], geometry=gpd.points_from_xy(pool["lon"], pool["lat"]), crs="EPSG:4326") \
    .to_file(PROCESSED / "active_learning_pool_v1.gpkg", layer="pool", driver="GPKG")
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump({"model": rf, "features": list(X.columns)}, ROOT / "models" / "rf_baseline_v1.joblib")
print("\nSaved data/processed/active_learning_pool_v1.csv/.gpkg and models/rf_baseline_v1.joblib")
