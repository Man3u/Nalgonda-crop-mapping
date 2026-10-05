"""Step 3.5a: Active learning round 2. Look where the model is UNSURE, not where it disagrees.
Targets hidden paddy inside the rule label 'Other kharif crop'. Blind, with control points.
P(paddy) for training points is OUT-OF-FOLD: the model never scores a point it was trained on."""
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict
from config import PROCESSED, ROOT
from review_tools import CLASSES, build_features, make_card

SEED = 42
BLOCK_KM = 10
N_UNSURE_OTHER = 15      # rule says Other, out-of-fold P(paddy) highest
N_REVIEW_PADDY = 5       # rule says Review, model says Paddy (random)
N_CONTROL = 5            # rule says Other, P(paddy) < 0.1, not in the unsure group (random)
CARD_DIR = ROOT / "outputs" / "review_cards_round2"
pd.set_option("display.width", 200)

cand = pd.read_csv(PROCESSED / "labels_v2_kharif2025.csv")
reviewed = cand["point_id"].isin(pd.read_csv(PROCESSED / "review_results_v2.csv")["point_id"])
bundle = joblib.load(ROOT / "models" / "rf_baseline_v1.joblib")
model, features = bundle["model"], bundle["features"]
X = build_features(cand)[features]

# Points that were NOT in training: ordinary predictions are fair
proba = pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=cand.index)

# Points that WERE in training: replace with out-of-fold predictions (spatial blocks, as in script 11)
train = cand["label"].isin(CLASSES) & ~reviewed
blocks = ((cand["lon"] * 106.4) // BLOCK_KM).astype(int).astype(str) + "_" + \
         ((cand["lat"] * 110.6) // BLOCK_KM).astype(int).astype(str)
oof = cross_val_predict(clone(model), X[train], cand.loc[train, "label"], groups=blocks[train],
                        cv=GroupKFold(n_splits=5), method="predict_proba")
proba.loc[train, np.unique(cand.loc[train, "label"])] = oof

cand["p_paddy"] = proba["Paddy"].round(2)
cand["model"] = proba.idxmax(axis=1)
free = cand[~reviewed]

other = free[free["label"] == "Other kharif crop"]
bands = pd.cut(other["p_paddy"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], include_lowest=True)
print("UNREVIEWED rule 'Other kharif crop' points by OUT-OF-FOLD P(paddy)")
print(bands.value_counts().sort_index().to_string())

unsure = other.nlargest(N_UNSURE_OTHER, "p_paddy").assign(stratum="unsure_other")
low = other[(other["p_paddy"] < 0.1) & ~other["point_id"].isin(unsure["point_id"])]
control = low.sample(n=min(N_CONTROL, len(low)), random_state=SEED).assign(stratum="control_low")
rev = free[(free["label"] == "Review") & (free["model"] == "Paddy")]
review_paddy = rev.sample(n=min(N_REVIEW_PADDY, len(rev)), random_state=SEED).assign(stratum="review_to_paddy")
sizes = {"unsure_other": len(other), "control_low": len(low), "review_to_paddy": len(rev)}

sample = pd.concat([unsure, control, review_paddy]).sample(frac=1, random_state=SEED).reset_index(drop=True)
assert sample["point_id"].is_unique, "a point was picked twice"
sample.insert(0, "review_id", [f"S{i:03d}" for i in range(1, len(sample) + 1)])
sample["stratum_size"] = sample["stratum"].map(sizes)
print(f"\nROUND-2 SAMPLE: {len(sample)} points")
print(sample.groupby("stratum")["p_paddy"].agg(points="size", min_p="min", max_p="max").to_string())

key_cols = ["review_id", "point_id", "stratum", "stratum_size", "label", "model", "p_paddy"]
sample[key_cols].to_csv(PROCESSED / "review_key_round2.csv", index=False)
blind = sample[["review_id", "orbit", "lon", "lat"]].copy()
gpd.GeoDataFrame(blind, geometry=gpd.points_from_xy(blind["lon"], blind["lat"]), crs="EPSG:4326") \
    .to_file(PROCESSED / "review_blind_round2.gpkg", layer="review", driver="GPKG")

CARD_DIR.mkdir(parents=True, exist_ok=True)
for old in CARD_DIR.glob("S*.png"):
    old.unlink()
for _, row in sample.iterrows():
    make_card(row, row["review_id"], CARD_DIR / f"{row['review_id']}.png")
print(f"\nSaved {len(sample)} cards to outputs/review_cards_round2/, "
      "review_blind_round2.gpkg (QGIS) and review_key_round2.csv (do not open)")
