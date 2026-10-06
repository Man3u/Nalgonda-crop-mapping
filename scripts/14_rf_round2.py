"""Step 3.6: Did uncertainty sampling find hidden paddy? Then retrain with all reviewed labels."""
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import PROCESSED, ROOT
from review_tools import CLASSES, build_features

SEED = 42
pd.set_option("display.width", 200)

# ------------------------------------------------------------------
# 1. Open the round-2 key: did the model's doubt point at hidden paddy?
# ------------------------------------------------------------------
key = pd.read_csv(PROCESSED / "review_key_round2.csv")
r2 = pd.DataFrame(gpd.read_file(PROCESSED / "review_blind_round2.gpkg", layer="review")
                  .drop(columns="geometry"))
r2 = key.merge(r2[["review_id", "review_label", "review_confidence"]], on="review_id")

print("ROUND 2: reviewer verdict by sampling group")
print(pd.crosstab(r2["stratum"], r2["review_label"], margins=True, margins_name="total").to_string())
rate = r2.assign(is_paddy=r2["review_label"].eq("Paddy")).groupby("stratum").agg(
    points=("is_paddy", "size"), paddy=("is_paddy", "sum"),
    paddy_pct=("is_paddy", lambda s: round(100 * s.mean())),
    group_size=("stratum_size", "first"))
print("\nPADDY RATE PER GROUP (round 1 found about 30% paddy among rule-'Other' points)")
print(rate.to_string())

# ------------------------------------------------------------------
# 2. One training set: rule labels, overridden by every reviewed verdict
# ------------------------------------------------------------------
cand = pd.read_csv(PROCESSED / "labels_v2_kharif2025.csv")
r1 = pd.read_csv(PROCESSED / "review_results_v2.csv")[["point_id", "review_label", "weight", "stratum"]]
r1["round"] = 1
r2x = r2[["point_id", "review_label", "stratum"]].assign(round=2)
r2x["weight"] = r2x["stratum"].map(rate["group_size"] / rate["points"])
gold = pd.concat([r1, r2x], ignore_index=True)
assert gold["point_id"].is_unique, "a point was reviewed twice"

df = cand.merge(gold, on="point_id", how="left")
df["truth"] = df["review_label"].where(df["review_label"].notna(), df["label"])
df["is_gold"] = df["review_label"].notna()
X = build_features(df)

# ------------------------------------------------------------------
# 3. Honest test: 5-fold cross-validation over the REVIEWED points only.
#    Each fold trains on rule labels + the other 4/5 of the gold points.
# ------------------------------------------------------------------
gold_idx = df.index[df["is_gold"] & df["truth"].isin(CLASSES)]
rng = np.random.default_rng(SEED)
fold = pd.Series(rng.permutation(len(gold_idx)) % 5, index=gold_idx)
rules_pred, old_pred, new_pred = {}, {}, {}
old_model = joblib.load(ROOT / "models" / "rf_baseline_v1.joblib")

for f in range(5):
    test_idx = gold_idx[fold == f]
    train_idx = df.index[(df["truth"].isin(CLASSES)) & ~df.index.isin(test_idx)]
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                random_state=SEED, n_jobs=-1).fit(X.loc[train_idx], df.loc[train_idx, "truth"])
    for i, p in zip(test_idx, rf.predict(X.loc[test_idx])):
        new_pred[i] = p
t = df.loc[gold_idx].copy()
t["new_model"] = pd.Series(new_pred)
t["old_model"] = old_model["model"].predict(X.loc[gold_idx][old_model["features"]])
t["rules"] = t["label"]
w = t["weight"]


def scores(pred):
    ok, paddy, says = pred == t["truth"], t["truth"] == "Paddy", pred == "Paddy"
    return {"accuracy %": 100 * (w * ok).sum() / w.sum(),
            "paddy recall %": 100 * (w * (paddy & says)).sum() / (w * paddy).sum(),
            "paddy precision %": 100 * (w * (paddy & says)).sum() / max((w * says).sum(), 1e-9)}


print(f"\nTEST on {len(t)} reviewed points (weighted; new model via 5-fold CV over the gold set)")
print(pd.DataFrame({"rules v2": scores(t["rules"]), "RF round 1": scores(t["old_model"]),
                    "RF round 2": scores(t["new_model"])}).round(1).to_string())
print("\nRF round 2 vs reviewer (rows = reviewer, columns = model)")
print(pd.crosstab(t["truth"], t["new_model"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_string())

# ------------------------------------------------------------------
# 4. Final model: trained on everything, for mapping the district
# ------------------------------------------------------------------
final_idx = df.index[df["truth"].isin(CLASSES)]
final = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                               random_state=SEED, n_jobs=-1).fit(X.loc[final_idx], df.loc[final_idx, "truth"])
print(f"\nFINAL MODEL trained on {len(final_idx)} points "
      f"({int(df.loc[final_idx, 'is_gold'].sum())} human-reviewed, the rest rule-labelled)")
print(df.loc[final_idx, "truth"].value_counts().to_string())
imp = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTOP 10 FEATURES")
print((100 * imp.head(10)).round(1).to_string())

joblib.dump({"model": final, "features": list(X.columns)}, ROOT / "models" / "rf_v2.joblib")
df[["point_id", "lon", "lat", "orbit", "label", "review_label", "truth", "is_gold"]] \
    .to_csv(PROCESSED / "training_set_v2.csv", index=False)
print("\nSaved models/rf_v2.joblib and data/processed/training_set_v2.csv")
