"""Step 3.3: Open the answer key. Compare the blind visual review with the rule labels (v2)."""
import geopandas as gpd
import numpy as np
import pandas as pd
from config import PROCESSED

CLASSES = ["Paddy", "Other kharif crop", "Perennial"]
REVIEW_LABELS = CLASSES + ["Not cropland", "Can't tell"]
RULE_LABEL = {                       # what the rules said, for each sampled group (stratum)
    "paddy_high": "Paddy", "paddy_medium": "Paddy",
    "other_high": "Other kharif crop", "other_medium": "Other kharif crop",
    "perennial": "Perennial",
    "review_moved_from_other": "Review", "review_other": "Review",
}
pd.set_option("display.width", 200)

# ------------------------------------------------------------------
# 1. Join the blind review with the answer key
# ------------------------------------------------------------------
review = gpd.read_file(PROCESSED / "review_blind_v2.gpkg", layer="review")
review = pd.DataFrame(review.drop(columns="geometry"))
key = pd.read_csv(PROCESSED / "review_key_v2.csv")
df = key.merge(review[["review_id", "review_label", "review_confidence", "review_notes"]],
               on="review_id", how="left", validate="one_to_one")
df["review_label"] = df["review_label"].fillna("").astype(str).str.strip()
df["rule_label"] = df["stratum"].map(RULE_LABEL)

bad = df[~df["review_label"].isin(REVIEW_LABELS)]
if len(bad):
    print("STOP: these points have a missing or misspelt review_label:")
    print(bad[["review_id", "review_label"]].to_string(index=False))
    raise SystemExit(1)

n_h = df.groupby("stratum").size()                    # points reviewed per group
N_h = df.groupby("stratum")["stratum_size"].first()   # points of that group in all 354 candidates
df["weight"] = df["stratum"].map(N_h / n_h)           # each reviewed point stands for this many candidates
print(f"Reviewed points: {len(df)} | they represent {int(N_h.sum())} candidates\n")

# ------------------------------------------------------------------
# 2. Raw table: what the reviewer said, per group
# ------------------------------------------------------------------
print("REVIEWER VERDICTS PER RULE GROUP (counts)")
print(pd.crosstab(df["stratum"], df["review_label"], margins=True, margins_name="total")
      .reindex(columns=REVIEW_LABELS + ["total"], fill_value=0).to_string())

# ------------------------------------------------------------------
# 3. Accuracy of the confident rule labels (Can't tell is left out)
# ------------------------------------------------------------------
decided = df[df["review_label"] != "Can't tell"].copy()
decided["correct"] = decided["review_label"] == decided["rule_label"]
labelled = decided[decided["rule_label"] != "Review"]

rows = []
for s, g in labelled.groupby("stratum"):
    rows.append({"group": s, "rule label": RULE_LABEL[s], "reviewed": len(g),
                 "agree": int(g["correct"].sum()), "accuracy %": 100 * g["correct"].mean(),
                 "group size": int(N_h[s])})
per_group = pd.DataFrame(rows).set_index("group")
print("\nACCURACY PER GROUP (rule label vs reviewer; Can't tell excluded)")
print(per_group.round(0).to_string())

# Stratified estimate: each group weighted by its real size (with finite-population correction)
W = per_group["group size"] / per_group["group size"].sum()
p = per_group["agree"] / per_group["reviewed"]
n = per_group["reviewed"]
fpc = 1 - n / per_group["group size"]
var = (W ** 2 * fpc * p * (1 - p) / (n - 1).clip(lower=1)).sum()
oa, se = (W * p).sum(), np.sqrt(var)
print(f"\nOVERALL ACCURACY of the {int(per_group['group size'].sum())} rule-labelled candidates "
      f"(stratified estimate): {100*oa:.1f}% ± {196*se:.1f} (95% CI)")
print(f"Unweighted, for comparison: {100*labelled['correct'].mean():.1f}% of {len(labelled)} reviewed points")

# ------------------------------------------------------------------
# 4. Area-weighted confusion matrix -> user's and producer's accuracy per class
# ------------------------------------------------------------------
cm = pd.crosstab(decided["rule_label"], decided["review_label"], values=decided["weight"],
                 aggfunc="sum").fillna(0)
cm = cm.reindex(index=CLASSES + ["Review"], columns=CLASSES + ["Not cropland"], fill_value=0)
print("\nWEIGHTED CONFUSION MATRIX (rows = rules, columns = reviewer; units = candidate points)")
print(cm.round(0).to_string())

acc = pd.DataFrame({
    "user's accuracy %": [100 * cm.loc[c, c] / cm.loc[c].sum() if cm.loc[c].sum() else np.nan for c in CLASSES],
    "producer's accuracy %": [100 * cm.loc[c, c] / cm[c].sum() if cm[c].sum() else np.nan for c in CLASSES],
}, index=CLASSES)
print("\nPER CLASS")
print("  user's accuracy     = when the rules say X, how often is it really X? (reliability of the label)")
print("  producer's accuracy = of all true X, how much did the rules label as X? (Review counts as missed)")
print(acc.round(0).to_string())

# ------------------------------------------------------------------
# 5. What is hiding in Review, and was v1 or v2 right on the disputed points?
# ------------------------------------------------------------------
print("\nWHAT THE REVIEW GROUP REALLY CONTAINS (estimated candidates, Can't tell included)")
rev = df[df["rule_label"] == "Review"]
print(rev.groupby("review_label")["weight"].sum().reindex(REVIEW_LABELS, fill_value=0).round(0).to_string())

disputed = df[df["stratum"] == "review_moved_from_other"]
v1_right = (disputed["review_label"] == "Other kharif crop").sum()
print(f"\nDISPUTED POINTS (v1 said Other kharif crop, v2 said Review): {len(disputed)} reviewed")
print(f"  reviewer agrees with v1 (Other kharif crop): {v1_right}")
print(f"  reviewer says something else: "
      f"{disputed.loc[disputed['review_label'] != 'Other kharif crop', 'review_label'].value_counts().to_dict()}")

# ------------------------------------------------------------------
# 6. Is WorldCover 'cropland' really cropland?
# ------------------------------------------------------------------
not_crop = df.loc[df["review_label"] == "Not cropland", "weight"].sum() / df["weight"].sum()
print(f"\nEstimated share of WorldCover-2021 core cropland that is NOT cropland now: {100*not_crop:.1f}%")

# ------------------------------------------------------------------
# 7. Reviewer confidence: are the labels better where the reviewer was sure?
# ------------------------------------------------------------------
print("\nAGREEMENT BY REVIEWER CONFIDENCE (rule-labelled groups only, unweighted)")
print(labelled.groupby("review_confidence")["correct"].agg(points="size", agree_pct="mean")
      .assign(agree_pct=lambda t: (100 * t["agree_pct"]).round(0)).to_string())

df.to_csv(PROCESSED / "review_results_v2.csv", index=False)
print("\nSaved data/processed/review_results_v2.csv")
