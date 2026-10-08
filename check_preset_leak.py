"""Replicate the web page's preset computation to check whether its AUC is honest."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

METRICS = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm",
           "dthpm", "skpm", "gold_share", "cs_share", "vision_share", "death_share"]

df = pd.read_parquet("data/processed/player_games.parquet")
mids, _ = pd.factorize(df["matchId"])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(df, groups=mids))

X = df[[m + "_zt" for m in METRICS]].fillna(0.0).values
y = df["win"].astype(int).values

# jungle preset: fit on train rows of jungle role only (unscaled, like the web build)
jmask = (df["role"] == "JUNGLE").values
model = LogisticRegression(max_iter=2000).fit(X[tr & jmask] if False else X[tr][jmask[tr]], y[tr][jmask[tr]])

score = X @ model.coef_[0] + model.intercept_

print(" Jungle preset, evaluated on TEST matches only:")
for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]:
    m = (df["role"] == role).values
    print(f"  {role:8s} AUC {roc_auc_score(y[te][m[te]], score[te][m[te]]):.4f}")
print(f"  POOLED (all roles, like the web page): {roc_auc_score(y[te], score[te]):.4f}")
print(f"  POOLED on ALL rows (train+test):       {roc_auc_score(y, score):.4f}")