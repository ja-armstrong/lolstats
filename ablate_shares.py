"""Ablation: which feature(s) cause the suspicious AUC jump to ~0.98?"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

LEVEL = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
EXTRA = ["gold_share", "cs_share", "vision_share", "death_share"]

df = pd.read_parquet("data/processed/player_games.parquet")
mids, _ = pd.factorize(df["matchId"])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(df, groups=mids))
y = df["win"].astype(int).values

def run(cols, label):
    X = df[[c + "_zt" for c in cols]].fillna(0.0).values
    m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    s = X @ m.coef_[0] + m.intercept_
    pooled = roc_auc_score(y[te], s[te])
    jungle = roc_auc_score(y[te][(df["role"] == "JUNGLE").values[te]],
                           s[te][(df["role"] == "JUNGLE").values[te]])
    print(f"{label:42s} pooled={pooled:.4f}  jungle={jungle:.4f}")

run(LEVEL, "10 level features")
for e in EXTRA:
    run(LEVEL + [e], f"10 level + {e}")
run(LEVEL + EXTRA, "all 14")
for e in EXTRA:
    run([e], f"{e} alone")