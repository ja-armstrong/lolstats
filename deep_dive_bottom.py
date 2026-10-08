"""
Deep dive: what predicts wins for BOTTOM (ADC)?
- single-feature AUCs (level-based z-scores and team shares)
- fitted logistic regression coefficients (level-based + share-based)
- win rate by score quartile for the strongest signals
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
LEVEL = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]
SEED = 42

import sys

ROLE = sys.argv[1].upper() if len(sys.argv) > 1 else "BOTTOM"

df = pd.read_parquet(ROOT / "data" / "processed" / "player_games.parquet")
df = df[df["role"] == ROLE].copy()
print(f"{ROLE} deep dive: {len(df):,} player-games, {df['matchId'].nunique():,} matches\n")

print(f"=== Single-feature AUCs (full sample, {ROLE} only) ===")
print(f"{'metric':14s} {'level':>7s} {'share':>7s}")
for m in LEVEL:
    lvl = roc_auc_score(df['win'], df[m + '_zt'].fillna(0))
    sh = roc_auc_score(df['win'], df[m + '_zt'].fillna(0)) if m in SHARES else np.nan
    print(f"{m:14s} {lvl:7.3f} {sh:7.3f}")
for m in SHARES:
    if m not in LEVEL:
        sh = roc_auc_score(df['win'], df[m + '_zt'].fillna(0))
        print(f"{m:14s} {'':7s} {sh:7.3f}")


def fit_report(cols, label):
    suffix = "_zt"
    X = df[[c + suffix for c in cols]].fillna(0.0).values
    y = df["win"].astype(int).values
    groups = df["matchId"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    scaler = StandardScaler().fit(X[tr])
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
    prob = model.predict_proba(scaler.transform(X[te]))[:, 1]
    print(f"\n=== {label} (test AUC {roc_auc_score(y[te], prob):.3f}) ===")
    coefs = pd.Series(model.coef_[0], index=cols).sort_values(key=np.abs, ascending=False)
    print(coefs.round(3).to_string())
    return model, scaler


fit_report(LEVEL, "Level-based model (raw stats, z-scored within role x tier)")
fit_report(SHARES, "Share-based model (team-relative slices)")

# win rate by gpm quartile
print(f"\n=== Win rate by gpm quartile ({ROLE}) ===")
q = pd.qcut(df["gpm"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print(df.groupby(q, observed=True)["win"].mean().round(3).to_string())

print(f"\n=== Win rate by kda quartile ({ROLE}) ===")
q = pd.qcut(df["kda"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print(df.groupby(q, observed=True)["win"].mean().round(3).to_string())

print(f"\n=== Win rate by vspm quartile ({ROLE}) ===")
q = pd.qcut(df["vspm"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print(df.groupby(q, observed=True)["win"].mean().round(3).to_string())