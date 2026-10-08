"""
Does 'what predicts wins' differ by rank, for a given role?

Usage: python deep_dive_tiers.py SUPPORT [BOTTOM ...]

- AUC-by-band matrix for headline signals (single features)
- Pooled LOW (IRON-GOLD) vs HIGH (EMERALD-APEX) fitted models
  (level-based and share-based), with coefficients side by side
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
BANDS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "APEX"]
APEX_TIERS = {"CHALLENGER", "GRANDMASTER", "MASTER"}
LEVEL = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]
SEED = 42


def band_of(tier: str) -> str:
    base = tier.split("_")[0]
    return "APEX" if base in APEX_TIERS else base


def fitted_coefs(df_role, cols, label):
    X = df_role[[c + "_zt" for c in cols]].fillna(0.0).values
    y = df_role["win"].astype(int).values
    groups = df_role["matchId"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    scaler = StandardScaler().fit(X[tr])
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
    auc = roc_auc_score(y[te], model.predict_proba(scaler.transform(X[te]))[:, 1])
    coefs = pd.Series(model.coef_[0], index=cols, name=label)
    print(f"{label}: test AUC {auc:.3f}")
    return coefs, auc


for role in [a.upper() for a in sys.argv[1:]] or ["SUPPORT"]:
    df = pd.read_parquet(ROOT / "data" / "processed" / "player_games.parquet")
    df = df[(df["role"] == role)].copy()
    df["band"] = df["tier"].map(band_of)
    print(f"\n{'='*70}\n{role}  —  {df['matchId'].nunique():,} matches\n{'='*70}")

    print("\n=== Single-feature AUC by rank band ===")
    metrics = ["kda", "gpm", "dthpm", "dpm", "csm", "vspm", "gold_share", "dmg_share", "kp", "vision_share"]
    rows = {}
    counts = {}
    for band in BANDS:
        r = df[df["band"] == band]
        counts[band] = r["matchId"].nunique()
        rows[band] = {m: roc_auc_score(r["win"], r[m + "_zt"].fillna(0)) for m in metrics}
    mat = pd.DataFrame(rows)[BANDS].T
    mat.insert(0, "n_matches", pd.Series(counts))
    print(mat.round(3).to_string())
    print("(AUC < 0.5 = LOW value of that stat predicts winning, e.g. death metrics)")

    low = df[df["band"].isin(["IRON", "BRONZE", "SILVER", "GOLD"])]
    high = df[df["band"].isin(["EMERALD", "DIAMOND", "APEX"])]
    print(f"\nLOW pool (Iron-Gold): {low['matchId'].nunique()} matches | "
          f"HIGH pool (Emerald+): {high['matchId'].nunique()} matches")

    for cols, name in [(LEVEL, "level-based"), (SHARES, "share-based")]:
        print(f"\n--- {name} model: LOW vs HIGH ---")
        c_low, _ = fitted_coefs(low, cols, f"LOW  {name}")
        c_high, _ = fitted_coefs(high, cols, f"HIGH {name}")
        cmp = pd.concat([c_low, c_high, c_high - c_low], axis=1)
        cmp.columns = ["low", "high", "shift"]
        print(cmp.round(3).to_string())