"""Check: is Model C contaminated by the share-residual degeneracy?
res(share) = (5*share_own - 1)/4 exactly (shares sum to 1 per team), so
share-residuals are affine transforms of own values, and pairing them with
level residuals lets the model reconstruct team totals (own/share = team total).
Compare: Model C full (levels+shares residuals) vs levels-residuals only.
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
LEVEL_ONLY = ["kda", "csm", "gpm", "dpm", "vspm", "wpm", "dthpm", "skpm"]  # kp is share-like; excluded here
SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]
COLS = LEVEL_ONLY + SHARES

df = pd.read_parquet(ROOT / "data" / "processed" / "player_games.parquet")
df = df[df["role"].isin(ROLES)].copy()
g = df.groupby(["matchId", "teamId"])
for c in COLS:
    col = c + "_zt"
    tm = g[col].transform("mean")
    n = g[col].transform("count")
    df[c + "_res"] = df[col] - (tm * n - df[col]) / (n - 1)

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
              .split(df, groups=df["matchId"]))

variants = {
    "C full (levels_res + shares_res)": [c + "_res" for c in COLS],
    "C levels-only (kp & shares excluded)": [c + "_res" for c in LEVEL_ONLY],
    "C shares_res only (degenerate check)": [c + "_res" for c in SHARES],
}
print(f"{'variant':40s}" + "".join(f"{r:>9s}" for r in ROLES))
for name, feats in variants.items():
    aucs = []
    for role in ROLES:
        r_tr, r_te = df.iloc[tr][df.iloc[tr]["role"] == role], df.iloc[te][df.iloc[te]["role"] == role]
        X_tr = r_tr[feats].fillna(0).values
        scaler = StandardScaler().fit(X_tr)
        m = LogisticRegression(max_iter=2000).fit(scaler.transform(X_tr),
                                                  r_tr["win"].astype(int).values)
        auc = roc_auc_score(r_te["win"].astype(int).values,
                            m.predict_proba(scaler.transform(r_te[feats].fillna(0).values))[:, 1])
        aucs.append(auc)
    print(f"{name:40s}" + "".join(f"{a:9.3f}" for a in aucs))