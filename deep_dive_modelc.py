"""Model C deep dive: coefficients of the team-residualized win models per role."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
LEVEL = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
COLS = LEVEL + [c for c in SHARES if c not in LEVEL]

df = pd.read_parquet(ROOT / "data" / "processed" / "player_games.parquet")
df = df[df["role"].isin(ROLES)].copy()
g = df.groupby(["matchId", "teamId"])
for c in COLS:
    col = c + "_zt"
    tm = g[col].transform("mean")
    n = g[col].transform("count")
    df[c + "_res"] = df[col] - (tm * n - df[col]) / (n - 1)

feat = [c + "_res" for c in COLS]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
              .split(df, groups=df["matchId"]))

print("Model C: win ~ (own stat - teammates' mean stat), z-units\n")
for role in ROLES:
    r_tr, r_te = df.iloc[tr][df.iloc[tr]["role"] == role], df.iloc[te][df.iloc[te]["role"] == role]
    X_tr = r_tr[feat].fillna(0).values
    scaler = StandardScaler().fit(X_tr)
    m = LogisticRegression(max_iter=2000).fit(scaler.transform(X_tr), r_tr["win"].astype(int).values)
    auc = roc_auc_score(r_te["win"].astype(int).values,
                        m.predict_proba(scaler.transform(r_te[feat].fillna(0).values))[:, 1])
    s = pd.Series(m.coef_[0], index=COLS).sort_values(key=np.abs, ascending=False)
    print(f"--- {role} (test AUC {auc:.3f}) ---")
    print(s.round(3).to_string(), "\n")