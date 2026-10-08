"""
Impact analysis: individual vs team performance.

The original analysis (analyze_roles.py) predicts a player's win from their
own normalized score. But end-of-game stats partly re-encode team dominance:
when a team stomps, all five players' stats are inflated, so a single
player's raw score reads the TEAM outcome as much as the individual's.

This script separates the two:

Pass A — three AUCs per role, using the equal-weight composite score:
  (a) own score            -> how well this player's performance predicts win
  (b) teammates' mean      -> how well the REST of the team predicts win
  (c) relative score       -> own minus teammates' mean: does this player
                              stand out among their teammates, and does
                              standing out predict winning?

Pass B — incremental lift (grouped train/test split by match):
  baseline model:  win ~ teammates' mean score
  full model:      win ~ teammates' mean + own score
  delta AUC        -> how much predictive information THIS player adds
                      beyond their team. Largest lift = most impactful role.

Outputs: results/impact_table.csv/.md, results/impact_aucs.png,
         results/impact_lift.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "data" / "processed" / "player_games.parquet"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
SEED = 42


def main():
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()

    # keep only matches where all 10 players have usable role rows
    counts = df.groupby("matchId")["puuid"].transform("count")
    df = df[counts == 10]
    print(f"Impact analysis on {len(df):,} player-games "
          f"({df['matchId'].nunique():,} complete matches)\n")

    score = "score_zt"  # equal-weight composite, z-scored within role x tier
    g = df.groupby(["matchId", "teamId"])[score]
    df["team_mean"] = g.transform("mean")
    df["n_team"] = g.transform("count")
    df["teammates_mean"] = (df["team_mean"] * df["n_team"] - df[score]) / (df["n_team"] - 1)
    df["relative"] = df[score] - df["teammates_mean"]

    rows = []
    for role in ROLES:
        r = df[df["role"] == role]
        y = r["win"].astype(int)
        rows.append({
            "role": role,
            "auc_own": roc_auc_score(y, r[score]),
            "auc_teammates": roc_auc_score(y, r["teammates_mean"]),
            "auc_relative": roc_auc_score(y, r["relative"]),
        })

    # ---- Pass B: incremental lift with grouped split ----
    groups_all = df["matchId"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(df, groups=groups_all))
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]

    for role in ROLES:
        # baseline uses every player's teammates_mean as one pooled feature
        Xb_tr = df_tr[["teammates_mean"]].values
        Xb_te = df_te[["teammates_mean"]].values
        y_tr, y_te = df_tr["win"].astype(int).values, df_te["win"].astype(int).values

        base = LogisticRegression(max_iter=2000).fit(Xb_tr, y_tr)
        auc_base = roc_auc_score(y_te, base.predict_proba(Xb_te)[:, 1])

        r_tr, r_te = df_tr[df_tr["role"] == role], df_te[df_te["role"] == role]
        X_tr = np.column_stack([r_tr["teammates_mean"], r_tr[score]])
        X_te = np.column_stack([r_te["teammates_mean"], r_te[score]])
        full = LogisticRegression(max_iter=2000).fit(X_tr, r_tr["win"].astype(int).values)
        auc_full = roc_auc_score(r_te["win"].astype(int).values,
                                 full.predict_proba(X_te)[:, 1])

        row = next(x for x in rows if x["role"] == role)
        row["auc_baseline_team"] = auc_base
        row["auc_team_plus_player"] = auc_full
        row["delta_auc"] = auc_full - auc_base

    res = pd.DataFrame(rows).set_index("role")
    res.to_csv(OUT_DIR / "impact_table.csv")
    md = res.round(4)
    md.to_markdown(OUT_DIR / "impact_table.md")
    print("=== Individual vs team impact (AUC) ===")
    print(md[["auc_own", "auc_teammates", "auc_relative",
              "delta_auc"]].to_string())

    by_rel = res["auc_relative"].idxmax()
    by_lift = res["delta_auc"].idxmax()
    print(f"\nMost impactful role by relative-score AUC: {by_rel}")
    print(f"Most impactful role by incremental lift (delta AUC): {by_lift}")

    # ---- plots ----
    x = np.arange(len(ROLES))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.27
    for i, (col, label) in enumerate([
            ("auc_own", "Own score"),
            ("auc_teammates", "Teammates' mean"),
            ("auc_relative", "Own vs teammates (relative)")]):
        ax.bar(x + (i - 1) * width, res[col], width, label=label)
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x, ROLES)
    ax.set_ylim(0.35, max(0.72, res[["auc_own", "auc_teammates", "auc_relative"]].values.max() + 0.03))
    ax.set_ylabel("AUC (predicting team win)")
    ax.set_title("Whose performance makes the win? Own vs teammates vs relative")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "impact_aucs.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lift = res["delta_auc"].sort_values(ascending=False)
    colors = ["#d62728" if i == lift.index[0] else "#1f77b4" for i in lift.index]
    ax.bar(lift.index, lift.values, color=colors)
    ax.set_ylabel("Delta AUC from adding this player's score")
    ax.set_title("Incremental impact: info this role adds beyond the team")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "impact_lift.png", dpi=150)

    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()