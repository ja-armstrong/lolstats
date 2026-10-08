"""
Team-relative ("share") impact analysis.

Motivation: the equal-weight composite treats "more of everything" as good,
which is wrong for some roles (e.g. a support with high damage share is often
losing; their value is vision + enabling). It also made the "you vs
teammates" comparison ill-formed, since it compared different roles' stat
levels directly.

Fix: every feature is a SLICE OF THE TEAM'S OUTPUT — gold share, damage
share, kill participation, CS share, vision share, death share. These are
"as compared to team" by construction. A per-role logistic regression then
learns which slices predict winning FOR THAT ROLE (support: vision share up,
damage share down; ADC: damage share up, etc.). Higher model score always
means "more likely to win", so AUC is well-formed by design.

Outputs:
  results/share_auc_table.csv/.md
  results/share_coefficients.csv
  results/share_auc.png
  results/share_calibration.png
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

SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
SEED = 42


def main():
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()
    print(f"Share-based analysis: {len(df):,} player-games "
          f"({df['matchId'].nunique():,} matches)\n")

    feat_cols = [c + "_zt" for c in SHARES]
    rows, coef_rows = [], []
    curves = {}

    for role in ROLES:
        r = df[df["role"] == role].copy()
        X = r[feat_cols].fillna(0.0).values
        y = r["win"].astype(int).values
        groups = r["matchId"].values

        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=SEED).split(X, y, groups))
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, prob)
        rows.append({"role": role, "n_test": len(te), "test_auc": auc})
        for c, w in zip(SHARES, model.coef_[0]):
            coef_rows.append({"role": role, "metric": c, "coef": w})

        # calibration: win rate by predicted-probability quintile
        qs = pd.qcut(prob, 5, labels=False, duplicates="drop")
        cal = pd.DataFrame({"q": qs, "win": y_te}).groupby("q")["win"].mean()
        curves[role] = cal

        # win/loss mean of raw shares (interpretability)
        print(f"--- {role} (test AUC {auc:.3f}) ---")
        for c in SHARES:
            w_mean = r.loc[r["win"], c].mean()
            l_mean = r.loc[~r["win"], c].mean()
            print(f"   {c:13s} win={w_mean:.3f}  loss={l_mean:.3f}  diff={w_mean - l_mean:+.3f}")

    res = pd.DataFrame(rows).set_index("role").sort_values("test_auc", ascending=False)
    res.to_csv(OUT_DIR / "share_auc_table.csv")
    res.round(4).to_markdown(OUT_DIR / "share_auc_table.md")

    coefs = pd.DataFrame(coef_rows).pivot(index="metric", columns="role", values="coef")
    coefs = coefs[ROLES]
    coefs.index.name = "metric"
    coefs.to_csv(OUT_DIR / "share_coefficients.csv")

    print("\n=== Share-model test AUC by role (higher = bigger slice of team output predicts win) ===")
    print(res.round(4).to_string())
    print("\n=== Learned coefficients (z-units; which team-slices matter per role) ===")
    print(coefs.round(3).to_string())

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    res["test_auc"].plot.bar(ax=ax, color="#1f77b4")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("Test AUC")
    ax.set_title("Impact as share of team output: whose slice predicts the win?")
    ax.set_ylim(0.4, max(0.7, res["test_auc"].max() + 0.02))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "share_auc.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for role, cal in curves.items():
        ax.plot(range(1, len(cal) + 1), cal.values, marker="o", label=role)
    ax.set_xlabel("Model-score quintile (low to high)")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Win rate rises with share-based impact score (monotonic = well-formed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "share_calibration.png", dpi=150)

    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()