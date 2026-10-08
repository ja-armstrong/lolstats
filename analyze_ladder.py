"""
Role impact across the ladder: where does the "best role to climb with" change?

Method
------
- Pool each rank band's divisions: IRON, BRONZE, SILVER, GOLD, PLATINUM,
  EMERALD, DIAMOND, APEX (MASTER+GRANDMASTER+CHALLENGER).
- In each band, fit the share-based per-role win model (same as
  analyze_shares.py) with 5 repeated grouped splits -> mean test AUC +/- SE.
- Plot one AUC-vs-rank curve per role; report the best role per band and
  where leadership changes ("inflection points").

Outputs: results/ladder_table.csv/.md, results/ladder_curve.png
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
BANDS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "APEX"]
APEX_TIERS = {"CHALLENGER", "GRANDMASTER", "MASTER"}
N_REPEATS = 5
SEED = 42


def band_of(tier: str) -> str:
    base = tier.split("_")[0]
    return "APEX" if base in APEX_TIERS else base


def main():
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()
    df["band"] = df["tier"].map(band_of)
    feat_cols = [c + "_zt" for c in SHARES]

    rows = []
    for band in BANDS:
        sub = df[df["band"] == band]
        n_m = sub["matchId"].nunique()
        for role in ROLES:
            r = sub[sub["role"] == role]
            if r["matchId"].nunique() < 40:
                rows.append({"band": band, "n_matches": n_m, "role": role,
                             "auc": np.nan, "se": np.nan})
                continue
            X, y, groups = r[feat_cols].fillna(0.0).values, r["win"].astype(int).values, r["matchId"].values
            aucs = []
            for rep in range(N_REPEATS):
                tr, te = next(GroupShuffleSplit(
                    n_splits=1, test_size=0.25,
                    random_state=SEED + rep).split(X, y, groups))
                scaler = StandardScaler().fit(X[tr])
                model = LogisticRegression(max_iter=2000).fit(
                    scaler.transform(X[tr]), y[tr])
                aucs.append(roc_auc_score(
                    y[te], model.predict_proba(scaler.transform(X[te]))[:, 1]))
            rows.append({"band": band, "n_matches": n_m, "role": role,
                         "auc": float(np.mean(aucs)), "se": float(np.std(aucs) / np.sqrt(N_REPEATS))})

    res = pd.DataFrame(rows)
    piv = res.pivot(index="band", columns="role", values="auc").reindex(BANDS)[ROLES]
    se_piv = res.pivot(index="band", columns="role", values="se").reindex(BANDS)[ROLES]
    n_piv = res.pivot(index="band", columns="role", values="n_matches").reindex(BANDS)

    piv.to_csv(OUT_DIR / "ladder_table.csv")

    best = piv.idxmax(axis=1)
    summary = pd.DataFrame({
        "matches": n_piv.max(axis=1),
        "best_role": best,
        "best_auc": piv.max(axis=1).round(4),
        "runner_up": [piv.loc[b].drop(b_).idxmax() for b, b_ in zip(piv.index, best)],
    })
    print("=== Best role per rank band (share-model AUC) ===")
    print(summary.to_string())
    print("\n=== Full AUC table ===")
    print(piv.round(4).to_string())

    # leadership changes between adjacent bands
    print("\n=== Inflection points (best role changes) ===")
    prev = None
    for band in piv.index:
        b = best[band]
        if prev is not None and b != prev:
            print(f"  {prev} -> {b} between {piv.index[piv.index.get_loc(band)-1]} and {band}")
        prev = b

    summary.to_markdown(OUT_DIR / "ladder_table.md")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(BANDS))
    colors = {"TOP": "#1f77b4", "JUNGLE": "#d62728", "MIDDLE": "#2ca02c",
              "BOTTOM": "#ff7f0e", "SUPPORT": "#9467bd"}
    for role in ROLES:
        ax.plot(x, piv[role], marker="o", label=role, color=colors[role])
        ax.fill_between(x, piv[role] - 2 * se_piv[role], piv[role] + 2 * se_piv[role],
                        color=colors[role], alpha=0.12)
        ax.scatter(x, piv[role].where(piv.idxmax(axis=1) == role), s=120,
                   color=colors[role], zorder=5, edgecolors="black")
    ax.set_xticks(x, BANDS)
    ax.set_ylabel("Share-model test AUC")
    ax.set_xlabel("Rank band")
    ax.set_title("Which role's performance predicts winning, across the ladder?\n(big dots = best role in that band)")
    ax.set_ylim(0.40, 0.75)
    ax.legend(ncol=5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ladder_curve.png", dpi=150)
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()