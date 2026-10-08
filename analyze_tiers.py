"""
Compare role impact across rank tiers (e.g. SILVER vs DIAMOND).

Runs the share-based per-role analysis within each tier group separately,
so each tier gets its own learned "definition of good" and its own AUCs.

Usage: python analyze_tiers.py SILVER DIAMOND
"""

import sys
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


def run_group(df: pd.DataFrame, label: str):
    feat_cols = [c + "_zt" for c in SHARES]
    rows, coef_rows = [], []
    for role in ROLES:
        r = df[df["role"] == role].copy()
        if len(r) < 200:
            rows.append({"role": role, "n": len(r), "test_auc": np.nan})
            continue
        X = r[feat_cols].fillna(0.0).values
        y = r["win"].astype(int).values
        groups = r["matchId"].values
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=SEED).split(X, y, groups))
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=2000).fit(
            scaler.transform(X[tr]), y[tr])
        auc = roc_auc_score(y[te], model.predict_proba(scaler.transform(X[te]))[:, 1])
        rows.append({"role": role, "n": len(r), "test_auc": auc})
        for c, w in zip(SHARES, model.coef_[0]):
            coef_rows.append({"role": role, "metric": c, "coef": w})

    res = pd.DataFrame(rows).set_index("role")
    coefs = pd.DataFrame(coef_rows).pivot(index="metric", columns="role", values="coef")
    coefs = coefs.reindex(columns=ROLES)
    print(f"\n===== {label} =====")
    print(res.round(4).to_string())
    print("coefficients:")
    print(coefs.round(3).to_string())
    return res, coefs


def main():
    tiers = sys.argv[1:] or ["SILVER", "DIAMOND"]
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()

    results, coef_maps = {}, {}
    for tier in tiers:
        sub = df[df["tier"].str.startswith(tier, na=False)]
        n_matches = sub["matchId"].nunique()
        print(f"\n{tier}: {len(sub):,} player-games, {n_matches} matches")
        if n_matches < 40:
            print("  (too little data — skipping)")
            continue
        res, coefs = run_group(sub, tier)
        results[tier] = res
        coef_maps[tier] = coefs

    if len(results) >= 2:
        comp = pd.concat({t: r["test_auc"] for t, r in results.items()}, axis=1)
        print("\n=== Side-by-side share-model AUC ===")
        print(comp.round(4).to_string())
        comp.round(4).to_markdown(OUT_DIR / "tier_comparison.md")

        # coefficient shift between tiers
        if len(coef_maps) == 2:
            t1, t2 = list(coef_maps)
            diff = coef_maps[t2] - coef_maps[t1]
            print(f"\n=== Coefficient shift ({t2} minus {t1}; what 'good' means differently) ===")
            print(diff.round(3).to_string())

        fig, ax = plt.subplots(figsize=(8, 4.8))
        comp.plot.bar(ax=ax)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylabel("Share-model test AUC")
        ax.set_title(f"Role impact by rank: {' vs '.join(tiers)}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "tier_comparison.png", dpi=150)


if __name__ == "__main__":
    main()