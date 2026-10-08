"""
Core analysis: which role's normalized performance score best predicts win rate?

Method
------
For each role (TOP, JUNGLE, MIDDLE, BOTTOM, SUPPORT):
  1. Take that role's player-games, features = z-scores within (role x tier).
  2. Split train/test GROUPED BY MATCH (players from the same match never
     straddle the split, so no outcome leakage).
  3. Fit three models: LogisticRegression, HistGradientBoosting, MLP.
  4. Evaluate on the test set: ROC-AUC (primary), accuracy, Brier score.
  5. Also score the simple equal-weight composite (no fitted model).

The role whose score most accurately predicts its own team's win rate —
i.e. highest test AUC — is judged the most impactful role.

Outputs:
  - results/role_auc_table.csv / .md
  - results/logreg_coefficients.csv
  - results/auc_by_role.png
  - results/winrate_by_score_decile.png
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "data" / "processed" / "player_games.parquet"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

METRICS = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
SEED = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.08, random_state=SEED),
        "Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
            max_iter=400, early_stopping=True, random_state=SEED),
    }


def main():
    if not IN_FILE.exists():
        raise SystemExit("Run prepare_features.py first.")
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()
    print(f"Analyzing {len(df):,} player-games across {df['matchId'].nunique():,} matches\n")

    feature_cols = [c + "_zt" for c in METRICS]
    rows = []
    coef_frames = {}
    roc_curves = {}

    for role in ROLES:
        rdf = df[df["role"] == role].copy()
        X, y = rdf[feature_cols].fillna(0.0).values, rdf["win"].astype(int).values
        groups = rdf["matchId"].values

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        tr, te = next(splitter.split(X, y, groups))
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

        scaler = StandardScaler().fit(X_tr)  # features are already z-like; harmless
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        # equal-weight composite (no fitting) — the "plain normalized score"
        comp = X_te_s.mean(axis=1)
        auc = roc_auc_score(y_te, comp)
        rows.append({"role": role, "model": "Equal-weight composite",
                     "n_train": len(tr), "n_test": len(te),
                     "test_auc": auc, "test_accuracy": accuracy_score(y_te, comp >= 0),
                     "brier": brier_score_loss(y_te, 1 / (1 + np.exp(-comp)))})

        for model_name, model in get_models().items():
            model.fit(X_tr_s, y_tr)
            prob = model.predict_proba(X_te_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, prob)
            roc_curves[(role, model_name)] = (fpr, tpr, roc_auc_score(y_te, prob))
            rows.append({"role": role, "model": model_name,
                         "n_train": len(tr), "n_test": len(te),
                         "test_auc": roc_auc_score(y_te, prob),
                         "test_accuracy": accuracy_score(y_te, prob >= 0.5),
                         "brier": brier_score_loss(y_te, prob)})
            if model_name == "Logistic Regression":
                coef_frames[role] = pd.Series(model.coef_[0], index=METRICS, name=role)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "role_auc_table.csv", index=False)

    md = res.pivot_table(index="model", columns="role", values="test_auc")
    md = md[ROLES].round(4)
    md.to_markdown(OUT_DIR / "role_auc_table.md")

    print("=== Test AUC by role and model (higher = score predicts win rate better) ===")
    print(md.to_string())
    best = md.loc["Logistic Regression"].idxmax()
    print(f"\nMost impactful role (logistic regression): {best} "
          f"(AUC {md.loc['Logistic Regression', best]:.4f})")

    # logistic coefficients per role
    coefs = pd.DataFrame(coef_frames)
    coefs.index.name = "metric"
    coefs.to_csv(OUT_DIR / "logreg_coefficients.csv")
    print("\n=== Logistic regression coefficients (z-units, per role) ===")
    print(coefs.round(3).to_string())

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    for i, model_name in enumerate(md.index):
        ax.bar(np.arange(len(ROLES)) + (i - 1) * width, md.loc[model_name, ROLES],
               width, label=model_name)
    ax.set_xticks(np.arange(len(ROLES)), ROLES)
    ax.set_ylim(0.5, max(0.68, np.nanmax(md.values) + 0.02))
    ax.set_ylabel("Test AUC")
    ax.set_title("How well does a player's normalized score predict their team's win?")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "auc_by_role.png", dpi=150)

    # decile calibration: win rate by composite-score decile, per role (logreg probs not needed)
    fig, ax = plt.subplots(figsize=(9, 5))
    for role in ROLES:
        rdf = df[df["role"] == role]
        dec = pd.qcut(rdf["score_zt"], 10, labels=False, duplicates="drop")
        wr = rdf.groupby(dec, observed=True)["win"].mean()
        ax.plot(wr.index + 0.5, wr.values, marker="o", label=role)
    ax.set_xlabel("Normalized score decile (within role x tier)")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Win rate by performance-score decile, per role")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "winrate_by_score_decile.png", dpi=150)

    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()