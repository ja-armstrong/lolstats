"""
Rigor audit: separate the estimands, quantify uncertainty, stress-test conclusions.

Three distinct questions get conflated when we say "performance predicts win":

  MODEL A (raw levels):      score = player's own stats.
        Answers: "is this player's performance associated with their team winning?"
        Contaminated by team dominance: when a team stomps, all 5 players' stats rise.

  MODEL B (team shares):     score = player's share of their team's output.
        Answers: "does this player's slice of the team's production predict winning?"
        Contamination: share = own / team_total, so a high share can mean a LOW
        team total (bad team) — the denominator still carries team state.

  MODEL C (team-residualized): feature = own stat minus teammates' mean stat.
        Answers: "does standing out AMONG YOUR OWN TEAMMATES predict winning?"
        The cleanest 'individual differential' signal — team-level component removed.

For each model x role: test AUC + 95% cluster-bootstrap CI (resampling matches,
so within-match label correlation can't fake precision), plus paired bootstrap
delta-AUC vs JUNGLE (same bootstrap replicates -> valid comparison).

Additional stress tests:
  - close games only (>= 32 min) for every model (fights outcome circularity)
  - player-clustered bootstrap (accounts for repeat players from snowballing)
  - champion-controlled models (removes pick-quality confound)
  - patch stability of the Model B ranking

Outputs: results/rigor_models.csv, results/rigor_summary.md, console report.
"""

from pathlib import Path

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
LEVEL = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]
B_BOOT = 400
SEED = 42
CLOSE_GAME_MIN = 32


def load():
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()
    # residualize within (match, team): own stat minus teammates' mean
    g = df.groupby(["matchId", "teamId"])
    for c in LEVEL + SHARES:
        col = c + "_zt"
        team_mean = g[col].transform("mean")
        n_team = g[col].transform("count")
        df[c + "_res"] = df[col] - (team_mean * n_team - df[col]) / (n_team - 1)
    df["patch"] = df["gameVersion"].astype(str).str.split(".").str[:2].str.join(".")
    return df


def fit_auc(df_tr_role, df_te_role, feat_cols, use_champion=False):
    X_tr_parts = []
    ch_cols = None
    if feat_cols:
        X_tr_parts.append(df_tr_role[feat_cols].fillna(0.0).values)
    if use_champion:
        dummies = pd.get_dummies(df_tr_role["champion"], prefix="ch")
        ch_cols = dummies.columns
        X_tr_parts.append(dummies.values)
    X_tr = np.hstack(X_tr_parts)
    y_tr = df_tr_role["win"].astype(int).values
    model = LogisticRegression(max_iter=3000, C=0.5).fit(X_tr, y_tr)

    X_te_parts = []
    if feat_cols:
        X_te_parts.append(df_te_role[feat_cols].fillna(0.0).values)
    if use_champion:
        d_te = pd.get_dummies(df_te_role["champion"], prefix="ch").reindex(columns=ch_cols, fill_value=0)
        X_te_parts.append(d_te.values)
    X_te = np.hstack(X_te_parts)
    prob = model.predict_proba(X_te)[:, 1]
    return roc_auc_score(df_te_role["win"].astype(int).values, prob), model


VARIANTS = {
    "A_raw_levels": LEVEL,
    "B_team_shares": SHARES,
    # Model C corrected: residualize only NON-degenerate features.
    # For the 5 share metrics (sum to 1 across a team), res(share) =
    # (5*share_own - 1)/4 exactly — an affine transform of own value with
    # zero teammate information — and paired with level residuals they let
    # the model reconstruct team totals. kp is NOT degenerate (kills+assists
    # don't sum to a constant), so it keeps a true residual.
    "C_team_residual": ["kda", "kp", "csm", "gpm", "dpm", "vspm", "wpm", "dthpm", "skpm"],
}


def main():
    df = load()
    print(f"Rigor audit on {df['matchId'].nunique():,} matches, {len(df):,} player-games\n")

    for subset_name, sub in [("ALL GAMES", df),
                             (f"CLOSE GAMES (>= {CLOSE_GAME_MIN} min)",
                              df[df["minutes"] >= CLOSE_GAME_MIN])]:
        tr_idx, te_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=SEED).split(sub, groups=sub["matchId"]))
        tr, te = sub.iloc[tr_idx], sub.iloc[te_idx]
        te_match_ids = te["matchId"].unique()
        match_pos = {m: i for i, m in enumerate(te_match_ids)}
        te = te.copy()
        te["mpos"] = te["matchId"].map(match_pos)

        rng = np.random.default_rng(SEED)
        boots = [rng.choice(len(te_match_ids), size=len(te_match_ids), replace=True)
                 for _ in range(B_BOOT)]

        rows = []
        per_boot = {}
        for vname, cols in VARIANTS.items():
            fc = [c + "_zt" for c in cols] if vname != "C_team_residual" \
                else [c + "_res" for c in cols]
            for role in ROLES:
                r_tr, r_te = tr[tr["role"] == role], te[te["role"] == role]
                auc, model = fit_auc(r_tr, r_te, fc)
                # bootstrap over test matches (CI of this fixed model's test AUC)
                order = np.argsort(r_te["mpos"].values)
                sorted_rows = r_te.iloc[order]
                boundaries = np.searchsorted(sorted_rows["mpos"].values,
                                             np.arange(len(te_match_ids) + 1))
                X_te = r_te[fc].fillna(0.0).values
                score = model.predict_proba(X_te)[:, 1]
                yv = r_te["win"].astype(int).values
                bs_vals = []
                for b in boots:
                    sel = np.concatenate([np.arange(boundaries[i], boundaries[i + 1]) for i in b])
                    if len(sel) < 50:
                        continue
                    bs_vals.append(roc_auc_score(yv[sel], score[sel]))
                bs = np.array(bs_vals)
                per_boot[(vname, role)] = bs
                rows.append({"subset": subset_name, "model": vname, "role": role,
                             "n_test": len(r_te), "auc": auc,
                             "ci_lo": np.percentile(bs, 2.5) if len(bs) else np.nan,
                             "ci_hi": np.percentile(bs, 97.5) if len(bs) else np.nan})

        res = pd.DataFrame(rows)
        res["delta_vs_jungle"] = res.apply(
            lambda r: r["auc"] - res[(res.model == r.model) & (res.role == "JUNGLE")]["auc"].iloc[0],
            axis=1)

        def delta_ci(model, role):
            b_r = per_boot[(model, role)]
            b_j = per_boot[(model, "JUNGLE")]
            d = b_r - b_j
            return np.percentile(d, 2.5), np.percentile(d, 97.5)

        print(f"===== {subset_name} =====")
        show = res.pivot_table(index="model", columns="role", values="auc")[ROLES]
        print("AUC (point estimates):")
        print(show.round(4).to_string())
        print("\n95% cluster-bootstrap CIs (resampled matches):")
        ci = res.copy()
        ci["ci"] = ci.apply(lambda r: f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
        print(ci.pivot_table(index="model", columns="role", values="ci", aggfunc="first")[ROLES].to_string())
        print("\nPaired delta-AUC vs JUNGLE (95% CI):")
        for model in VARIANTS:
            parts = []
            for role in ROLES:
                if role == "JUNGLE":
                    parts.append(f"{role}: —")
                    continue
                lo, hi = delta_ci(model, role)
                parts.append(f"{role}: {res[(res.model==model)&(res.role==role)].auc.iloc[0] - res[(res.model==model)&(res.role=='JUNGLE')].auc.iloc[0]:+.3f} [{lo:+.3f},{hi:+.3f}]")
            print(f"  {model:16s} " + " | ".join(parts))
        res.to_csv(OUT_DIR / ("rigor_models.csv" if subset_name == "ALL GAMES" else "rigor_models_close.csv"),
                   index=False)
        print()

    # ---- player-clustered bootstrap (pseudoreplication from repeat players) ----
    print("===== Player-clustered bootstrap (Model A pooled, ALL GAMES) =====")
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED).split(df, groups=df["matchId"]))
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    X_tr, y_tr = tr[[c + "_zt" for c in LEVEL]].fillna(0.0).values, tr["win"].astype(int).values
    model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    X_te, y_te = te[[c + "_zt" for c in LEVEL]].fillna(0.0).values, te["win"].astype(int).values
    score = model.predict_proba(X_te)[:, 1]
    point = roc_auc_score(y_te, score)
    rng = np.random.default_rng(SEED)
    players = te["puuid"].dropna().unique()
    pmap = {p: i for i, p in enumerate(players)}
    pidx = te["puuid"].map(pmap).values
    bs = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(players), size=len(players), replace=True)
        # map each selected player to its rows
        counts = np.bincount(pick, minlength=len(players))
        sel = np.repeat(np.arange(len(te)), counts[pidx])
        bs.append(roc_auc_score(y_te[sel], score[sel]))
    bs = np.array(bs)
    print(f"  naive point AUC {point:.4f}")
    print(f"  player-clustered 95% CI: [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
    print(f"  distinct players in test set: {len(players):,} for {len(te):,} rows "
          f"({len(te)/max(len(players),1):.1f} rows/player avg)")

    # ---- champion control ----
    print("\n===== Champion-controlled models (does pick quality explain the signal?) =====")
    for subset_name, sub in [("ALL", df), ("CLOSE", df[df["minutes"] >= CLOSE_GAME_MIN])]:
        tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED).split(sub, groups=sub["matchId"]))
        t_r, t_e = sub.iloc[tr_i], sub.iloc[te_i]
        out = {}
        for model_name, cols in [("A_raw", LEVEL), ("B_shares", SHARES)]:
            for champ in [False, True]:
                aucs = {}
                for role in ROLES:
                    r_tr, r_te = t_r[t_r["role"] == role], t_e[t_e["role"] == role]
                    fc = [c + "_zt" for c in cols]
                    auc, _ = fit_auc(r_tr, r_te, fc, use_champion=champ)
                    aucs[role] = auc
                out[f"{model_name}{'_champ' if champ else ''}"] = aucs
        t = pd.DataFrame(out).T[ROLES]
        print(f"--- {subset_name} ---")
        print(t.round(4).to_string())

    # ---- patch stability of Model B ranking ----
    print("\n===== Model B (shares) AUC by patch (rank stability across metas) =====")
    counts = df.groupby("patch")["matchId"].nunique()
    big = counts[counts >= 120].index
    rows = []
    for patch in big:
        sub = df[df["patch"] == patch]
        tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED).split(sub, groups=sub["matchId"]))
        t_r, t_e = sub.iloc[tr_i], sub.iloc[te_i]
        for role in ROLES:
            auc, _ = fit_auc(t_r[t_r["role"] == role], t_e[t_e["role"] == role],
                             [c + "_zt" for c in SHARES])
            rows.append({"patch": patch, "n": int(counts[patch]), "role": role, "auc": auc})
    if rows:
        pt = pd.DataFrame(rows).pivot(index="patch", columns="role", values="auc")[ROLES]
        pt["n_matches"] = pt.index.map(counts)
        print(pt.round(3).to_string())
        pt.to_csv(OUT_DIR / "rigor_patch_stability.csv")
    else:
        print("  (no patch has >= 120 matches yet)")
    print(f"\nPatch distribution: {counts.sort_index().to_dict()}")


if __name__ == "__main__":
    main()