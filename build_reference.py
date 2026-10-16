"""
Freeze the collected corpus into a reference distribution, so one player's
~40 games can be scored against ~96k player-games instead than against itself.

Why this exists: z-scoring 40 games uses that player's own mean and spread, so
"you are 1.4 SD above average" would be meaningless. The baseline has to come
from the corpus. This script publishes, per role and per role x rank band:
  - mean / sd of every metric (to turn raw box stats into z-scores)
  - a 21-point quantile grid per metric (to turn a z-free value into a percentile)
  - quantile grids for the equal-weight composite score and for
    "outperform" (own composite minus the mean of your four teammates)
  - a small sample of z-scored rows, so the report page can re-rank you under
    custom metric weights without shipping the whole corpus

Outputs:
  data/processed/reference.json   (authoritative, read by the Python side)
  web/reference.js                (browser copy: window.LOL_REFERENCE)

Usage:
    python build_reference.py                 # ladder corpus
    python build_reference.py --tag _pro      # tier-1 pro corpus instead
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from prepare_features import METRICS, SHARES

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "data" / "processed" / "player_games.parquet"
OUT_JSON = ROOT / "data" / "processed" / "reference.json"
OUT_JS = ROOT / "web" / "reference.js"

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
BANDS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "APEX"]
APEX_TIERS = {"CHALLENGER", "GRANDMASTER", "MASTER"}

LABELS = {
    "kda": "KDA (k+a)/d", "kp": "Kill participation", "csm": "CS per minute",
    "gpm": "Gold per minute", "dpm": "Damage to champs / min",
    "dmg_share": "Damage share of team", "vspm": "Vision score / min",
    "wpm": "Wards placed / min", "dthpm": "Deaths per minute",
    "skpm": "Solo kills / min", "gold_share": "Gold share of team",
    "cs_share": "CS share of team", "vision_share": "Vision share of team",
    "death_share": "Death share of team",
}
# metrics where a higher raw value is worse (mirrors build_web.py DEFAULT_W)
SIGNS = {m: (-1 if m in ("dthpm", "death_share") else 1) for m in METRICS}

# Feature sets, copied from analyze_rigor.py so report presets and the research
# models mean the same thing:
#   A = raw levels (partly re-encodes the win: 0.93 -> 0.81 in 32min+ games)
#   B = team shares (the headline honest estimand)
#   C = within-team residuals of the NON-degenerate metrics only. res(share) is an
#       affine transform of your own value (zero teammate information), so the 5
#       share metrics are excluded; kp stays because kills+assists do not sum to a
#       constant. This is the "how good were YOU" axis.
LEVEL = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm", "dthpm", "skpm"]
C_RESIDUAL = ["kda", "kp", "csm", "gpm", "dpm", "vspm", "wpm", "dthpm", "skpm"]
AXES = {
    "equal": {"kind": "level", "feats": METRICS},
    "A": {"kind": "level", "feats": LEVEL},
    "B": {"kind": "share", "feats": SHARES},
    "C": {"kind": "residual", "feats": C_RESIDUAL},
}
AXIS_LABEL = {
    "equal": "Equal weight",
    "A": "Model A (raw levels)",
    "B": "Model B (team shares)",
    "C": "Model C (vs teammates)",
}
# analyze_rigor.py fits C with a tighter penalty than the A/B presets in build_web.py
LR = {"A": {"C": 1.0, "max_iter": 2000}, "B": {"C": 1.0, "max_iter": 2000},
      "C": {"C": 0.5, "max_iter": 3000}}

PCTS = list(range(0, 101, 5))     # 21-point quantile grid
MIN_GROUP = 150                   # rows needed to publish a role x band baseline
SAMPLE_ROLE = 250                 # z rows kept per role (custom-weight ranking)
SAMPLE_BAND = 80                  # ... and per role x band (grids cover the 4 presets)
SEED = 42


def band_of(tier) -> str:
    base = str(tier).split("_")[0]
    return "APEX" if base in APEX_TIERS else base


def qgrid(v: np.ndarray) -> list[float]:
    return [round(float(np.quantile(v, p / 100.0)), 4) for p in PCTS]


def _zmat(sub: pd.DataFrame, prefix: str) -> np.ndarray:
    return sub[[f"z_{prefix}_{m}" for m in METRICS]].values


def _rmat(sub: pd.DataFrame, prefix: str) -> np.ndarray:
    return sub[[f"r_{prefix}_{m}" for m in METRICS]].values


def univariate_logit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Win-logit per 1 SD of one standardized stat: coefficient + SE.

    SE comes from the observed information at the MLE, 1/sqrt(sum p(1-p) x^2),
    which is the right scale for "how much more likely is a win per SD here".
    """
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    if len(x) < 30 or y.min() == y.max() or x.std() < 1e-9:
        return 0.0, 9.99
    model = LogisticRegression(C=1e6, max_iter=1000).fit(x.reshape(-1, 1), y)
    b = float(model.coef_[0][0])
    p = model.predict_proba(x.reshape(-1, 1))[:, 1]
    info = float(np.sum(p * (1 - p) * x * x))
    se = 1.0 / np.sqrt(info) if info > 1e-9 else 9.99
    # a coefficient that ran away (separation) is not a number worth publishing
    if abs(b) > 6 or not np.isfinite(se):
        return 0.0, 9.99
    return b, se


def _sample(sub: pd.DataFrame, cols: list[str], cap: int, rng) -> list[list[int]]:
    """Scored rows as x100 ints, for client-side custom-weight percentiles."""
    z = sub[cols].values
    if len(z) > cap:
        z = z[rng.choice(len(z), size=cap, replace=False)]
    return [[int(round(x * 100)) for x in row] for row in z]


def build_block(sub: pd.DataFrame, prefix: str, cap: int, rng) -> dict:
    blk = {"n": int(len(sub)), "m": {}}
    for m in METRICS:
        v = sub[m].values.astype(float)
        sd = float(v.std(ddof=0))
        blk["m"][m] = {
            "mean": round(float(v.mean()), 4),
            "sd": round(sd if sd > 1e-9 else 1.0, 4),
            "q": qgrid(v),
        }
    blk["score"] = {ax: {"q": qgrid(sub[f"s_{prefix}_{ax}"].values.astype(float))}
                    for ax in AXES}
    blk["diff"] = {ax: {"q": qgrid(sub[f"d_{prefix}_{ax}"].values.astype(float))}
                   for ax in AXES}
    # per-metric win logits: "how much does one SD of this stat raise a win for
    # peers like you", on both the level axis and the vs-teammates (residual) axis.
    # Compared against the same number fitted on the looked-up player's own games,
    # this is what shows whether you win by doing what your role wins by.
    blk["slope"] = {}
    wins = sub["win"].astype(int).values
    for kind, tmpl in (("level", "z_{p}_{m}"), ("residual", "r_{p}_{m}")):
        d = {}
        for m in METRICS:
            b, se = univariate_logit(sub[tmpl.format(p=prefix, m=m)].values, wins)
            d[m] = [round(b, 3), round(se, 3)]
        blk["slope"][kind] = d
    # kept for older pages / quick reference: equal-weight versions
    blk["comp"] = blk["score"]["equal"]
    blk["outperform"] = blk["diff"]["equal"]
    blk["z"] = _sample(sub, [f"z_{prefix}_{m}" for m in METRICS], cap, rng)
    blk["rz"] = _sample(sub, [f"r_{prefix}_{m}" for m in METRICS], cap, rng)
    return blk


def add_research_columns(df: pd.DataFrame) -> pd.DataFrame:
    """residual of the corpus's own role x tier z, within (match, team).

    Same formula as analyze_rigor.load(): own z minus the mean of the other
    teammates' z. Needs no new moments - the parquet already carries _zt.
    """
    g = df.groupby(["matchId", "teamId"], observed=True)
    for c in C_RESIDUAL:
        col = c + "_zt"
        if col not in df:
            continue
        tm, n = g[col].transform("mean"), g[col].transform("count")
        df["res_" + c] = df[col] - (tm * n - df[col]) / (n - 1).clip(lower=1)
    return df


def fit_presets(df: pd.DataFrame) -> dict:
    """Per-role logistic win models on a train split (grouped by match).

    The report applies these coefficients to z-scores computed against its own
    role / role x band groups rather than the corpus's role x tier _zt columns.
    That transfer is safe because every input is standardized, and it is the
    player's ranking within a peer group that is being reported, not a coefficient.
    """
    mids, _ = pd.factorize(df["matchId"].values)
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED).split(df, groups=mids))
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    presets, meta = {}, {}
    for axis in ("A", "B", "C"):
        feats = AXES[axis]["feats"]
        cols = [("res_" + c) if axis == "C" else (c + "_zt") for c in feats]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[presets] skipping {axis}: missing {missing[:3]}")
            continue
        presets[axis], meta[axis] = {}, {}
        for role in ROLES:
            a, b = tr[tr["role"] == role], te[te["role"] == role]
            Xa, ya = a[cols].fillna(0.0).values, a["win"].astype(int).values
            model = LogisticRegression(C=LR[axis]["C"], max_iter=LR[axis]["max_iter"]).fit(Xa, ya)
            auc = None
            if len(b) and b["win"].nunique() > 1:
                p = model.predict_proba(b[cols].fillna(0.0).values)[:, 1]
                auc = round(float(roc_auc_score(b["win"].astype(int).values, p)), 3)
            presets[axis][role] = {"w": {f: round(float(w), 3) for f, w in zip(feats, model.coef_[0])},
                                   "intercept": round(float(model.intercept_[0]), 3)}
            meta[axis][role] = {"auc": auc, "nTrain": int(len(a)), "nTest": int(len(b)),
                                "feats": feats}
    return {"presets": presets, "fit": {"splitBy": "matchId", "testSize": 0.25,
                                        "seed": SEED, "auc": {ax: {r: meta[ax][r]["auc"]
                                                                   for r in meta[ax]}
                                                              for ax in meta}}}


def add_axis_scores(df: pd.DataFrame, presets: dict) -> pd.DataFrame:
    """Per-row score and teammate-differential for every weighting, both baselines.

    score = weighted mean of z (level/share) or of residual z (C).
    diff  = the same weights applied to the residual axis, i.e. how much the
            metric said you beat your own teammates. For C, diff == score.
    """
    for prefix in ("all", "band"):
        zv = {m: df[f"z_{prefix}_{m}"].values for m in METRICS}
        rv = {m: df[f"r_{prefix}_{m}"].values for m in METRICS}
        df[f"s_{prefix}_equal"] = sum(SIGNS[m] * zv[m] for m in METRICS) / len(METRICS)
        df[f"d_{prefix}_equal"] = sum(SIGNS[m] * rv[m] for m in METRICS) / len(METRICS)
        for axis in ("A", "B", "C"):
            if axis not in presets:
                continue
            src = rv if AXES[axis]["kind"] == "residual" else zv
            s = np.zeros(len(df))
            d = np.zeros(len(df))
            for role in ROLES:
                sel = (df["role"] == role).values
                w = presets[axis][role]["w"]
                feats = [f for f in AXES[axis]["feats"] if f in w]
                if not feats:
                    continue
                s[sel] = sum(w[f] * src[f][sel] for f in feats) / len(feats)
                d[sel] = sum(w[f] * rv[f][sel] for f in feats) / len(feats)
            df[f"s_{prefix}_{axis}"] = s
            df[f"d_{prefix}_{axis}"] = d
    return df


def residualize_within_group(df: pd.DataFrame, prefix: str) -> None:
    """r_{prefix}_{m} = own z minus the mean of the four teammates' z."""
    g = df.groupby(["matchId", "teamId"], observed=True)
    for m in METRICS:
        col = f"z_{prefix}_{m}"
        tm, n = g[col].transform("mean"), g[col].transform("count")
        df[f"r_{prefix}_{m}"] = df[col] - (tm * n - df[col]) / (n - 1).clip(lower=1)


def add_scored_columns(df: pd.DataFrame, presets: dict) -> pd.DataFrame:
    """z per metric and per baseline, teammate residuals, then every weighting."""
    df = df.copy()
    for prefix, group_cols in (("all", ["role"]), ("band", ["role", "band"])):
        g = df.groupby(group_cols, observed=True)[METRICS]
        mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
        z = (df[METRICS] - mu) / sd
        for m in METRICS:
            df[f"z_{prefix}_{m}"] = z[m].fillna(0.0)
        residualize_within_group(df, prefix)
    return add_axis_scores(df, presets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_FILE)
    ap.add_argument("--tag", default="", help="suffix for the source parquet, e.g. _pro")
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-js", type=Path, default=OUT_JS)
    ap.add_argument("--region", default="NA", help="label only: where the corpus came from")
    args = ap.parse_args()

    src = args.input
    if args.tag and src == IN_FILE:
        src = IN_FILE.with_name(f"player_games{args.tag}.parquet")
    df = pd.read_parquet(src)
    df = df[df["role"].isin(ROLES)].copy()
    df = df.dropna(subset=METRICS)
    df["band"] = df["tier"].map(band_of)

    df = add_research_columns(df)
    fitted = fit_presets(df)
    print(f"[presets] fitted per role on {int(0.75 * len(df)):,} train player-games "
          f"(split by match): " + " | ".join(
              ax + " " + ",".join(f"{r[0]}{(fitted['fit']['auc'][ax][r] or 0):.2f}" for r in ROLES)
              for ax in fitted["presets"]))
    df = add_scored_columns(df, fitted["presets"])

    rng = np.random.default_rng(SEED)
    groups, skipped = {}, []

    # role baselines: every role pooled across the ladder
    for role in ROLES:
        sub = df[df["role"] == role]
        groups[role] = build_block(sub, "all", SAMPLE_ROLE, rng)

    # role x band baselines
    for role in ROLES:
        for band in BANDS:
            sub = df[(df["role"] == role) & (df["band"] == band)]
            if len(sub) < MIN_GROUP:
                skipped.append(f"{role}|{band}")
                continue
            groups[f"{role}|{band}"] = build_block(sub, "band", SAMPLE_BAND, rng)

    ref = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": src.name,
        "queue": "RANKED_SOLO_5x5 (420)" if args.tag == "" else args.tag.lstrip("_"),
        "region": args.region,
        "nRows": int(len(df)),
        "nMatches": int(df["matchId"].nunique()),
        "nPlayers": int(df["puuid"].nunique()) if "puuid" in df else None,
        "roles": ROLES,
        "bands": BANDS,
        "pcts": PCTS,
        "metrics": [{"key": m, "label": LABELS[m], "sign": SIGNS[m]} for m in METRICS],
        # weighting schemes the report can switch between; C lives on the residual
        # axis (own minus teammates), which is why its "diff" and "score" coincide
        "axes": [{"key": k, "label": AXIS_LABEL[k], "kind": v["kind"], "feats": v["feats"]}
                 for k, v in AXES.items()],
        "presets": fitted["presets"],
        "fit": fitted["fit"],
        # corpus min/max = the 0.1%/99.9% trim prepare_features.py applied, so a
        # looked-up player gets clipped to the same range before scoring
        "bounds": {m: [round(float(df[m].min()), 4), round(float(df[m].max()), 4)]
                   for m in METRICS},
        "minGroup": MIN_GROUP,
        "skipped": skipped,
        "groups": groups,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(ref), encoding="utf-8")
    args.out_js.parent.mkdir(parents=True, exist_ok=True)
    args.out_js.write_text(
        "window.LOL_REFERENCE = " + json.dumps(ref, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Reference from {len(df):,} player-games / {ref['nMatches']:,} matches "
          f"({ref['nPlayers']} players)")
    print(f"  groups: {len(groups)} published, {len(skipped)} skipped (<{MIN_GROUP} rows)")
    print(f"  {args.out_json} ({args.out_json.stat().st_size/1e6:.2f} MB)")
    print(f"  {args.out_js} ({args.out_js.stat().st_size/1e6:.2f} MB)")
    if skipped:
        print("  skipped:", ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""))


if __name__ == "__main__":
    main()
