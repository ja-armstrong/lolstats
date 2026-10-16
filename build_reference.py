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

from prepare_features import METRICS

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

PCTS = list(range(0, 101, 5))     # 21-point quantile grid
MIN_GROUP = 150                   # rows needed to publish a role x band baseline
SAMPLE_ROLE = 400                 # z rows kept per role (custom-weight ranking)
SAMPLE_BAND = 120                 # ... and per role x band
SEED = 42


def band_of(tier) -> str:
    base = str(tier).split("_")[0]
    return "APEX" if base in APEX_TIERS else base


def qgrid(v: np.ndarray) -> list[float]:
    return [round(float(np.quantile(v, p / 100.0)), 4) for p in PCTS]


def _zmat(sub: pd.DataFrame, prefix: str) -> np.ndarray:
    return sub[[f"z_{prefix}_{m}" for m in METRICS]].values


def _sample(sub: pd.DataFrame, prefix: str, cap: int, rng) -> list[list[int]]:
    """z-scored rows as x100 ints, for client-side custom-weight percentiles."""
    z = _zmat(sub, prefix)
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
    blk["comp"] = {"q": qgrid(sub[f"comp_{prefix}"].values.astype(float))}
    blk["outperform"] = {"q": qgrid(sub[f"out_{prefix}"].values.astype(float))}
    blk["z"] = _sample(sub, prefix, cap, rng)
    return blk


def add_scored_columns(df: pd.DataFrame) -> pd.DataFrame:
    """z per metric, composite, and outperform-vs-teammates, for both baselines."""
    df = df.copy()
    for prefix, group_cols in (("all", ["role"]), ("band", ["role", "band"])):
        g = df.groupby(group_cols, observed=True)[METRICS]
        mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
        z = (df[METRICS] - mu) / sd
        for m in METRICS:
            df[f"z_{prefix}_{m}"] = z[m].fillna(0.0)
        df[f"comp_{prefix}"] = (
            sum(SIGNS[m] * df[f"z_{prefix}_{m}"] for m in METRICS) / len(METRICS)
        )
        team = df.groupby(["matchId", "teamId"], observed=True)[f"comp_{prefix}"]
        tsum, tsize = team.transform("sum"), team.transform("size")
        mate_mean = (tsum - df[f"comp_{prefix}"]) / (tsize - 1).clip(lower=1)
        df[f"out_{prefix}"] = df[f"comp_{prefix}"] - mate_mean
    return df


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
    df = add_scored_columns(df)

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
