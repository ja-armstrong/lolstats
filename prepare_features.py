"""
Turn raw collected matches into a per-player-game table with normalized metrics.

Outputs: data/processed/player_games.parquet

Normalization is the core idea:
- Raw stats are not comparable across roles (junglers farm differently than
  ADCs) or across ranks (Diamond games have higher CS/min than Bronze).
- So every metric is z-scored within (role) AND within (role x tier).
- The normalized score for a player is comparable across the whole dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "matches.jsonl"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "player_games.parquet"

MIN_DURATION_S = 600  # drop remakes / 3-minute surrenders

ROLE_MAP = {  # standardize teamPosition labels
    "TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE",
    "BOTTOM": "BOTTOM", "UTILITY": "SUPPORT", "SUPPORT": "SUPPORT",
}


def load_matches() -> pd.DataFrame:
    rows = []
    with RAW_FILE.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("queueId") != 420:
                continue
            dur = m.get("gameDuration") or 0
            if dur < MIN_DURATION_S:
                continue
            minutes = dur / 60.0
            players = m["players"]
            # team totals (kills and damage) for share metrics
            team_kills = {100: 0, 200: 0}
            team_dmg = {100: 0, 200: 0}
            team_gold = {100: 0, 200: 0}
            team_cs = {100: 0, 200: 0}
            team_vision = {100: 0, 200: 0}
            team_deaths = {100: 0, 200: 0}
            for p in players:
                t = p.get("teamId")
                team_kills[t] = team_kills.get(t, 0) + (p.get("kills") or 0)
                team_dmg[t] = team_dmg.get(t, 0) + (p.get("totalDamageDealtToChampions") or 0)
                team_gold[t] = team_gold.get(t, 0) + (p.get("goldEarned") or 0)
                team_cs[t] = team_cs.get(t, 0) + (p.get("totalMinionsKilled") or 0) + (p.get("neutralMinionsKilled") or 0)
                team_vision[t] = team_vision.get(t, 0) + (p.get("visionScore") or 0)
                team_deaths[t] = team_deaths.get(t, 0) + (p.get("deaths") or 0)
            for p in players:
                role = ROLE_MAP.get(p.get("teamPosition") or "")
                if role is None:
                    continue
                k, d, a = (p.get("kills") or 0), (p.get("deaths") or 0), (p.get("assists") or 0)
                cs = (p.get("totalMinionsKilled") or 0) + (p.get("neutralMinionsKilled") or 0)
                gold = p.get("goldEarned") or 0
                dmg = p.get("totalDamageDealtToChampions") or 0
                tk, td = team_kills.get(p.get("teamId"), 1), team_dmg.get(p.get("teamId"), 1)
                tgold = team_gold.get(p.get("teamId"), 1)
                tcs = team_cs.get(p.get("teamId"), 1)
                tvis = team_vision.get(p.get("teamId"), 1)
                tdeaths = team_deaths.get(p.get("teamId"), 1)
                rows.append({
                    "matchId": m["matchId"],
                    "puuid": p.get("puuid"),
                    "teamId": p.get("teamId"),
                    "tier": m.get("seedTier"),
                    "role": role,
                    "champion": p.get("championName"),
                    "win": bool(p.get("win")),
                    "minutes": minutes,
                    "gameVersion": m.get("gameVersion"),
                    "kills": k, "deaths": d, "assists": a,
                    "kda": (k + a) / max(d, 1),
                    "kp": (k + a) / max(tk, 1),
                    "csm": cs / minutes,
                    "gpm": gold / minutes,
                    "dpm": dmg / minutes,
                    "dmg_share": dmg / max(td, 1),
                    "vspm": (p.get("visionScore") or 0) / minutes,
                    "wpm": (p.get("wardsPlaced") or 0) / minutes,
                    "dthpm": d / minutes,
                    "skpm": (p.get("soloKills") or 0) / minutes,
                    # team-relative share features ("as compared to team")
                    "gold_share": gold / max(tgold, 1),
                    "cs_share": cs / max(tcs, 1),
                    "vision_share": (p.get("visionScore") or 0) / max(tvis, 1),
                    "death_share": d / max(tdeaths, 1),
                })
    return pd.DataFrame(rows)


def zscore_group(df: pd.DataFrame, cols: list[str], group: list[str], suffix: str) -> pd.DataFrame:
    g = df.groupby(group, observed=True)[cols]
    mu = g.transform("mean")
    sd = g.transform("std")
    out = (df[cols] - mu) / sd.replace(0, np.nan)
    return df.join(out.add_suffix(suffix))


METRICS = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm",
           "dthpm", "skpm", "gold_share", "cs_share", "vision_share", "death_share"]

# the team-relative shares — the "as compared to team" feature set
SHARES = ["gold_share", "dmg_share", "kp", "cs_share", "vision_share", "death_share"]


def main():
    df = load_matches()
    if df.empty:
        print("No usable matches found. Run collect_matches.py first.")
        return
    n_matches = df["matchId"].nunique()
    print(f"Loaded {len(df):,} player-games from {n_matches:,} matches")

    # drop rows with missing core metrics
    df = df.dropna(subset=METRICS)
    # trim pathological outliers before z-scoring (e.g. 20-min 30-kill stomp stats)
    for c in METRICS:
        lo, hi = df[c].quantile([0.001, 0.999])
        df[c] = df[c].clip(lo, hi)

    df = zscore_group(df, METRICS, ["role"], "_z")          # within role
    df = zscore_group(df, METRICS, ["role", "tier"], "_zt") # within role x tier

    # simple equal-weight composite: mean of z-scores (higher = better performance)
    z_cols = [c + "_zt" for c in METRICS]
    df["score_zt"] = df[z_cols].mean(axis=1)
    df["score_z"] = df[[c + "_z" for c in METRICS]].mean(axis=1)

    df.to_parquet(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}")
    print("\nRows per role x tier (sample):")
    print(df.groupby(["role", "tier"], observed=True).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()