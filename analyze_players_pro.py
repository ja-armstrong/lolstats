"""
Player-focused analysis on the tier-1 pro track.

Players change teams, so every player's record is split into STINTS:
consecutive stretches with the same team (sorted by game timestamp). All
metrics are computed per stint; the career table aggregates across stints.

Per-stint metrics:
- outperform: own composite score (z-scored within role x league) minus the
  mean of the four teammates' scores, per game, averaged. Team strength
  cancels out — "does this player beat the level of the team they play on?"
- carryRate: share of games where they posted the team's best score
- swing: win rate in their above-median games minus below-median games
- playerAUC: AUC of own score for win within the stint (>= 25 games only)
- withMinusWithout: team win rate with vs without them (real absences only)
- level, consistency, playoffDelta, champion pool, signature champion

Outputs:
  results/player_stints_pro.csv    one row per player x team stint
  results/player_impact_pro.csv   career aggregate (current team labeled)
  results/player_impact_pro.md    leaderboards
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "data" / "processed" / "player_games_pro.parquet"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)


def stint_metrics(r: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Aggregate one player-team stint (r = that player's rows, time-ordered)."""
    med = r["diff"].median()
    hi, lo = r[r["diff"] >= med], r[r["diff"] < med]
    names = r["playerName"].dropna()
    auc = np.nan
    if len(r) >= 25 and 0 < r["win"].mean() < 1:
        try:
            auc = roc_auc_score(r["win"].astype(int), r["score_zt"])
        except ValueError:
            pass
    team = r["teamName"].iloc[0]
    puuid = r["puuid"].iloc[0]
    # with/without: team's win rate in games this player did not play
    team_games = df[df["teamName"] == team]
    without = team_games[~team_games["matchId"].isin(set(r["matchId"]))]
    n_without = without["matchId"].nunique()
    wr_without = (without.groupby("matchId")["win"].first().mean()
                  if n_without else np.nan)
    champ_counts = r["champion"].value_counts()
    return {
        "player": names.iloc[-1] if len(names) else puuid,
        "puuid": puuid,
        "role": r["role"].mode().iloc[0],
        "rolesPlayed": r["role"].nunique(),
        "team": r["teamName"].iloc[0],
        "league": r["league"].mode().iloc[0],
        "games": len(r),
        "firstGame": r["ts"].min(), "lastGame": r["ts"].max(),
        "winRate": r["win"].mean(),
        "level": r["score_zt"].mean(),
        "outperform": r["diff"].mean(),
        "consistency": r["diff"].std(),
        "carryRate": r["is_carry"].mean(),
        "swing": (hi["win"].mean() - lo["win"].mean())
                 if len(hi) and len(lo) else np.nan,
        "playerAUC": auc,
        "teamWR_with": team_games[team_games["puuid"] == puuid]
                        .groupby("matchId")["win"].first().mean(),
        "teamWR_without": wr_without,
        "nTeamGamesWithout": n_without,
        "playoffDelta": (r[r["playoffs"]]["diff"].mean()
                         - r[~r["playoffs"]]["diff"].mean())
                        if r["playoffs"].any() and (~r["playoffs"]).any() else np.nan,
        "champPool": r["champion"].nunique(),
        "signature": champ_counts.index[0],
        "signatureShare": champ_counts.iloc[0] / len(r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_FILE)
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--tag", default="_pro")
    args = ap.parse_args()

    df = pd.read_parquet(args.input).sort_values("ts")

    # ---- per-game teammate-relative differential -------------------------
    team_sum = df.groupby(["matchId", "teamId"], observed=True)["score_zt"].transform("sum")
    df["teammate_mean"] = (team_sum - df["score_zt"]) / 4.0
    df["diff"] = df["score_zt"] - df["teammate_mean"]
    df["team_max"] = df.groupby(["matchId", "teamId"], observed=True)["score_zt"].transform("max")
    df["is_carry"] = df["score_zt"] >= df["team_max"] - 1e-9

    # ---- split players into team stints ----------------------------------
    # a new stint starts when the team changes OR after a >=60-day gap
    # (separates international guest appearances / returns to a team)
    prev = df.groupby("puuid", observed=True)[["teamName", "ts"]].shift()
    df["new_stint"] = (prev["teamName"].isna()
                       | (df["teamName"] != prev["teamName"])
                       | ((df["ts"] - prev["ts"]) > 60 * 86400 * 1000))
    df["stint_id"] = df.groupby("puuid", observed=True)["new_stint"].cumsum()
    stint_key = ["puuid", "stint_id"]

    stints = pd.DataFrame(
        [stint_metrics(g, df)
         for _, g in df.groupby(stint_key, observed=True, sort=False)])
    stints = stints.sort_values(["player", "firstGame"]).reset_index(drop=True)
    stints["stint"] = stints.groupby("puuid").cumcount() + 1
    stints = stints.sort_values(["outperform"], ascending=False).reset_index(drop=True)
    stints.to_csv(OUT_DIR / f"player_stints{args.tag}.csv", index=False)

    # ---- career aggregate across stints (weighted by games) --------------
    def agg_career(g: pd.DataFrame) -> pd.Series:
        w = g["games"].values.astype(float)
        last_stint = g.sort_values("lastGame").iloc[-1]
        auc_vals = g["playerAUC"].dropna()
        auc_w = g.loc[auc_vals.index, "games"].values.astype(float)
        return pd.Series({
            "player": last_stint["player"], "puuid": g["puuid"].iloc[0],
            "role": g["role"].mode().iloc[0], "rolesPlayed": g["rolesPlayed"].max(),
            "team": last_stint["team"], "league": last_stint["league"],
            "nStints": len(g),
            "games": int(w.sum()),
            "winRate": np.average(g["winRate"], weights=w),
            "level": np.average(g["level"], weights=w),
            "outperform": np.average(g["outperform"], weights=w),
            "carryRate": np.average(g["carryRate"], weights=w),
            "playerAUC": np.average(auc_vals, weights=auc_w) if len(auc_vals) else np.nan,
            "champPool": g["champPool"].max(),
        })

    career = pd.DataFrame(
        [agg_career(g) for _, g in stints.groupby("puuid", observed=True)])
    career = career[career["games"] >= args.min_games]
    career = career.sort_values("outperform", ascending=False).reset_index(drop=True)
    career.to_csv(OUT_DIR / f"player_impact{args.tag}.csv", index=False)

    # ---- leaderboards ----------------------------------------------------
    pd.set_option("display.width", 220)
    print(f"{len(stints)} stints | {len(career)} players with >= "
          f"{args.min_games} games | {df['matchId'].nunique()} games\n")
    cols = ["player", "role", "team", "league", "games", "winRate", "level",
            "outperform", "carryRate"]
    print("=== Top 15 career outperformers (vs their own teammates) ===")
    print(career.head(15)[cols].round(3).to_string(index=False))
    print("\n=== Bottom 15 ===")
    print(career.tail(15)[cols].round(3).to_string(index=False))
    print("\n=== Top 15 carry rate (career) ===")
    print(career.sort_values("carryRate", ascending=False).head(15)[
        ["player", "role", "team", "games", "carryRate", "outperform"]].round(3).to_string(index=False))
    print("\n=== Multi-team players (stint view) ===")
    multi = stints[stints["puuid"].isin(
        stints["puuid"].value_counts()[lambda s: s > 1].index)]
    print(multi[multi["games"] >= 5].sort_values(["player", "firstGame"])[
        ["player", "team", "league", "games", "outperform", "carryRate", "winRate"]
    ].round(3).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / f'player_stints{args.tag}.csv'}")
    print(f"Saved: {OUT_DIR / f'player_impact{args.tag}.csv'}")


if __name__ == "__main__":
    main()