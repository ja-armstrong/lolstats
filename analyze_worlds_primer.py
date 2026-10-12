"""
Worlds 2026 primer: rank the players actually attending Worlds by how
interesting / impactful their 2025-2026 play has been, using the tier-1 pro
track.

Method:
- Only players on the official Worlds rosters are ranked (no "last guy who
  played the role" artifacts).
- Each player's history is split into per-team stints (analyze_players_pro).
  A player's score uses their games WITH THE WORLDS TEAM: mean-like stats
  are game-weighted across all stints on that team; swing/AUC come from the
  largest stint on that team.
    outperform  own composite score minus teammates' mean (team cancels out)
    carryRate   share of games as team's best performer
    swing       win-rate gap between their good and bad games
    playerAUC   how predictive their own score is of the team winning
    level       raw performance level vs role x league peers
- watchScore = mean percentile of the five stats within the roster pool.
- careerOutperform (all stints, game-weighted) is shown for context.

Outputs:
  results/worlds_primer_2026.md
  results/worlds_primer_2026.csv
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STINTS = ROOT / "results" / "player_stints_pro.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

# Worlds 2026 qualified teams (display name -> region, seed, stage)
WORLDS_TEAMS = {
    "Gen.G": ("LCK", 1, "Swiss"),
    "Hanwha Life Esports": ("LCK", 2, "Swiss"),
    "T1": ("LCK", 3, "Swiss"),
    "Dplus KIA": ("LCK", 4, "Swiss"),
    "Anyone's Legend": ("LPL", 1, "Swiss"),
    "Bilibili Gaming": ("LPL", 2, "Swiss"),
    "Top Esports": ("LPL", 3, "Swiss"),
    "Invictus Gaming": ("LPL", 4, "Swiss"),
    "G2 Esports": ("LEC", 1, "Swiss"),
    "Movistar KOI": ("LEC", 2, "Swiss"),
    "Karmine Corp": ("LEC", 3, "Play-In"),
    "Team Liquid": ("LCS", 1, "Swiss"),
    "LYON": ("LCS", 2, "Swiss"),
    "Cloud9": ("LCS", 3, "Play-In"),
    "Team Secret Whales": ("LCP", 1, "Swiss"),
    "CTBC Flying Oyster": ("LCP", 2, "Swiss"),
    "MVK Esports": ("LCP", 3, "Play-In"),
    "LOS": ("CBLOL", 1, "Swiss"),
    "FURIA": ("CBLOL", 2, "Play-In"),
}

# official Worlds rosters (starters). notes flag roster caveats.
ROSTERS = {
    "Gen.G": ["Kiin", "Canyon", "Chovy", "Ruler", "Duro"],
    "Hanwha Life Esports": ["Zeus", "Kanavi", "Zeka", "Gumayusi", "Delight"],
    "T1": ["Doran", "Oner", "Faker", "Peyz", "Keria"],
    "Dplus KIA": ["Siwoo", "Lucid", "ShowMaker", "Smash", "Career"],
    "Anyone's Legend": ["Breathe", "Tarzan", "Shanks", "hope", "Kael"],
    "Bilibili Gaming": ["bin", "Xun", "Knight", "Viper", "ON"],
    "Top Esports": ["369*", "Tian", "Creme", "JackeyLove", "Zhuo"],
    "Invictus Gaming": ["TheShy", "Wei", "Rookie", "JiaQi", "meiko"],
    "G2 Esports": ["BrokenBlade", "SkewMond", "Caps", "hans sama", "Labrov"],
    "Movistar KOI": ["Myrwn", "Elyoya", "jojopyun", "Supa", "Alvaro"],
    "Karmine Corp": ["Canna", "Yike", "Kyeahoo", "Caliste", "Busio"],
    "Team Liquid": ["Morgan", "Josedeodo", "Quid", "Yeon", "CoreJJ"],
    "LYON": ["Dhokla", "Inspired", "Saint", "Berserker", "Isles"],
    "Cloud9": ["Thanatos", "Blaber", "Loki", "Tactical", "Vulcan"],
    "Team Secret Whales": ["Pun", "Hizto", "Dire", "Eddie", "Bie"],
    "CTBC Flying Oyster": ["Rest", "Shad0w", "Pout", "Doggo", "KIno"],
    "MVK Esports": ["Kratos", "Gury", "Chika", "Harky", "SiuLoong"],
    "LOS": ["ZEST", "Curse", "Feisty", "Duduhh", "Ackerman"],
    "FURIA": ["Guigo", "Tatu", "Tutsz", "Ayu", "Jojo"],
}
NOTES = {"369": "stepping in at top for ZUIAN (visa issue)"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(ch for ch in s if ch.isascii() and ch.isalnum())


def merge_stints(grp: pd.DataFrame) -> dict:
    """One roster-team record from a player's stints on that team."""
    w = grp["games"].values.astype(float)
    big = grp.loc[grp["games"].idxmax()]
    names = grp["player"].dropna()
    return {
        "player": names.iloc[0] if len(names) else "?",
        "role": big["role"], "league": big["league"],
        "games": int(w.sum()),
        "nStintsOnTeam": len(grp),
        "winRate": np.average(grp["winRate"], weights=w),
        "level": np.average(grp["level"], weights=w),
        "outperform": np.average(grp["outperform"], weights=w),
        "carryRate": np.average(grp["carryRate"], weights=w),
        "swing": big["swing"],
        "playerAUC": big["playerAUC"],
        "playoffDelta": big["playoffDelta"],
        "consistency": big["consistency"],
        "signature": big["signature"],
        "signatureShare": big["signatureShare"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_pro")
    args = ap.parse_args()

    st = pd.read_csv(STINTS).sort_values("lastGame")
    st["teamKey"] = st["team"].map(norm)
    st["nameKey"] = st["player"].map(norm)

    # resolve roster -> data
    team_by_key = {norm(t): (t, *WORLDS_TEAMS[t]) for t in WORLDS_TEAMS}
    roster_rows = []      # resolved (roster player -> data puuid)
    unmatched_names = []  # roster names not found in data at all
    for team, players in ROSTERS.items():
        disp, region, seed, stage = team_by_key[norm(team)]
        tkey = norm(team)
        for p in players:
            clean = p.rstrip("*")
            note = NOTES.get(clean, "")
            cand = st[(st["nameKey"] == norm(clean))]
            on_team = cand[cand["teamKey"] == tkey]
            if len(on_team):
                roster_rows.append({"rosterName": clean, "puuid": on_team["puuid"].iloc[0],
                                    "team": disp, "region": region, "seed": seed,
                                    "stage": stage, "note": note, "resolved": "team"})
            elif len(cand):
                roster_rows.append({"rosterName": clean, "puuid": cand["puuid"].iloc[0],
                                    "team": disp, "region": region, "seed": seed,
                                    "stage": stage,
                                    "note": (note + "; " if note else "")
                                            + f"data shows latest stint on {cand['team'].iloc[-1]}",
                                    "resolved": "other-team"})
            else:
                unmatched_names.append((disp, clean, note))

    rp = pd.DataFrame(roster_rows)
    print("=== Roster resolution ===")
    print(f"resolved to a stint on the roster team: "
          f"{(rp['resolved'] == 'team').sum()}")
    if (rp["resolved"] == "other-team").any():
        print("resolved via other-team stint:")
        print(rp[rp["resolved"] == "other-team"][
            ["rosterName", "team", "note"]].to_string(index=False))
    if unmatched_names:
        print("NO DATA AT ALL for:")
        print(pd.DataFrame(unmatched_names, columns=["team", "player", "note"]
                           ).to_string(index=False))

    # aggregate stints per rostered player ON THE ROSTER TEAM
    rows = []
    for _, rr in rp.iterrows():
        grp = st[(st["puuid"] == rr["puuid"]) & (st["teamKey"] == norm(rr["team"]))]
        if not len(grp):
            # fallback: their other-team stint (flagged)
            grp = st[st["puuid"] == rr["puuid"]]
            if not len(grp):
                continue
        rec = merge_stints(grp)
        rec.update({"rosterName": rr["rosterName"], "team": rr["team"],
                    "region": rr["region"], "seed": rr["seed"],
                    "stage": rr["stage"], "note": rr["note"],
                    "resolved": rr["resolved"]})
        # career context across ALL stints
        g = st[st["puuid"] == rr["puuid"]]
        rec["careerOutperform"] = np.average(g["outperform"], weights=g["games"])
        rec["careerGames"] = int(g["games"].sum())
        rows.append(rec)
    pool = pd.DataFrame(rows)
    pool["gamesOnTeam"] = pool["games"]

    # watch score within the roster pool
    comps = ["outperform", "carryRate", "swing", "playerAUC", "level"]
    pct = {c: pool[c].rank(pct=True) for c in comps}
    pool["watchScore"] = pd.concat(pct, axis=1).mean(axis=1) * 100
    pool["sample"] = np.where(pool["games"] >= 15, "", "low-n")
    pool = pool.sort_values("watchScore", ascending=False).reset_index(drop=True)
    pool.insert(0, "rank", range(1, len(pool) + 1))

    # ---- outputs ----------------------------------------------------------
    def table(df: pd.DataFrame, cols_: list[str], title: str) -> str:
        out = [f"## {title}", "",
               "| " + " | ".join(cols_) + " |",
               "|" + "---|" * len(cols_)]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(
                ("" if pd.isna(r[c]) else
                 (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])))
                for c in cols_) + " |")
        out.append("")
        return "\n".join(out)

    main_cols = ["rank", "rosterName", "role", "team", "region", "seed",
                 "games", "winRate", "outperform", "careerOutperform",
                 "carryRate", "swing", "playerAUC", "watchScore", "sample"]
    md = ["# Worlds 2026 Player Primer",
          "",
          f"Tier-1 pro track, 2025-2026. **Only official Worlds rosters are "
          f"ranked** ({len(pool)} of 95 rostered players have data; "
          f"{len(unmatched_names)} rostered players have no games in the "
          "tier-1 window). Stats use each player's games WITH THEIR WORLDS "
          "TEAM. `outperform` = own composite score minus teammates' mean "
          "(team strength cancels out); `watchScore` = mean percentile of "
          "outperform / carryRate / swing / playerAUC / level within this "
          "pool; `low-n` = < 15 games on the team.",
          ""]
    md.append(table(pool[main_cols], main_cols,
                    "Full ranking — all rostered players with data, top to bottom"))

    md.append(table(pool[pool["sample"] == ""].head(25),
                    ["rosterName", "role", "team", "games", "winRate",
                     "outperform", "carryRate", "swing", "watchScore"],
                    "The watch list — top 25 (full-sample players)"))

    for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]:
        r = pool[pool["role"] == role]
        if len(r):
            md.append(table(r.head(8),
                            ["rosterName", "team", "games", "winRate",
                             "outperform", "carryRate", "swing", "watchScore"],
                            f"Best at {role}"))

    md.append("## Team by team: who to watch")
    md.append("")
    for team, tr in pool.groupby("team"):
        top2 = tr.head(2)
        line = ", ".join(
            f"{r['rosterName']} ({r['role']}, outperform {r['outperform']:+.2f}, "
            f"carry {r['carryRate']:.0%})" for _, r in top2.iterrows())
        region = tr["region"].iloc[0]
        stage = tr["stage"].iloc[0]
        md.append(f"- **{team}** ({region}, {stage}): {line}")
    md.append("")

    md += ["## Caveats",
           "",
           "- Only official 5-man rosters are ranked; substitute players "
           "(T1 Painter, HLE Bluffing, LYON Armao, ...) and players replaced "
           "during the season (BLG Flandre/Wenbo, TES ZUIAN*) are excluded "
           "even if they logged games.",
           "- Stats are box-score-based association, not causation; "
           "shot-calling, drafting and resource routing are invisible "
           "(this hits historic midlaners/supports hardest — read the bottom "
           "of the ranking with that in mind).",
           f"- 'low-n' rows have < 15 games with the team (subs, visa "
           "stand-ins, guest appearances) and are statistically noisy.",
           "- CBLOL domestic games are outside the tier-1 track; FURIA players "
           "only have international-game data, LOS has none.",
           ""]

    (OUT_DIR / "worlds_primer_2026.md").write_text("\n".join(md))
    pool.to_csv(OUT_DIR / "worlds_primer_2026.csv", index=False)

    pd.set_option("display.width", 230)
    print("\n=== FULL RANKING (rostered players with data) ===")
    print(pool[main_cols].round(3).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'worlds_primer_2026.md'}")
    print(f"Saved: {OUT_DIR / 'worlds_primer_2026.csv'}")


if __name__ == "__main__":
    main()