"""
Outlier scan over the Worlds 2026 roster pool.

For every player-stat axis in the primer table, flags the players at the
extremes (top and bottom), and builds a master "storyline list": everyone
who is extreme (|z| >= 1.5) on ANY axis. Also renders two quadrant charts:

  1. outperform (x) vs swing (y)     -> engines vs X-factors vs passengers
  2. level (x) vs outperform (y)     -> stars on stacked teams vs weak-team engines

Outputs:
  results/outliers_pro.md
  results/outliers_quadrants.png
  results/outliers_level_vs_outperform.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "results" / "worlds_primer_2026.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

ROLE_COLORS = {"TOP": "#e6194b", "JUNGLE": "#3cb44b", "MIDDLE": "#4363d8",
               "BOTTOM": "#f58231", "SUPPORT": "#911eb4"}

# axis -> (display, higher-is interpretation, lower-is interpretation)
AXES = {
    "outperform": ("Outperformance vs teammates (z)",
                   "consistently beats his own teammates' level",
                   "trails his own teammates' level"),
    "careerOutperform": ("Career outperformance (all stints, z)",
                         "career-long outperformer of every team he's been on",
                         "career-long underperformer relative to teammates"),
    "carryRate": ("Carry rate (share of games as team's best)",
                  "his team's best performer unusually often",
                  "rarely his team's top performer"),
    "swing": ("Swing (win-rate gap: good games vs bad games)",
              "boom-bust: his good games are disproportionately wins",
              "metronome: his form barely tracks the result"),
    "playerAUC": ("Player AUC (own score predicts the win)",
                  "the game flows through him — his statline decides outcomes",
                  "his statline is decoupled from the result (system player)"),
    "level": ("Raw level (vs role x league peers, z)",
              "elite in absolute terms",
              "well below peers in absolute terms"),
    "consistency": ("Differential std across games (lower = steadier)",
                    "high-variance performer (wild game-to-game swings)",
                    "remarkably steady performer"),
    "playoffDelta": ("Playoffs minus regular season (outperform, z)",
                     "levels up in playoffs",
                     "drops off in playoffs"),
    "signatureShare": ("Signature-champion share of games",
                       "extreme one-trick / signature pick",
                       "huge champion pool"),
    "winRate": ("Team win rate in his games",
                "plays on winners (team effect, not individual!)",
                "plays on losers (team effect, not individual!)"),
}

Z_OUT = 1.5   # |z| threshold for the master storyline list
TOP_K = 5     # per-axis leaderboard depth


def main():
    df = pd.read_csv(IN_FILE)
    df["label"] = df["rosterName"]

    md = ["# Outlier scan — Worlds 2026 roster pool",
          "",
          f"{len(df)} rostered players. 'z' = standard score within this "
          f"pool (how many pool-standard-deviations from the pool mean). "
          f"Master storyline list = players with |z| >= {Z_OUT} on at least "
          "one axis.", ""]

    # ---- per-axis leaderboards ------------------------------------------
    extreme_rows = []
    for col, (disp, hi_story, lo_story) in AXES.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        mu, sd = vals.mean(), vals.std()
        if sd == 0 or np.isnan(sd):
            continue
        df.loc[vals.index, col + "_z"] = (df.loc[vals.index, col] - mu) / sd

        top = df.nlargest(TOP_K, col)
        bot = df.nsmallest(TOP_K, col)
        md.append(f"## {disp}")
        md.append("")
        md.append(f"**High — {hi_story}:**")
        for _, r in top.iterrows():
            md.append(f"- **{r['label']}** ({r['role']}, {r['team']}) "
                      f"{r[col]:+.3f} (z {r[col + '_z']:+.1f}) "
                      f"— {int(r['games'])} games")
        md.append("")
        md.append(f"**Low — {lo_story}:**")
        for _, r in bot.iterrows():
            md.append(f"- **{r['label']}** ({r['role']}, {r['team']}) "
                      f"{r[col]:+.3f} (z {r[col + '_z']:+.1f}) "
                      f"— {int(r['games'])} games")
        md.append("")

        for _, r in df.iterrows():
            z = r.get(col + "_z")
            if pd.notna(z) and abs(z) >= Z_OUT:
                extreme_rows.append({
                    "player": r["label"], "role": r["role"], "team": r["team"],
                    "axis": disp, "value": r[col], "z": z,
                    "direction": "high" if z > 0 else "low",
                    "games": int(r["games"]),
                })

    # ---- master storyline list ------------------------------------------
    ex = pd.DataFrame(extreme_rows)
    if len(ex):
        counts = ex.groupby("player").agg(
            n_axes=("axis", "nunique"), axes=("axis", lambda s: "; ".join(s)))
        counts = counts.sort_values("n_axes", ascending=False)
        md.append("## Master storyline list (|z| >= "
                  f"{Z_OUT} on any axis)")
        md.append("")
        for player, row in counts.iterrows():
            r0 = df[df["label"] == player].iloc[0]
            md.append(f"- **{player}** ({r0['role']}, {r0['team']}) — extreme "
                      f"on {row['n_axes']} ax"
                      f"{'is' if row['n_axes'] == 1 else 'es'}: "
                      f"{row['axes']}")
        md.append("")
        ex_out = ex.copy()
        ex_out["_k"] = ex_out["z"].abs()
        ex_out = ex_out.sort_values(["player", "_k"], ascending=[True, False])
        ex_out.drop(columns="_k").to_csv(
            OUT_DIR / "outliers_pro.csv", index=False)

    # ---- quadrant charts --------------------------------------------------
    def scatter(x, y, xlabel, ylabel, title, fname):
        d = df.dropna(subset=[x, y]).copy()
        fig, ax = plt.subplots(figsize=(11, 8.5))
        for role, color in ROLE_COLORS.items():
            rd = d[d["role"] == role]
            ax.scatter(rd[x], rd[y], s=20 + 1.2 * rd["games"], alpha=0.55,
                       color=color, label=role, edgecolors="white", lw=0.5)
        # quadrant guides at pool medians
        ax.axvline(d[x].median(), color="gray", ls=":", lw=1)
        ax.axhline(d[y].median(), color="gray", ls=":", lw=1)
        # label extremes and watch-list names
        zx = (d[x] - d[x].mean()) / d[x].std()
        zy = (d[y] - d[y].mean()) / d[y].std()
        for _, r in d.iterrows():
            if abs(zx[r.name]) > 1.4 or abs(zy[r.name]) > 1.4:
                ax.annotate(r["label"], (r[x], r[y]), fontsize=7.5,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=150)
        plt.close(fig)

    scatter("outperform", "swing",
            "Outperformance vs own teammates (z) ->",
            "Swing: his good games are wins ->",
            "Engines vs X-factors: performance gap to teammates vs result-sensitivity",
            "outliers_quadrants.png")
    scatter("level", "outperform",
            "Raw level (z) ->",
            "Outperformance vs own teammates (z) ->",
            "Stars on stacked teams (top-right) vs weak-team engines (bottom-right)",
            "outliers_level_vs_outperform.png")

    md += ["## Quadrant charts",
           "",
           "- `outliers_quadrants.png` — x: outperformance vs teammates, "
           "y: swing. Top-right = engines who also decide games; bottom-right "
           "= steady outperformers; top-left = volatile underliers.",
           "- `outliers_level_vs_outperform.png` — x: raw level, y: "
           "outperformance. Top-right = genuinely elite AND the gap on his "
           "team; bottom-right = outperforming a weak roster; bottom-left = "
           "dragged by strong teammates.",
           ""]

    (OUT_DIR / "outliers_pro.md").write_text("\n".join(md))

    # ---- console ----------------------------------------------------------
    pd.set_option("display.width", 230)
    if len(ex):
        lines = []
        for player, grp in ex.groupby("player"):
            r0 = df[df["label"] == player].iloc[0]
            parts = [f"{row['axis'].split('(')[0].strip()} "
                     f"{'HIGH' if row['direction'] == 'high' else 'LOW'} "
                     f"(z {row['z']:+.1f})" for _, row in
                     grp.sort_values("z", key=lambda s: -s.abs()).iterrows()]
            lines.append((len(grp), f"{r0['role']:8s} {player:12s} "
                          f"({r0['team']}): " + " | ".join(parts)))
        lines.sort(key=lambda t: -t[0])
        print(f"Players extreme (|z|>={Z_OUT}) on at least one axis:")
        for _, line in lines:
            print("  " + line)
    print(f"\nSaved: {OUT_DIR / 'outliers_pro.md'}")
    print(f"Saved: {OUT_DIR / 'outliers_quadrants.png'}")
    print(f"Saved: {OUT_DIR / 'outliers_level_vs_outperform.png'}")


if __name__ == "__main__":
    main()