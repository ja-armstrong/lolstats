"""Dump a per-team data pack for writing the Worlds primer post."""
import pandas as pd

pd.set_option("display.width", 260)

df = pd.read_csv("results/worlds_primer_2026.csv")
st = pd.read_csv("results/player_stints_pro.csv")


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


st["k"] = st["player"].map(norm)
df["k"] = df["rosterName"].map(norm)
cp = (st.sort_values("games", ascending=False)
        .groupby("k")[["champPool"]]
        .first())
df = df.join(cp, on="k")

cols = ["rosterName", "role", "games", "winRate", "outperform",
        "careerOutperform", "carryRate", "swing", "playerAUC",
        "playoffDelta", "consistency", "champPool", "signature",
        "signatureShare", "watchScore", "note"]
for team in df["team"].unique():
    t = df[df.team == team].sort_values("watchScore", ascending=False)
    print("== " + team + " ==")
    print(t[cols].round(3).to_string(index=False))
    print()