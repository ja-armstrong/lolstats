"""
Dev tool: fabricate a player folder from the already-collected corpus, so the
single-player pipeline can be exercised without spending Riot API requests.

    python make_offline_player.py                 # -> data/players/offline-demo/
    python make_offline_player.py --games 60 --bucket CHALLENGER

It picks the corpus puuid that appears in the most collected matches, writes
their matches in the real record schema, and invents a profile (the Riot ID is
NOT a real account). Useful for testing scoring changes and for eyeballing the
report page; never publish the output - web/players/ is gitignored for that
reason, and this slug is excluded by name too.

Delete afterwards:  Remove-Item -Recurse data\players\offline-demo
"""

import argparse
import collections
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "matches.jsonl"
OUT = ROOT / "data" / "players" / "offline-demo"
SOLO_QUEUE = 420
MIN_DURATION_S = 600
DEMO_ID = "Offline Demo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--bucket", default=None, help="only seed matches from this tier, e.g. CHALLENGER")
    ap.add_argument("--slug", default="offline-demo")
    ap.add_argument("--clean", action="store_true", help="delete the demo folder and exit")
    args = ap.parse_args()

    out = ROOT / "data" / "players" / args.slug
    if args.clean:
        shutil.rmtree(out, ignore_errors=True)
        stale = ROOT / "web" / "players" / f"{args.slug}.html"
        if stale.exists():
            stale.unlink()
        print(f"[clean] removed {out} and {stale if stale.exists() else stale}")
        return

    if not RAW.exists():
        raise SystemExit(f"no corpus at {RAW} - run collect_matches.py first")

    by_player = collections.defaultdict(list)
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("queueId") != SOLO_QUEUE or (m.get("gameDuration") or 0) < MIN_DURATION_S:
                continue
            if args.bucket and not str(m.get("seedTier", "")).startswith(args.bucket):
                continue
            for p in m["players"]:
                if p.get("puuid") and p.get("teamPosition"):
                    by_player[p["puuid"]].append((line, m.get("seedTier")))
    if not by_player:
        raise SystemExit("no players matched that filter")

    puuid, recs = max(by_player.items(), key=lambda kv: len(kv[1]))
    recs = recs[: args.games]
    tier = collections.Counter(t for _, t in recs).most_common(1)[0][0]
    lines = [l for l, _ in recs]

    out.mkdir(parents=True, exist_ok=True)
    (out / "matches.jsonl").write_text("".join(lines), encoding="utf-8")

    wins = 0
    for l in lines:
        m = json.loads(l)
        pl = next(p for p in m["players"] if p["puuid"] == puuid)
        wins += bool(pl.get("win"))
    base, _, div = str(tier).partition("_")
    (out / "profile.json").write_text(json.dumps({
        "slug": args.slug,
        "gameName": DEMO_ID, "tagLine": "NA1", "riotId": f"{DEMO_ID}#NA1",
        "puuid": puuid, "summonerId": "OFFLINE-TEST", "summonerLevel": 100,
        "profileIconId": 5245, "region": "na1", "route": "americas",
        "soloRank": {"tier": base, "division": div or "I", "leaguePoints": 0,
                     "wins": wins, "losses": len(lines) - wins, "rank": None,
                     "veteran": False, "inactive": False, "freshBlood": False},
        "tier": tier, "queue": SOLO_QUEUE, "gamesStored": len(lines),
        "requestedN": args.games, "skipped": {}, "apiCalls": 0,
        "collectedAt": int(time.time()),
        "collectedAtIso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "OFFLINE DEV ARTIFACT - games are a real corpus player's, identity is invented",
    }, indent=2), encoding="utf-8")

    print(f"[demo] puuid={puuid[:10]}... {len(lines)} games, {wins} wins, tier={tier}")
    print(f"[demo] wrote {out}")
    print(f"[demo] then: python build_player_web.py --slug {args.slug}")
    print("[demo] identity is fabricated - do not publish this report")


if __name__ == "__main__":
    main()
