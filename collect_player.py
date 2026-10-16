"""
Collect ONE player's recent ranked solo games, looked up by Riot ID (name#TAG).

    python collect_player.py "killed cat#NA1"
    python collect_player.py "SomeName#KR1" --region kr --n 60

Writes, under data/players/<slug>/ (data/ is gitignored — it holds PUUIDs):
  matches.jsonl   same compact schema as data/raw/matches.jsonl, so
                  prepare_features.load_matches() can parse it unchanged
  profile.json    puuid, Riot ID, level, icon, current rank, collection stats

Re-running only fetches games that are not already on disk, so it is cheap to
refresh a player before rebuilding their report.

Dev keys allow 100 requests / 2 min: a first 40-game pull is ~42 requests
(roughly a minute of waiting); refreshes are ~2.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from riot_client import RiotAPI, slim_match

ROOT = Path(__file__).resolve().parent
PLAYER_DIR = ROOT / "data" / "players"
SOLO_QUEUE = 420
MIN_DURATION_S = 600          # same remake filter as prepare_features
SOLO_LEAGUE = "RANKED_SOLO_5x5"


def split_riot_id(text: str, default_tag: str | None) -> tuple[str, str]:
    """'killed cat#NA1' -> ('killed cat', 'NA1'); bare names need --tag."""
    text = text.strip()
    if "#" in text:
        name, _, tag = text.rpartition("#")
        return name.strip(), tag.strip()
    if not default_tag:
        sys.exit("Give a Riot ID like 'Name#TAG', or pass --tag (e.g. --tag NA1).")
    return text, default_tag.strip()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "player"


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["matchId"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def find_solo_rank(entries, summoner_id: str, api: RiotAPI):
    """Prefer league-v4 by-puuid; fall back to the by-summoner route if absent."""
    if entries is None and summoner_id:
        entries = api.league_entries_by_summoner(summoner_id)
    for e in entries or []:
        if e.get("queueType") == SOLO_LEAGUE:
            return {
                "tier": e.get("tier"), "division": e.get("division"),
                "leaguePoints": e.get("leaguePoints"),
                "wins": e.get("wins"), "losses": e.get("losses"),
                "rank": e.get("rank"),          # present for apex tiers
                "veteran": bool(e.get("veteran")),
                "inactive": bool(e.get("inactive")),
                "freshBlood": bool(e.get("freshBlood")),
            }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("riot_id", help="Riot ID, e.g. \"killed cat#NA1\"")
    ap.add_argument("--tag", default=None, help="tag line, if riot_id has no #TAG")
    ap.add_argument("--region", default="na1",
                    help="shard: na1 br1 la1 la2 euw1 eun1 tr1 ru kr jp1 oc1 ph2 sg2 th2 tw2 vn2")
    ap.add_argument("--n", type=int, default=40, help="recent ranked solo games to keep (max 100)")
    ap.add_argument("--out", type=Path, default=None, help="override the output folder")
    args = ap.parse_args()

    game_name, tag_line = split_riot_id(args.riot_id, args.tag)
    api = RiotAPI(region=args.region)
    slug = slugify(game_name)
    out_dir = args.out or PLAYER_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file = out_dir / "matches.jsonl"

    print(f'[lookup] "{game_name}#{tag_line}" on {api.region} ({api.route}) ...')
    acct = api.account_by_riot_id(game_name, tag_line)
    if not acct or not acct.get("puuid"):
        sys.exit(
            f"[lookup] no account found. Check the name/tag spelling and the region "
            f"(--region). Riot IDs are shard-specific: '{game_name}#{tag_line}' exists "
            f"only in {args.region} if it exists at all."
        )
    puuid = acct["puuid"]
    print(f"[lookup] puuid={puuid[:12]}...  summoner level/icon ...")

    summ = api.summoner_by_puuid(puuid) or {}
    rank = find_solo_rank(api.league_entries_by_puuid(puuid), summ.get("id"), api)
    tier_label = f"{rank['tier']}_{rank['division']}" if rank else "UNRANKED"
    print(f"[lookup] rank: {tier_label}"
          + (f" ({rank['leaguePoints']} LP, {rank['wins']}-{rank['losses']})" if rank else ""))

    ids = api.match_ids(puuid, count=min(max(args.n, 1), 100), queue=SOLO_QUEUE) or []
    print(f"[matchlist] {len(ids)} ranked solo game ids returned")

    have = existing_ids(raw_file)
    todo = [m for m in ids if m not in have]
    print(f"[collect] {len(have)} already on disk, {len(todo)} to fetch")

    t0, got, skipped = time.time(), 0, {"surrender": 0, "short": 0, "other": 0}
    with raw_file.open("a", encoding="utf-8") as f:
        for i, mid in enumerate(todo, 1):
            raw = api.match(mid)
            if raw is None:
                skipped["other"] += 1
                continue
            slim = slim_match(raw, tier_label)
            if slim is None or slim.get("queueId") != SOLO_QUEUE:
                skipped["other"] += 1
                continue
            if slim.get("gameEndedInEarlySurrender"):
                skipped["surrender"] += 1
                continue
            if (slim.get("gameDuration") or 0) < MIN_DURATION_S:
                skipped["short"] += 1
                continue
            if not any(p.get("puuid") == puuid for p in slim["players"]):
                skipped["other"] += 1
                continue
            f.write(json.dumps(slim) + "\n")
            got += 1
            if i % 10 == 0:
                el = time.time() - t0
                print(f"[collect] {i}/{len(todo)} fetched "
                      f"({got} kept, {api.calls_made} api calls, {el:.0f}s)")

    stored = existing_ids(raw_file)
    profile = {
        "slug": slug,
        "gameName": acct.get("gameName", game_name),
        "tagLine": acct.get("tagLine", tag_line),
        "riotId": f"{acct.get('gameName', game_name)}#{acct.get('tagLine', tag_line)}",
        "puuid": puuid,
        "summonerId": summ.get("id"),
        "summonerLevel": summ.get("summonerLevel"),
        "profileIconId": summ.get("profileIconId"),
        "region": api.region,
        "route": api.route,
        "soloRank": rank,
        "tier": tier_label,
        "queue": SOLO_QUEUE,
        "gamesStored": len(stored),
        "requestedN": args.n,
        "skipped": skipped,
        "apiCalls": api.calls_made,
        "collectedAt": int(time.time()),
        "collectedAtIso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"\n[done] {profile['gamesStored']} games in {raw_file}")
    if skipped:
        print(f"[done] skipped: {skipped}")
    print(f"[next] python build_player_web.py --slug {slug}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"[error] {e}")
    except KeyboardInterrupt:
        sys.exit("\n[stopped] games fetched so far are already saved; re-run to continue.")
