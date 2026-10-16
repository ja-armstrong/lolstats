"""
Collect ONE player's recent ranked solo games, looked up by Riot ID (name#TAG).

    python collect_player.py "killed cat#NA1"
    python collect_player.py "SomeName#KR1" --region kr --n 60

Writes, under data/players/<slug>/ (data/ is gitignored - it holds PUUIDs):
  matches.jsonl   same compact schema as data/raw/matches.jsonl, so
                  prepare_features.load_matches() can parse it unchanged
  profile.json    puuid, Riot ID, level, icon, current rank, collection stats

Re-running only fetches games that are not already on disk, so it is cheap to
refresh a player before rebuilding their report. The collect() function is
also what server.py drives for the in-browser lookup (it takes a progress
callback for the UI).

Dev keys allow 100 requests / 2 min: a first 40-game pull is ~42 requests
(roughly a minute of waiting); refreshes are ~2.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from riot_client import RiotAPI, ROUTES, slim_match

ROOT = Path(__file__).resolve().parent
PLAYER_DIR = ROOT / "data" / "players"
SOLO_QUEUE = 420
MIN_DURATION_S = 600          # same remake filter as prepare_features
SOLO_LEAGUE = "RANKED_SOLO_5x5"
MAX_GAMES = 500               # pages of 100 ids; dev key = ~100 requests / 2 min
PAGE = 100


def split_riot_id(text: str, default_tag: str | None) -> tuple[str, str]:
    """'killed cat#NA1' -> ('killed cat', 'NA1'); bare names need --tag."""
    text = text.strip()
    if "#" in text:
        name, _, tag = text.rpartition("#")
        return name.strip(), tag.strip()
    if not default_tag:
        raise ValueError("Give a Riot ID like 'Name#TAG', or pass a tag (--tag NA1).")
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
    """Prefer league-v4 by-puuid; fall back to the by-summoner route if absent.

    Schema note: division I-IV used to live in `division`; current responses put
    it in `rank` for tiers below Apex, while for Apex tiers `rank` is the numeric
    ladder position. Read both, and never treat a number as a division.
    """
    if entries is None and summoner_id:
        entries = api.league_entries_by_summoner(summoner_id)
    for e in entries or []:
        if e.get("queueType") == SOLO_LEAGUE:
            tier = e.get("tier")
            division = e.get("division")
            rank_field = e.get("rank")
            if not division and rank_field not in (None, "") and not str(rank_field).isdigit():
                division = rank_field
            return {
                "tier": tier, "division": division,
                "leaguePoints": e.get("leaguePoints"),
                "wins": e.get("wins"), "losses": e.get("losses"),
                "rank": rank_field,           # numeric ladder position in Apex
                "veteran": bool(e.get("veteran")),
                "inactive": bool(e.get("inactive")),
                "freshBlood": bool(e.get("freshBlood")),
                "hotStreak": bool(e.get("hotStreak")),
            }
    return None


def fetch_ids(api: RiotAPI, puuid: str, want: int, queue: int, report=None):
    """Most recent `want` match ids, paging 100 at a time until Riot runs out."""
    ids, start, exhausted = [], 0, False
    while len(ids) < want:
        count = min(PAGE, want - len(ids))
        page = api.match_ids(puuid, start=start, count=count, queue=queue)
        if not page:
            exhausted = True
            break
        ids.extend(page)
        if report:
            report("ids", got=len(ids), want=want)
        if len(page) < count:          # short page == end of available history
            exhausted = True
            break
        start += len(page)
    return ids, exhausted


def collect(game_name: str, tag_line: str, region: str = "na1", n: int = 100,
            out_dir: Path | None = None, progress=None) -> dict:
    """Resolve a Riot ID, fetch that player's ranked solo games, return profile.

    progress(stage: str, **info) is called at each phase so server.py can show
    what is happening in the browser.
    """
    def report(stage, **info):
        if progress:
            try:
                progress(stage, **info)
            except Exception:
                pass

    region = region.lower()
    if region not in ROUTES:
        raise ValueError(f"unknown region '{region}'. Valid: {', '.join(sorted(ROUTES))}")
    n = max(1, min(int(n), MAX_GAMES))

    api = RiotAPI(region=region)
    slug = slugify(game_name)
    out_dir = out_dir or PLAYER_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file = out_dir / "matches.jsonl"

    report("resolve", region=region, riotId=f"{game_name}#{tag_line}")
    acct = api.account_by_riot_id(game_name, tag_line)
    if not acct or not acct.get("puuid"):
        raise LookupError(
            f"no account found for '{game_name}#{tag_line}' in {region}. "
            "Check the name/tag spelling and the region: Riot IDs are shard-specific, "
            "so a NA1 account looks 'not found' when queried on euw1."
        )
    puuid = acct["puuid"]

    report("profile")
    summ = api.summoner_by_puuid(puuid) or {}
    rank = find_solo_rank(api.league_entries_by_puuid(puuid), summ.get("id"), api)
    if rank is None:
        tier_label = "UNRANKED"
    elif rank.get("division"):
        tier_label = f"{rank['tier']}_{rank['division']}"
    else:                                    # apex tiers carry no division
        tier_label = rank["tier"]

    report("matchlist")
    ids, exhausted = fetch_ids(api, puuid, n, SOLO_QUEUE, report)
    have = existing_ids(raw_file)
    todo = [m for m in ids if m not in have]
    if todo:
        print_eta = round(len(todo) / 50.0, 1)   # ~100 match calls per 2 minutes
        report("planning", toFetch=len(todo), already=len(have), minutes=print_eta)

    t0, got, skipped = time.time(), 0, {"surrender": 0, "short": 0, "other": 0}
    throttle = max(5, len(todo) // 20)
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
            if i % throttle == 0 or i == len(todo):
                report("collect", done=i, total=len(todo), kept=got,
                       seconds=round(time.time() - t0))

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
        "requestedN": n,
        "idsFound": len(ids),
        "historyExhausted": bool(exhausted),
        "skipped": skipped,
        "apiCalls": api.calls_made,
        "collectedAt": int(time.time()),
        "collectedAtIso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    report("done", games=profile["gamesStored"], fetched=got, skipped=skipped)
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("riot_id", help="Riot ID, e.g. \"killed cat#NA1\"")
    ap.add_argument("--tag", default=None, help="tag line, if riot_id has no #TAG")
    ap.add_argument("--region", default="na1",
                    help="shard: " + " ".join(sorted(ROUTES)))
    ap.add_argument("--n", type=int, default=100,
                    help=f"recent ranked solo games to keep (max {MAX_GAMES}; 100 per id page)")
    ap.add_argument("--out", type=Path, default=None, help="override the output folder")
    args = ap.parse_args()

    try:
        game_name, tag_line = split_riot_id(args.riot_id, args.tag)
    except ValueError as e:
        sys.exit(f"[error] {e}")

    def progress(stage, **info):
        if stage == "resolve":
            print(f'[lookup] {info["riotId"]} on {info["region"]} ...')
        elif stage == "profile":
            print("[lookup] summoner + rank ...")
        elif stage == "matchlist":
            print(f"[matchlist] asking for up to {args.n} recent ranked solo ids ...")
        elif stage == "ids":
            print(f"[matchlist] {info['got']}/{info['want']} ids")
        elif stage == "planning":
            if info["toFetch"] == 0:
                print(f"[collect] all {info['already']} games already on disk")
            else:
                print(f"[collect] {info['already']} cached, {info['toFetch']} to fetch "
                      f"(~{info['minutes']} min at dev-key limits)")
        elif stage == "collect":
            pct = info["done"] / max(1, info["total"]) * 100
            print(f'[collect] {info["done"]}/{info["total"]} ({pct:.0f}%) '
                  f'kept {info["kept"]} | {info["seconds"]}s')
        elif stage == "done":
            print(f"[done] {info['games']} games stored "
                  f"({info['fetched']} fetched this run, skipped {info['skipped']})")

    try:
        profile = collect(game_name, tag_line, args.region, args.n,
                          out_dir=args.out, progress=progress)
    except (RuntimeError, LookupError, ValueError) as e:
        sys.exit(f"[error] {e}")

    print(f"[done] data/players/{profile['slug']}/ "
          f"(rank {profile['tier']}, {profile['apiCalls']} api calls)")
    print(f"[next] python build_player_web.py --slug {profile['slug']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n[stopped] games fetched so far are already saved; re-run to continue.")
