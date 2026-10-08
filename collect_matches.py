"""
Collect ranked solo queue (queue 420) matches from the Riot API.

Strategy
--------
1. Bootstrap: sample seed players from every ladder bucket
   (CHALLENGER/GRANDMASTER/MASTER + IRON..DIAMOND x divisions I-IV).
2. Pull each seed's last 100 ranked solo games -> match IDs (tagged with
   the seed's tier, since solo queue matches players of similar MMR).
3. Fetch full match details, write compact records to data/raw/matches.jsonl.
4. Snowball: participants of fetched matches become new seeds in the same
   bucket, so match discovery is cheap (~1 request per new match).
5. Everything checkpoints to data/state.json; safe to stop and resume.

Usage:
    python collect_matches.py --target 5000
    python collect_matches.py --target 2000 --max-minutes 60
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from riot_client import RiotAPI, slim_match

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "matches.jsonl"
STATE_FILE = RAW_DIR / "state.json"

QUEUE = 420  # ranked solo queue
IDS_PER_CALL = 100
SEEDS_PER_BUCKET_BOOTSTRAP = 12
SEED_POOL_CAP = 500          # max seed players per bucket
FRONTIER_TOPUP = 60          # fetch ids until frontier >= this many
CHECKPOINT_EVERY = 25        # matches

APEX = ["CHALLENGER", "GRANDMASTER", "MASTER"]
DIVISIONS = ["I", "II", "III", "IV"]
TIERS = ["EMERALD", "DIAMOND", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]

BUCKETS = APEX + [f"{t}_{d}" for t in TIERS for d in DIVISIONS]


def new_bucket_state():
    return {"seeds": {}, "collected": 0, "next_seed": 0, "frontier": []}


def load_state():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        for b in BUCKETS:
            state.setdefault("buckets", {}).setdefault(b, new_bucket_state())
        state.setdefault("done", [])
        state.setdefault("matches_written", 0)
        return state
    return {"buckets": {b: new_bucket_state() for b in BUCKETS},
            "done": [], "matches_written": 0}


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE_FILE)


def bootstrap_bucket(api: RiotAPI, state, bucket: str, seeds_per_bucket: int):
    """Fill a bucket's seed pool from the ladder listings (entries include puuids)."""
    b = state["buckets"][bucket]
    if b["seeds"]:
        return True
    if bucket in APEX:
        data = api.apex_league(QUEUE, bucket) or {}
        entries = data.get("entries", [])
        entries.sort(key=lambda e: e.get("leaguePoints", 0), reverse=True)
        puuids = [e["puuid"] for e in entries[: seeds_per_bucket * 3] if e.get("puuid")]
    else:
        tier, _, division = bucket.partition("_")
        entries = api.league_entries(QUEUE, tier, division, page=1) or []
        puuids = [e["puuid"] for e in entries[: seeds_per_bucket * 3] if e.get("puuid")]
    for p in puuids[:seeds_per_bucket]:
        b["seeds"][p] = {"start": 0, "exhausted": False}
    b["next_seed"] = 0
    return len(b["seeds"]) > 0


def top_up_frontier(api: RiotAPI, state, bucket: str):
    """Pull match IDs from the bucket's seed queue until frontier is big enough."""
    b = state["buckets"][bucket]
    seed_items = list(b["seeds"].items())
    if not seed_items:
        return
    attempts = 0
    while len(b["frontier"]) < FRONTIER_TOPUP and attempts < len(seed_items):
        idx = b["next_seed"] % len(seed_items)
        puuid, info = seed_items[idx]
        if info["exhausted"] or info["start"] >= 1000:  # cap history depth
            b["next_seed"] += 1
            attempts += 1
            # pool exhausted? rotate keeps trying others
            if all(s["exhausted"] or s["start"] >= 1000 for s in b["seeds"].values()):
                break
            continue
        ids = api.match_ids(puuid, start=info["start"], count=IDS_PER_CALL, queue=QUEUE)
        info["start"] += IDS_PER_CALL
        if ids is None or len(ids) == 0:
            info["exhausted"] = True
        else:
            done = set(state["done"])
            queued = {mid for bb in state["buckets"].values() for mid, _ in bb["frontier"]}
            for mid in ids:
                if mid not in done and mid not in queued:
                    b["frontier"].append([mid, bucket])
                    queued.add(mid)
        b["next_seed"] += 1
        attempts += 1


def snowball(state, match: dict):
    """Add match participants as seeds in the match's bucket."""
    bucket = match["seedTier"]
    b = state["buckets"].get(bucket)
    if b is None or len(b["seeds"]) >= SEED_POOL_CAP:
        return
    all_seeds = {p for bb in state["buckets"].values() for p in bb["seeds"]}
    for player in match["players"]:
        p = player.get("puuid")
        if p and p not in all_seeds and len(b["seeds"]) < SEED_POOL_CAP:
            b["seeds"][p] = {"start": 0, "exhausted": False}


def pick_bucket(state, target: int) -> str | None:
    """Bucket furthest behind its fair share of the target."""
    best, best_ratio = None, None
    for bucket in BUCKETS:
        b = state["buckets"][bucket]
        share = target / len(BUCKETS)
        ratio = b["collected"] / share if share > 0 else 0
        if len(b["frontier"]) == 0 and all(
            s["exhausted"] or s["start"] >= 1000 for s in b["seeds"].values()
        ) and b["seeds"]:
            ratio += 1.0  # deprioritize starved buckets
        if best_ratio is None or ratio < best_ratio:
            best, best_ratio = bucket, ratio
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000, help="total matches to collect")
    ap.add_argument("--max-minutes", type=int, default=None, help="stop after N minutes")
    ap.add_argument("--seeds-per-bucket", type=int, default=SEEDS_PER_BUCKET_BOOTSTRAP)
    args = ap.parse_args()

    api = RiotAPI()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    stop = {"flag": False}

    def on_sig(_s, _f):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    t0 = time.time()
    print(f"[collect] region={api.region} target={args.target} buckets={len(BUCKETS)}")

    # bootstrap any empty buckets
    for bucket in BUCKETS:
        if not state["buckets"][bucket]["seeds"]:
            ok = bootstrap_bucket(api, state, bucket, args.seeds_per_bucket)
            print(f"[collect] bootstrapped {bucket}: {len(state['buckets'][bucket]['seeds'])} seeds" if ok
                  else f"[collect] WARNING: no seeds for {bucket}")

    new_matches = 0
    try:
        while state["matches_written"] < args.target:
            if args.max_minutes and (time.time() - t0) > args.max_minutes * 60:
                print("[collect] time limit reached")
                break
            bucket = pick_bucket(state, args.target)
            if bucket is None:
                break
            b = state["buckets"][bucket]
            if not b["frontier"]:
                top_up_frontier(api, state, bucket)
                if not b["frontier"]:
                    # this bucket is drained for now; pick again (loop guard below)
                    starved = all(
                        s["exhausted"] or s["start"] >= 1000
                        for s in b["seeds"].values()
                    )
                    if starved and b["seeds"]:
                        print(f"[collect] bucket {bucket} exhausted; will rely on snowball")
                        time.sleep(1)
                        if all(
                            state["buckets"][bb]["frontier"] == [] and all(
                                s["exhausted"] or s["start"] >= 1000
                                for s in state["buckets"][bb]["seeds"].values()
                            )
                            for bb in BUCKETS
                        ):
                            print("[collect] all buckets exhausted — stopping")
                            break
                        continue
                    continue

            match_id, tier = b["frontier"].pop(0)
            if match_id in state["done"]:
                continue
            raw = api.match(match_id)
            state["done"].append(match_id)

            if raw is not None:
                slim = slim_match(raw, tier)
                if slim and slim.get("queueId") == QUEUE and not slim.get("gameEndedInEarlySurrender"):
                    with RAW_FILE.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(slim) + "\n")
                    state["matches_written"] += 1
                    b["collected"] += 1
                    new_matches += 1
                    snowball(state, slim)

            if new_matches and new_matches % CHECKPOINT_EVERY == 0:
                save_state(state)
                el = time.time() - t0
                rate = new_matches / (el / 60) if el > 0 else 0
                print(f"[collect] {state['matches_written']}/{args.target} matches | "
                      f"{rate:.0f}/min | {api.calls_made} api calls | {el/60:.1f} min elapsed")
    finally:
        save_state(state)
        print(f"[collect] DONE/STOPPED: {state['matches_written']} total matches "
              f"({api.calls_made} api calls, {new_matches} new this run). State saved.")


if __name__ == "__main__":
    main()