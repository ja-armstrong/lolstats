"""
Riot API client with rate limiting, retries, and region routing.

Personal (dev) keys: 20 requests/sec, 100 requests/2 minutes.
The limiter below enforces both windows conservatively.
"""

import os
import time
import json
from collections import deque
from pathlib import Path
from urllib.parse import quote

import requests

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except ImportError:
    # minimal .env loader fallback
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

# Platform -> regional routing cluster
ROUTES = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}


class RateLimiter:
    """Token-free sliding-window limiter for two concurrent windows."""

    def __init__(self, per_second: int = 20, per_two_minutes: int = 100):
        self.per_second = per_second
        self.per_two_minutes = per_two_minutes
        self.calls = deque()

    def wait(self):
        while True:
            now = time.monotonic()
            while self.calls and self.calls[0] < now - 120:
                self.calls.popleft()
            in_window = len(self.calls)
            in_second = sum(1 for t in self.calls if t > now - 1.0)
            if in_window < self.per_two_minutes and in_second < self.per_second:
                self.calls.append(now)
                return
            # sleep until the oldest call exits the binding window
            if in_window >= self.per_two_minutes:
                sleep_for = max(0.05, self.calls[0] + 120 - now)
            else:
                sleep_for = max(0.05, 1.0 - (now - max(t for t in self.calls if t > now - 1.0)))
            time.sleep(min(sleep_for, 2.0))


class RiotAPI:
    def __init__(self, region: str = "na1", api_key: str | None = None):
        self.region = region
        self.route = ROUTES[region]
        self.key = api_key or os.environ.get("RIOT_API_KEY")
        if not self.key:
            raise RuntimeError(
                "RIOT_API_KEY not set. Put it in a .env file next to the scripts:\n"
                "  RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
            )
        self.limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.key})
        self.calls_made = 0

    def _url(self, host: str, path: str) -> str:
        return f"https://{host}.api.riotgames.com{path}"

    def get(self, path: str, regional: bool = False, params: dict | None = None,
            max_retries: int = 6) -> dict | list | None:
        """GET an endpoint. regional=True uses the routing cluster (match-v5)."""
        host = self.route if regional else self.region
        url = self._url(host, path)
        net_fails = 0
        for attempt in range(1, max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                net_fails = 0
            except requests.RequestException as e:
                # Three in a row is an outage/proxy problem, not a blip: fail the
                # call now instead of sleeping through 6 retries for every game.
                net_fails += 1
                if net_fails >= 3:
                    raise RuntimeError(
                        f"cannot reach {host}.api.riotgames.com "
                        f"({type(e).__name__}: {e}). Gave up after 3 tries - "
                        "that looks like a network/proxy issue, not Riot.")
                time.sleep(2 * attempt)
                continue
            self.calls_made += 1

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "2") or 2)
                time.sleep(retry_after + 1.0)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"{resp.status_code} from {url}: Riot rejected this API key "
                    f"({'missing' if not self.key else self.key[:6] + '...'}). "
                    "Dev keys expire 24h after they are created - request a new one at "
                    "https://developer.riotgames.com/ and update RIOT_API_KEY in .env."
                )
            if resp.status_code == 400:
                # e.g. legacy-format puuids from the 2025 corpus, which current
                # Account-V1 rejects; surface Riot's own wording instead of a bare 400
                raise RuntimeError(f"400 bad request from {url}: {resp.text[:180]}")
            resp.raise_for_status()
        return None

    # ---- convenience wrappers ----
    # league-v4 uses the queue type string, match-v5 uses the numeric queue id
    LEAGUE_QUEUE = "RANKED_SOLO_5x5"

    def apex_league(self, queue: int, tier: str):
        """CHALLENGER / GRANDMASTER / MASTER league lists."""
        return self.get(f"/lol/league/v4/{tier.lower()}leagues/by-queue/{self.LEAGUE_QUEUE}")

    def league_entries(self, queue: int, tier: str, division: str, page: int = 1):
        return self.get(
            f"/lol/league/v4/entries/{self.LEAGUE_QUEUE}/{tier}/{division}",
            params={"page": page},
        )

    def summoner_by_id(self, summoner_id: str):
        return self.get(f"/lol/summoner/v4/summoners/{summoner_id}")

    # ---- single-player lookup: Riot ID -> puuid -> matches ----
    # Summoner-V4 by-name is deprecated; Riot ID (gameName + tagLine) is the
    # supported way to resolve a player, and it is shard-scoped: looking a
    # NA1 account up on the eu route returns 404, not "wrong region" advice.
    def account_by_riot_id(self, game_name: str, tag_line: str):
        g, t = quote(str(game_name), safe=""), quote(str(tag_line), safe="")
        return self.get(f"/riot/account/v1/accounts/by-riot-id/{g}/{t}", regional=True)

    def account_by_puuid(self, puuid: str):
        return self.get(f"/riot/account/v1/accounts/by-puuid/{quote(puuid, safe='')}",
                        regional=True)

    def summoner_by_puuid(self, puuid: str):
        return self.get(f"/lol/summoner/v4/summoners/by-puuid/{quote(puuid, safe='')}")

    def league_entries_by_puuid(self, puuid: str):
        return self.get(f"/lol/league/v4/entries/by-puuid/{quote(puuid, safe='')}")

    def league_entries_by_summoner(self, summoner_id: str):
        return self.get(f"/lol/league/v4/entries/by-summoner/{summoner_id}")

    def match_ids(self, puuid: str, start: int = 0, count: int = 100, queue: int = 420):
        return self.get(
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
            regional=True,
            params={"start": start, "count": count, "queue": queue},
        )

    def match(self, match_id: str):
        return self.get(f"/lol/match/v5/matches/{match_id}", regional=True)


# fields we keep per participant to keep raw files compact
PARTICIPANT_FIELDS = [
    "puuid", "teamId", "teamPosition", "championName", "championId",
    "kills", "deaths", "assists", "win",
    "totalMinionsKilled", "neutralMinionsKilled", "goldEarned",
    "totalDamageDealtToChampions", "visionScore",
    "wardsPlaced", "wardsKilled",
    # Riot ID fields as far as match-v5 discloses them: riotIdGameName comes back,
    # riotIdTagLine is masked to None for other players, and summonerName is now "".
    # Storing the game name still makes collected games half-addressable by name;
    # any lookup needs the tag from whoever is asking.
    "riotIdGameName", "riotIdTagLine",
]


def slim_match(raw: dict, seed_tier: str) -> dict | None:
    """Reduce a full match-v5 payload to what we need."""
    try:
        info = raw["info"]
        participants = info["participants"]
    except (KeyError, TypeError):
        return None
    if len(participants) != 10:
        return None
    players = []
    for p in participants:
        row = {k: p.get(k) for k in PARTICIPANT_FIELDS}
        ch = p.get("challenges") or {}
        row["soloKills"] = ch.get("soloKills")
        row["killParticipation"] = ch.get("killParticipation")
        players.append(row)
    return {
        "matchId": raw["metadata"]["matchId"],
        "queueId": info.get("queueId"),
        "gameDuration": info.get("gameDuration"),
        "gameVersion": info.get("gameVersion"),
        "gameStartTimestamp": info.get("gameStartTimestamp"),
        "gameEndedInEarlySurrender": info.get("gameEndedInEarlySurrender", False),
        "seedTier": seed_tier,
        "players": players,
    }