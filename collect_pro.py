"""
Pro-play data track: Oracle's Elixir -> pipeline-native records.

OE publishes one CSV per calendar year (one row per participant-game, plus a
'team' row per side). We filter to tier-1 leagues only, keep complete-data
games, and convert to the same compact record shape that collect_matches.py /
slim_match() produce, so prepare_features and the analysis scripts can run on
either population.

Tier 1 = LCK, LPL, LEC, LCS, LCP (the Big 5) + international events
(Worlds / MSI / First Stand / EWC). Academy and regional leagues are excluded.

Usage:
    python collect_pro.py                     # download (if needed) + convert
    python collect_pro.py --csv data/pro/oe_2025.csv data/pro/oe_2026.csv
    python collect_pro.py --download-only     # just fetch the CSVs
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PRO_DIR = ROOT / "data" / "pro"
RAW_PRO_FILE = ROOT / "data" / "raw" / "pro_matches.jsonl"

# per-year Google Drive file ids (files are updated in place, ids are stable)
OE_FILES = {
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}

YEARS = [2025, 2026]
MIN_DURATION_S = 600  # drop remakes / 3-minute forfeits, same as ladder track

TIER1_LEAGUES = {"LCK", "LPL", "LEC", "LCS", "LCP"}
# OE league tags for international events (case-insensitive prefix match)
# WLDs = Worlds, WLRQ = Worlds regional qualifiers
INTERNATIONAL = ["WLD", "WLRQ", "WCS", "MSI", "FST", "EWC"]

POSITION_MAP = {"top": "TOP", "jgl": "JUNGLE", "jng": "JUNGLE", "mid": "MIDDLE",
                "bot": "BOTTOM", "sup": "SUPPORT", "support": "SUPPORT"}


def is_tier1(league: str) -> bool:
    lg = (league or "").strip().upper()
    return lg in TIER1_LEAGUES or any(lg.startswith(p) for p in INTERNATIONAL)


def download_year(year: int, dest: Path, max_retries: int = 3) -> bool:
    """Fetch the OE csv for a year via Google Drive. Detects quota errors."""
    import urllib.request

    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    file_id = OE_FILES[year]
    url = (f"https://drive.usercontent.google.com/download?"
           f"id={file_id}&export=download&confirm=t")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            if b"Quota exceeded" in data[:2048] or len(data) < 10_000:
                print(f"[collect_pro] {year}: Drive quota block "
                      f"(attempt {attempt}/{max_retries})")
            else:
                dest.write_bytes(data)
                print(f"[collect_pro] {year}: saved {len(data)/1e6:.1f} MB -> {dest}")
                return True
        except Exception as e:
            print(f"[collect_pro] {year}: {type(e).__name__}: {e}")
        time.sleep(10 * attempt)
    return False


def pick(cols: dict, *names):
    for n in names:
        if n in cols:
            return cols[n]
    return None


def convert(csv_path: Path) -> int:
    """Convert one OE csv to pipeline-native records. Returns games written."""
    use_guess = None
    header = pd.read_csv(csv_path, nrows=0)
    # OE column names are stable-ish; normalize for lookup
    norm = {c.strip().lower().replace(" ", "").replace("_", ""): c for c in header.columns}
    # sanity: a few columns we consider mandatory
    for req in ["gameid", "league", "position", "result"]:
        if req not in norm:
            raise SystemExit(f"{csv_path}: missing expected column '{req}' "
                             f"(has {list(header.columns)[:30]}...)")

    get = lambda k: norm.get(k)  # noqa: E731
    def one(*names):  # first available column across OE naming generations
        for n in names:
            if norm.get(n):
                return norm[n]
        return None
    usecols = [c for c in [
        get("gameid"), get("datacompleteness"), get("league"), get("year"),
        get("split"), get("playoffs"), get("date"), get("patch"),
        get("game"), get("participantid"), get("side"), get("position"),
        get("playername"), get("playerid"), get("teamname"), get("champion"),
        get("gamelength"), get("result"),
        get("kills"), get("deaths"), get("assists"),
        get("teamkills"), get("teamdeaths"),
        one("minionkills", "minionskilled"),
        one("monsterkills", "monsterskilled"),
        one("totalcs"),
        get("earnedgold"), get("totalgold"),
        get("damagetochampions"), get("visionscore"),
        get("wardsplaced"), get("wardskilled"), get("kp"),
        one("solokills"), get("url"),
    ] if c is not None]

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    # OE renamed/spaced some columns across years ("total cs", "minionkills",
    # "jng") — normalize once so all downstream access is uniform
    df.columns = [c.strip().lower().replace(" ", "").replace("_", "")
                  for c in df.columns]
    total_rows = len(df)

    if "position" in df.columns:  # drop the per-team summary rows
        df = df[df["position"].astype(str).str.lower() != "team"]
    # OE's completeness flag is unreliable (e.g. all LPL 2025 games are marked
    # 'partial' while every core stat is present). Filter on core fields
    # instead; games that lose rows to this get dropped whole below.
    core = [c for c in ["gameid", "position", "playerid", "champion",
                        "gamelength", "result", "kills", "deaths", "assists",
                        "totalcs", "earnedgold", "totalgold",
                        "damagetochampions", "visionscore"]
            if c in df.columns]
    df = df.dropna(subset=core)
    df = df[df["league"].map(is_tier1)]
    if "year" in df.columns:
        df = df[df["year"].isin(YEARS)]
    if "playerid" in df.columns:
        df = df[df["playerid"].notna()]  # subs recorded without player ids
    df = df.drop_duplicates(subset=["gameid", "participantid"])

    written = 0
    RAW_PRO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PRO_FILE.open("a", encoding="utf-8") as out:
        for game_id, g in df.groupby("gameid", sort=False):
            if len(g) != 10:
                continue  # incomplete side/rows -> skip the game entirely
            dur = float(pick_col(g, "gamelength") or 0)
            if dur < MIN_DURATION_S:
                continue
            league = g["league"].iloc[0]
            date = str(pick_col(g, "date") or "")
            ts = 0
            if date:
                try:
                    ts = int(pd.Timestamp(date).value // 1e6)
                except Exception:
                    ts = 0
            players = []
            for _, r in g.iterrows():
                side = str(r.get("side", "")).lower()
                players.append({
                    "puuid": _id(r),
                    "playerName": str(r.get("playername", "")),
                    "teamName": str(r.get("teamname", "")),
                    "teamId": 100 if side == "blue" else 200,
                    "teamPosition": POSITION_MAP.get(str(r.get("position", "")).lower()),
                    "championName": str(r.get("champion", "")),
                    "kills": int(r.get("kills") or 0),
                    "deaths": int(r.get("deaths") or 0),
                    "assists": int(r.get("assists") or 0),
                    "win": bool(int(r.get("result") or 0) == 1),
                    "totalMinionsKilled": _num(r, ["minionkills", "minionskilled"]),
                    "neutralMinionsKilled": _num(r, ["monsterkills", "monsterskilled"]),
                    "csTotal": _num(r, ["totalcs"]),
                    "goldEarned": _num(r, ["earnedgold", "totalgold"]),
                    "totalDamageDealtToChampions": _num(r, ["damagetochampions"]),
                    "visionScore": _num(r, ["visionscore"]),
                    "wardsPlaced": _num(r, ["wardsplaced"]),
                    "wardsKilled": _num(r, ["wardskilled"]),
                    "soloKills": _num(r, ["solokills"]),
                    "killParticipation": _num(r, ["kp"]),
                })
            rec = {
                "matchId": str(game_id),
                "queueId": None,  # pro track marker (ladder uses 420)
                "gameDuration": int(dur),
                "gameVersion": str(pick_col(g, "patch") or ""),
                "gameStartTimestamp": ts,
                "gameEndedInEarlySurrender": False,
                "seedTier": league,           # tier field = league for pro
                "league": league,
                "year": int(pick_col(g, "year") or 0),
                "split": str(pick_col(g, "split") or ""),
                "playoffs": bool(pick_col(g, "playoffs") in (1, True, "1", "True")),
                "url": str(pick_col(g, "url") or ""),
                "players": players,
            }
            out.write(json.dumps(rec) + "\n")
            written += 1

    print(f"[collect_pro] {csv_path.name}: {total_rows:,} rows -> "
          f"{written} tier-1 games")
    return written


def pick_col(g: pd.DataFrame, name: str):
    """Case/space-insensitive column access on a game sub-frame."""
    for c in g.columns:
        if c.strip().lower().replace(" ", "").replace("_", "") == name:
            v = g[c].iloc[0]
            return None if pd.isna(v) else v
    return None


def _num(r, names):
    for n in names:
        try:
            v = r.get(n)
            if v is not None and not pd.isna(v):
                f = float(v)
                return int(f) if f.is_integer() else f
        except Exception:
            pass
    return None


def _id(r):
    v = r.get("playerid")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    # modern OE ids are strings ("oe:player:<hash>"); older files used ints
    if s.replace(".", "", 1).isdigit():
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    return s or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", help="existing OE csv paths to convert")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    args = ap.parse_args()

    PRO_DIR.mkdir(parents=True, exist_ok=True)

    csvs = []
    if args.csv:
        csvs = [Path(p) for p in args.csv]
    else:
        for y in args.years:
            dest = PRO_DIR / f"oe_{y}.csv"
            if download_year(y, dest):
                csvs.append(dest)
        if not csvs:
            raise SystemExit(
                "Could not download (Google Drive quota). Download manually:\n"
                + "\n".join(f"  {y}: https://drive.google.com/file/d/{OE_FILES[y]}/view"
                            for y in args.years)
                + "\nthen run: python collect_pro.py --csv data/pro/oe_2025.csv ...")
    if args.download_only:
        return

    if RAW_PRO_FILE.exists():
        RAW_PRO_FILE.unlink()  # full rebuild from csvs each run
    n = sum(convert(c) for c in csvs)
    print(f"[collect_pro] wrote {n} games -> {RAW_PRO_FILE}")


if __name__ == "__main__":
    main()