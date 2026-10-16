"""
Local web server: search a Riot ID in the browser and get a report page.

    python server.py                    ->  http://127.0.0.1:8765/search
    python server.py --port 9000 --ttl 1800 --n 60

Why a server is needed at all: Riot deliberately sends no CORS headers, because
a key in client-side JS would be harvested off the page. So live lookup needs a
backend that keeps the key secret. This one is stdlib-only (nothing to install),
binds 127.0.0.1 unless you pass --host, and reuses the same code as the CLI:
collect_player.collect() to fetch and build_player_web.write_page() to render.

A lookup runs as a background job, because 40 games is ~42 API requests and a
dev key allows 100 per 2 min - roughly a minute of waiting. The search page polls
/api/lookup while the worker drains the queue. Games are cached under
data/players/<slug>/, so looking the same player up again within --ttl seconds is
instant and costs no API calls.

Routes
  /                    landing page          /players/        report list
  /search              Riot ID search box    /players/<slug>  one report
  /explorer            corpus role-impact explorer
  /api/health          status + whether a key is configured
  /api/lookup          ?riotId=Name#TAG&region=na1&n=40[&force=1]
  /api/players         built reports (from web/players/registry.json)
"""

import argparse
import json
import os
import queue
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import build_player_web as bpw
import collect_player as cp
from riot_client import ROUTES

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
PLAYERS_WEB = WEB / "players"
DATA_PLAYERS = ROOT / "data" / "players"
REF_FILE = ROOT / "data" / "processed" / "reference.json"

CFG = {"ttl": 900, "n": 40, "region": "na1"}
REF: dict = {}
JOBS: dict[str, dict] = {}
QUEUE: queue.Queue[str] = queue.Queue()
LOCK = threading.Lock()

TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".json": "application/json; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


# ---------------------------------------------------------------- lookups ----
def page_exists(slug: str) -> bool:
    return (PLAYERS_WEB / f"{slug}.html").is_file()


def classify_error(exc: Exception, msg: str) -> str:
    if isinstance(exc, LookupError):
        return ("Riot IDs are shard-specific: check the #TAG and that --region/region "
                "matches where they play. Spaces and unicode are fine.")
    if "cannot reach" in msg or isinstance(exc, (ConnectionError, OSError)):
        return ("This machine cannot reach the Riot API right now (proxy/VPN/firewall). "
                "The server holds the key, but it still needs outbound internet.")
    if "401" in msg or "403" in msg or "rejected" in msg.lower():
        return ("Dev keys expire 24h after creation. Get a fresh one at "
                "developer.riotgames.com and update RIOT_API_KEY in .env, then retry.")
    if "429" in msg or "rate" in msg.lower():
        return "Riot rate-limited us (100 requests / 2 min). Try again in a couple of minutes."
    return ""


def make_progress(job: dict):
    def progress(stage: str, **info):
        with LOCK:
            job["stage"] = stage
            job["progress"] = info
            text = stage
            if stage == "collect":
                text = f"fetching games {info.get('done')}/{info.get('total')}"
            elif stage == "matchlist":
                text = "reading recent game list"
            elif stage == "resolve":
                text = f'resolving {info.get("riotId")} in {info.get("region")}'
            elif stage == "profile":
                text = "reading summoner + rank"
            if not job["stages"] or job["stages"][-1] != text:
                job["stages"].append(text)
                job["stages"] = job["stages"][-8:]
    return progress


def job_view(key: str, job: dict) -> dict:
    return {"key": key, "status": job["status"], "stage": job.get("stage"),
            "steps": job.get("stages", []), "progress": job.get("progress"),
            "error": job.get("error"), "hint": job.get("hint"),
            "warning": job.get("warning"), "url": job.get("url"),
            "staleUrl": job.get("staleUrl"), "seconds": round(time.time() - job["started"], 1)}


def process_job(key: str) -> None:
    job = JOBS.get(key)
    if not job:
        return
    with LOCK:
        job["status"] = "running"
    try:
        prof = cp.collect(job["game"], job["tag"], job["region"], job["n"],
                          progress=make_progress(job))
        payload = bpw.player_payload(DATA_PLAYERS / prof["slug"], REF)
        if payload is None:
            raise RuntimeError("their recent games produced no usable rows "
                               "(queue/remake/position filters)")
        card = bpw.write_page(payload)
        with LOCK:
            job.update(status="done", url=f"/players/{prof['slug']}.html",
                       games=card["games"], wr=card["wr"], rank=card["rank"])
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        with LOCK:
            if page_exists(job["slug"]):
                job.update(status="done", url=f"/players/{job['slug']}.html",
                           stale=True,
                           warning=f"refresh failed - {msg}. Showing the last snapshot.")
            else:
                job.update(status="error", error=msg, hint=classify_error(e, str(e)))


def worker() -> None:
    """One worker so Riot's 100-req/2-min window is never raced. It must not die:
    a lost thread turns every later lookup into a permanently queued job."""
    while True:
        try:
            key = QUEUE.get()
        except Exception as e:                        # queue closed / interrupted
            print(f"[worker] queue read failed: {e}")
            return
        try:
            process_job(key)
        except Exception as e:                        # bug in the paths above
            job = JOBS.get(key)
            if job is not None:
                with LOCK:
                    job.update(status="error",
                               error=f"worker failure: {type(e).__name__}: {e}")
            print(f"[worker] unexpected failure for {key}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                QUEUE.task_done()
            except Exception:
                pass


def handle_lookup(qs: dict) -> tuple[int, dict]:
    riot_id = (qs.get("riotId") or [""])[0].strip()
    if not riot_id:
        return 400, {"status": "error", "error": "riotId is required (Name#TAG)"}
    region = ((qs.get("region") or [CFG["region"]])[0] or CFG["region"]).lower()
    if region not in ROUTES:
        return 400, {"status": "error",
                     "error": f"unknown region '{region}'",
                     "hint": "valid: " + " ".join(sorted(ROUTES))}
    try:
        n = int((qs.get("n") or [CFG["n"]])[0])
    except ValueError:
        n = CFG["n"]
    n = max(1, min(n, 100))
    force = (qs.get("force") or [""])[0] not in ("", "0", "false")
    try:
        game, tag = cp.split_riot_id(riot_id, None)
    except ValueError as e:
        return 400, {"status": "error", "error": str(e)}

    slug = cp.slugify(game)
    key = f"{game.lower()}#{tag.lower()}@{region}"
    stale_url = f"/players/{slug}.html" if page_exists(slug) else None

    prof_file = DATA_PLAYERS / slug / "profile.json"
    if not force and prof_file.is_file():
        try:
            prof = json.loads(prof_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prof = {}
        same = (str(prof.get("riotId", "")).lower() == f"{game}#{tag}".lower()
                and prof.get("region") == region)
        age = time.time() - (prof.get("collectedAt") or 0)
        if same and age < CFG["ttl"]:
            if stale_url:
                return 200, {"status": "done", "url": stale_url, "cached": True,
                             "ageSeconds": int(age), "key": key}
            payload = bpw.player_payload(DATA_PLAYERS / slug, REF)
            if payload:
                bpw.write_page(payload)
                return 200, {"status": "done", "url": stale_url or f"/players/{slug}.html",
                             "cached": True, "built": True, "key": key}

    with LOCK:
        job = JOBS.get(key)
        if job and job["status"] in ("queued", "running"):
            job["staleUrl"] = job.get("staleUrl") or stale_url
            return 200, job_view(key, job)
        JOBS[key] = job = {"status": "queued", "game": game, "tag": tag,
                           "region": region, "n": n, "slug": slug,
                           "started": time.time(), "stages": [], "staleUrl": stale_url}
    QUEUE.put(key)
    return 202, job_view(key, job)


def registry() -> dict:
    path = PLAYERS_WEB / "registry.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------- routing ----
def resolve_static(path: str) -> Path | None:
    p = urllib.parse.unquote(path).lstrip("/")
    if p.startswith("web/"):
        p = p[4:]
    if p in ("", "index.html"):
        return ROOT / "index.html"
    if p == "explorer":
        return WEB / "index.html"
    if p in ("search", "search.html"):
        return WEB / "search.html"
    if p in ("players", "players/", "players/index.html"):
        return PLAYERS_WEB / "index.html"
    if p.startswith("players/"):
        p = PLAYERS_WEB / (p.split("players/", 1)[1] or "index.html")
    else:
        p = WEB / p
    if p.suffix not in TYPES and p.is_dir():
        p = p / "index.html"
    if not p.suffix:
        p = p.with_suffix(".html")
    try:
        p = p.resolve()
    except OSError:
        return None
    return p if p.is_file() and str(p).startswith(str(WEB.resolve())) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "lolstats/1.0"

    def log_message(self, fmt, *args):
        print(f"[http] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode("utf-8"), TYPES[".json"])

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                code, obj = self._api(path, qs)
                return self._json(code, obj)
            if path in ("/favicon.ico", "/robots.txt"):
                return self._send(204, b"", "text/plain")
            f = resolve_static(path)
            if f is None:
                return self._send(404, b"not found\n", "text/plain; charset=utf-8")
            return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
        except BrokenPipeError:
            pass
        except Exception as e:
            return self._json(500, {"status": "error", "error": f"{type(e).__name__}: {e}"})

    def _api(self, path: str, qs: dict):
        if path == "/api/health":
            return 200, {"ok": True, "corpusRows": REF.get("nRows"),
                         "corpusMatches": REF.get("nMatches"),
                         "baseline": bool(REF), "reports": len(registry()),
                         "apiKeySet": bool(os.environ.get("RIOT_API_KEY")),
                         "ttl": CFG["ttl"], "defaultGames": CFG["n"],
                         "regions": sorted(ROUTES)}
        if path == "/api/players":
            return 200, {"players": sorted(registry().values(),
                                           key=lambda c: c["riotId"].lower())}
        if path == "/api/jobs":                      # debugging: queue + job states
            with LOCK:
                items = list(JOBS.items())[-20:]
                jobs = {k: job_view(k, v) for k, v in items}
            return 200, {"queued": QUEUE.qsize(), "workers": 1, "jobs": jobs}
        if path == "/api/lookup":
            if not REF:
                return 503, {"status": "error",
                             "error": "no baseline loaded",
                             "hint": "run: python build_reference.py"}
            return handle_lookup(qs)
        return 404, {"status": "error", "error": f"unknown endpoint {path}"}

    def do_HEAD(self):
        self.do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 keeps this to your machine; 0.0.0.0 exposes it to your network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ttl", type=int, default=900,
                    help="seconds a cached player is served without re-fetching")
    ap.add_argument("--n", type=int, default=40, help="games per lookup (max 100)")
    ap.add_argument("--region", default="na1", help="default shard for the search page")
    args = ap.parse_args()
    CFG.update({"ttl": args.ttl, "n": max(1, min(args.n, 100)), "region": args.region})

    if REF_FILE.is_file():
        REF.update(json.loads(REF_FILE.read_text(encoding="utf-8")))
    else:
        print(f"[warn] {REF_FILE} missing - lookups disabled until you run build_reference.py")
    if not os.environ.get("RIOT_API_KEY"):
        print("[warn] RIOT_API_KEY not set in .env - live lookups will fail (cached reports still serve)")

    threading.Thread(target=worker, daemon=True, name="lookup-worker").start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"[serve] http://{shown}:{args.port}/")
    print(f"[serve] search: http://{shown}:{args.port}/search"
          f"   explorer: /explorer   reports: /players/")
    if args.host != "127.0.0.1":
        print("[warn] bound beyond localhost: anyone who can reach this port can make "
              "Riot API calls with your key and read every report. Dev keys are not "
              "meant to be shared.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
        srv.server_close()


if __name__ == "__main__":
    main()
