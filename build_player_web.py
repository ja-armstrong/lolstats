"""
Build a single-player report page, scored against the collected corpus.

    python build_player_web.py                 # every player in data/players/
    python build_player_web.py --slug killed-cat
    python build_player_web.py --no-html       # just print the summary

For each player we take the games collect_player.py stored, rebuild the same
metrics prepare_features.py uses, and score them against the frozen baseline in
data/processed/reference.json at two levels:

  * overall  -> every player-game in the corpus for that role
  * band     -> the player's own rank band and role (e.g. JUNGLE|GOLD)

Percentiles are computed per game inside the role actually played that game,
then averaged - so a jungler's CS/min is judged against junglers, not ADCs.

Output is plain static files (no backend, no API key anywhere in the page):
  web/players/<slug>.html   the report (loads ../reference.js + ../player.js)
  web/players/index.html    list of generated reports
  web/player.js             shared app code, regenerated every build
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from prepare_features import load_matches, METRICS

ROOT = Path(__file__).resolve().parent
PLAYER_DIR = ROOT / "data" / "players"
REF_FILE = ROOT / "data" / "processed" / "reference.json"
WEB_DIR = ROOT / "web"
PLAYERS_OUT = WEB_DIR / "players"

DDAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDAGON_CHAMPS_URL = "https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json"
DDAGON_FALLBACK = "12.4.1"

_CDN = {}


def cdn() -> dict:
    """Version + champion-name -> ddragon key, for icon URLs. Cached per build.
    versions.json has been both an object with "default" and a bare list."""
    if not _CDN:
        version, keys = DDAGON_FALLBACK, {}
        try:
            import requests
            d = requests.get(DDAGON_VERSIONS_URL, timeout=6).json()
            version = d["default"] if isinstance(d, dict) else d[0]
            data = requests.get(DDAGON_CHAMPS_URL.format(v=version), timeout=8).json()["data"]
            keys = {v["name"]: k for k, v in data.items()}
        except Exception as e:
            print(f"[cdn] ddragon unavailable ({type(e).__name__}); report renders without icons")
        _CDN.update({"version": version, "keys": keys})
    return _CDN


def player_payload(pdir: Path, ref: dict) -> dict | None:
    profile = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
    raw_file = pdir / "matches.jsonl"
    if not raw_file.exists() or not raw_file.stat().st_size:
        print(f"[{profile.get('slug')}] no games collected yet - run collect_player.py")
        return None

    df = load_matches(raw_file, queue_filter=True)
    puuid = profile["puuid"]
    mine = df[df["puuid"] == puuid]
    if mine.empty:
        print(f"[{profile['slug']}] no usable rows (queue 420 / remake / position filter)")
        return None

    bounds, champ_keys = ref["bounds"], cdn()["keys"]
    games = []
    for r in mine.itertuples(index=False):
        sub = df[(df["matchId"] == r.matchId) & (df["teamId"] == r.teamId)]
        mates = sub[sub["puuid"] != puuid]
        raw, mate = [], []
        for m in METRICS:
            lo, hi = bounds[m]
            raw.append(round(float(np.clip(getattr(r, m), lo, hi)), 5))
            # teammate averages are clipped too, so z-scores stay on the same scale
            mate.append(round(float(np.clip(mates[m].mean(), lo, hi)), 5)
                        if len(mates) else 0.0)
        games.append({
            "id": r.matchId,
            "ts": int(r.ts),
            "date": time.strftime("%Y-%m-%d", time.gmtime((r.ts or 0) / 1000)),
            "role": r.role,
            "champ": r.champion,
            "champKey": champ_keys.get(r.champion),
            "win": bool(r.win),
            "min": round(float(r.minutes), 1),
            "k": int(r.kills), "d": int(r.deaths), "a": int(r.assists),
            "raw": raw, "mate": mate,
            "mates": int(len(mates)),
        })
    games.sort(key=lambda g: g["ts"])
    # keep the report stable as history piles up on disk: score the most
    # recent requestedN games only (collect_player.py never deletes the rest)
    want = int(profile.get("requestedN") or 0)
    trimmed = 0
    if want and len(games) > want:
        trimmed = len(games) - want
        games = games[-want:]

    tier = (profile.get("soloRank") or {}).get("tier") or profile.get("tier") or ""
    band = str(tier).split("_")[0]
    if band in ("CHALLENGER", "GRANDMASTER", "MASTER"):
        band = "APEX"
    seen_roles = {g["role"] for g in games}
    band_available = band in ref["bands"] and any(
        f"{role}|{band}" in ref["groups"] for role in seen_roles)

    roles = {}
    for g in games:
        rr = roles.setdefault(g["role"], {"games": 0, "wins": 0})
        rr["games"] += 1
        rr["wins"] += 1 if g["win"] else 0

    return {
        "slug": profile["slug"],
        "profile": profile,
        "band": band if band_available else None,
        "bandAvailable": band_available,
        "nGames": len(games),
        "nWins": sum(1 for g in games if g["win"]),
        "roles": roles,
        "games": games,
        "corpus": {"nRows": ref["nRows"], "nMatches": ref["nMatches"],
                   "region": ref["region"], "generated": ref["generated"]},
        "meta": {"gamesOnDisk": profile.get("gamesStored"),
                 "trimmedOlder": trimmed,
                 "rowsAfterFilters": int(len(df)),
                 "ddragon": cdn()["version"]},
    }


def console_summary(payload: dict, ref: dict) -> str:
    """Text summary, so the build log says something useful per player."""
    signs = {m["key"]: m["sign"] for m in ref["metrics"]}
    band = payload["band"]
    comps, outs = [], []
    for g in payload["games"]:
        grp = ref["groups"].get(f"{g['role']}|{band}") if band else None
        grp = grp or ref["groups"][g["role"]]
        z = [signs[m] * (v - grp["m"][m]["mean"]) / grp["m"][m]["sd"]
             for m, v in zip(METRICS, g["raw"])]
        zm = [signs[m] * (v - grp["m"][m]["mean"]) / grp["m"][m]["sd"]
              for m, v in zip(METRICS, g["mate"])]
        c = sum(z) / len(z)
        comps.append(c)
        outs.append(c - sum(zm) / len(zm))
    wr = payload["nWins"] / payload["nGames"] * 100
    top = max(payload["roles"].items(), key=lambda kv: kv[1]["games"])
    return (f"{payload['slug']}: {payload['nGames']} games, WR {wr:.1f}%, "
            f"main {top[0]} ({top[1]['games']}), "
            f"vs {band or 'all ranks'}: score z {np.mean(comps):+.2f}, "
            f"vs teammates {np.mean(outs):+.2f}")


def write_page(payload: dict) -> dict:
    """Write one report page, update the registry, rebuild the reports index.

    Registry-driven so building a single player never drops the others from
    web/players/index.html (this is what server.py calls after each lookup).
    """
    PLAYERS_OUT.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "player.js").write_text(APP_JS, encoding="utf-8")
    prof = payload["profile"]
    rank = prof.get("soloRank") or {}
    data = json.dumps(payload, separators=(",", ":"))
    html = (PAGE_TEMPLATE
            .replace("__TITLE__", f"{prof['riotId']} - LoL player report")
            .replace("__PLAYER__", data))
    out = PLAYERS_OUT / f"{payload['slug']}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[web] {out} ({out.stat().st_size / 1024:.0f} KB)")

    reg_path = PLAYERS_OUT / "registry.json"
    registry = {}
    if reg_path.exists():
        try:
            registry = json.loads(reg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = {}
    registry[payload["slug"]] = {
        "slug": payload["slug"], "riotId": prof["riotId"],
        "rank": (f"{rank.get('tier', '')} {rank.get('division', '')}".strip() or "Unranked"),
        "region": prof.get("region"), "games": payload["nGames"],
        "wr": round(payload["nWins"] / payload["nGames"] * 100, 1),
        "updated": prof.get("collectedAtIso", "")[:10],
    }
    registry = {k: v for k, v in registry.items()
                if (PLAYERS_OUT / f"{k}.html").exists()}
    reg_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    _write_index(registry)
    return registry[payload["slug"]]


def _write_index(registry: dict) -> None:
    rows = "\n".join(
        f'<tr><td><a href="{c["slug"]}.html">{c["riotId"]}</a></td>'
        f'<td>{c["rank"]}</td><td>{c["region"]}</td><td>{c["games"]}</td>'
        f'<td>{c["wr"]}%</td><td>{c["updated"]}</td></tr>'
        for _, c in sorted(registry.items(), key=lambda kv: kv[1]["slug"]))
    (PLAYERS_OUT / "index.html").write_text(
        INDEX_TEMPLATE
        .replace("__ROWS__", rows or
                 '<tr><td colspan="6">No reports yet &mdash; run '
                 '<code>python collect_player.py "Name#TAG"</code>, or start '
                 '<code>python server.py</code> and search in the browser.</td></tr>')
        .replace("__SEARCH__", SEARCH_HINT), encoding="utf-8")
    print(f"[web] {PLAYERS_OUT / 'index.html'} ({len(registry)} reports)")


SEARCH_HINT = (
    '<p class="sub">Static hosting: these pages are generated locally. '
    'Run <code>python server.py</code> for the in-browser search box.</p>')


def write_pages(payloads: list[dict]) -> None:
    for p in sorted(payloads, key=lambda x: x["slug"]):
        write_page(p)


PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0f1115;--panel:#171a21;--panel2:#1d222b;--text:#e6e9ef;--dim:#8b93a7;
  --accent:#4f9cf9;--good:#3fb96f;--bad:#e05656;--line:#2a3040;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
a{color:var(--accent);text-decoration:none}
#wrap{max-width:1180px;margin:0 auto}
header{padding:18px 24px 10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:21px;margin:0 0 4px}
#avatar img{width:56px;height:56px;border-radius:8px;display:block}
.sub{color:var(--dim);font-size:13px}
main{padding:0 24px 30px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:14px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .big{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .lbl{color:var(--dim);font-size:12px;margin-top:2px}
.btn{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;
  padding:5px 10px;cursor:pointer;font-size:12.5px;margin:0 6px 6px 0}
.btn.on{border-color:var(--accent);background:#20304a}
.mrow{display:grid;grid-template-columns:190px 1fr 190px;gap:10px;align-items:center;margin:6px 0;font-size:12.5px}
.mrow .vals{color:var(--dim);font-size:11.5px;text-align:right;font-variant-numeric:tabular-nums}
.pbar{position:relative;height:17px;background:#20242e;border-radius:4px;overflow:hidden}
.pbar>span{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#2c6fbb,#4f9cf9)}
.pbar>em{position:absolute;right:5px;top:0;line-height:17px;font-style:normal;font-size:11px;color:#dbe6f7;
  font-variant-numeric:tabular-nums}
.pbar>i{position:absolute;top:-1px;bottom:-1px;width:2px;background:#f0a13a}
.wrow{display:grid;grid-template-columns:175px 1fr 46px;gap:8px;align-items:center;margin:5px 0}
.wrow .name{font-size:12.5px}.wrow .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--dim)}
input[type=range]{width:100%;accent-color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
img.champ{width:22px;height:22px;border-radius:4px;vertical-align:middle}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}.mrow{grid-template-columns:130px 1fr 130px}}
svg{display:block;width:100%}
.note{color:var(--dim);font-size:12px;line-height:1.5}
.win{color:var(--good)}.lose{color:var(--bad)}
</style>
</head>
<body>
<div id="wrap">
<header>
  <div id="avatar"></div>
  <div>
    <h1 id="pname">loading&hellip;</h1>
    <div class="sub" id="pmeta"></div>
  </div>
</header>
<main>
  <div class="tiles" id="tiles"></div>

  <div class="panel">
    <h2>Compared against</h2>
    <div id="baselineBtns"></div>
    <div class="note" id="baselineNote" style="margin-top:6px"></div>
  </div>

  <div class="panel">
    <h2>Role</h2>
    <div id="roleBtns"></div>
    <div class="note">Every game is scored inside the role you played in it &mdash; jungler CS never gets compared to ADC CS.</div>
  </div>

  <div class="panel">
    <h2>Where you rank on each metric</h2>
    <div class="note" style="margin-bottom:8px">Bar = your average per-game percentile (longer is better;
      &darr; metrics are flipped so "fewer deaths" reads as good). Orange tick = percentile of your <i>average</i> value &mdash;
      if it sits far from the bar end, one outlier game is carrying the stat.</div>
    <div id="metricBars"></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Score per game</h2>
      <div class="note" style="margin-bottom:6px">Composite z-score, shown as its peer percentile. Line = 5-game average.</div>
      <svg id="trend" height="240"></svg>
    </div>
    <div class="panel">
      <h2>Score distribution vs peers</h2>
      <div class="note" style="margin-bottom:6px">Bars = your games (z-score). Shaded = peer middle 50%, line = peer median.</div>
      <svg id="hist" height="240"></svg>
    </div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Teammate differential</h2>
      <div class="note" style="margin-bottom:6px">Your score minus your four teammates' average, so team strength cancels out.
        Closest box scores come to "how better were <i>you</i>".</div>
      <svg id="outChart" height="190"></svg>
      <div class="note" id="outNote" style="margin-top:6px"></div>
    </div>
    <div class="panel">
      <h2>Champion pool</h2>
      <div id="champTable"></div>
    </div>
  </div>

  <div class="panel">
    <h2>Your score weights</h2>
    <div class="note" style="margin-bottom:8px">Sliders re-rank everything above. At non-default weights the peer percentile is
      read off a sampled baseline (120&ndash;400 peer games), so it moves by ~1%; the differential percentile stays on the
      equal-weight peer scale.</div>
    <div><button class="btn" id="resetW">Reset to equal weights</button></div>
    <div id="weights"></div>
  </div>

  <div class="panel">
    <h2>Recent games</h2>
    <div id="gameTable" style="overflow-x:auto"></div>
  </div>

  <div class="panel">
    <h2>Method and caveats</h2>
    <div class="note" id="method"></div>
  </div>
</main>
</div>
<script src="../reference.js"></script>
<script>window.LOL_PLAYER = __PLAYER__;</script>
<script src="../player.js"></script>
</body>
</html>
"""

INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoL player reports</title>
<style>
body{margin:0;background:#0f1115;color:#e6e9ef;font:14px/1.5 "Segoe UI",system-ui,sans-serif;padding:24px}
h1{font-size:20px}table{border-collapse:collapse;width:100%;max-width:900px;font-size:13px}
td,th{border-bottom:1px solid #2a3040;padding:6px 8px;text-align:left}
a{color:#4f9cf9;text-decoration:none}.sub{color:#8b93a7;font-size:12.5px;margin-bottom:14px}
</style></head><body>
<h1>LoL player reports</h1>
<div class="sub">Single-player lookups scored against the collected ranked-solo corpus.
Built by <code>build_player_web.py</code> &middot; also see the
<a href="../index.html">role impact explorer / home</a> and the
<a href="../search.html">player search</a>.</div>
__SEARCH__
<table><tr><th>player</th><th>rank</th><th>region</th><th>games</th><th>win rate</th><th>updated</th></tr>
__ROWS__
</table>
</body></html>
"""

APP_JS = r"""
(function () {
const REF = window.LOL_REFERENCE, P = window.LOL_PLAYER;
if (!REF || !P) { document.getElementById('pname').textContent = 'reference.js not found'; return; }

const METS = REF.metrics, MK = METS.map(m => m.key);
const DEF_W = {}; METS.forEach(m => DEF_W[m.key] = m.sign);
const ROLE_COLORS = { TOP:'#4f9cf9', JUNGLE:'#e05656', MIDDLE:'#3fb96f', BOTTOM:'#f0a13a', SUPPORT:'#b07cf0' };
const ROLE_ORDER = REF.roles;
const state = { baseline: P.bandAvailable ? 'band' : 'overall', role: 'ALL', w: Object.assign({}, DEF_W) };
for (const k in REF.groups) REF.groups[k].__key = k;

// ---------- reference math ----------
function groupFor(role) {
  if (state.baseline === 'band' && P.band) {
    const g = REF.groups[role + '|' + P.band];
    if (g) return g;
  }
  return REF.groups[role];
}
function pct(q, v) {   // value -> percentile, interpolating the quantile grid
  const n = q.length, step = 100 / (n - 1);
  // Ties get mid-rank, exactly like the Mann-Whitney AUC in build_web.py. Without
  // this, a metric that is 0 in 58% of peer games (solo kills) reports every tied
  // game as 0% and drags the averaged percentile down to nonsense.
  if (v <= q[0]) { let h = 0; while (h + 1 < n && q[h + 1] === q[0]) h++; return h ? h * step / 2 : 0; }
  if (v >= q[n - 1]) { let l = n - 1; while (l - 1 >= 0 && q[l - 1] === q[n - 1]) l--; return 100 - (n - 1 - l) * step / 2; }
  let i = 1; while (i < n && v > q[i]) i++;
  const a = q[i - 1], b = q[i];
  return (i - 1) * step + (b > a ? step * (v - a) / (b - a) : 0);
}
function z(v, grp, k) { const s = grp.m[k]; return (v - s.mean) / s.sd; }
function compOf(vals, grp) {
  let s = 0, d = 0;
  for (let i = 0; i < MK.length; i++) {
    const w = state.w[MK[i]]; if (!w) continue;
    s += w * z(vals[i], grp, MK[i]); d += Math.abs(w);
  }
  return d ? s / d : 0;
}
function isDefaultW() { for (let i = 0; i < MK.length; i++) if (state.w[MK[i]] !== DEF_W[MK[i]]) return false; return true; }

// peer percentile of a custom-weight composite, from the sampled baseline
const scache = new Map();
function sampleComps(grp) {
  const sig = grp.__key + '|' + MK.map(k => state.w[k]).join(',');
  let a = scache.get(sig);
  if (!a) {
    a = [];
    for (const row of grp.z) {
      let s = 0, d = 0;
      for (let i = 0; i < MK.length; i++) {
        const w = state.w[MK[i]]; if (!w) continue;
        s += w * (row[i] / 100 - grp.m[MK[i]].mean) / grp.m[MK[i]].sd; d += Math.abs(w);
      }
      a.push(d ? s / d : 0);
    }
    a.sort((x, y) => x - y); scache.set(sig, a);
  }
  return a;
}
function sampPct(grp, v) {
  const a = sampleComps(grp);
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
  return a.length ? lo / a.length * 100 : 50;
}
function compPct(grp, v) { return isDefaultW() ? pct(grp.comp.q, v) : sampPct(grp, v); }

function rows() {
  let gs = P.games;
  if (state.role !== 'ALL') gs = gs.filter(g => g.role === state.role);
  return gs.map(g => {
    const grp = groupFor(g.role);
    const comp = compOf(g.raw, grp), mate = compOf(g.mate, grp);
    return { g, grp, comp, out: comp - mate, cp: compPct(grp, comp), op: pct(grp.outperform.q, comp - mate) };
  });
}
function avg(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
function majorityRole(rs) {
  const c = {}; rs.forEach(r => c[r.g.role] = (c[r.g.role] || 0) + 1);
  return Object.keys(c).sort((a, b) => c[b] - c[a])[0];
}
const nf = (v, d) => (v === null || v === undefined || isNaN(v)) ? '-' : v.toFixed(d === undefined ? 1 : d);
const pc = v => Math.round(v) + '%';

// ---------- svg helpers ----------
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text; return e;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

// ---------- header + tiles ----------
function header() {
  const pr = P.profile, rank = pr.soloRank || {}, cdn = P.meta.ddragon;
  document.getElementById('pname').textContent = pr.riotId;
  const bits = [];
  if (rank.tier) bits.push(rank.tier + ' ' + (rank.division || '') +
    (rank.leaguePoints != null ? ' \u00b7 ' + rank.leaguePoints + ' LP' : ''));
  if (rank.wins != null) bits.push(rank.wins + '-' + rank.losses + ' W-L');
  bits.push('level ' + (pr.summonerLevel ?? '?'));
  bits.push((pr.region || '').toUpperCase());
  document.getElementById('pmeta').innerHTML = bits.join(' &nbsp;\u00b7&nbsp; ') +
    '<br><span style="font-size:12px">Last ' + P.nGames + ' ranked solo games' +
    (P.meta.trimmedOlder ? ' (of ' + (P.nGames + P.meta.trimmedOlder) + ' stored)' : '') +
    ' \u00b7 collected ' +
    (pr.collectedAtIso || '').slice(0, 10) + ' \u00b7 <a href="index.html">all reports</a> \u00b7 ' +
    '<a href="../index.html">home / explorer</a></span>';
  if (pr.profileIconId != null)
    document.getElementById('avatar').innerHTML =
      '<img src="https://ddragon.leagueoflegends.com/cdn/' + cdn + '/img/profileicon/' + pr.profileIconId + '.png" alt="">';
}

function tiles(rs) {
  const wins = rs.filter(r => r.g.win).length;
  const top20 = rs.filter(r => r.cp >= 80).length;
  const items = [
    [rs.length, 'games in view'],
    [rs.length ? (wins / rs.length * 100).toFixed(1) + '%' : '-', 'win rate'],
    [rs.length ? pc(avg(rs.map(r => r.cp))) : '-', 'score percentile vs peers'],
    [rs.length ? (top20 / rs.length * 100).toFixed(0) + '%' : '-', 'games in peers\u2019 top 20%'],
    [rs.length ? pc(avg(rs.map(r => r.op))) : '-', 'percentile vs own teammates'],
  ];
  document.getElementById('tiles').innerHTML = items.map(([big, lbl]) =>
    '<div class="tile"><div class="big">' + big + '</div><div class="lbl">' + lbl + '</div></div>').join('');
}

// ---------- baseline / role controls ----------
function baselineBtns() {
  const box = document.getElementById('baselineBtns');
  const opts = [['overall', 'All ranks (whole corpus)']];
  if (P.bandAvailable) opts.push(['band', 'My band: ' + P.band]);
  box.innerHTML = '';
  opts.forEach(([v, label]) => {
    const b = document.createElement('button');
    b.className = 'btn' + (state.baseline === v ? ' on' : '');
    b.textContent = label;
    b.onclick = () => { state.baseline = v; render(); };
    box.appendChild(b);
  });
  const rolesInView = state.role === 'ALL' ? ROLE_ORDER.filter(r => P.roles[r]) : [state.role];
  const parts = rolesInView.map(r => {
    const g = groupFor(r);
    return r + ' n=' + g.n.toLocaleString() + (g.__key.includes('|') ? '' : ' (all ranks)');
  });
  document.getElementById('baselineNote').innerHTML =
    'Baseline groups used: ' + parts.join(' \u00b7 ') +
    (state.baseline === 'band' && !P.bandAvailable ? ' <b class="lose">(band baseline unavailable \u2014 using all ranks)</b>' : '') +
    '. Your band is taken from your <i>current</i> league tier and applied to every game here, ' +
    'so games played at a lower tier are scored a bit generously.';
}

function roleBtns() {
  const box = document.getElementById('roleBtns');
  box.innerHTML = '';
  ['ALL'].concat(ROLE_ORDER.filter(r => P.roles[r])).forEach(r => {
    const b = document.createElement('button');
    b.className = 'btn' + (state.role === r ? ' on' : '');
    b.textContent = r === 'ALL' ? 'All roles (' + P.nGames + ')' :
      r + ' ' + P.roles[r].games + 'g ' + Math.round(P.roles[r].wins / P.roles[r].games * 100) + '%W';
    if (r !== 'ALL') b.style.borderColor = ROLE_COLORS[r] + '66';
    b.onclick = () => { state.role = r; render(); };
    box.appendChild(b);
  });
}

// ---------- metric percentile bars ----------
function metricBars(rs) {
  const box = document.getElementById('metricBars');
  const role = majorityRole(rs) || 'JUNGLE';
  const grp = groupFor(role);
  let html = '';
  METS.forEach((m, i) => {
    const ps = rs.map(r => {
      const g = groupFor(r.g.role);
      const p = pct(g.m[m.key].q, r.g.raw[i]);
      return m.sign < 0 ? 100 - p : p;
    });
    if (!ps.length) return;
    const mean = avg(ps);
    // percentile of your *average value* per role, games-weighted: differs from the
    // bar whenever your per-game percentiles are lopsided (one stomp vs nine close games)
    const byRole = {};
    rs.forEach(r => (byRole[r.g.role] = byRole[r.g.role] || []).push(r.g.raw[i]));
    let num = 0, den = 0;
    for (const rl in byRole) {
      const g = groupFor(rl), mv = avg(byRole[rl]);
      let p = pct(g.m[m.key].q, mv); if (m.sign < 0) p = 100 - p;
      num += p * byRole[rl].length; den += byRole[rl].length;
    }
    const aggP = den ? num / den : 50;
    const mine = avg(rs.map(r => r.g.raw[i]));
    // flag metrics where a big block of peers is tied at one value (solo kills,
    // wards killed): the percentile then says less than the tie share does
    const qq = grp.m[m.key].q; let tie = 0;
    while (tie + 1 < qq.length && qq[tie + 1] === qq[0]) tie++;
    const tiePct = tie * (100 / (qq.length - 1));
    html += '<div class="mrow">' +
      '<div>' + m.label + (m.sign < 0 ? ' <span class="note">\u2193</span>' : '') + '</div>' +
      '<div class="pbar" title="average of per-game percentiles ' + pc(mean) +
        '; percentile of your average value ' + pc(aggP) + '">' +
        '<span style="width:' + mean.toFixed(1) + '%"></span>' +
        '<i style="left:' + aggP.toFixed(1) + '%"></i><em>' + pc(mean) + '</em></div>' +
      '<div class="vals">you ' + nf(mine, 2) + ' \u00b7 peers ' + nf(qq[10], 2) +
      (tiePct >= 10 ? ' <span class="note">(' + Math.round(tiePct) + '% tied at ' + nf(qq[0], 2) + ')</span>' : '') +
      '</div></div>';
  });
  box.innerHTML = html || '<div class="note">no games in this filter</div>';
}

// ---------- score per game ----------
function drawTrend(rs) {
  const svg = document.getElementById('trend'); clear(svg);
  const W = svg.clientWidth || 860, H = 240, pad = { l: 34, r: 10, t: 10, b: 22 };
  const n = rs.length; if (!n) return;
  const ys = v => pad.t + (H - pad.t - pad.b) * (1 - v / 100);
  [0, 25, 50, 75, 100].forEach(v => {
    svg.appendChild(el('line', { x1: pad.l, x2: W - pad.r, y1: ys(v), y2: ys(v),
      stroke: v === 50 ? '#3a4252' : '#2a3040', 'stroke-dasharray': v === 50 ? '' : '3 3' }));
    svg.appendChild(el('text', { x: pad.l - 5, y: ys(v) + 3, 'text-anchor': 'end', 'font-size': 10, fill: '#8b93a7' }, v));
  });
  const bw = (W - pad.l - pad.r) / n;
  rs.forEach((r, i) => {
    const x = pad.l + i * bw, top = ys(r.cp), base = ys(0);
    svg.appendChild(el('rect', { x: x + bw * 0.18, y: top, width: Math.max(1.5, bw * 0.64),
      height: Math.max(1, base - top), fill: r.g.win ? '#3fb96f' : '#e05656', opacity: .75 }));
  });
  const win = 5, pts = [];
  for (let i = 0; i < n; i++) {
    const s = rs.slice(Math.max(0, i - win + 1), i + 1);
    pts.push([pad.l + i * bw + bw / 2, ys(avg(s.map(x => x.cp)))]);
  }
  svg.appendChild(el('polyline', { points: pts.map(p => p.join(',')).join(' '),
    fill: 'none', stroke: '#4f9cf9', 'stroke-width': 2 }));
  svg.appendChild(el('text', { x: pad.l, y: H - 5, 'font-size': 10, fill: '#8b93a7' }, 'oldest'));
  svg.appendChild(el('text', { x: W - pad.r, y: H - 5, 'font-size': 10, fill: '#8b93a7', 'text-anchor': 'end' }, 'newest'));
}

function drawHist(id, vals, grp, grid) {
  const svg = document.getElementById(id); clear(svg);
  const W = svg.clientWidth || 860, H = +svg.getAttribute('height') || 240, pad = { l: 26, r: 12, t: 10, b: 22 };
  if (!vals.length) return;
  const q = grid.q, lo = Math.min(q[1], Math.min.apply(null, vals)), hi = Math.max(q[19], Math.max.apply(null, vals));
  const xs = v => pad.l + (W - pad.l - pad.r) * (v - lo) / (hi - lo + 1e-9);
  const NB = 16, counts = new Array(NB).fill(0);
  vals.forEach(v => counts[Math.min(NB - 1, Math.floor((v - lo) / (hi - lo + 1e-9) * NB))]++);
  const mx = Math.max.apply(null, counts) || 1;
  svg.appendChild(el('rect', { x: xs(q[5]), y: pad.t, width: Math.max(2, xs(q[15]) - xs(q[5])),
    height: H - pad.t - pad.b, fill: '#4f9cf9', opacity: .10 }));
  svg.appendChild(el('line', { x1: xs(q[10]), x2: xs(q[10]), y1: pad.t, y2: H - pad.b, stroke: '#f0a13a', 'stroke-width': 2 }));
  counts.forEach((c, b) => {
    const h = c / mx * (H - pad.t - pad.b), x = pad.l + (W - pad.l - pad.r) * b / NB;
    if (c) svg.appendChild(el('rect', { x: x + 1, y: H - pad.b - h, width: (W - pad.l - pad.r) / NB - 2,
      height: h, fill: '#4f9cf9', opacity: .8 }));
  });
  svg.appendChild(el('text', { x: xs(q[10]) + 4, y: pad.t + 10, 'font-size': 10, fill: '#f0a13a' }, 'peer median'));
  [lo, (lo + hi) / 2, hi].forEach(v => svg.appendChild(el('text',
    { x: xs(v), y: H - 6, 'text-anchor': 'middle', 'font-size': 10, fill: '#8b93a7' }, nf(v, 2))));
}

function drawOut(rs) {
  drawHist('outChart', rs.map(r => r.out), groupFor(majorityRole(rs)), groupFor(majorityRole(rs)).outperform);
  const better = rs.filter(r => r.out > 0).length;
  document.getElementById('outNote').innerHTML =
    'You outscored your own teammates in <b>' + better + ' of ' + rs.length + '</b> games \u00b7 average differential ' +
    (avg(rs.map(r => r.out)) >= 0 ? '<b class="win">' : '<b class="lose">') + nf(avg(rs.map(r => r.out)), 2) + '</b> SD \u00b7 ' +
    'median peer differential is 0 by construction.';
}

// ---------- tables ----------
function champTable(rs) {
  const by = {};
  rs.forEach(r => {
    const c = by[r.g.champ] || (by[r.g.champ] = { n: 0, w: 0, cp: 0, role: r.g.role, key: r.g.champKey });
    c.n++; c.w += r.g.win ? 1 : 0; c.cp += r.cp;
  });
  const list = Object.entries(by).sort((a, b) => b[1].n - a[1].n).slice(0, 14);
  const cdn = P.meta.ddragon;
  let html = '<table><tr><th>champion</th><th></th><th>role</th><th>games</th><th>win</th><th>avg score pct</th></tr>';
  list.forEach(([name, c]) => {
    const img = c.key ? '<img class="champ" src="https://ddragon.leagueoflegends.com/cdn/' + cdn +
      '/img/champion/' + c.key + '.png" alt="">' : '';
    html += '<tr><td>' + img + '</td><td style="text-align:left">' + name + '</td><td>' + c.role + '</td><td>' + c.n +
      '</td><td>' + Math.round(c.w / c.n * 100) + '%</td><td>' + pc(c.cp / c.n) + '</td></tr>';
  });
  document.getElementById('champTable').innerHTML =
    (list.length ? html + '</table>' : '<div class="note">no games in this filter</div>');
}

function gameTable(rs) {
  const idx = k => MK.indexOf(k);
  const cols = ['date', 'champion', 'role', 'result', 'KDA', 'CS/min', 'dmg/min', 'vision/min', 'gold/min', 'score pct', 'vs team'];
  let html = '<table><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr>';
  rs.slice().reverse().slice(0, 20).forEach(r => {
    const g = r.g, i = idx;
    html += '<tr><td>' + g.date + '</td><td>' + g.champ + '</td><td style="color:' + ROLE_COLORS[g.role] + '">' + g.role +
      '</td><td class="' + (g.win ? 'win">W' : 'lose">L') + '</td><td>' + g.k + '/' + g.d + '/' + g.a + '</td>' +
      '<td>' + nf(g.raw[i('csm')], 1) + '</td><td>' + Math.round(g.raw[i('dpm')]).toLocaleString() + '</td>' +
      '<td>' + nf(g.raw[i('vspm')], 1) + '</td><td>' + Math.round(g.raw[i('gpm')]).toLocaleString() + '</td>' +
      '<td>' + pc(r.cp) + '</td><td>' + (r.out >= 0 ? '<span class="win">+' : '<span class="lose">') + nf(r.out, 2) + '</span></td></tr>';
  });
  document.getElementById('gameTable').innerHTML = html + '</table>';
}

// ---------- weights ----------
function weightsBox() {
  const box = document.getElementById('weights');
  if (box.dataset.built) { syncSliders(); return; }
  box.dataset.built = '1';
  box.innerHTML = '';
  METS.forEach(m => {
    const row = document.createElement('div'); row.className = 'wrow';
    row.innerHTML = '<span class="name">' + m.label + '</span>' +
      '<input type="range" min="-3" max="3" step="0.1" value="' + DEF_W[m.key] + '">' +
      '<span class="val">' + DEF_W[m.key].toFixed(1) + '</span>';
    const inp = row.querySelector('input'), val = row.querySelector('.val');
    inp.dataset.key = m.key;
    inp.oninput = () => {
      state.w[m.key] = +inp.value; val.textContent = (+inp.value).toFixed(1);
      scache.clear(); render();
    };
    box.appendChild(row);
  });
  document.getElementById('resetW').onclick = () => {
    METS.forEach(m => state.w[m.key] = m.sign); scache.clear(); syncSliders(); render();
  };
}
function syncSliders() {
  document.querySelectorAll('#weights input').forEach(inp => {
    inp.value = state.w[inp.dataset.key];
    inp.parentElement.querySelector('.val').textContent = (+state.w[inp.dataset.key]).toFixed(1);
  });
}

function methodText() {
  const c = P.corpus, pr = P.profile;
  document.getElementById('method').innerHTML =
    '<b>Baseline.</b> Every metric is z-scored and percentiled against ' + c.nRows.toLocaleString() +
    ' player-games from ' + c.nMatches.toLocaleString() + ' ' + c.region +
    ' ranked solo queue games (collected via the Riot API, baseline frozen ' + c.generated.slice(0, 10) + '). ' +
    'Two baselines are offered: all ranks for your role, and your rank band \u00d7 role. ' +
    'Values are trimmed to the corpus 0.1%/99.9% range before scoring.<br>' +
    '<b>Score.</b> Equal-weight mean of the 14 signed z-scores (deaths and death share count negatively). ' +
    'Percentiles are computed per game inside the role played in that game, then averaged \u2014 averaging raw stats first ' +
    'would let one 25-0 stomp dominate the result. Ties are given mid-rank (the same Mann-Whitney convention the ' +
    'AUC code uses), so a metric that is zero in most peer games does not report every tied game as 0%.<br>' +
    '<b>Teammate differential.</b> Own score minus the mean of your teammates\u2019 scores (same baseline), so team strength ' +
    'cancels out; this is the within-team signal the project\u2019s Model C uses.<br>' +
    '<b>Caveats worth knowing.</b> (1) End-of-game box scores partly <i>re-encode</i> the win \u2014 stomps inflate everything, ' +
    'so read the differential panel and the distribution, not just the win-rate tile. ' +
    '(2) ' + P.nGames + ' games is a small sample: a \u00b15 percentile change is noise. ' +
    '(3) Your band is your <i>current</i> tier (' + (pr.soloRank ? pr.soloRank.tier + ' ' + pr.soloRank.division : 'unranked') +
    ') applied to all games shown. ' + '(4) The corpus is ' + c.region + ' solo queue, so cross-region comparisons inherit ' +
    c.region + '-specific playstyle. ' + '(5) Percentiles describe association with winning in the corpus, not causal carry value.' +
    '<br><b>Refresh.</b> <code>python collect_player.py "' + pr.riotId + '" --region ' + (pr.region || 'na1') +
    '</code> then <code>python build_player_web.py --slug ' + P.slug + '</code>';
}

// ---------- render ----------
let tm = null;
function render() {
  const rs = rows();
  baselineBtns(); roleBtns(); weightsBox();
  tiles(rs);
  if (!rs.length) {
    ['trend', 'hist', 'outChart'].forEach(id => clear(document.getElementById(id)));
    document.getElementById('metricBars').innerHTML = '<div class="note">no games in this filter</div>';
    document.getElementById('champTable').innerHTML = document.getElementById('gameTable').innerHTML = '';
    return;
  }
  metricBars(rs);
  drawTrend(rs);
  const grp = groupFor(majorityRole(rs));
  drawHist('hist', rs.map(r => r.comp), grp, grp.comp);
  drawOut(rs);
  champTable(rs);
  gameTable(rs);
  methodText();
}
header();
render();
window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(render, 150); });
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", default=None,
                    help="only build this player (repeatable)")
    ap.add_argument("--players-dir", type=Path, default=PLAYER_DIR)
    ap.add_argument("--ref", type=Path, default=REF_FILE)
    ap.add_argument("--no-html", action="store_true", help="print summary only")
    args = ap.parse_args()

    if not args.ref.exists():
        raise SystemExit(f"missing {args.ref} - run: python build_reference.py")
    ref = json.loads(args.ref.read_text(encoding="utf-8"))

    dirs = []
    if args.players_dir.exists():
        dirs = [d for d in sorted(args.players_dir.iterdir())
                if (d / "profile.json").exists() and (d / "matches.jsonl").exists()]
    if args.slug:
        dirs = [d for d in dirs if d.name in set(args.slug)]
    if not dirs:
        # still publish an empty index, so the landing page's "built reports"
        # link is never a 404 before the first lookup
        if not args.no_html:
            PLAYERS_OUT.mkdir(parents=True, exist_ok=True)
            (WEB_DIR / "player.js").write_text(APP_JS, encoding="utf-8")
            _write_index({})
        raise SystemExit(
            f"no players found in {args.players_dir}. "
            f'Run: python collect_player.py "Name#TAG"')

    payloads = []
    for d in dirs:
        p = player_payload(d, ref)
        if p:
            print(f"[summary] {console_summary(p, ref)}")
            payloads.append(p)
    if payloads and not args.no_html:
        write_pages(payloads)
    elif payloads:
        print("[web] skipped (--no-html)")


if __name__ == "__main__":
    main()
