# Which LoL Role Has the Most Impact?

> Two tracks now share one corpus. **(1)** The role-impact study, answered below.
> **(2)** Single-player reports: look someone up by Riot ID (`Name#TAG`) and score their
> games against the collected population — see [Single-player reports](#single-player-reports).

## Question
Build a normalized performance score for each role, see how accurately it
predicts that player's win rate, and rank roles by predictive power. The role
whose performance best predicts winning is judged the most impactful.

## Answer (final, audit-aware, ~9,600 matches)
"Impact" depends on which question you ask — the project separates three
estimands (see `analyze_rigor.py`):

1. **Raw performance -> win** (Model A): Jungle leads (0.938 vs 0.91-0.92),
   but the raw signal is largely team dominance re-read through end-of-game
   stats — in 32min+ close games all roles compress to 0.81-0.84.
2. **Share of team output -> win** (Model B): **Jungle is significantly ahead
   of ALL four roles** (0.667; deltas vs TOP -0.112, MID -0.121, BOT -0.077,
   SUP -0.073, every 95% CI excludes 0) — including in 32min+ close games.
   The lead is stable across 23 patches and robust to champion controls.
3. **Performance relative to your own teammates -> win** (Model C,
   corrected): Only two roles carry real within-team differential signal:
   **Support 0.690 and Jungle 0.675 — statistically tied** (delta +0.015,
   CI [-0.004,+0.035]). All others are significantly below jungle
   (Top -0.078, Bottom -0.097, Mid -0.110). Same story in close games.
   (An earlier version of Model C that also residualized the five share
   metrics reported "Support 0.877, decisive" — that was an artifact: for
   share metrics the residual is provably an affine transform of the own
   value with zero teammate information, letting the model reconstruct team
   totals. See `check_modelc_degeneracy.py`.)

**Across the ladder** (`analyze_ladder.py`, ~1,200 matches per rank band):
Jungle is the best role in **every** band from Iron to Apex (0.63-0.69 AUC)
with Support/Bottom as runners-up. There are no rank "inflection points" —
the earlier small-sample crossovers were noise.

**Model benchmark:** at ~10k matches the MLP neural net and gradient
boosting slightly outperform logistic regression (MLP 0.925-0.948 vs LR
0.908-0.938 test AUC), answering the original "shouldn't this be a neural
net?" question: with enough data the NN ekes out a small edge, but the role
ranking is identical across all model classes.

## Data
- Source: Riot Games API, ranked solo queue (queue 420), NA ladder
- Sampling: seeds from every bucket (Iron IV -> Challenger), snowball expansion
- One row per player-game, tagged with the rank bucket of the player whose
  history surfaced the match (solo queue matches by MMR, so this is a good
  proxy for the match's rank band)
- Raw storage: `data/raw/matches.jsonl` (one compact JSON per match)

### Pro-play track (tier 1)
- Source: Oracle's Elixir public match CSVs (community-aggregated pro box
  scores, updated daily), 2025-2026
- Tier 1 filter: LCK, LPL, LEC, LCS, LCP + internationals (Worlds, MSI,
  First Stand, EWC); academy/regional leagues excluded
- `collect_pro.py` converts OE rows into the same record schema as the ladder
  track (`data/raw/pro_matches.jsonl`), with `seedTier` = league name so the
  role x tier normalization becomes role x league
- OE's `datacompleteness` flag is ignored (all LPL 2025 games are marked
  "partial" while every core stat is present); instead rows missing any core
  field are dropped, and games that lose rows to that are dropped whole
- Player identity = OE `playerid` (stable string ids), team = `teamname`
- Feature building reuses `prepare_features.py` via `--input/--output/
  --no-queue-filter`; analyses take `--input/--tag` to run on the pro parquet
- `analyze_players_pro.py` — player-level: outperformance vs own teammates,
  carry rate, swing, with/without team win rate, champion pool (tier-1 only).
  Players are split into per-team STINTS (team change or >=60-day gap), so
  mid-season moves are scored per spell
- `analyze_worlds_primer.py` — ranks the official Worlds 2026 rosters (95
  rostered players) by a watch score (team-stint outperform/carry/swing/AUC/
  level percentiles); excludes subs and replaced players; outputs
  `results/worlds_primer_2026.md`
- Known pro-specific caveats: fixed roles (no autofill), box scores only
  (same end-of-game-stomp caveat applies), players repeat across games
  (cluster bootstrap uses matchId), patch/league meta mixing

## Pro results (tier 1, 2025-2026, 4,670 games / 46,700 player-games)

Role ranking flips vs the ladder. Model B (share-of-team-output -> win,
well-formed per role): **Support 0.735 > Jungle 0.674 > Bottom 0.620 >
Top 0.609 > Mid 0.603** — Support leads, which it never did in solo queue.
Holds in the 32min+ close-game cut (Support 0.737, Jungle 0.645). Reading of
the coefficients: support winning = *less* damage share, *more* gold share
(reverse-causal: ahead supports get gold), positive death share; jungle
winning = big cs_share + vision_share. Raw Model A hits 0.93-0.98 AUC —
even stronger stomp artifact than ladder (pro leads close out); the honest
number there is the equal-weight composite: Jungle 0.770 / Bottom 0.770 /
Mid 0.719 / Support 0.708 / Top 0.693.

Player-level (`results/player_impact_pro.csv`, min 20 games, 349 players):
`outperform` = own composite minus teammates' mean (team strength cancels).
Headliners: Adam (NAV top, +0.56, carried 73% of games), Castle (LYON),
Keshi, Kiaya, and Chovy as the elite big-sample case (282 games, +0.245,
0.727 win rate — Gen.G's engine). Wenbo (BLG top) posts the largest
*negative* differential among winners' teams. See
`results/player_impact_pro.md` for all leaderboards and
`results/player_impact_teams_pro.csv` for each team's most impactful player.

## Single-player reports

Look up one player by Riot ID and compare their games against the collected population:

```powershell
python build_reference.py                                   # freeze corpus -> baseline distributions
python collect_player.py "killed cat#NA1" --region na1 --n 100
python build_player_web.py                                  # -> web/players/<slug>.html
```

### How deep to go, and why the report has a window control

`collect_player.py` pages the id list 100 at a time up to `--n 500` and stops when Riot runs out of
history (`profile.json` records `historyExhausted`). Cost is linear in games - about one request per
match - so at a dev key's 100 requests / 2 min: 100 games ~2 min, 300 ~6 min, 500 ~10 min. Everything
is cached per player, so widening the pull later only fetches the difference.

Pulling deep is *not* automatically better, which is why the page slices it. Your own first report
showed the problem immediately: 37 games already spanned **4 patches**, and a Diamond IV player's
games from four months ago may predate their current rank. So the collector stores everything and the
report fits its personal models on **last 50 / last 100 / all** separately, and the header names the
date range and patch list of the window you're looking at - a window crossing >2 patches is flagged
as mixing metas. Peer baselines are still keyed to your *current* band, which stays a stated
limitation for the older games.

- `build_reference.py` writes `data/processed/reference.json` (Python) and `web/reference.js`
  (browser): mean/sd, a 21-point quantile grid, the equal-weight composite distribution and the
  teammate-differential distribution for every role and every role x rank band (45 groups at
  96k player-games), plus 120-400 z-scored sample rows per group so a page can re-rank you under
  custom weights without shipping the corpus.
- `collect_player.py` resolves `Name#TAG` through Account-V1 (**Summoner-V4 `by-name` is
  deprecated**; Riot IDs are shard-scoped, so a wrong `--region` reads as "player not found"),
  pulls the last N ranked solo (420) games, and writes them in the *same* compact schema as
  `data/raw/matches.jsonl` — so `prepare_features.load_matches()` parses player files unchanged.
  Per-player cache: re-running only fetches games not already on disk.
- `build_player_web.py` scores each player against the baseline and emits static HTML. No
  backend, and no API key anywhere in the page.

### Live API notes (found by looking up a real account)

| Quirk | Handling |
|---|---|
| League-V4 entries put the division in **`rank`**, not `division`, for tiers below Apex (e.g. `tier: DIAMOND, rank: IV`). For Master+ `rank` is the numeric ladder position instead. | `find_solo_rank()` reads `division` first, then `rank` only if it is not a number, so a Challenger position of `250` can never be printed as a division. |
| `riotIdTagLine` is masked to `null` and `summonerName` is `""` for other participants; only `riotIdGameName` comes back. | Collected games are half-addressable by name; lookup always needs the tag from the person asking. Identity resolution in the pipeline is by **puuid**, never by name. |
| `challenges` still carries `soloKills` and `killParticipation` (126 keys in a 2026 game), so `slim_match()` keeps working. | Verified against a live match rather than assumed. |
| Match IDs are now `NA1_5653773454`-style, not GUIDs. | The pipeline stores IDs as opaque strings, so both formats coexist in `data/`. |

| Panel | Method, and why |
|---|---|
| Metric percentiles | Every metric is percentiled **per game, inside the role played in that game**, then averaged. A jungler's CS/min is never judged against ADC CS. Averaging per-game ranks rather than taking the percentile of the average stops one 25-0 stomp from carrying the number (the orange tick shows how far apart those two are). Ties are given **mid-rank**, like the Mann-Whitney AUC code, because ~55% of peer games are tied at zero on stats such as solo kills — without it every tied game reads as 0%. |
| Two baselines | *All ranks* = every corpus game for that role. *My band* = role x the player's rank band. Same player can read 56th percentile overall but 45th against Apex peers — rank matching is the whole point. |
| **Four weightings** | *Equal* (neutral signed mean), *Model A* (raw levels), *Model B* (team shares), *Model C* (within-team residuals) — the latter three fitted **per role** by logistic regression on the corpus, train/test split by match, and the button shows that role's held-out AUC. Default is C. A scores ~0.92 per role almost entirely because box scores re-encode stomps, so as a "how good were you" score it is the least honest of the four; B and C are the estimands the research track actually stands behind. |
| Teammate differential | Own score minus the mean of your four teammates' scores on the same baseline. Under Model C this *is* the score (the residual axis), which is why the two panels agree there by construction. |
| **Your weights vs your role's** | Refits a win model on the player's own games alone — ridge-shrunk (C=0.2) and scored **leave-one-out** so it cannot grade data it saw, with bootstrap SEs — then prints cosine similarity to the role's vector and both AUCs side by side. On 33 games the answer is usually "your own fit is worse", which is the honest result: 30-odd games cannot out-evaluate 19,000. |
| **What you actually win with** | Per-metric univariate win logits (log-odds per SD) for the player vs the same number for their role x band peers, with both SEs and z = gap / combined SE. Better powered than the multivariate weights. Because 14 metrics are tested at once, the note flags |z| >= 2.9 (Bonferroni) rather than every |z| >= 2. |
| Score weights | Sliders re-rank everything live. Touching one switches to *custom* mode, whose peer percentiles come off a sampled baseline (80-250 games per group), so they move by ~1-2%; the four published weightings use exact quantile grids. |

Fitted per-role held-out AUCs on the 96k corpus, which double-check the research
track's own rankings: **A** 0.91-0.94 (all roles, stomp-inflated), **B** TOP 0.56 / JUNGLE
**0.66** / MID 0.54 / BOT 0.59 / SUP 0.60 (jungle leads, as published), **C** TOP 0.59 /
JUNGLE 0.67 / MID 0.56 / BOT 0.58 / SUP **0.70** (support and jungle on top, as published).

Stated limits: 40 games is a small sample (a 5-point percentile move is noise); the band is the
player's **current** tier applied to all their games, so promotions score old games generously;
end-of-game box scores partly re-encode the win (the stomp caveat from the assumptions audit),
which is why the differential panel is the one to trust; the baseline is NA ranked solo, so
other regions inherit NA's playstyle; percentiles describe association with winning in the
corpus, not causal carry value.

### Live lookup in the browser (`server.py`)

```powershell
python server.py            # -> http://127.0.0.1:8765/search
```

Static pages cannot do live lookup: Riot deliberately omits CORS headers so that a key can never
be shipped in browser JS. `server.py` is the small backend that holds the key - stdlib
`http.server` only, nothing to install, one worker thread so Riot's 100-requests/2-min window is
never raced, and it calls the same `collect()` / `write_page()` functions as the CLI.

| Route | |
|---|---|
| `/search` | Riot ID box: region, game count, force-refresh; polls while the worker fetches, then opens the report |
| `/players/<slug>` | a report; `/players/` lists everything built |
| `/explorer`, `/` | corpus role-impact explorer, landing page |
| `/api/health`, `/api/players`, `/api/lookup?riotId=Name#TAG&region=na1&n=40&force=1` | JSON used by the search page |

A fresh lookup costs roughly one request per game (100 games is ~105 requests, about 2 minutes at a
dev key's 100-per-2-min window; 500 games is about 10 minutes), so the worker runs jobs one at a time
and the search page polls. Results are cached in `data/players/<slug>/` and re-served for `--ttl`
seconds (default 900), which makes a repeat search instant and free; the report then slices that
cache into last-50 / last-100 / all windows without asking Riot again. If the refresh fails
mid-lookup (expired key, rate limited) you still get the last snapshot with a warning instead of an
error page.

It binds `127.0.0.1` by default. `--host 0.0.0.0` hands your key's rate budget to everyone who can
reach the port and shows every built report, so it needs a real deployment story: a Riot
**production** key for traffic that isn't just you, plus per-IP throttling and some limit on how
many distinct players get pulled per hour. `collect_player.py` and the search box both accept
`Name#TAG` plus a region, and the collector caches by player, so an hour of lookups costs about
450 requests against a 100/2-min dev key.

### Publishing to GitHub Pages

The landing page (`index.html`), `web/index.html` (explorer), `web/reference.js` and
`web/player.js` are committed, so Pages from the branch root serves the landing at `/`, the
explorer at `/web/index.html`, and reports at `/web/players/`. **`web/players/` is gitignored on
purpose**: a report puts a named player's stats on a public site, so publishing one is a
deliberate command -

```powershell
python build_player_web.py --slug killed-cat
git add -f web/players/killed-cat.html web/players/index.html web/players/registry.json
```

Looked-up games always stay in gitignored `data/` (they contain PUUIDs). Turning Pages on is a
one-time repo setting (or `gh api -X POST repos/<you>/lolstats/pages`). Letting *strangers* search
arbitrary players needs more than that: a host for `server.py` holding the key, a Riot
**production** key for traffic that isn't just you, and per-IP throttling - Riot deliberately sends
no CORS headers, which is precisely why a static page cannot do live lookup itself.

### Dev tool

`python make_offline_player.py` fabricates `data/players/offline-demo/` from the already
collected corpus, with zero API calls, so scoring or template changes can be tested against a real
40-game history. The Riot ID in it is invented and that report must never be published - another
reason `web/players/` is ignored.

## Pipeline
1. `collect_matches.py` — Riot API collector with rate limiting + checkpoints
2. `prepare_features.py` — per-player metrics, z-scored within **role** and
   within **role x tier**; raw levels, team shares, and team-residualized
   features (own stat minus teammates' mean)
3. `analyze_shares.py` — Model B headline (share-based, well-formed per role)
4. `analyze_roles.py` — Model A with model benchmarks (logistic regression,
   gradient boosting, MLP neural net, equal-weight composite)
5. `analyze_impact.py` — own vs teammates vs relative decomposition
6. `analyze_tiers.py`, `analyze_ladder.py` — rank-stratified analyses
7. `analyze_rigor.py` — the audit: three estimands, cluster-bootstrap CIs,
   paired role comparisons, close-game robustness, player-clustered
   pseudoreplication check, champion controls, patch stability
8. `build_reference.py` — freeze the corpus into baseline distributions
   (`reference.json` for Python, `reference.js` for the browser)
9. `collect_player.py` + `build_player_web.py` — one player by Riot ID, scored
   against those baselines as a static report page
10. `server.py` — the same lookup as a local web app with a search box, so the
    API key stays server-side (Riot sends no CORS headers on purpose)

## Assumptions audit (what we checked and what we found)

| Assumption | Check | Verdict |
|---|---|---|
| End-of-game stats re-encode the win (stomps) | Stratify by game length | Confirmed: Model A drops 0.93->0.79 in 32min+ games; role differences vanish. Headline claims must use share/residual models or close games. |
| Raw stats measure the player vs their team | Model C corrected (non-degenerate residuals only) | Share-residual degeneracy found and fixed (res(share) is an affine transform of own value). Clean differential signal is far smaller than first estimated; Jungle and Support tie at the top, all other roles significantly below. |
| Share features are "clean" team-relative measures | Shares-only vs shares+levels | Mixing shares with levels lets a model reconstruct team totals (gold ~ share x team gold) and inflate AUC to 0.98 — documented; never mix without acknowledging. |
| Rows are independent | Player-clustered bootstrap | 1.2 rows/player on average; CIs barely widen. Snowballing did not concentrate the sample. |
| No uncertainty reported | Match-clustered bootstrap, paired delta-AUC CIs | Jungle's Model B lead is significant vs TOP/MID/BOT; not vs SUPPORT. Model A role differences are NOT significant in close games. |
| Champion pick confounds performance | Champion one-hot controls | Negligible for raw levels; slightly reduces jungle's share signal (0.611->0.557 in close games); ranking stable. |
| Meta shifts across patches | Per-patch models (16.18, 16.19) | Jungle top in both; ranking stable. |
| Multicollinearity (negative kp/csm coefficients) | Treated as artifacts | Coefficients with collinear inputs are for narrative only; AUC comparisons are the decision metric. |

### Remaining limitations (documented, not fixable with current data)
- **Rank is a proxy**: matches are tagged with the seed player's tier, not
  verified per-participant rank.
- **Association, not causation**: AUC measures predictive association. Roles
  are not randomly assigned; "main X to climb" also reflects who plays X.
- **Vision quality** is invisible in box scores; vision stat weakness is
  likely measurement failure, not proof vision doesn't matter.
- **Patch mixing**: pooled models span patches 14.22-16.20 (a year of metas).
- **Normalization transduction**: z-scores use full-dataset moments before
  the train/test split (feature-space only; mild).

## How to run
```powershell
python collect_matches.py --target 10000      # collects (resumable, Ctrl+C safe)
python prepare_features.py                    # builds player_games.parquet
python analyze_rigor.py                       # audit: 3 estimands + CIs
python analyze_shares.py                      # Model B headline
python analyze_roles.py                       # Model A + model benchmarks
python analyze_impact.py                      # own vs teammates decomposition
python analyze_tiers.py SILVER DIAMOND        # rank comparisons
python analyze_ladder.py                      # role impact across the ladder
python build_web.py                           # interactive explorer -> web/index.html

# ---- pro-play track (tier 1 only) ----
python collect_pro.py                         # download OE 2025+2026 CSVs, convert
python prepare_features.py --input data/raw/pro_matches.jsonl `
    --output data/processed/player_games_pro.parquet --no-queue-filter
python analyze_shares.py --input data/processed/player_games_pro.parquet --tag _pro
python analyze_roles.py  --input data/processed/player_games_pro.parquet --tag _pro
python analyze_players_pro.py                 # player outperformance/importance

# ---- single-player track (needs a fresh-ish Riot ID lookup) ----
python build_reference.py                     # re-run whenever the corpus changes
python collect_player.py "killed cat#NA1" --region na1 --n 40
python build_player_web.py                    # all players, or --slug <name>
python build_player_web.py --no-html          # print the summary only
python server.py                              # search box -> http://127.0.0.1:8765/search
```
You can run steps 2+ at any time on partial data.

## Caveats
- A single player's stats cap out around ~0.60-0.65 AUC for *honest*
  individual-differential prediction — 9 other players determine most of a
  match. What matters is the *relative* ranking across roles and estimand.
- Dev API keys rate-limit to 100 requests/2 min; collection of ~10k matches
  takes a few hours. State persists in `data/raw/state.json`.
- Dev keys also **expire 24h after they are created**. A `401/403` from any script
  means generate a new one at developer.riotgames.com and paste it into `.env`
  (`RIOT_API_KEY=...`) — the lookup paths now fail with that advice instead of a
  bare traceback.
