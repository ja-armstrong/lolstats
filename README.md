# Which LoL Role Has the Most Impact?

## Question
Build a normalized performance score for each role, see how accurately it
predicts that player's win rate, and rank roles by predictive power. The role
whose performance best predicts winning is judged the most impactful.

## Answer (current, audit-aware, 8,340 matches)
"Impact" depends on which question you ask — the project separates three
estimands (see `analyze_rigor.py`):

1. **Raw performance -> win** (Model A): Jungle leads (0.928 vs 0.90-0.92),
   but the raw signal is largely team dominance re-read through end-of-game
   stats — in 32min+ close games all roles compress to 0.80-0.84.
2. **Share of team output -> win** (Model B): **Jungle is significantly ahead
   of ALL four roles** (0.661; deltas vs TOP -0.108, MID -0.126, BOT -0.072,
   SUP -0.069, every 95% CI excludes 0) — including in 32min+ close games,
   where the previously ambiguous jungle-vs-bottom comparison resolved
   (-0.050, CI [-0.088,-0.011]). The lead is stable across 21 patches and
   robust to champion controls.
3. **Performance relative to your own teammates -> win** (Model C):
   **Support leads decisively** (0.871; delta vs jungle +0.145, CI
   [+0.129,+0.161]), stable in close games (0.774) and across every patch
   sampled. Bottom/ADC is second (+0.085), and TOP is now also significantly
   above jungle (+0.030). Mid is the least individually-differential role.

**Across the ladder** (`analyze_ladder.py`, ~1,050 matches per rank band):
Jungle is the best role in **every** band from Iron to Apex (0.63-0.69 AUC)
with Support consistently second. The earlier "inflection points" (bottom in
bronze/silver, mid in gold) were small-sample noise — with proper sample
sizes there are none.

**Model benchmark note:** at 8.3k matches the MLP neural net now slightly
outperforms logistic regression (0.921-0.931 vs 0.905-0.928 test AUC),
answering the original "shouldn't this be a neural net?" question: with
enough data the NN ekes out a small edge, but the ranking of roles is
identical across all model classes.

## Data
- Source: Riot Games API, ranked solo queue (queue 420), NA ladder
- Sampling: seeds from every bucket (Iron IV -> Challenger), snowball expansion
- One row per player-game, tagged with the rank bucket of the player whose
  history surfaced the match (solo queue matches by MMR, so this is a good
  proxy for the match's rank band)
- Raw storage: `data/raw/matches.jsonl` (one compact JSON per match)

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

## Assumptions audit (what we checked and what we found)

| Assumption | Check | Verdict |
|---|---|---|
| End-of-game stats re-encode the win (stomps) | Stratify by game length | Confirmed: Model A drops 0.93->0.79 in 32min+ games; role differences vanish. Headline claims must use share/residual models or close games. |
| Raw stats measure the player vs their team | Model C (team-residualized features) | Raw-level AUC is mostly team-shared signal; the *differential* signal is real but smaller, and largest for support. |
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
```
You can run steps 2+ at any time on partial data.

## Caveats
- A single player's stats cap out around ~0.60-0.65 AUC for *honest*
  individual-differential prediction — 9 other players determine most of a
  match. What matters is the *relative* ranking across roles and estimand.
- Dev API keys rate-limit to 100 requests/2 min; collection of ~10k matches
  takes a few hours. State persists in `data/raw/state.json`.
