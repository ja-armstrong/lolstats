# Which LoL Role Has the Most Impact?

## Question
Build a normalized performance score for each role, see how accurately it
predicts that player's win rate, and rank roles by predictive power. The role
whose performance best predicts winning is judged the most impactful.

## Data
- Source: Riot Games API, ranked solo queue (queue 420), NA ladder
- Sampling: seeds from every bucket (Iron IV → Challenger), snowball expansion
- One row per player-game, tagged with the rank bucket of the player whose
  history surfaced the match (solo queue matches by MMR, so this is a good
  proxy for the match's rank band)
- Raw storage: `data/raw/matches.jsonl` (one compact JSON per match)

## Pipeline
1. `collect_matches.py` — Riot API collector with rate limiting + checkpoints
2. `prepare_features.py` — per-player metrics, z-scored within **role** and
   within **role × tier**:
   kda, kp (kill participation), csm, gpm, dpm, dmg_share, vspm, wpm,
   dthpm (deaths/min), skpm (solo kills/min) + equal-weight composite score
3. `analyze_roles.py` — per-role win-prediction models:
   - Logistic regression (primary, interpretable)
   - Gradient boosting (nonlinear benchmark)
   - MLP neural net (deep-learning benchmark)
   - Equal-weight composite (no fitting at all)
   - Train/test split **grouped by match** to prevent leakage
   - Primary metric: test ROC-AUC per role

## How to run
```powershell
python collect_matches.py --target 10000      # collects (resumable, Ctrl+C safe)
python prepare_features.py                    # builds player_games.parquet
python analyze_roles.py                       # results in results/
```
You can run steps 2-3 at any time on partial data.

## Caveats
- A single player's stats cap out around ~0.60-0.65 AUC — 9 other players
  determine most of a match. What matters is the *relative* ranking across roles.
- Match rank is a proxy (seed player's tier), not verified per-participant ranks.
- Dev API keys rate-limit to 100 requests/2 min; collection of ~10k matches
  takes a few hours. State persists in `data/raw/state.json`.
