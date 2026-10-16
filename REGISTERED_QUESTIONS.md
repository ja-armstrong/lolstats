# Registered Questions v1

Nine behaviour questions and two placebos, fixed **before** the timeline corpus is queried. The
family size is 11; Benjamini-Hochberg is applied at q = 0.10 across the nine real questions, and the
two placebos are the pipeline's own false-positive detector. Nothing here may be added, re-worded or
re-thresholded after seeing outcomes without bumping the document version and re-running the family.

Every claim has two halves, each measured by whoever has the sample for it (ADR-0002):

- **Price** - win-rate difference, estimated on the cohort in like games (ADR-0004), reported with CI.
- **Gap** - the subject's behaviour rate on eligible instances vs the winning share of that cohort,
  with its own CI. The subject's 410 games are powered for this, not for the price.

Screening runs on the intermediate target; the match result is only the validation step (ADR-0003).

## Language used below

| Term | Meaning here |
|---|---|
| minute frame `F(t)` | the timeline frame at `floor(t / 60000)`; positions are accurate to within 60s |
| distance | Euclidean map units between two players' frame positions, walls ignored (~4,000 units is roughly a screen) |
| available ally | `championStats.health > 0` in `F(death)` - measured, not inferred, but blurred by the 60s frame |
| contestable state | the subject's team gold differential at `F(14)` is within the cohort's interquartile band around 0 - the game was still live |
| like games | same rank band, role, side, and gold/XP differential at `F(14)` (ADR-0004) |
| cut-point | the named cohort percentile, computed on the training half only (ADR-0006) |

## Verified against a real timeline before any rule was written

Measured on one 41-minute game (41 frames, 1,914 events):

- Positions move per frame for living players; `championStats.health` is present every frame, so aliveness
  is measured at 60s resolution rather than proxied by a death-then-40-seconds guess.
- `damageStats` are cumulative, so **per-minute damage deltas** are available - used for commitment
  instead of the kill event's `victimDamageDealt` arrays, which came back **empty** for the sampled kill.
- Travel distance per minute across that game: p25 1,764, p50 4,051, p75 7,121, p95 11,734 units - so
  "could you have gotten there" cut-points come from this distribution, not a guessed constant. The top
  of that range is not running: it is respawn and shop teleports, so any reachability rule must exclude
  frames where the player was at a fountain or had `health = 0` in the previous frame, or the cut-points
  silently measure deaths instead of movement.
- `F(0)` puts each team in its fountain cluster (~600,600 vs ~14,300,14,400) and minutes 1-4 separate
  lane homes, so zones and side are derivable without hard-coded map geometry.
- Assist lists live on the kill event as `assistingParticipantIds` (absent when there were none).
- Roles come from the match **detail** record joined by `participantId`; the timeline's own participant
  list carries only `participantId` and `puuid`, so `participantId` is now stored in the slim record.

## Family A - death geography

**A1 - Isolated death.** *Agency unit: presence.*
Eligible instance: a `CHAMPION_KILL` where `victimId` is the subject.
Behaviour: nearest **available** ally's distance at `F(death)`. Cut-point: cohort **p75** of that
distance (p25 as the mirror). Intermediate: deaths per 10 minutes in contestable state. Validation: win.
Falsifier: the price exists in the cohort but the subject's rate is already at the winning share.

**A2 - Committed death.** *Agency unit: selection.*
Eligible instance: same as A1.
Behaviour: in the minute of death, the subject's share of their own team's damage dealt to champions
(per-minute `damageStats` deltas). Cut-point: cohort **p75** of that share - a death where you were the
one fighting is your decision, not your team's.
Intermediate: trades where damage taken exceeded damage dealt. Validation: win.
The kill event's own `victimDamageDealt` arrays are **not** used: they came back empty in verification,
and cumulative frame damage is both present and differentiable.

**A3 - Deep death zone.** *Agency unit: presence.*
Eligible instance: same as A1. Behaviour: the death's `position` classified into a zone anchored on
that game's own structure coordinates (from `BUILDING_KILL` / `TURRET_PLATE_DESTROYED` positions), so
"enemy-side jungle" needs no hard-coded map constants. Cut-point: cohort p75 of deaths in enemy-half
zones. Intermediate: gold differential lost in the 60s after the death. Validation: win.

## Family B - pressure and recall tempo (wave state is out of scope, ADR-0005)

**B1 - Hoarded gold.** *Agency unit: timing.*
Eligible instance: the subject's `currentGold` first exceeding the cohort p50 cost of a boot or component
tier, while alive. Behaviour: minutes from that moment to the next `ITEM_PURCHASED` for that player.
Cut-point: cohort **p75** of that gap. Intermediate: XP lost during the gap (frames give per-minute `xp`).
Validation: win.

**B2 - Kill-to-structure conversion.** *Agency unit: selection.*
Eligible instance: a `CHAMPION_KILL` in the subject's lane region (zone from `laneType` on nearby plate
events) where the subject was killer or assistant. Behaviour: a `TURRET_PLATE_DESTROYED` or
`BUILDING_KILL` in that region within 90s. Cut-point: cohort **p25** (failure to convert). Intermediate:
turret plates taken per 10 minutes. Validation: win.

**B3 - Lane presence before 14.** *Agency unit: presence.*
Eligible instance: minutes 0-14 while alive. Behaviour: share of frames with the subject inside their
own lane's region. Cut-point: cohort **p25**. Intermediate: CS+XP differential at `F(14)`. Validation: win.
Note: B3 is the question most likely to be confounded by role proxies (an ADC who roams once is not
"absent"), so it reports cohort size per decile of presence.

## Family C - objective contests, kill-anchored (ADR-0005)

**C1 - Pit attendance.** *Agency unit: presence.*
Eligible instance: an `ELITE_MONSTER_KILL` whose pit was in a half the subject could reach given their
`F(t-1)` position (distance under cohort p75 of travel distance per minute). Behaviour: subject inside
the pit zone (radius from the kill's `position`) at `F(t-1)`. Cut-point: cohort **p75** attendance.
Intermediate: whether the monster was stolen/contested (`killerTeamId` vs subject's team). Validation: win.

**C2 - Approach distance.** *Agency unit: timing.*
Eligible instance: same as C1, subject not in the pit. Behaviour: distance to pit at `F(t-1)`.
Cut-point: cohort **p75** (far). Intermediate: assists credited on the contest
(`assistingParticipantIds`). Validation: win.

**C3 - Baron discipline.** *Agency unit: selection.*
Eligible instance: `ELITE_MONSTER_KILL` with `monsterType = BARON`. Behaviour: subject attending while
their team's `F(14)`-state is behind (contestable-or-worse). Cut-point: cohort p75 of attending-while-behind.
Intermediate: deaths on the baron pit in the following 60s (join A1's death list). Validation: win.

## Placebos (registered negatives)

**P1 - Nearest-ally distance at `F(3)`.** Too early to be a decision anyone can make on purpose.
**P2 - Number of `ITEM_PURCHASED` events in the first 60 seconds.** Determined by opening build and
spawn timing, not by in-game judgement.

If either placebo earns a non-trivial price, the pipeline is producing artifacts and the whole family's
results are reported as untrustworthy rather than interesting (ADR-0006).

## Holdout and reporting

- Subject: fit cut-points and rates on the oldest 70% of games; test on the newest 30%. Never shuffled.
- Cohort: split by match date for the same reason, then report price CIs from a match-clustered bootstrap.
- Every displayed claim carries: cohort n, subject eligible-instance count, price with CI, subject rate
  with CI, winning-cohort rate, and the placebo status. A claim missing a denominator is not displayed.
