# LoL performance analysis

One corpus of ranked solo queue games serving two questions: which role's performance predicts
winning (research track), and what one player should change (coaching track). The research track reads
a game as one end-of-game row per player; the coaching track reads it as timed state, because almost
nothing coachable survives to the final scoreline.

## Language

### The game, as data

**Match**:
One 5v5 game, identified by Riot's match id. The unit every other fact hangs from.
_Avoid_: Game (reserved for a player's appearance in one), Replay

**Player-Game**:
One player's participation in one match - the row type the research track analyses.
_Avoid_: Row, Record

**Minute Frame**:
A 60-second sample of live state carrying every player's position, health, gold, XP, level and combat
stats. The finest thing anyone here can observe, and therefore the ceiling on every timing claim:
"arrived late" can never mean fewer than about a minute late.
_Avoid_: Tick, Snapshot, Timestamp (events have second-level timestamps; frames do not)

**Event**:
A discrete happening reported at second-level precision: a kill, an elite-monster kill, a structure or
plate destroy, a ward placed or killed, an item purchased, a level up.
_Avoid_: Action, Occurrence

**Geolocated Event**:
An Event carrying a map `position`. Verified present for champion kills, special kills, elite-monster
kills, building kills and turret-plate destroys. Ward placed, ward killed and item purchased are **not**
geolocated on the event itself.
_Avoid_: Positional event

**Zone**:
A named map region derived from the game's own structure and spawn coordinates rather than hard-coded
map constants, so it survives patch geometry changes.
_Avoid_: Area, Location, Side-of-map (too coarse to price anything)

**Lane Assignment**:
The role Riot recorded for a player in that match, which is not the same as the place they were
standing. A claim about being "in lane" is about position, not assignment.

**Alive**:
Measured, not inferred: a player's health in the frame nearest the moment in question. Respawn state is
never available, so "could an ally have reached you" is bounded by the frame resolution.
_Avoid_: Dead-within-40-seconds style proxies

### Comparing

**Cohort**:
The set of player-games a comparison is made against, named explicitly by role, rank band and any
state conditioning. A number without its cohort is not a result.
_Avoid_: Peers, baseline, "average player"

**Like Game**:
A cohort match whose state *before* the decision under study was comparable: same rank band, role,
side, and gold/XP differential at the 14-minute frame.
_Avoid_: Similar match (undefined similarity), same-comp-only matching

**Contestable State**:
The game was still in the balance at the moment in question. Behaviour claims are made only from
contestable states, because a habit that only appears in decided games is an echo of the score.
_Avoid_: Close game (that's duration, not state)

**Eligible Instance**:
One occurrence of a situation where the behaviour was actually choosable - a contest that spawned while
the subject was alive and reachable, not every contest in every game. Every rate is per eligible
instance or it is not a rate.
_Avoid_: Opportunity, denominator-free percentage

### Claims

**Price**:
What a behaviour is worth in win rate, estimated on the cohort in like games, with a confidence
interval. Only the cohort can produce a price. A player's own games cannot, at any sample size they
will realistically reach.
_Avoid_: Impact, Value, Lift

**Gap**:
The distance between the subject's behaviour rate and the winning share of their cohort on the same
eligible instances. This is the coaching signal; the Price says what closing it is worth.
_Avoid_: Weakness, Area to improve

**Association**:
A relationship between a behaviour and how games ended, with no claim that doing it caused the win.
Everything here is an association unless a specific design earns more.
_Avoid_: Insight, Pattern, Cause

**Endogenous Feature**:
A measurement that partly *is* the outcome - gold after a shutdown, damage dealt in a won fight. The
research track's share-and-residual apparatus exists to handle these; the coaching track handles them
by conditioning on Contestable State and by preferring intermediates.

**Actionable**:
A behaviour that is chosen by the subject rather than by the matchup, observable at minute-frame
resolution, defined on eligible instances, and testable on games it was not derived from. Failing any
of those, it is a description.
_Avoid_: Recommendation, Tip, Fix

**Agency Unit**:
What a claim is about: *presence* (were you there), *timing* (when you got there), or *selection*
(what you chose to fight). These are different measurements and cannot share one rule.
_Avoid_: Behaviour (too coarse to test)

**Registered Question**:
A hypothesis and its measurement rule, written down before the data is queried, so the number of tests
is known and false discoveries can be controlled.
_Avoid_: Exploration, "let's see what shows up"

**Placebo Question**:
A registered question that cannot plausibly be a decision - included to detect whether the pipeline is
manufacturing findings. If a placebo earns a Price, the family's other results are untrustworthy.
_Avoid_: Null test (too vague to pre-register)

**Discovery Set / Holdout Set**:
Games split by *time*, never shuffled: behaviour travels in streaks, so shuffling leaks the answer.

**Tempo**:
Pressure and recall rhythm measured through per-minute CS/XP differentials, structure and plate events,
shop-purchase timing and lane presence. Explicitly not wave state, which the data does not contain.
_Avoid_: Wave management, freezing, slow-pushing (unmeasurable here)

**Scouting Report**:
The coaching track's artifact: a short list of registered questions answered about one player, each with
its Price, its Gap, denominators on both, and the placebo status. Few tested claims beat many
plausible ones.
_Avoid_: Dashboard, Insights feed
