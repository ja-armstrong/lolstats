# Three claim families are out of scope: wave state, ward geography, spawn lateness

Explicit no-s, because each is a thing a future reader will assume we simply hadn't got around to.
All three were checked against a real timeline payload, not assumed.

**Wave state.** The feed carries no unit or minion events (`CHAMPION_KILL`, `WARD_PLACED`,
`TURRET_PLATE_DESTROYED`, `ELITE_MONSTER_KILL`, `BUILDING_KILL`, `ITEM_PURCHASED`, `LEVEL_UP`,
`GAME_END`, `PAUSE_END` - no `MINION_KILL`, no minion positions). Frozen/pushing/crashing waves do
not exist in the data. We measure *tempo* (per-minute CS/XP differential, shop-event recall timing,
plate and structure destruction by lane, time-in-lane share) and never describe a wave.

**Ward placement point.** `WARD_PLACED` carries `creatorId`, `wardType`, `timestamp` and no position, so
the ward's own coordinates do not exist. What *is* available: the creator's position in the same minute
frame (verified present in 6/6 sampled ward events), which gives where the player was standing when they
placed it. So ward claims are limited to "you were in the enemy jungle when you warded" and cannot assert
the ward's exact sightline. `WARD_KILL` has the same shape.

**Spawn lateness.** No spawn events exist, so "you arrived 40 seconds late to dragon" would require
modelling spawn schedules, which drift across patches - a player's stored games already span 15
patches. Objective contests are anchored kill-wise: attendance and distance-to-pit in the minute
before `ELITE_MONSTER_KILL`.

## Consequences

Any of these can be revisited only if Riot adds the fields, or if we accept a named inference layer
with its own error bars. Until then a report containing "your wave was frozen" is a bug.
