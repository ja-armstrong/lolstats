# Timeline data is fetched per match and kept as its own raw layer

The coaching track needs what `matches.jsonl` throws away: per-minute positions, health, gold, XP and
per-minute combat stats for all ten players, plus second-level events. That lives on a **separate
endpoint** (`/lol/match/v5/matches/{id}/timeline`) - one extra request and ~120KB per game, with a
different field vocabulary from the detail endpoint (`type` not `eventType`; participant frames keyed by
participantId string).

Decision: re-fetch the existing corpus (10,029 matches; ids already on disk, so no rediscovery) and
store raw timeline JSON as its own gzipped layer under `data/timelines/`, never merged into
`matches.jsonl`. Derived interval and event tables are built from it and re-built freely. Verified that
a 727-day-old match still serves a complete timeline, so the whole corpus is reachable; the cost is
~3.3 hours of rate-limited calls and roughly 1.2GB raw, which is a one-time spend that makes every
later feature derivation free.

## Consequences

- Two raw layers with different shapes are a permanent hazard. `participantId` is now stored in the slim
  record because it is the only join key between them.
- Nothing derived is trusted over raw: any new coaching metric re-runs from `data/timelines/`, not from a
  previous derivation.
- Re-fetching is idempotent and resumable, so partial coverage is a normal state, not an error.
