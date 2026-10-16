# Timeline data is fetched per match and kept as its own raw layer

The coaching track needs what the collected per-player rows throw away: per-minute positions, health,
gold, XP and per-minute combat stats for all ten players, plus second-level events. That lives on a
**separate endpoint** (`/lol/match/v5/matches/{id}/timeline`) - one extra request and ~120KB per game,
with a different field vocabulary from the detail endpoint (`type` not `eventType`; participant frames
keyed by participantId string).

Decision: re-fetch the existing corpus (10,029 matches; ids already on disk, so no rediscovery) and
store raw timeline JSON as its own gzipped layer, never merged into the per-player rows. Derived
interval and event tables are built from it and re-built freely. Verified that a 727-day-old match
still serves a complete timeline, so the whole corpus is reachable; the cost is ~3.3 hours of
rate-limited calls and roughly 1.2GB raw, a one-time spend that makes every later feature derivation
free.

## Join path (verified, not assumed)

The timeline's own participant list carries only `participantId` and `puuid` - no role, no champion, no
team. So frames join to identities by **puuid**: the timeline maps `participantId -> puuid`, and the
already-stored player rows map `puuid -> role/champion/team`. Per-minute frames are then indexed by the
participantId key. `participantId` is additionally stored in the slim record for new collections
because it is cheaper than re-deriving it, but it is not load-bearing: existing corpus matches need no
detail re-fetch.

## Consequences

- **This layer is machine-local and is not synced through git** (like the rest of `data/`). The
  refetch therefore has to run on the machine that will do the coaching-track work, with a dev key
  generated there - keys expire 24h after creation, so a key obtained elsewhere will not survive the
  transfer anyway. Coverage state lives in a manifest so the job is resumable and idempotent.
- Two raw layers with different shapes are a permanent hazard, and any field-name mismatch between them
  is a data bug, not a code bug.
- Nothing derived is trusted over raw: any new coaching metric re-runs from the raw timeline layer, not
  from a previous derivation.
- Re-fetching is idempotent, so partial coverage is a normal state, not an error. Claims must state the
  coverage they were priced on.
