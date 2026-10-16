# Registered questions, time-split holdout, and discovery on intermediates

"Let's see what shows up" over a year of events per player is a false-discovery machine: a hundred
slices of 410 games guarantees a handful of 2-3 SE results. Three constraints, accepted together:

1. **Registered questions.** The behaviours to test are written down, with their measurement rule,
   before the timeline data is queried, and the family size is known so BH-FDR can be applied honestly.
   The first list is small: death geography, lane tempo, objective approach.
2. **Time-split holdout.** A player's games are split by date, not shuffled - form travels in streaks,
   so shuffling leaks. Patterns are found on older games and scored on newer ones.
3. **Discover on intermediates, validate on wins.** Candidate relationships are screened against
   less-endogenous intermediate outcomes (CS differential in still-contestable games, deaths per 10
   minutes, objective attendance rate), and only survivors are priced against the match result in the
   cohort. The win-rate step is the *test*, never the *search*.

## Consequences

Adding a new behaviour to the report reopens registration and adds a hypothesis to the multiplicity
correction. This is deliberate friction: the alternative is a report that quietly grows claims.
