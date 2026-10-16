# Behavioural cut-points are cohort percentiles, fixed on the training half

"Isolated death" needs a distance, "late approach" needs a threshold, and either can be tuned until a
result appears. So every cut-point is defined as a **cohort percentile** (default: the cohort's 75th
for a behaviour, 25th for an absence) computed on the training half of the data before any outcome is
looked at, and recorded in `REGISTERED_QUESTIONS.md` with the number it landed on.

This also solves two secondary problems: map units are patch- and position-dependent (a support's
"far" is not a jungler's "far"), and percentiles are automatically re-derived per role cohort rather
than hard-coded.

## Consequences

A claim's threshold is part of its identity. Re-running the pipeline after adding corpus data can move
a cut-point, which must be reported as a change of question, not a refinement of one. Placebo
questions (registered negatives) run alongside the real ones; if a placebo earns a price, the family's
results are treated as untrustworthy rather than interesting.
