# Cohort prices an effect; a player's own games only measure behaviour

At 410 usable games and a 50% baseline win rate, the smallest win-rate difference the subject's own
games can detect at 80% power is 13.8pp at 50% behaviour prevalence, 17.3pp at 20% and 23pp at 10%;
BH-FDR over a family of ~10 registered questions raises that floor to roughly 18-21pp. Real coaching
effects are 2-8pp. No amount of modelling sophistication changes this: the labels are the sample.

The same 410 games estimate the subject's *own behaviour rate* with an SE near 2.5pp, and the refetched
cohort (~100k player-games, ADR-0001) prices effects down to ~2pp. So every claim is assembled from two
different datasets, each used where it has power:

> **Price** (cohort, like games): doing X when contestable is worth +4.2pp win rate [2.1, 6.3].
> **Gap** (subject): you did X in 31% of eligible instances; winning peers did in 58%.

A claim whose price comes from the subject's own wins is rejected by construction, not warned about.

## Consequences

The app needs a cohort at all, which makes the refetch a precondition rather than an optimisation, and
means a player with 40 games still gets gaps measured (wide ones) but no personal win-rate verdicts
ever. Every displayed number carries its cohort name and its denominator.
