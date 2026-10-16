# Like games are matched on rank band, role, side, and state at the 14-minute frame

A cohort claim is only as credible as its counterfactual. Conditioning on rank band, role and side
(blue/red) alone leaves the claim saturated with outcome-echo: a cohort game already won by 5k gold
looks nothing like one already lost by 5k, yet both enter the same average. Conditioning on comp and
lane matchup alone ignores the same fact, and cells get large but wrong.

We match on the observable game state **before** the decision under study: the gold and XP
differentials at the 14-minute frame (`participantFrames`), plus rank band, role, side, **and patch
era**. 14 minutes is late enough that lane matchups and jungle pathing have expressed themselves,
early enough that the objective fights the report is about have mostly not happened yet.

Patch era is part of the definition because the corpus spans 2024-10 to 2026-10 - more than 15
patches, across which structure rules, plate counts, dragon timings and map geometry moved. Pricing
across all of them averages over different games, so a Price is estimated within one era (the
subject's current split window) and the era is named wherever the Price is shown. If a cell gets too
small inside one era, the honest answer is "we cannot price this yet", not "widen the era".

## Considered options

- kNN over an early-game feature vector: highest nominal fidelity, but "we compared you to numerically
  close games" is unauditable, and a similarity metric quietly makes the causal claim for us.
- Band + role only: cheap and big cells, and it prices behaviours that merely accompany winning.
- Pool every patch and report the per-patch spread as a diagnostic: bigger cells, but the headline
  number is a weighted average over rule sets that no longer exist.
- A full causal model of the match: this is the shape we would want eventually, and it is not a
  first step - the matching above is the version we can check by hand.

## Consequences

State-at-14 must be derivable for every cohort match, so the timeline refetch (ADR-0001) is a
precondition for any priced claim. Games that ended before 14 minutes cannot be used for pricing.
Every cohort query also reports its cell size, because era-restriction shrinks them.
