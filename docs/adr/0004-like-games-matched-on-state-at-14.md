# Like games are matched on rank band, role, side, and state at the 14-minute frame

A cohort claim is only as credible as its counterfactual. Conditioning on rank band, role and side
(blue/red) alone leaves the claim saturated with outcome-echo: a cohort game already won by 5k gold
looks nothing like one already lost by 5k, yet both enter the same average. Conditioning on comp and
lane matchup alone ignores the same fact, and cells get large but wrong.

We match on the observable game state **before** the decision under study: the gold and XP
differentials at the 14-minute frame (`participantFrames`), plus band, role and side. 14 minutes is
late enough that lane matchups and jungle pathing have expressed themselves, early enough that the
objective fights the report is about have mostly not happened yet.

## Considered options

- kNN over an early-game feature vector: highest nominal fidelity, but "we compared you to numerically
  close games" is unauditable, and a similarity metric quietly makes the causal claim for us.
- Band + role only: cheap and big cells, and it prices behaviours that merely accompany winning.
- A full causal model of the match: this is the shape we would want eventually, and it is not a
  first step - the matching above is the version we can check by hand.

## Consequences

State-at-14 must be derivable for every cohort match, so the timeline refetch (ADR-0001) is a
precondition for any priced claim. Games that ended before 14 minutes cannot be used for pricing.
