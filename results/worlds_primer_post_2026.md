# The Worlds 2026 Statistical Primer

*Every roster, ranked by what the box scores actually say. Built from 4,670 tier-1 games across 2025 and 2026 (LCK, LPL, LEC, LCS, LCP, plus internationals), 46,700 player-games, Oracle's Elixir data.*

**How to read this.** Every player is scored on his games with the team he's actually going to Worlds with. The core stat is **outperform**: a player's composite performance score minus the average of his four teammates' scores, in standard-deviation units. It exists to cancel out team quality, because "high stats on a good team" tells you nothing about the player. Around it sit four more numbers: **carry%** (how often he was his team's best performer), **swing** (win rate in his good games minus win rate in his bad ones, which catches boom-bust players), **AUC** (how well his own statline predicts the result), and a **watch score** that blends all of it into one percentile ranking.

**Two findings frame everything below.** First, in pro play the share-of-team-output model ranks support as the most win-predictive role (0.735 AUC), ahead of jungle at 0.674. That's a flip from solo queue, where jungle led. Second, raw statline models hit absurd 0.97 AUCs in pro play because leads actually close, so everything here leans on the team-adjusted view instead. One more standing disclaimer: box scores can't see shot-calling, drafting, or resource routing. When a famous name sits near the bottom of a ranking, that's a statement about the stats, not the player.

---

## LCK

### Gen.G (1st seed, Swiss), .727

The deepest roster in the field, and the interesting part is how that shows up: almost nobody on Gen.G outperforms, because everyone is the standard. Chovy is the exception at +0.245 outperform with a 37.6% carry rate over 282 games, the largest individual-to-team gap anywhere on an elite roster. Canyon reads the opposite way, -0.05 outperform but a 0.78 AUC, meaning Gen.G's results track his form even though his averages look ordinary. He also dips in playoffs (-0.18). Ruler is the quiet playoff riser, +0.26 in postseason play, the biggest rise of any bot laner in the sample. Duro posts the largest negative differential on any winning team (-0.26). Read that with the support caveat in mind, but it's a real number opponents will test.

**Watch for:** whether Chovy's gap survives international mids, Canyon's early game, Ruler in long series.

### Hanwha Life Esports (2nd seed and MSI champion, Swiss), .634

Kanavi is the engine: +0.14 outperform, 0.74 AUC, carried in 27% of games. It's the classic jungle-first profile and it matches what the role-level model says wins pro games. Gumayusi has elite raw level (+0.10) but the least win-predictive statline of any tier-one bot laner (AUC 0.57). His numbers look great exactly when HLE is already winning, which is a different thing from driving wins. Delight is the steadiest player in the whole 95-man pool, with the lowest game-to-game variance. Two flags: Zeus drops 0.20 from regular season to playoffs, and Zeka is solid rather than spectacular (-0.05) for a team with title expectations.

**Watch for:** Kanavi's pathing against international junglers, whether Zeus's playoff dip was a blip, Delight holding the map together when things get ugly.

### T1 (3rd seed, Swiss), .627

The roster the stats describe most strangely. Peyz is T1's statistical leader: +0.09 outperform, 31% carry rate, and the highest-variance bot laner in the pool. T1's games swing hardest on his good days. Oner owns the single highest AUC in the pool at 0.81. When Oner's statline is good, T1 wins, full stop, and he rises in playoffs (+0.14). Keria (+0.09) is still the benchmark at support. Faker sits near the bottom of every box-score axis (-0.16 outperform, team-best performer in just 10% of games). That is the cleanest example in esports of what these stats cannot see. T1's early game deliberately routes resources away from mid. Doran (-0.02) is structurally fine.

**Watch for:** Oner's first 10 minutes as T1's win condition, Peyz's variance cutting both ways, and every Faker highlight being statistically invisible.

### Dplus KIA (4th seed, Swiss), .556

The most average profile in the field. Nobody on DK is a statistical outlier in either direction. Smash leads at +0.09, Lucid has the team's best AUC (0.73) plus a heavy Jarvan signature (22% of his games), ShowMaker sits at -0.09, and Career runs a 24-champion pool. DK has the lowest win rate of the LCK quartet and the stat profile offers no obvious lever: no engine, no boom-bust, no weak link. That's either balance or a team that needs a plan to beat better rosters.

**Watch for:** Lucid and ShowMaker's mid-jungle sync, and whether DK can manufacture an edge rather than waiting for one.

---

## LPL

### Anyone's Legend (1st seed, Swiss), .615

A fortress built around two players. Kael is the pool's second-best support by outperform (+0.12). Tarzan posts a 0.73 AUC on the field's most durable sample, 325 games. The jungle-support axis is exactly where the role-level model says pro games are decided. hope (+0.08, 0.72 AUC) makes it three outperformers in the top four. The stress points: Shanks at -0.16 is the roster's statistical floor, and Breathe's current stint (-0.08 over 52 games) sits well below his career mark (+0.10). A player trending the wrong way into the hardest tournament of the year.

**Watch for:** the Tarzan-Kael two-man game, whether Shanks holds up against Chovy-class mids, Breathe's form.

### Bilibili Gaming (2nd seed, Swiss), .661

The support-is-king thesis, embodied. ON has 357 games, the largest sample in the field, and outperforms his teammates +0.20 with a 31% carry rate. By these numbers BLG's most important player is their support. Viper pairs elite raw level (+0.29, top three in the pool) with a +0.19 outperform. That's genuinely elite, not just fed. The wrinkle is Knight: the headline mid is dead even with his teammates (-0.02). When everyone is a star, nobody stands out. Xun is the X-factor at -0.16 outperform but +0.21 swing. BLG wins the games he pops off in. bin (-0.10) rounds out a roster whose numbers all point the same way: enormous team quality, driven by ON and Viper.

**Watch for:** ON's engage timing as the real engine, Xun's highs in Swiss tiebreakers, Viper against the world's bot lanes.

### Top Esports (3rd seed, Swiss), .604

Tian or bust. He is the roster's only real outperformer (+0.16) with a 0.79 AUC, the clear statistical center of gravity. Creme (323 games) is the metronome: +0.01 outperform but a 0.72 AUC, steady and quietly predictive. JackeyLove runs slightly below teammates (-0.04) with a strong 0.70 AUC. Two flags. Zhuo drops -0.35 from regular season to playoffs, the worst playoff decline of any support in the pool. And top lane is in flux, with 369 stepping in for ZUIAN on visa grounds. 369 has 226 TES games, so this is a role shuffle more than a call-up.

**Watch for:** Tian's early game setting TES's ceiling, the reshuffled top lane, JackeyLove in five-game series.

### Shanghai Invictus Gaming (4th seed, Swiss), .560

The youngest statistical profile in the LPL group, with a legend attached. Wei is the engine (+0.07, 0.79 AUC, +0.19 swing). JiaQi is the pool's #4 watch score, +0.13 outperform on only 71 games, the highest-upside bot laner here. Rookie (+0.05 across 230 games) is still net positive against LPL mid competition. TheShy (-0.09) is the second straight legendary name whose box score undersells him. meiko (-0.02, 0.72 AUC) remains quietly predictive.

**Watch for:** Wei's form as IG's thermometer, JiaQi's Worlds debut as a potential breakout, and how much of TheShy's laning dominance still shows up where the stats can see it.

---

## LEC

### G2 Esports (1st seed, Swiss), .631

SkewMond is the statistical story: +0.20 outperform, 33% carry rate over 244 games, the biggest engine gap in the LEC and a top-seven watch score overall. Then the inversion everyone will talk about. Caps, the greatest Western mid ever, reads below his teammates (-0.05). G2 is good enough that Caps's baseline is the team baseline. Labrov is the pool's biggest playoff riser (+0.15) despite a rough regular-season number (-0.14), a pattern of showing up when it counts. hans sama dips in playoffs (-0.18). BrokenBlade (+0.04, 28 champions) is the flexible floor.

**Watch for:** SkewMond as G2's win condition, Labrov's postseason gear, and whether Caps's statistical invisibility turns into highlight visibility.

### Movistar KOI (2nd seed, Swiss), .532

Elyoya is the engine (+0.07 outperform, 0.78 AUC). MKOI's results flow through his jungle. The stat anomaly of the tournament is jojopyun: he outperforms his teammates (+0.12, 30% carry rate) but his good games barely correlate with wins (swing -0.21, AUC 0.56), and he drops -0.20 in playoffs. MKOI wins even when their mid's statline is quiet, which raises a real question about what his numbers are measuring. Supa is the boom-bust bot at +0.23 swing. MKOI's highs come through him. Alvaro rises in playoffs (+0.19). Myrwn (-0.08) is the quietest link.

**Watch for:** which way the jojopyun paradox resolves, Supa's ceiling games, and Elyoya's drafting, which this data can't see but defines MKOI's identity.

### Karmine Corp (3rd seed, Play-In), .614

A top-driven team with a soft underbelly. Canna (+0.12, 0.76 AUC, Rumble in 22% of his games, one of the pool's heaviest signatures) and Caliste (+0.13, 30% carry) both outperform. The bottom half does not: Kyeahoo (-0.21) and Busio (-0.21) both trail their teammates. KC's mid-support axis is the statistical weak side. Yike is volatile, with the highest game-to-game variance among LEC junglers, and drops -0.23 in playoffs. That's the profile of a player you gameplan against.

**Watch for:** whether the Canna-Caliste spine can carry a Play-In run, Yike's playoff form, and opponents attacking the Kyeahoo-Busio lane.

---

## LCS

### Team Liquid (1st seed, Swiss), .644

Josedeodo is the top-rated jungler in the pool by watch score: +0.14 outperform, +0.21 swing, 0.78 AUC. TL's games go where his form goes. Quid (+0.03, Ryze in 18% of games) is steady. Morgan (+0.04, Rumble signature) quietly outperforms. The questions sit bot side. Yeon (-0.09) trails his teammates, though his statline still predicts TL's results (0.73 AUC). CoreJJ (-0.07) reads as a system player. TL has the best team win rate in the LCS quartet and the stats say it's built jungle-first.

**Watch for:** Josedeodo against the Kanavi/Tarzan/Shad0w tier, and whether Yeon's numbers convert on the big stage.

### LYON (2nd seed, Swiss), .623

Saint is this team's engine, +0.21 outperform and a 34% carry rate, the largest mid-lane gap in the LCS, and almost nobody is talking about him. Berserker improved with the move: +0.08 on LYON against +0.00 for his career. He has the LCS's best bot-lane AUC (0.76) but flags a -0.22 playoff decline. Inspired is the exact inverse shape: underperforming on the season (-0.07) yet rising in playoffs (+0.10). Dhokla (-0.17) and Isles (-0.04) round out a roster whose story is one mid carrying and one bot needing October form.

**Watch for:** Saint as the tournament's least-hyped outperformer, Berserker reversing his playoff trend, Inspired's postseason gear.

### Cloud9 (3rd seed, Play-In), .581

Blaber (+0.09, 0.73 AUC) remains the engine. Thanatos (+0.10, 29% carry) owns the LCS's biggest playoff riser number (+0.30), though his statline barely tracks wins (AUC 0.53). Huge games, in either direction. Tactical is the most committed one-trick among full-sample players, Ezreal in 28% of his games out of a 12-champion pool. C9's drafts will telegraph. Vulcan (-0.18) shows the biggest negative gap on the team. Small-sample note: Loki (34 games) and Tactical (32) have the thinnest data of any LCS starters.

**Watch for:** Tactical's Ezreal priority in drafts, Thanatos's October form, Blaber's early game against LCK and LPL pathing.

---

## LCP

### Team Secret Whales (1st seed, Swiss), .639

Home of the #1 watch score in the tournament: Bie. The support outperforms his teammates +0.16 over 102 games with a 0.78 AUC and the highest raw level of any support in the pool. He is the single-player version of the role-level finding that support impact wins pro games. Eddie (0.79 AUC) means TSW's results also flow through bot lane. Hizto is the most boom-bust jungler in the field (+0.28 swing). TSW's highs arrive through his games. The vulnerability is Dire, -0.26 outperform with a playoff decline, the one lane opponents can target. Pun (-0.03) rises in playoffs (+0.14).

**Watch for:** Bie as the quietest elite player in the tournament, Hizto's highs and lows, and whether Dire's lane gets international attention.

### CTBC Flying Oyster (2nd seed, Swiss), .583

Two engines, two flagged lanes. Shad0w (+0.19 this stint, 0.73 AUC) and Doggo (+0.17, 40% carry rate over 206 games) both genuinely outperform. CFO's spine is real. But Rest (-0.21 outperform, K'Sante in 24% of games, heavy signature, modest results) and KIno (-0.11) are the floor, and Pout is a committed one-trick, Akali in 23% of his games. CFO's team win rate sits mid-pack in the LCP. Their ceiling depends on two engines dragging four others upward.

**Watch for:** Shad0w's mid-year surge in form, Doggo against the world's bot lanes, and how hard teams punish Rest's predictability.

### MVK Esports (3rd seed, Play-In), .468

The most systemic roster in the field. Nobody outperforms, but three players have elite result-sensitivity. SiuLoong has the second-highest AUC in the pool (0.80) plus the lowest variance of anyone in the tournament, on a Seraphine signature no less. Kratos (0.80 AUC) and Harky (0.77) match him. Gury is the engine (+0.17) with the pool's biggest playoff riser number (+0.33). The anomaly is Chika, whose statline is the most win-decoupled in the entire pool (AUC 0.37). MVK wins and loses independently of their midlaner's numbers. Either the system routes around him or the stats can't see what he does. Play-In underdogs with a defined identity.

**Watch for:** the SiuLoong-Gury system against LEC and LCS opponents, whether Chika is a roleplayer or a hidden gem, Gury's trajectory.

---

## CBLOL

### LOS (1st seed, Swiss), .632 in international games only

CBLOL domestic games are not in the tier-1 dataset, so all five LOS players appear here on 19 international games. Treat everything as provisional. What those 19 games show: ZEST is the most extreme outlier in the entire tournament pool, +0.48 outperform with his team's best performer in 58% of games and elite raw level. That's a watch-him-immediately flag. Feisty owns the largest playoff riser number in the pool (+0.65). Ackerman (swing -0.49) and Curse (-0.25) read as the pressure points.

**Watch for:** ZEST against tier-1 top laners. Biggest unknown quantity at Worlds.

### FURIA (2nd seed, Play-In), .227 in international games only

Same 22-game caveat, and a harder one: those games mostly came in losses, which makes these outperform numbers the least reliable flavor, good stats on a struggling team. Guigo (+0.27 outperform, 41% carry rate in defeats) was the bright spot. Tatu is the pool's most signature-defined jungler (Xin Zhao in 31% of games) and its most boom-bust (+0.41 swing). When his game is on, FURIA lives. Ayu (-0.30) and Tutsz (highest-variance mid) are the volatility.

**Watch for:** Guigo holding up against tier-1 tops, Tatu's Xin Zhao priority in drafts, and whether the Play-In bracket gives them a winnable path.

---

## Storylines the stats are begging you to write

1. **Supports are the story of pro LoL in 2026.** The role-level model puts support first, and it shows up top-down: Bie is the #1 player in the pool, ON is BLG's engine, Kael and Keria anchor their rosters.
2. **Jungle is still the raw-skill throne.** Eight of the top ten watch scores. Josedeodo, Tian, SkewMond, Shad0w. This tournament may be decided by that tier.
3. **The outperformers nobody hypes:** Saint (LYON), Doggo (CFO), JiaQi (IG), Caliste (KC). Each outperforms established teammates by a margin that should make them household names by November.
4. **The stats-invisible legends:** Faker, TheShy, Caps, Rookie, ShowMaker, all at or below their teammates' level by box score. That's a statement about measurement, not greatness. Watch them anyway; the stats can't see why you should.
5. **Boom-bust X-factors who decide ties:** Xun, Hizto, Supa, Peyz, Tatu. When their games come online, their teams win at a completely different rate.
6. **Playoff-form wildcards:** Ruler, Thanatos, Gury, Feisty, and Alvaro rise. Zeus, Berserker, Yike, jojopyun, Zhuo, and hans sama drop. October form is not September form, in either direction.

## Glossary

- **outperform**: own composite performance score (z-scored within role and league) minus teammates' mean, averaged per game. Team strength cancels out.
- **carry%**: share of games where the player posted the team's best score.
- **swing**: win rate in the player's above-median games minus below-median games.
- **AUC**: within a player's games, how well his own score separates wins from losses.
- **watch score**: mean percentile across outperform, carry%, swing, AUC, and raw level, within the 95-player pool.
- **Caveats**: box scores only. Players repeat across games. CBLOL rows are international-only. The composite is descriptive, not causal.
