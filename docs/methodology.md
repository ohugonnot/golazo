# Methodology

This note documents the parts the README compresses: how goal intensities are
recovered from market prices, how the rarity bonus is computed, and how the x2
multiplier placement is decided.

## 1. From 1X2 + over/under to goal intensities

The market gives us three de-vigged outcome probabilities `(p_H, p_D, p_A)` and,
when available, an over/under line. We want the Poisson rates `(λ_home, λ_away)`
whose Dixon-Coles scoreline matrix reproduces them.

There is no closed form, so we solve numerically in two coupled steps:

1. **Total intensity.** The over/under line constrains `λ_home + λ_away`: the total
   goals `G ~ Poisson(λ_total)`, and the line's implied `P(under)` pins `λ_total`
   via `P(G ≤ line)`.
2. **The split.** With the total fixed, one free parameter (the home/away split)
   remains. We search it so the matrix's marginal `(p_H, p_D, p_A)` best matches the
   de-vigged market, then **calibrate** the matrix exactly onto the market 1X2 so
   the model never contradicts the odds it was built from.

When no odds are supplied the engine falls back to the pool's own quotations,
de-vigged with the power method (a tuned exponent reduces the favourite-longshot
bias of the raw quotations). This is explicitly a degraded mode — real bookmaker
odds are the source of edge.

## 2. The rarity bonus is conditional

The bonus for an exact score depends on **how many of the players who got the
result right also played that exact score** — not on the whole community. So the
relevant quantity is `P(score | correct result)`, not `P(score)`.

`conditional_popularity()` computes this by partitioning the community popularity
prior by 1X2 outcome and renormalising within the outcome. The five-tier schedule
then maps that conditional share to a bonus:

| Share among correct-result players | Bonus |
| --- | --- |
| > 30%      | +20 |
| 20–30%     | +30 |
| 5–20%      | +50 |
| 0.5–5%     | +70 |
| < 0.5%     | +100 |

A practical consequence the engine exploits: on a heavy favourite, the *modal*
score (say 2-0) is played by a large fraction of the correct-result players, so it
sits in the +20 tier. Nudging one tier up (3-0) can drop into +50 — lower
probability, but the expected bonus `bonus(s) · P(exact = s)` can be higher.

When real community data for a score is available it overrides the prior and the
full +100 tier is reachable; without it, synthetic-prior scores are capped at +70
(we do not claim "ultra-rare" from a guess).

## 3. Placing the one-shot x2

Doubling a forecast is a single irreversible decision over a multi-round
tournament. We initially modelled it as optimal stopping with a Cayley–Moser
prophet threshold; on a short, noisy sequence of matchdays the thresholds
degenerated (the candidate match is the top of its own observed distribution, so
the rule never waits). We replaced it with two explicit gates:

- **Timing (a patience schedule).** Place only if the best match *stands out* from
  its slate — `best_gain / median_gain ≥ r`, where the required ratio `r` falls as
  the tournament progresses (1.8 → 1.4 → 1.15). The final window forces a placement
  (an unused x2 scores nothing).
- **Quality (triple convergence).** Independently require high confidence in the
  result (`P(result) > 0.65`), low ownership of the exact score (`< 20%`), and fresh
  bookmaker odds. All three must hold.

The verdict is `place_now = last_window OR (timing AND quality)`. The two gates are
reported separately so the decision is auditable rather than a black box.

## References

- Dixon, M. J. & Coles, S. G. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market.* JRSS-C.
- Shin, H. S. (1993). *Measuring the Incidence of Insider Trading in a Market for
  State-Contingent Claims.* The Economic Journal.
- Clair, B. & Letscher, D. (2007). *Optimal Strategies for Sports Betting Pools.*
  Operations Research.
- Gaba, A. & Tsetlin, I. (2004). *Modifying Variability and Correlations in
  Winner-Take-All Contests.* Operations Research.
