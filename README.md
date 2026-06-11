# golazo

**A quantitative score-prediction engine for football forecasting pools.**

Turn bookmaker odds into calibrated exact-score probabilities, then play the
points game optimally — including when to maximise expected value versus your
actual probability of finishing first.

[![Python 3](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#running-it)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

---

## What it does

`golazo` predicts the most *valuable* exact scoreline to forecast for a football
match in a points-based forecasting pool. In these pools you score points indexed
on the bookmaker quotation when you call the right 1X2 outcome, earn a rarity
bonus when you nail the exact score, and hold a single one-shot "double points"
multiplier to spend across the tournament.

The engine starts from raw bookmaker odds, removes the margin to recover true
probabilities, fits a Dixon-Coles scoreline model (Poisson goals with a low-score
correlation correction), and ranks every candidate score by its **expected points**.
A separate Monte-Carlo layer simulates thousands of tournaments against a model of
the field to optimise **probability of finishing first** rather than expected value
alone — because the two are not the same objective.

## Why it's interesting

- **De-vigging done properly.** Bookmaker odds embed an overround. `golazo` strips
  it with both Shin's (1993) insider-trading model and the power method, rather than
  naive normalisation — recovering probabilities that respect the
  favourite-longshot bias.
- **Goal intensities from market data.** It solves for the Poisson rates
  λ_home / λ_away implied jointly by the 1X2 prices and the over/under line, so the
  scoreline matrix is anchored to the market, not to historical averages.
- **Dixon-Coles correction.** A τ adjustment inflates the probability mass on
  0-0, 1-0, 0-1 and 1-1, fixing the well-known independent-Poisson underestimate of
  low-scoring draws and one-goal games.
- **Calibration.** The exact-score matrix is rescaled so its marginal 1X2
  probabilities match the de-vigged market — the model never disagrees with the
  odds on the outcome it was built from.
- **EV ≠ P(win).** Maximising expected points and maximising the chance of *winning
  a contest* are different problems. `golazo` makes the distinction explicit and
  ships an optimiser for each.
- **Multiplier placement as optimal stopping.** Deciding when to spend the one-shot
  x2 over a multi-round tournament is a sequential decision under uncertainty —
  framed here as a "standout" patience schedule gated by a triple-convergence
  quality check.

## Architecture

Four self-contained modules, pure Python standard library, no third-party
dependencies.

| Module | Responsibility |
| --- | --- |
| `src/model.py` | Core modelling. De-vigging (Shin + power), λ solve from 1X2 + O/U line, Dixon-Coles scoreline matrix, 1X2 calibration, expected-points ranking, 5-tier rarity bonus, x2 placement logic. |
| `src/optimize_winprob.py` | Monte-Carlo tournament optimiser. Simulates thousands of tournaments against a popularity-driven model of rival forecasters to estimate and maximise P(finish 1st). |
| `src/odds_api.py` | Pure-stdlib client for an external odds API (The Odds API). Aggregates ~24 bookmakers by median, with multi-language team-name mapping. |
| `src/stakes_context.py` | Contextual adjustment for final group-stage matchdays — "dead rubber" convenient draws, plus the goal-scoring incentive specific to the 48-team World Cup 2026 format (best third-placed teams qualify on goal difference). |
| `tests/` | 75 deterministic unit tests (stdlib `unittest`), zero network calls. |

### The expected-points ranking

For each candidate scoreline `s` the engine computes

```
EV(s) = points(outcome(s)) · P(outcome(s))  +  rarity_bonus(s) · P(exact = s)
```

The first term rewards calling the right 1X2 result (weighted by the points
indexed on the bookmaker quotation); the second rewards landing the exact score,
where the bonus grows as fewer of the forecasters who got the *result* right also
played that *exact* score. The 5-tier rarity schedule encodes that ownership curve.

## The math

**Shin de-vigging.** Given quoted prices with overround, Shin recovers the
insider-trading parameter `z` and the true probabilities `p_i`:

```
        sqrt( z² + 4(1 - z) · π_i² / Σπ )  −  z
p_i  =  ──────────────────────────────────────────
                       2(1 − z)
```

where `π_i = 1/odds_i` are the implied (booked) probabilities. `z` is solved so
the `p_i` sum to 1. The power method is offered as an alternative:
`p_i ∝ π_i^(1/k)` with `k` chosen so `Σ p_i = 1`.

**Dixon-Coles low-score correction.** Independent Poisson with rates λ (home), μ
(away) is multiplied by the τ factor on the four low scores:

```
τ(x, y) =
   1 − λ·μ·ρ     if (x, y) = (0, 0)
   1 + λ·ρ       if (x, y) = (0, 1)
   1 + μ·ρ       if (x, y) = (1, 0)
   1 − ρ         if (x, y) = (1, 1)
   1             otherwise

P(X=x, Y=y) = τ(x, y) · Poisson(x; λ) · Poisson(y; μ)
```

with ρ controlling the dependence at low scores. The full matrix is then
renormalised and calibrated so its 1X2 marginals match the de-vigged market.

**Objective.** Per match the engine maximises `EV(s)` above; across the tournament
the Monte-Carlo layer instead maximises `P( rank = 1 )` estimated over simulated
fields.

## Design decisions

A few deliberate choices, and one honest reversal.

- **Expected value first, win-probability when it matters.** Early in a tournament,
  chasing P(1st) on a single round degenerates into a lottery (everyone forecasting
  0-0). So the default objective is expected points; the Monte-Carlo P(1st)
  optimiser is reserved for when maximising the chance of winning genuinely diverges
  from maximising EV — e.g. when trailing late. The two optimisers are separate by
  design.
- **Optimal stopping: tried, then rejected.** The x2-multiplier-placement problem
  looks like a textbook optimal-stopping problem, and a Cayley–Moser
  prophet-style stopping rule was implemented first. On real data it degenerated —
  the thresholds it produced were not robust to the small, noisy sequence of
  matchdays. It was replaced with a transparent "standout" heuristic gated by a
  triple-convergence quality check (confidence above threshold **and** low score
  ownership **and** fresh odds). Judgment over formula-fitting.
- **Standard library only.** Zero third-party dependencies — `model.py`,
  `optimize_winprob.py` and the odds client all run on a bare Python install. Easy
  to audit, trivial to deploy, nothing to pin.
- **Deterministic, offline test suite.** 75 `unittest` cases, no network, fixed
  seeds. The whole maths pipeline is regression-tested without hitting any API.
- **Defensive by default.** Inputs are validated before any network call; the odds
  client backs off on rate-limit responses.

## Running it

The core model reads a slate of matches as JSON on stdin and prints, per match, the
ranked scorelines and the recommended pick (plus an x2-placement verdict). A ready-made
example lives in [`examples/sample_input.json`](examples/sample_input.json):

```bash
python src/model.py < examples/sample_input.json
```

Input format:

```json
{
  "gameweek": 1,
  "total_gameweeks": 9,
  "matches": [
    {
      "id": "match_001",
      "home": "Brazil",
      "away": "Serbia",
      "odds_home": 1.45,
      "odds_draw": 4.20,
      "odds_away": 7.50,
      "total_hint": 2.5,
      "points_home": 22,
      "points_draw": 60,
      "points_away": 95,
      "field_bets": {"home": 0.82, "draw": 0.11, "away": 0.07}
    }
  ]
}
```

- `odds_home / odds_draw / odds_away` — decimal bookmaker odds for the 1X2 market.
- `total_hint` — the over/under goals line, used to pin the total goal intensity.
- `points_home / points_draw / points_away` — the pool's points indexed on each
  outcome (the quotation you score if you call that result).
- `field_bets` *(optional)* — how the field splits across 1X2, for leverage/ownership.
- `gameweek / total_gameweeks / x2_holding` *(optional)* — enable the x2-placement verdict.

Run the Monte-Carlo win-probability optimiser:

```bash
python src/optimize_winprob.py < examples/sample_input.json
```

Run the test suite (deterministic, no network):

```bash
python -m unittest discover -s tests
```

## References

The model is grounded in the literature rather than improvised:

- **Dixon, M. J. & Coles, S. G. (1997).** *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical
  Society: Series C. — the low-score correlation correction.
- **Shin, H. S. (1993).** *Measuring the Incidence of Insider Trading in a Market
  for State-Contingent Claims.* The Economic Journal. — odds de-vigging under
  insider trading.
- **Clair, B. & Letscher, D. (2007).** *Optimal Strategies for Sports Betting
  Pools.* Operations Research. — strategy in pool/contest play.
- **Gaba, A. & Tsetlin, I.** Contest theory — when to maximise expected value
  versus probability of winning a contest.
- **Prophet inequalities / optimal stopping** (Cayley–Moser and successors) — the
  sequential-decision framing for one-shot multiplier placement.

## License

MIT — see [`LICENSE`](LICENSE).
