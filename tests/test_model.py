#!/usr/bin/env python3
"""
Tests unitaires du coeur de modélisation (src/model.py).
"""
import contextlib
import io
import math
import os
import sys
import unittest

# Inject src/ into path before any import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import model


# ---------------------------------------------------------------------------
# 1. poisson_pmf
# ---------------------------------------------------------------------------
class TestPoissonPmf(unittest.TestCase):

    def test_pmf_zero_lambda_is_one_for_k0(self):
        # poisson_pmf(0, lam<=0) == 1
        self.assertAlmostEqual(model.poisson_pmf(0, 0.0), 1.0)

    def test_pmf_zero_lambda_is_zero_for_k_positive(self):
        self.assertAlmostEqual(model.poisson_pmf(1, 0.0), 0.0)

    def test_pmf_k0_lam1(self):
        # pmf(0, 1.0) == e^-1
        self.assertAlmostEqual(model.poisson_pmf(0, 1.0), math.exp(-1), delta=1e-9)

    def test_pmf_sums_to_one(self):
        # sum over k=0..50 with lam=2.3 ≈ 1
        total = sum(model.poisson_pmf(k, 2.3) for k in range(51))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_pmf_non_negative(self):
        for k in range(10):
            self.assertGreaterEqual(model.poisson_pmf(k, 1.5), 0.0)


# ---------------------------------------------------------------------------
# 2. dc_tau
# ---------------------------------------------------------------------------
class TestDcTau(unittest.TestCase):
    """Dixon-Coles standard — rho < 0, lambdas modérés."""

    def setUp(self):
        self.lh = 1.5
        self.la = 1.0
        self.rho = -0.13

    def test_tau_00_greater_than_one(self):
        # tau(0,0) = 1 - lh*la*rho  → > 1 quand rho < 0
        tau = model.dc_tau(0, 0, self.lh, self.la, self.rho)
        expected = 1.0 - self.lh * self.la * self.rho
        self.assertAlmostEqual(tau, expected, places=9)
        self.assertGreater(tau, 1.0)

    def test_tau_01_less_than_one(self):
        # tau(0,1) = 1 + lh*rho  → < 1 quand lh>0, rho<0
        tau = model.dc_tau(0, 1, self.lh, self.la, self.rho)
        expected = 1.0 + self.lh * self.rho
        self.assertAlmostEqual(tau, expected, places=9)
        self.assertLess(tau, 1.0)

    def test_tau_10_less_than_one(self):
        # tau(1,0) = 1 + la*rho  → < 1 quand la>0, rho<0
        tau = model.dc_tau(1, 0, self.lh, self.la, self.rho)
        expected = 1.0 + self.la * self.rho
        self.assertAlmostEqual(tau, expected, places=9)
        self.assertLess(tau, 1.0)

    def test_tau_11_greater_than_one(self):
        # tau(1,1) = 1 - rho  → > 1 quand rho < 0
        tau = model.dc_tau(1, 1, self.lh, self.la, self.rho)
        expected = 1.0 - self.rho
        self.assertAlmostEqual(tau, expected, places=9)
        self.assertGreater(tau, 1.0)

    def test_tau_22_is_one(self):
        # tau(i,j) = 1 for i>=2 or j>=2
        self.assertAlmostEqual(model.dc_tau(2, 2, self.lh, self.la, self.rho), 1.0)
        self.assertAlmostEqual(model.dc_tau(3, 0, self.lh, self.la, self.rho), 1.0)
        self.assertAlmostEqual(model.dc_tau(0, 2, self.lh, self.la, self.rho), 1.0)


# ---------------------------------------------------------------------------
# 3. score_matrix
# ---------------------------------------------------------------------------
class TestScoreMatrix(unittest.TestCase):

    def setUp(self):
        self.lh = 1.5
        self.la = 1.0
        self.matrix = model.score_matrix(self.lh, self.la, rho=-0.13)

    def test_sum_approximately_one(self):
        total = sum(self.matrix.values())
        self.assertAlmostEqual(total, 1.0, delta=1e-6)

    def test_no_negative_probabilities(self):
        for p in self.matrix.values():
            self.assertGreaterEqual(p, 0.0)

    def test_dc_boosts_00_vs_independent_poisson(self):
        # With rho=-0.13, P(0-0) should be HIGHER than product of marginal Poisson pmfs
        p_dc = self.matrix[(0, 0)]
        p_indep = model.poisson_pmf(0, self.lh) * model.poisson_pmf(0, self.la)
        self.assertGreater(p_dc, p_indep)

    def test_keys_cover_expected_range(self):
        # default max_goals=10 → keys go from (0,0) to (10,10)
        self.assertIn((0, 0), self.matrix)
        self.assertIn((10, 10), self.matrix)
        self.assertNotIn((11, 0), self.matrix)


# ---------------------------------------------------------------------------
# 4. demargin
# ---------------------------------------------------------------------------
class TestDemargin(unittest.TestCase):

    def test_proportional_roundtrip_fair_odds(self):
        # "Fair" odds: 1/p exactly, sum(inv)=1 → proportional should recover p
        p_true = [0.55, 0.25, 0.20]
        odds = [1.0 / p for p in p_true]
        result = model.demargin(odds[0], odds[1], odds[2], method="proportional")
        for got, expected in zip(result, p_true):
            self.assertAlmostEqual(got, expected, places=6)

    def test_proportional_sums_to_one(self):
        result = model.demargin(1.50, 4.0, 6.0, method="proportional")
        self.assertAlmostEqual(sum(result), 1.0, places=8)

    def test_shin_sums_to_one(self):
        result = model.demargin(1.50, 4.0, 6.0, method="shin")
        self.assertAlmostEqual(sum(result), 1.0, places=6)

    def test_shin_inflates_favourite_vs_proportional(self):
        # Shin corrects favourite-longshot bias → favourite gets higher prob than proportional
        prop = model.demargin(1.50, 4.0, 6.0, method="proportional")
        shin = model.demargin(1.50, 4.0, 6.0, method="shin")
        # index 0 is the favourite (shortest odds)
        self.assertGreaterEqual(shin[0], prop[0])

    def test_power_explicit_k127(self):
        # CONTRAT: power with k=1.27 explicit: p_i = inv_i^1.27 / sum(inv_j^1.27)
        odds = [1.50, 4.0, 6.0]
        inv = [1.0 / o for o in odds]
        expected_raw = [x ** 1.27 for x in inv]
        s = sum(expected_raw)
        expected = [x / s for x in expected_raw]
        result = model.demargin(odds[0], odds[1], odds[2], method="power", k=1.27)
        for got, exp in zip(result, expected):
            self.assertAlmostEqual(got, exp, places=6)

    def test_power_sums_to_one(self):
        result = model.demargin(1.50, 4.0, 6.0, method="power", k=1.27)
        self.assertAlmostEqual(sum(result), 1.0, places=8)

    def test_power_k127_accentuates_favourite_vs_proportional(self):
        # power k>1 accentuates the favourite relative to proportional
        prop = model.demargin(1.50, 4.0, 6.0, method="proportional")
        power = model.demargin(1.50, 4.0, 6.0, method="power", k=1.27)
        self.assertGreater(power[0], prop[0])


# ---------------------------------------------------------------------------
# 5. solve_lambdas round-trip
# ---------------------------------------------------------------------------
class TestSolveLambdas(unittest.TestCase):

    def _roundtrip(self, p_home, p_draw, p_away, total_hint=None, tol=0.02):
        lh, la = model.solve_lambdas(p_home, p_draw, p_away, total_hint=total_hint)
        m = model.score_matrix(lh, la)
        ph, pd, pa = model.outcome_probs(m)
        self.assertAlmostEqual(ph, p_home, delta=tol,
                               msg=f"pH: got {ph:.3f} expected {p_home}")
        self.assertAlmostEqual(pd, p_draw, delta=tol,
                               msg=f"pD: got {pd:.3f} expected {p_draw}")
        self.assertAlmostEqual(pa, p_away, delta=tol,
                               msg=f"pA: got {pa:.3f} expected {p_away}")

    def test_roundtrip_balanced(self):
        self._roundtrip(0.45, 0.29, 0.26)

    def test_roundtrip_home_favoured(self):
        self._roundtrip(0.65, 0.21, 0.14)

    def test_roundtrip_strong_favourite(self):
        self._roundtrip(0.85, 0.10, 0.05)

    def test_roundtrip_with_total_hint_preserves_1x2(self):
        # total_hint must NOT override the 1X2 fit — round-trip stays within tol=0.03
        self._roundtrip(0.65, 0.21, 0.14, total_hint=2.5, tol=0.03)

    def test_roundtrip_balanced_with_total_hint(self):
        self._roundtrip(0.45, 0.29, 0.26, total_hint=2.8, tol=0.03)


# ---------------------------------------------------------------------------
# 6. rarity_bonus — CRITIQUE
# ---------------------------------------------------------------------------
class TestRarity(unittest.TestCase):

    PROBE_POINTS = [0.001, 0.003, 0.02, 0.1, 0.25, 0.35, 0.6, 1.0]

    def test_monotonic(self):
        """rarity_bonus must be monotone non-increasing as share increases."""
        bonuses = [model.rarity_bonus(s) for s in self.PROBE_POINTS]
        for i in range(len(bonuses) - 1):
            self.assertGreaterEqual(
                bonuses[i], bonuses[i + 1],
                msg=(f"Not monotone: bonus({self.PROBE_POINTS[i]})={bonuses[i]} "
                     f"< bonus({self.PROBE_POINTS[i+1]})={bonuses[i+1]}")
            )

    # Barème (paliers d'un pool typique) : <0.5%→100, 0.5-5%→70, 5-20%→50, 20-30%→30, >30%→20
    def test_exact_value_001(self):
        self.assertEqual(model.rarity_bonus(0.001), 100)

    def test_exact_value_003(self):
        # 0.03 is in [0.005, 0.05) → 70
        self.assertEqual(model.rarity_bonus(0.03), 70)

    def test_exact_value_010(self):
        # 0.10 is in [0.05, 0.20) → 50
        self.assertEqual(model.rarity_bonus(0.10), 50)

    def test_exact_value_025(self):
        # 0.25 is in [0.20, 0.30) → 30
        self.assertEqual(model.rarity_bonus(0.25), 30)

    def test_exact_value_035(self):
        # 0.35 > 0.30 → 20
        self.assertEqual(model.rarity_bonus(0.35), 20)

    def test_exact_value_050(self):
        # 0.50 → 20
        self.assertEqual(model.rarity_bonus(0.50), 20)


# ---------------------------------------------------------------------------
# 7. evaluate_match fallback — favori net, CRITIQUE
# ---------------------------------------------------------------------------
class TestEvaluateMatchFallback(unittest.TestCase):

    def _minimal_match(self, **overrides):
        """Build a minimal match dict for testing fallback (no odds)."""
        base = {
            "id": "mpp_championship_match_1",
            "home": "TeamA",
            "away": "TeamB",
            "points_home": 30,    # net favourite (low points = high probability)
            "points_draw": 130,
            "points_away": 250,
            "odds_home": None,
            "odds_draw": None,
            "odds_away": None,
            "total_hint": None,
            "field_bets": {"home": 0.70, "draw": 0.20, "away": 0.10},
        }
        base.update(overrides)
        return base

    def test_fallback_prob_source(self):
        result = model.evaluate_match(self._minimal_match())
        self.assertEqual(result["prob_source"], "pool_quotations")

    def test_fallback_pick_result_is_home(self):
        # Net favourite (points_home=30 << draw=130, away=250) → pick_result == 'H'
        result = model.evaluate_match(self._minimal_match())
        self.assertEqual(result["pick_result"], "H")

    def test_fallback_pick_recommended_not_draw_score(self):
        # Old bug: always returned '1-1'. With correct logic it should be a home-win score.
        result = model.evaluate_match(self._minimal_match())
        # pick_recommended should NOT be '1-1' for a net home favourite
        self.assertNotEqual(result["pick_recommended"], "1-1")

    def test_fallback_p_home_above_threshold(self):
        # After power decompress with k=1.27, p_home > 0.80 for a clear favourite (pts_home=30).
        # Regression anchor: would be ~0.74 at k=1.0 — test must redden if k falls back to 1.0.
        result = model.evaluate_match(self._minimal_match())
        self.assertGreater(result["p_home"], 0.80)

    def test_output_schema(self):
        result = model.evaluate_match(self._minimal_match())
        required_keys = [
            "id", "home", "away", "prob_source",
            "lambda_home", "lambda_away",
            "p_home", "p_draw", "p_away",
            "expected_total",
            "points",
            "pick_ev", "pick_recommended", "pick_result",
            "ev_recommended",
            "contrarian_play",
            "top_candidates",
        ]
        for key in required_keys:
            self.assertIn(key, result, msg=f"Missing key: {key}")


# ---------------------------------------------------------------------------
# 8. evaluate_match — invalid odds → fallback + stderr warning
# ---------------------------------------------------------------------------
class TestEvaluateMatchInvalidOdds(unittest.TestCase):

    def _match_with_zero_home_odds(self):
        return {
            "id": "mpp_championship_match_999",
            "home": "TeamX",
            "away": "TeamY",
            "points_home": 60,
            "points_draw": 110,
            "points_away": 200,
            "odds_home": 0,       # invalid
            "odds_draw": 4.0,
            "odds_away": 6.0,
            "total_hint": None,
            "field_bets": None,
        }

    def test_invalid_odds_triggers_fallback(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = model.evaluate_match(self._match_with_zero_home_odds())
        self.assertEqual(result["prob_source"], "pool_quotations")

    def test_invalid_odds_emits_warning_to_stderr(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            model.evaluate_match(self._match_with_zero_home_odds())
        warning = buf.getvalue()
        self.assertTrue(len(warning) > 0, "Expected a warning on stderr, got nothing")
        self.assertIn("mpp_championship_match_999", warning)


# ---------------------------------------------------------------------------
# 9. evaluate_match — valid bookmaker odds
# ---------------------------------------------------------------------------
class TestEvaluateMatchValidOdds(unittest.TestCase):

    def _match_with_valid_odds(self):
        return {
            "id": "mpp_championship_match_42",
            "home": "TeamA",
            "away": "TeamB",
            "points_home": 55,
            "points_draw": 120,
            "points_away": 210,
            "odds_home": 1.48,
            "odds_draw": 4.33,
            "odds_away": 6.5,
            "total_hint": None,
            "field_bets": {"home": 0.60, "draw": 0.25, "away": 0.15},
        }

    def test_valid_odds_prob_source_is_bookmaker(self):
        result = model.evaluate_match(self._match_with_valid_odds())
        self.assertEqual(result["prob_source"], "bookmaker")


# ---------------------------------------------------------------------------
# 10. recommend_x2
# ---------------------------------------------------------------------------
class TestRecommendX2(unittest.TestCase):

    def _make_result(self, match_id, home, away, ev=50.0, p_result=0.60):
        return {
            "id": match_id,
            "home": home,
            "away": away,
            "pick_recommended": "2-1",
            "pick_result": "H",
            "ev_recommended": ev,
            "top_candidates": [
                {"score": "2-1", "result": "H", "p_result": p_result, "ev": ev,
                 "p_exact": 0.12, "diff_score": ev * 1.5}
            ],
        }

    def test_x2_hint_ev_only_key_exists(self):
        results = [self._make_result("mpp_championship_match_1", "A", "B")]
        out = model.recommend_x2(results)
        self.assertIn("x2_hint_ev_only", out)

    def test_marginal_score_key_absent(self):
        results = [self._make_result("mpp_championship_match_1", "A", "B")]
        out = model.recommend_x2(results)
        self.assertNotIn("marginal_score", out)

    def test_note_key_exists(self):
        results = [self._make_result("mpp_championship_match_1", "A", "B")]
        out = model.recommend_x2(results)
        self.assertIn("note", out)

    def test_picks_highest_ev(self):
        results = [
            self._make_result("mpp_championship_match_1", "A", "B", ev=30.0),
            self._make_result("mpp_championship_match_2", "C", "D", ev=90.0),
        ]
        out = model.recommend_x2(results)
        self.assertEqual(out["id"], "mpp_championship_match_2")

    def test_empty_results_returns_none_or_dict(self):
        # Should not raise; may return None
        out = model.recommend_x2([])
        # Just assert it doesn't crash — None is acceptable
        self.assertTrue(out is None or isinstance(out, dict))


if __name__ == "__main__":
    unittest.main()
