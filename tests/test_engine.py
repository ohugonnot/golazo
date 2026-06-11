#!/usr/bin/env python3
"""
Tests du moteur : décision de placement du x2 (recommend_x2), module d'ajustement
contextuel (stakes_context), draw_boost dans evaluate_match, et client odds_api
(matching multi-langue, consensus médian, enrich) — réseau mocké.
"""
import os
import sys
import json
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import model
import stakes_context as j3_context
import odds_api


# ---------------------------------------------------------------------------
# recommend_x2 — décision poser/attendre (calendrier de patience par standout)
# ---------------------------------------------------------------------------
def _fake_result(name, ev, p_result, ownership=0.10, fresh=True):
    return {"home": name, "away": "X", "id": f"id_{name}",
            "pick_recommended": "2-0", "ev_recommended": ev,
            "prob_source": "bookmaker" if fresh else "pool_quotations",
            "recommended": {"score": "2-0", "result": "H", "p_result": p_result,
                            "p_exact": 0.1, "share_est": ownership, "ev": ev},
            "top_candidates": [{"p_result": p_result}]}


class TestRecommendX2Decision(unittest.TestCase):

    def test_no_decision_without_gameweek(self):
        res = [_fake_result("A", 100, 0.6)]
        out = model.recommend_x2(res)
        self.assertNotIn("decision", out)
        self.assertIn("x2_hint_ev_only", out)

    def test_last_gameweek_forces_place(self):
        res = [_fake_result("A", 50, 0.5), _fake_result("B", 40, 0.5)]
        out = model.recommend_x2(res, gameweek=9, total_gameweeks=9)
        self.assertTrue(out["place_now"])
        self.assertEqual(out["decision"], "POSER")
        self.assertEqual(out["remaining_gameweeks"], 1)
        self.assertIsNone(out["required_standout"])

    def test_not_holding_no_decision(self):
        res = [_fake_result("A", 100, 0.7)]
        out = model.recommend_x2(res, gameweek=3, holding=False)
        self.assertNotIn("decision", out)

    def test_picks_highest_gain_match(self):
        res = [_fake_result("low", 30, 0.5), _fake_result("high", 200, 0.8)]
        out = model.recommend_x2(res, gameweek=5)
        self.assertEqual(out["id"], "id_high")

    def test_early_homogeneous_slate_waits(self):
        # J1/9, slate homogène (aucun standout) → ATTENDRE (exige ×1.8 la médiane)
        res = [_fake_result(f"m{i}", 40, 0.6) for i in range(10)]
        out = model.recommend_x2(res, gameweek=1, total_gameweeks=9)
        self.assertEqual(out["decision"], "ATTENDRE")
        self.assertFalse(out["place_now"])

    def test_early_standout_places(self):
        # J1/9 mais un match domine très nettement (×4 la médiane) → POSER
        res = [_fake_result(f"m{i}", 40, 0.6) for i in range(9)] + [_fake_result("star", 200, 0.85)]
        out = model.recommend_x2(res, gameweek=1, total_gameweeks=9)
        self.assertEqual(out["decision"], "POSER")
        self.assertEqual(out["id"], "id_star")

    def test_required_standout_decreases_over_tournament(self):
        # même slate modérément contrastée → ATTENDRE tôt, POSER tard (qualité OK)
        res = [_fake_result(f"m{i}", 40, 0.6) for i in range(9)] + [_fake_result("x", 52, 0.70)]
        early = model.recommend_x2(res, gameweek=1, total_gameweeks=9)
        late = model.recommend_x2(res, gameweek=7, total_gameweeks=9)
        self.assertGreater(early["required_standout"], late["required_standout"])
        self.assertEqual(early["decision"], "ATTENDRE")  # 52/40=1.3 < 1.8
        self.assertEqual(late["decision"], "POSER")      # 1.3 >= 1.15 et qualité OK

    # --- porte de QUALITÉ (triple convergence) ---
    def test_quality_blocks_low_confidence(self):
        # standout énorme (timing OK) mais confiance < 0.65 → ATTENDRE
        res = [_fake_result(f"m{i}", 20, 0.5) for i in range(9)] + [_fake_result("star", 200, 0.60)]
        out = model.recommend_x2(res, gameweek=5, total_gameweeks=9)
        self.assertFalse(out["quality_ok"])
        self.assertEqual(out["decision"], "ATTENDRE")

    def test_quality_blocks_high_ownership(self):
        # confiance OK mais score trop couru (ownership 0.4) → ATTENDRE
        res = [_fake_result(f"m{i}", 20, 0.5) for i in range(9)] + \
              [_fake_result("star", 200, 0.8, ownership=0.40)]
        out = model.recommend_x2(res, gameweek=5, total_gameweeks=9)
        self.assertFalse(out["quality_ok"])
        self.assertEqual(out["decision"], "ATTENDRE")

    def test_quality_blocks_stale_odds(self):
        # confiance + ownership OK mais cotes non fraîches (fallback) → ATTENDRE
        res = [_fake_result(f"m{i}", 20, 0.5) for i in range(9)] + \
              [_fake_result("star", 200, 0.8, fresh=False)]
        out = model.recommend_x2(res, gameweek=5, total_gameweeks=9)
        self.assertFalse(out["quality_ok"])
        self.assertEqual(out["decision"], "ATTENDRE")

    def test_last_window_places_despite_poor_quality(self):
        # dernière fenêtre : on pose même si la qualité est mauvaise (use-it-or-lose-it)
        res = [_fake_result("a", 50, 0.5, ownership=0.5, fresh=False)]
        out = model.recommend_x2(res, gameweek=9, total_gameweeks=9)
        self.assertTrue(out["place_now"])
        self.assertFalse(out["quality_ok"])


# ---------------------------------------------------------------------------
# Module enjeu J3
# ---------------------------------------------------------------------------
class TestJ3Context(unittest.TestCase):

    def test_no_scenario_no_adjustment(self):
        self.assertEqual(j3_context.adjust_for_stakes({"home": "A", "away": "B"}), {})

    def test_unknown_scenario_no_adjustment(self):
        self.assertEqual(j3_context.adjust_for_stakes({"j3_scenario": "bogus"}), {})

    def test_draw_qualifies_both_boosts_draw_lowers_total(self):
        m = {"j3_scenario": "draw_qualifies_both", "total_hint": 2.6}
        adj = j3_context.adjust_for_stakes(m)
        self.assertGreater(adj["draw_boost"], 1.0)
        self.assertLess(adj["total_hint_suggested"], 2.6)

    def test_chasing_goaldiff_raises_total_lowers_draw(self):
        m = {"j3_scenario": "one_chasing_goaldiff", "total_hint": 2.4}
        adj = j3_context.adjust_for_stakes(m)
        self.assertLess(adj["draw_boost"], 1.0)
        self.assertGreater(adj["total_hint_suggested"], 2.4)

    def test_total_hint_bounded(self):
        m = {"j3_scenario": "draw_qualifies_both", "total_hint": 1.9}
        adj = j3_context.adjust_for_stakes(m)
        self.assertGreaterEqual(adj["total_hint_suggested"], 1.8)

    def test_scenario_hint_draw_qualifies(self):
        self.assertEqual(
            j3_context.j3_scenario_hint("qualified", "alive", draw_qualifies_both=True),
            "draw_qualifies_both")

    def test_scenario_hint_both_qualified(self):
        self.assertEqual(j3_context.j3_scenario_hint("qualified", "qualified"), "both_qualified")

    def test_scenario_hint_chasing(self):
        self.assertEqual(j3_context.j3_scenario_hint("chasing", "qualified"), "one_chasing_goaldiff")


# ---------------------------------------------------------------------------
# draw_boost dans evaluate_match
# ---------------------------------------------------------------------------
class TestDrawBoost(unittest.TestCase):

    def _match(self, **extra):
        m = {"id": "m1", "home": "A", "away": "B",
             "points_home": 50, "points_draw": 100, "points_away": 120,
             "odds_home": 2.0, "odds_draw": 3.3, "odds_away": 3.8,
             "total_hint": 2.5}
        m.update(extra)
        return m

    def test_draw_boost_raises_draw_probability(self):
        base = model.evaluate_match(self._match())
        boosted = model.evaluate_match(self._match(draw_boost=1.3))
        self.assertGreater(boosted["p_draw"], base["p_draw"])

    def test_draw_boost_one_is_neutral(self):
        base = model.evaluate_match(self._match())
        same = model.evaluate_match(self._match(draw_boost=1.0))
        self.assertEqual(base["p_draw"], same["p_draw"])

    def test_probabilities_still_sum_to_one(self):
        b = model.evaluate_match(self._match(draw_boost=1.4))
        self.assertAlmostEqual(b["p_home"] + b["p_draw"] + b["p_away"], 1.0, places=2)


# ---------------------------------------------------------------------------
# Client odds_api — réseau mocké
# ---------------------------------------------------------------------------
class TestOddsApiMatching(unittest.TestCase):

    def test_norm_fr_to_en_alias(self):
        self.assertEqual(odds_api._norm("Corée du Sud"), odds_api._norm("South Korea"))
        self.assertEqual(odds_api._norm("Afrique du Sud"), odds_api._norm("South Africa"))
        self.assertEqual(odds_api._norm("Côte d’Ivoire"), odds_api._norm("Ivory Coast"))

    def test_norm_identity_for_unmapped(self):
        self.assertEqual(odds_api._norm("France"), "france")

    def test_median_even_odd(self):
        self.assertEqual(odds_api._median([1, 2, 3]), 2)
        self.assertEqual(odds_api._median([1, 3]), 2)
        self.assertIsNone(odds_api._median([]))

    def test_consensus_odds_aggregates_medians(self):
        event = {
            "home_team": "Mexico", "away_team": "South Africa",
            "bookmakers": [
                {"markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Mexico", "price": 1.5},
                        {"name": "South Africa", "price": 6.0},
                        {"name": "Draw", "price": 4.0}]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.0},
                        {"name": "Under", "point": 2.5, "price": 1.8}]}]},
                {"markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Mexico", "price": 1.6},
                        {"name": "South Africa", "price": 6.4},
                        {"name": "Draw", "price": 4.2}]}]},
            ],
        }
        co = odds_api.consensus_odds(event)
        self.assertAlmostEqual(co["home"], 1.55)
        self.assertAlmostEqual(co["draw"], 4.1)
        self.assertEqual(co["total_hint"], 2.5)
        self.assertEqual(co["n_books"], 2)

    def test_enrich_matches_by_name_and_backs_up(self):
        import tempfile
        doc = {"matches": [
            {"home": "Corée du Sud", "away": "Tchéquie", "total_hint": 2.5},
            {"home": "Inconnue", "away": "Autre", "total_hint": 2.5},
        ]}
        fake_events = [{
            "home_team": "South Korea", "away_team": "Czech Republic",
            "bookmakers": [{"markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "South Korea", "price": 2.6},
                    {"name": "Czech Republic", "price": 3.0},
                    {"name": "Draw", "price": 3.05}]}]}],
        }]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enr.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with patch("odds_api.fetch", return_value=fake_events):
                rep = odds_api.enrich(path)
            self.assertIn("Corée du Sud-Tchéquie", rep["updated"])
            self.assertIn("Inconnue-Autre", rep["not_found"])
            with open(path, encoding="utf-8") as f:
                out = json.load(f)
            kr = out["matches"][0]
            self.assertEqual(kr["odds_home"], 2.6)
            self.assertTrue(kr["odds_source"].startswith("the-odds-api/"))
            self.assertTrue(os.path.exists(rep["backup"]))

    def test_missing_key_raises_clear_error(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("odds_api.KEY_FILE", "/nonexistent/key.txt"):
            with self.assertRaises(SystemExit) as cm:
                odds_api.api_key()
        self.assertIn("the-odds-api", str(cm.exception))

    def test_env_var_key_takes_priority(self):
        with patch.dict(os.environ, {"ODDS_API_KEY": "abc123"}):
            self.assertEqual(odds_api.api_key(), "abc123")


if __name__ == "__main__":
    unittest.main()
