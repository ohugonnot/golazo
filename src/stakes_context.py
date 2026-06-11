#!/usr/bin/env python3
"""
Module "enjeu J3" — ajuste les matchs de la 3e journée de poules selon le contexte
sportif que les cotes brutes ne capturent pas entièrement.

Fondé sur la recherche (cf. docs/methodology.md) :
- Équipes DÉJÀ QUALIFIÉES → rotations massives (×3 vs J1/J2) + nul "arrangeant" si le nul
  qualifie les deux → P(nul) en hausse, total bas.
- Format 2026 : les 8 meilleurs 3es se qualifient à la différence de buts → une équipe en
  COURSE pour la 3e place a intérêt à MARQUER → total en hausse, moins de nul.
- Une équipe DÉJÀ ÉLIMINÉE → démobilisée → encaisse plus (total en hausse côté adverse).

Ne remplace pas le modèle : produit des AJUSTEMENTS (total_hint delta, multiplicateur de
nul) à appliquer manuellement/automatiquement sur l'enriched, à valider par les compos
confirmées. À utiliser UNIQUEMENT en J3, et seulement quand l'enjeu est connu.

Usage :
  python3 src/stakes_context.py annotate <input.json>   # ajoute des champs *_j3 indicatifs
  (ou importer adjust_for_stakes() / j3_scenario_hint())

Scénarios reconnus (champ "j3_scenario" à poser sur chaque match d'un enriched J3) :
  "draw_qualifies_both"   nul qualifie les 2 → P(nul)↑↑, total↓↓ (le piège classique)
  "both_qualified"        2 déjà qualifiées, classement figé → rotations, total↓, nul↑
  "one_chasing_goaldiff"  ≥1 équipe en course 3e place (diff de buts) → total↑, nul↓
  "one_eliminated"        1 démobilisée → total↑ (l'autre se balade)
  "high_stakes"           qualif en jeu des deux côtés → match "vrai", pas d'ajustement
  None / absent           pas d'ajustement
"""
import json, sys

# (delta_total_hint, mult_proba_nul) par scénario. mult appliqué à p_draw post-demargin
# côté optimiseur/modèle via le champ "draw_boost". Calibrés sur l'ordre de grandeur de la
# recherche (nul +5-8 pts ; rotations baissent le total ~0.3-0.5).
SCENARIOS = {
    "draw_qualifies_both": (-0.45, 1.30),
    "both_qualified":      (-0.30, 1.18),
    "one_chasing_goaldiff": (+0.35, 0.85),
    "one_eliminated":      (+0.30, 0.90),
    "high_stakes":         (0.0, 1.0),
}


def adjust_for_stakes(match):
    """Renvoie un dict d'ajustements indicatifs pour un match selon match['j3_scenario'].
    Ne modifie pas le match. {} si aucun scénario ou scénario inconnu."""
    sc = match.get("j3_scenario")
    if not sc or sc not in SCENARIOS:
        return {}
    d_total, mult_draw = SCENARIOS[sc]
    out = {"j3_scenario": sc, "draw_boost": mult_draw}
    base_total = match.get("total_hint")
    if base_total is not None and d_total:
        # borne raisonnable [1.8, 3.6] pour éviter les valeurs aberrantes
        out["total_hint_suggested"] = round(max(1.8, min(3.6, base_total + d_total)), 2)
    return out


def j3_scenario_hint(home_status, away_status, draw_qualifies_both=False):
    """Aide à choisir le scénario depuis les statuts d'équipe.
    *_status ∈ {"qualified","chasing","eliminated","alive"}. Renvoie une clé de SCENARIOS."""
    s = {home_status, away_status}
    if draw_qualifies_both and s <= {"qualified", "alive", "chasing"}:
        return "draw_qualifies_both"
    if s == {"qualified"}:
        return "both_qualified"
    if "chasing" in s:
        return "one_chasing_goaldiff"
    if "eliminated" in s:
        return "one_eliminated"
    return "high_stakes"


def annotate(enriched_path):
    """Ajoute un bloc 'j3_adjustment' à chaque match porteur d'un j3_scenario. Backup préalable."""
    with open(enriched_path, encoding="utf-8") as f:
        doc = json.load(f)
    matches = doc.get("matches", doc if isinstance(doc, list) else [])
    report = {"annotated": [], "skipped": 0}
    for m in matches:
        adj = adjust_for_stakes(m)
        if adj:
            m["j3_adjustment"] = adj
            report["annotated"].append(f'{m.get("home")}-{m.get("away")}: {adj.get("j3_scenario")}')
        else:
            report["skipped"] += 1
    if report["annotated"]:
        with open(enriched_path + ".bak_j3", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    return report


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "annotate":
        print(__doc__); return
    print(json.dumps(annotate(sys.argv[2]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
