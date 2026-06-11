#!/usr/bin/env python3
"""
golazo — moteur de prédiction & optimiseur pour pools de pronostics de football.

Un pool de pronostics : on prédit le score exact d'un match. On gagne des points
indexés sur la cote du résultat (1X2) si le résultat est correct, plus un bonus de
rareté si on trouve le score exact. Un multiplicateur x2 unique se place sur un match.

Pipeline:
  1. De-vigging des cotes 1X2 (retrait de la marge bookmaker, méthode de Shin /
     Power) -> probabilités vraies P(H/N/A).
  2. Inversion: on retrouve les buts attendus (lambda_home, lambda_away) qui
     reproduisent ces probabilités, via un modèle Dixon-Coles (Poisson corrigé
     pour les petits scores). Si une ligne over/under est fournie, on la respecte.
  3. Matrice des scores exacts P(i,j) via Dixon-Coles.
  4. Espérance de points pour chaque score candidat:
        EV(s) = points_resultat(s) * P(resultat(s))  +  bonus_rarete(s) * P(exact = s)
     puis score de "leverage" pour une stratégie agressive (maximiser P(finir 1er)).
  5. Recommandation par match + placement du bonus x2 (timing + porte de qualité).

Stdlib uniquement. Entrée/sortie JSON. Voir docs/methodology.md pour la théorie.
"""
import json
import math
import sys
from itertools import product

if hasattr(sys.stdout, 'reconfigure') and (sys.stdout.encoding or '').lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --------------------------------------------------------------------------
# Barème du bonus "score exact" selon la rareté (paliers d'un pool typique, vérifiés
# empiriquement : un score commun rapporte +20, un score ultra-rare +100).
# CLEF = part des joueurs ayant le score exact PARMI ceux qui ont le bon RÉSULTAT
# (pas sur toute la communauté). conditional_popularity() calcule déjà ce P(score|résultat).
# Chaque tuple = (seuil_haut_de_part, points_bonus). Trié du plus rare au + commun.
# --------------------------------------------------------------------------
RARITY_TIERS = [
    (0.005, 100),   # < 0.5%   -> ultra rare
    (0.05,   70),   # 0.5-5%   -> very rare
    (0.20,   50),   # 5-20%    -> rare
    (0.30,   30),   # 20-30%   -> peu commun
    (1.01,   20),   # > 30%    -> commun (catches all remaining share)
]

def rarity_bonus(share):
    """Points bonus pour un score exact dont `share` = part de joueurs l'ayant."""
    share = max(share, 0.0)
    for thresh, pts in RARITY_TIERS:
        if share <= thresh:
            return pts
    # unreachable: (1.01, 20) catches share <= 1.0 always
    return 20

# --------------------------------------------------------------------------
# Popularité publique des scores (prior). Sert à estimer la rareté quand on n'a
# pas la vraie distribution communautaire. Fréquences indicatives des scores les + pronostiqués
# par le grand public (toutes issues confondues). Override possible via l'API du pool
# si elle expose le "% de joueurs ayant choisi ce score".
# --------------------------------------------------------------------------
BASE_SCORE_POPULARITY = {
    (1, 1): 0.135, (2, 1): 0.115, (1, 0): 0.110, (0, 0): 0.090, (2, 0): 0.080,
    (1, 2): 0.070, (0, 1): 0.065, (2, 2): 0.055, (3, 1): 0.045, (3, 0): 0.040,
    (1, 3): 0.030, (0, 2): 0.030, (3, 2): 0.025, (2, 3): 0.020, (4, 0): 0.012,
    (0, 3): 0.012, (4, 1): 0.012, (1, 4): 0.008, (4, 2): 0.006, (3, 3): 0.010,
}
POP_FLOOR = 0.0015  # part plancher pour un score non listé (rare mais possible)

# --------------------------------------------------------------------------
# Probabilités de base
# --------------------------------------------------------------------------
def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)

def dc_tau(i, j, lh, la, rho):
    """Correction Dixon-Coles pour les scores 0-0,1-0,0-1,1-1."""
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0

def score_matrix(lh, la, rho=-0.13, max_goals=10):
    """Matrice P(i,j) normalisée des scores exacts (Dixon-Coles)."""
    m = {}
    total = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_pmf(i, lh) * poisson_pmf(j, la) * dc_tau(i, j, lh, la, rho)
            p = max(p, 0.0)
            m[(i, j)] = p
            total += p
    if total > 0:
        for k in m:
            m[k] /= total
    return m

def calibrate_to_outcomes(matrix, p_home, p_draw, p_away):
    """
    Recale la matrice de scores pour que ses probabilités marginales 1/N/2 égalent
    EXACTEMENT (p_home, p_draw, p_away) issues des cotes. Corrige l'incohérence entre
    la matrice (Dixon-Coles booste les nuls + inversion λ approchée) et les cotes —
    sinon un optimiseur "exploite" un faux edge sur le résultat sur-représenté.
    """
    cur = {"H": 0.0, "D": 0.0, "A": 0.0}
    for (i, j), p in matrix.items():
        cur[result_of(i, j)] += p
    target = {"H": p_home, "D": p_draw, "A": p_away}
    factor = {r: (target[r] / cur[r] if cur[r] > 1e-12 else 0.0) for r in cur}
    m2 = {k: matrix[k] * factor[result_of(k[0], k[1])] for k in matrix}
    s = sum(m2.values())
    return {k: v / s for k, v in m2.items()} if s > 0 else matrix

def outcome_probs(matrix):
    """P(home win), P(draw), P(away win) à partir de la matrice de scores."""
    ph = pd = pa = 0.0
    for (i, j), p in matrix.items():
        if i > j:
            ph += p
        elif i == j:
            pd += p
        else:
            pa += p
    return ph, pd, pa

# --------------------------------------------------------------------------
# De-margin des cotes 1X2 -> probabilités vraies
# --------------------------------------------------------------------------
def demargin(c_home, c_draw, c_away, method="proportional", k=None):
    """
    Retire la marge bookmaker des cotes décimales -> probas normalisées.
    'proportional': normalisation simple (rapide, robuste).
    'power': réduit le biais favori-outsider (favorite-longshot bias).
      k fourni -> applique directement p_i = inv_i^k / sum(inv_j^k) sans bissection.
      k absent  -> bissection pour trouver k tel que sum(p_i^k)=1.
    'shin': méthode de Shin (modélise la part de parieurs informés z) — état de l'art
            pour retirer la marge en corrigeant le biais favori-outsider.
    """
    inv = [1.0 / c_home, 1.0 / c_draw, 1.0 / c_away]
    overround = sum(inv)
    if method == "shin":
        book = overround
        def probs(z):
            d = 2.0 * (1.0 - z) if z < 1 else 1e-9
            return [(math.sqrt(z * z + 4.0 * (1.0 - z) * (pi * pi) / book) - z) / d for pi in inv]
        # bisection sur z dans [0,1) pour que sum(probs)=1 (sum décroît avec z)
        lo, hi = 0.0, 0.9999
        for _ in range(80):
            z = (lo + hi) / 2
            if sum(probs(z)) > 1.0:
                lo = z
            else:
                hi = z
        ps = probs((lo + hi) / 2)
        s = sum(ps)
        return [p / s for p in ps]
    if method == "power":
        if k is not None:
            # direct application — no bisection needed
            ps = [p ** k for p in inv]
            s = sum(ps)
            return [p / s for p in ps]
        # bisect for k such that sum(inv_i^k) = 1
        lo, hi = 0.5, 1.5
        for _ in range(60):
            k = (lo + hi) / 2
            s = sum(p ** k for p in inv)
            if s > 1:
                lo = k
            else:
                hi = k
        k = (lo + hi) / 2
        ps = [p ** k for p in inv]
        s = sum(ps)
        return [p / s for p in ps]
    return [p / overround for p in inv]

# --------------------------------------------------------------------------
# Inversion: (P_home, P_draw, P_away) -> (lambda_home, lambda_away)
# --------------------------------------------------------------------------
# calibrated on 11 GW1 matches with real odds, error /4.4 vs proportional
FALLBACK_POWER_K = 1.27

def solve_lambdas(p_home, p_draw, p_away, total_hint=None, rho=-0.13):
    """
    Recherche (lh, la) qui reproduit les probabilités 1X2 cibles sous Dixon-Coles.
    Grid search 2 étages (grossier puis fin) — robuste sans scipy.
    total_hint: if provided, soft constraint on expected total goals (weight 0.01
    so 1X2 dominates the fit).
    """
    def err(lh, la):
        m = score_matrix(lh, la, rho, max_goals=8)
        ph, pd, pa = outcome_probs(m)
        e = (ph - p_home) ** 2 + (pd - p_draw) ** 2 + (pa - p_away) ** 2
        if total_hint is not None:
            delta = (lh + la) - total_hint
            # Soft pull toward hint only within ±1 goal; no gradient beyond — trust 1X2 when hint diverges.
            e += 0.01 * min(delta * delta, 1.0)
        return e

    def grid(lo, hi, step):
        best, berr = (1.0, 1.0), 1e9
        n = int((hi - lo) / step) + 1
        for a in range(n):
            for b in range(n):
                lh = lo + a * step
                la = lo + b * step
                if lh <= 0 or la <= 0:
                    continue
                e = err(lh, la)
                if e < berr:
                    berr, best = e, (lh, la)
        return best, berr

    (lh, la), _ = grid(0.1, 4.5, 0.10)
    # raffinage local
    (lh, la), e = grid_local(err, lh, la, span=0.12, step=0.02)
    (lh, la), e = grid_local(err, lh, la, span=0.03, step=0.005)
    return lh, la

def grid_local(err, lh0, la0, span, step):
    best, berr = (lh0, la0), err(lh0, la0)
    n = int(2 * span / step) + 1
    for a in range(n):
        for b in range(n):
            lh = lh0 - span + a * step
            la = la0 - span + b * step
            if lh <= 0 or la <= 0:
                continue
            e = err(lh, la)
            if e < berr:
                berr, best = e, (lh, la)
    return best, berr

# --------------------------------------------------------------------------
# Popularité conditionnelle & rareté
# --------------------------------------------------------------------------
def result_of(i, j):
    return "H" if i > j else ("D" if i == j else "A")

def conditional_popularity(matrix_keys, popularity_override=None):
    """
    Renvoie (share, capped_rarity_bonus) — deux callables (i,j) -> float.
    share(i,j): part estimée des joueurs ayant le score (i,j) PARMI ceux ayant
    le bon résultat. Utilise l'override communautaire si fourni.
    capped_rarity_bonus(i,j): bonus rareté capé PAR SCORE selon la qualité des données :
      score présent dans popularity_override -> aucun cap (100 possible, données réelles)
      score dans BASE_SCORE_POPULARITY mais sans override -> cap 70 (prior synthétique :
        on ne revendique pas le palier ultra-rare <0,5% sans vraie distribution communautaire)
      score absent des deux -> cap 30 (hors-prior, données très limitées)
    Les clés de popularity_override peuvent être des strings "i-j" (format JSON)
    ou des tuples (i,j) — normalisées en tuples une seule fois à l'entrée.
    """
    # Normalise les clés override en tuples (i,j) — JSON ne permet pas de tuples.
    overridden_scores = set()
    pop = dict(BASE_SCORE_POPULARITY)
    if popularity_override:
        for k, v in popularity_override.items():
            if isinstance(k, str) and "-" in k:
                parts = k.split("-")
                key = (int(parts[0]), int(parts[1]))
            elif isinstance(k, (list, tuple)):
                key = (int(k[0]), int(k[1]))
            else:
                key = k
            pop[key] = v
            overridden_scores.add(key)

    def raw(i, j):
        return pop.get((i, j), POP_FLOOR)

    def in_base_prior(i, j):
        return (i, j) in BASE_SCORE_POPULARITY

    # masse par résultat
    mass = {"H": 0.0, "D": 0.0, "A": 0.0}
    for (i, j) in matrix_keys:
        mass[result_of(i, j)] += raw(i, j)

    def share(i, j):
        r = result_of(i, j)
        denom = mass[r] if mass[r] > 0 else 1.0
        return raw(i, j) / denom

    def capped_rarity_bonus(i, j):
        s = share(i, j)
        bonus = rarity_bonus(s)
        if (i, j) in overridden_scores:
            # Real community data for this score — no cap, top tier 100 is valid
            pass
        elif not in_base_prior(i, j):
            # Score absent from prior entirely — cap hard
            bonus = min(bonus, 30)
        else:
            # Score in prior but no real community data — cap below top (ultra-rare) tier
            bonus = min(bonus, 70)
        return bonus

    return share, capped_rarity_bonus

# --------------------------------------------------------------------------
# Remplissage field_bets partiel
# --------------------------------------------------------------------------
def fill_field_bets(fb, model_probs):
    """
    Complète un dict field_bets partiellement connu {'H','D','A'} -> dict complet.
    Jambes None : remplies au prorata des model_probs sur le reliquat (1 - somme_connues).
    Jambes négatives : clampées à 0. Somme connue >= 1 : jambes None <- model_probs, renormalisé.
    Toutes None : retourne model_probs directement.
    Résultat toujours normalisé à 1.
    """
    # Clamp known values to [0, 1]
    result = {k: (max(0.0, min(1.0, v)) if v is not None else None) for k, v in fb.items()}
    known_sum = sum(v for v in result.values() if v is not None)
    missing_keys = [k for k, v in result.items() if v is None]
    if not missing_keys:
        # All present — just normalize
        total = sum(result.values())
        if total > 0:
            return {k: v / total for k, v in result.items()}
        return dict(model_probs)
    if not any(v is not None for v in result.values()):
        # All None — use model probs; if model_probs is all zeros, fall back to uniform.
        s = sum(model_probs.values())
        if s > 0:
            return dict(model_probs)
        return {"H": 1/3, "D": 1/3, "A": 1/3}
    if known_sum >= 1.0:
        # Known legs already sum to >= 1: fill missing with model_probs, then normalize
        for k in missing_keys:
            result[k] = model_probs[k]
    else:
        remainder = max(0.0, 1.0 - known_sum)
        model_missing_sum = sum(model_probs[k] for k in missing_keys)
        for k in missing_keys:
            result[k] = (model_probs[k] / model_missing_sum * remainder) if model_missing_sum > 0 else remainder / len(missing_keys)
    total = sum(result.values())
    if total > 0:
        return {k: v / total for k, v in result.items()}
    return dict(model_probs)


# --------------------------------------------------------------------------
# Optimisation par match
# --------------------------------------------------------------------------
def evaluate_match(match, posture="aggressive", popularity_override=None):
    """
    match = {
      "id": ..., "home": "...", "away": "...",
      # POINTS pool gagnés si résultat correct (= quotations du pool, échelle ~15-222):
      "points_home": float, "points_draw": float, "points_away": float,
      # VRAIES cotes décimales bookmaker (pour les probabilités). Optionnel mais
      # FORTEMENT recommandé: c'est la source d'alpha. À défaut, on dérive les
      # probas des points du pool (moins précis: beaucoup de pools compriment les favoris).
      "odds_home": float|None, "odds_draw": float|None, "odds_away": float|None,
      "total_hint": float|None,   # somme de buts attendue (ligne over/under book)
      "field_bets": {"home","draw","away"}|None,  # répartition votes ligue (stats.bets)
      "rho": float|None
    }
    Retourne le détail + le meilleur prono selon EV et selon leverage.
    """
    points = {"H": match["points_home"], "D": match["points_draw"], "A": match["points_away"]}
    mid = match.get("id", "unknown")
    # Quotations must be positive numbers — zero or negative cause ZeroDivisionError or complex probas.
    for label, val in (("points_home", points["H"]), ("points_draw", points["D"]), ("points_away", points["A"])):
        if not (isinstance(val, (int, float)) and val > 0):
            raise SystemExit(f"Invalid quotation {label}={val!r} for match {mid} — must be a number > 0")
    rho = match.get("rho", -0.13)
    # Bookmaker odds take priority — they are the alpha source.
    # Require all three odds present AND > 1.0 (sentinel for missing/corrupt values).
    raw_odds = (match.get("odds_home"), match.get("odds_draw"), match.get("odds_away"))
    use_bookmaker = all(o is not None and o > 1.0 for o in raw_odds)
    if not use_bookmaker and any(o is not None for o in raw_odds):
        print(f"WARNING: partial odds for {match.get('id')} — values {raw_odds} — falling back to pool quotations (alpha lost)", file=sys.stderr)
    if use_bookmaker:
        p_home, p_draw, p_away = demargin(raw_odds[0], raw_odds[1], raw_odds[2], method="shin")
        prob_source = "bookmaker"
    else:
        # Fallback: power de-margin on pool quotations reduces favourite-longshot bias vs proportional.
        p_home, p_draw, p_away = demargin(points["H"], points["D"], points["A"],
                                          method="power", k=FALLBACK_POWER_K)
        prob_source = "pool_quotations"
    # Ajustement enjeu J3 (cf. j3_context.py) : booste P(nul) puis renormalise. >1 = nul
    # arrangeant/rotations ; <1 = course aux buts. Neutre (1.0) ou absent = sans effet.
    draw_boost = match.get("draw_boost", 1.0)
    if draw_boost and draw_boost != 1.0 and draw_boost > 0:
        p_draw *= draw_boost
        s = p_home + p_draw + p_away
        if s > 0:
            p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    lh, la = solve_lambdas(p_home, p_draw, p_away, match.get("total_hint"), rho)
    matrix = score_matrix(lh, la, rho, max_goals=10)
    matrix = calibrate_to_outcomes(matrix, p_home, p_draw, p_away)
    ph, pd, pa = outcome_probs(matrix)
    presult = {"H": ph, "D": pd, "A": pa}
    # Répartition des votes du field (stats de popularité du field) par résultat — sert à la
    # leverage: jouer un résultat peu choisi par la ligue = différenciation.
    fb = match.get("field_bets") or {}
    field = fill_field_bets(
        {"H": fb.get("home"), "D": fb.get("draw"), "A": fb.get("away")},
        presult,
    )
    share_fn, capped_bonus_fn = conditional_popularity(matrix.keys(), popularity_override)

    cands = []
    for (i, j), pexact in matrix.items():
        if i > 8 or j > 8:
            continue
        r = result_of(i, j)
        share_est_val = share_fn(i, j)      # popularité du SCORE exact (communauté de joueurs)
        bonus = capped_bonus_fn(i, j)
        base_ev = points[r] * presult[r]    # points du pool * proba vraie du résultat
        bonus_ev = bonus * pexact
        ev = base_ev + bonus_ev
        contrarian_result = (1.0 - field.get(r, 0.33))      # 0..1, haut = peu couru
        # diff_score (sélection agressive) : EV + prime à l'UPSIDE du bonus rare.
        # On NE flippe PAS vers un résultat improbable — la prime porte sur le choix
        # du score exact rare-mais-plausible, pas sur le résultat.
        diff_score = ev + 0.8 * bonus_ev
        # contrarian_value (levier SLATE-level, pour choisir où dévier sciemment) :
        # gain attendu d'un pari peu couru = (1-popularité résultat) * EV * proba résultat.
        contrarian_value = contrarian_result * ev * presult[r]
        cands.append({
            "score": f"{i}-{j}", "i": i, "j": j, "result": r,
            "p_exact": round(pexact, 4), "p_result": round(presult[r], 4),
            "field_result": round(field.get(r, 0.0), 3),
            "share_est": round(share_est_val, 4), "bonus": bonus,
            "base_ev": round(base_ev, 3), "bonus_ev": round(bonus_ev, 3),
            "ev": round(ev, 3), "diff_score": round(diff_score, 3),
            "contrarian_value": round(contrarian_value, 3),
        })

    cands.sort(key=lambda c: c["ev"], reverse=True)
    best_ev = cands[0]
    if posture == "aggressive":
        # Vraies cotes : EV significative. Parmi les scores à forte EV (<=10% du max)
        # et plausibles, prendre celui qui maximise EV + upside bonus rare.
        max_ev = best_ev["ev"]
        pool = [c for c in cands
                if c["ev"] >= max_ev - 0.10 * abs(max_ev) and c["p_exact"] >= 0.015]
        if not pool:
            pool = cands[:5]
        best_play = max(pool, key=lambda c: c["diff_score"])
    else:
        best_play = best_ev
    # meilleur pari contrarien du match (pour la sélection slate-level des leviers)
    best_contrarian = max(cands, key=lambda c: c["contrarian_value"])

    return {
        "id": match.get("id"), "home": match["home"], "away": match["away"],
        "prob_source": prob_source,
        "lambda_home": round(lh, 3), "lambda_away": round(la, 3),
        "p_home": round(ph, 3), "p_draw": round(pd, 3), "p_away": round(pa, 3),
        "expected_total": round(lh + la, 2),
        "points": {"H": points["H"], "D": points["D"], "A": points["A"]},
        "pick_ev": best_ev["score"], "pick_recommended": best_play["score"],
        "pick_result": best_play["result"],
        "ev_recommended": best_play["ev"],
        # candidat recommandé en entier : sert à la porte de qualité du x2
        # (p_result = confiance, share_est = ownership du score, prob_source = fraîcheur cotes).
        "recommended": {"score": best_play["score"], "result": best_play["result"],
                        "p_result": best_play["p_result"], "p_exact": best_play["p_exact"],
                        "share_est": best_play["share_est"], "ev": best_play["ev"]},
        "contrarian_play": {"score": best_contrarian["score"], "result": best_contrarian["result"],
                            "field_result": best_contrarian["field_result"],
                            "ev": best_contrarian["ev"], "value": best_contrarian["contrarian_value"]},
        "top_candidates": cands[:6],
    }

def _x2_gain(r):
    """Gain marginal espéré FIABLE de doubler le prono recommandé d'un match :
    EV du prono pondérée par la confiance dans le résultat (anti-gâchis)."""
    confidence = r["top_candidates"][0]["p_result"]
    return r["ev_recommended"] * (0.5 + confidence)


# Ratio de "standout" exigé du candidat (gain du meilleur match / gain médian de la slate)
# pour POSER le x2, selon la fraction du tournoi déjà écoulée. Décroît : tôt il faut un coup
# qui DOMINE nettement sa journée (favori net + bon exact rare) ; tard on accepte un standout
# modeste ; à la dernière fenêtre on pose quoi qu'il arrive (géré séparément).
# Stratégie : ne pas brûler le x2 en poules sauf coup exceptionnel ;
# fenêtre idéale = phases finales (affiches déséquilibrées). Avec UNE slate observée, le vrai
# arrêt optimal inter-journées dégénère (le candidat est le sommet de sa propre loi) → on
# encode la patience par un calendrier transparent plutôt qu'un seuil empirique trompeur.
X2_STANDOUT_SCHEDULE = [
    (0.33, 1.8),   # premier tiers du tournoi : exiger un net standout (×1.8 la médiane)
    (0.66, 1.4),   # deuxième tiers : standout modéré
    (1.00, 1.15),  # dernier tiers : léger standout suffit
]
# Porte de qualité du x2 (triple convergence, cf. CLAUDE.md) : confiance dans le résultat,
# ownership max du score dans le field, fraîcheur des cotes (gérée à part).
X2_MIN_CONFIDENCE = 0.65
X2_MAX_OWNERSHIP = 0.20


def _required_standout(progress):
    for thresh, ratio in X2_STANDOUT_SCHEDULE:
        if progress <= thresh:
            return ratio
    return X2_STANDOUT_SCHEDULE[-1][1]


def recommend_x2(results, gameweek=None, total_gameweeks=9, holding=True):
    """
    Recommande le match porteur du x2 ET tranche poser-maintenant vs attendre.

    Sélection : le match qui maximise le gain marginal fiable (_x2_gain).
    Décision (si gameweek fourni et qu'on détient encore le x2) : calendrier de patience.
    On POSE si (a) c'est la dernière fenêtre, OU (b) le candidat DOMINE nettement sa journée
    — gain du meilleur / gain médian ≥ ratio exigé, lequel décroît au fil du tournoi. La pose
    réelle reste soumise aux garde-fous qualité de CLAUDE.md (P>0.65, ownership<20%, cotes
    fraîches) : cette décision ne fait que gérer le TIMING, pas la qualité.
    """
    if not results:
        return None
    best = max(results, key=_x2_gain)
    best_gain = _x2_gain(best)
    out = {"match": f'{best["home"]}-{best["away"]}', "id": best["id"],
           "prono": best["pick_recommended"], "x2_hint_ev_only": round(best_gain, 2),
           "note": "EV-only hint; real x2 placement via optimize_winprob"}
    if gameweek is not None and holding:
        remaining = max(1, total_gameweeks - gameweek + 1)
        gains = sorted(_x2_gain(r) for r in results)
        med = gains[len(gains) // 2] if gains else 0.0
        standout = (best_gain / med) if med > 0 else float("inf")
        progress = gameweek / total_gameweeks if total_gameweeks else 1.0
        required = _required_standout(progress)
        timing_ok = remaining <= 1 or standout >= required
        # Porte de QUALITÉ (triple convergence, CLAUDE.md) : confiance haute, score peu
        # couru par le field, cotes fraîches. Indépendante du timing.
        rec = best.get("recommended", {})
        p_result = rec.get("p_result", best["top_candidates"][0]["p_result"])
        ownership = rec.get("share_est", 1.0)
        fresh_odds = best.get("prob_source") == "bookmaker"
        quality_ok = (p_result > X2_MIN_CONFIDENCE
                      and ownership < X2_MAX_OWNERSHIP
                      and fresh_odds)
        # Verdict final autonome : poser si (timing ET qualité), OU si c'est la DERNIÈRE
        # fenêtre (use-it-or-lose-it : un x2 médiocre vaut mieux qu'un x2 jamais posé = 0).
        place = bool(remaining <= 1 or (timing_ok and quality_ok))
        out.update({
            "decision": "POSER" if place else "ATTENDRE",
            "place_now": place,
            "timing_ok": timing_ok,
            "quality_ok": quality_ok,
            "standout_ratio": (round(standout, 2) if standout != float("inf") else None),
            "required_standout": (None if remaining <= 1 else required),
            "remaining_gameweeks": remaining,
            "quality_gate": {
                "p_result": round(p_result, 3), "min_confidence": X2_MIN_CONFIDENCE,
                "ownership_est": round(ownership, 3), "max_ownership": X2_MAX_OWNERSHIP,
                "fresh_odds": fresh_odds,
            },
            "rationale": _x2_rationale(timing_ok, quality_ok, remaining, standout, required,
                                       p_result, ownership, fresh_odds),
        })
    return out


def _x2_rationale(timing_ok, quality_ok, remaining, standout, required, p_result, ownership, fresh_odds):
    if timing_ok and quality_ok:
        return "POSER : timing OK + triple convergence (confiance, ownership bas, cotes fraîches)"
    reasons = []
    if not timing_ok:
        reasons.append(f"timing trop tôt (standout {standout:.2f} < {required})")
    if not quality_ok:
        if p_result <= X2_MIN_CONFIDENCE:
            reasons.append(f"confiance {p_result:.2f} ≤ {X2_MIN_CONFIDENCE}")
        if ownership >= X2_MAX_OWNERSHIP:
            reasons.append(f"score trop couru ({ownership:.2f} ≥ {X2_MAX_OWNERSHIP})")
        if not fresh_odds:
            reasons.append("cotes non fraîches (fallback)")
    return "ATTENDRE : " + " ; ".join(reasons)

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    if sys.stdin.isatty():
        print("Usage: python model.py < enriched.json", file=sys.stderr)
        sys.exit(1)
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON on stdin: {e}")
    if "matches" not in data:
        raise SystemExit('Key "matches" not found — expected {"matches": [...], "posture": "aggressive"}')
    posture = data.get("posture", "aggressive")
    pop_override = data.get("popularity_override")
    results = [evaluate_match(m, posture, pop_override) for m in data["matches"]]
    # gameweek/holding optionnels : activent la décision poser/attendre du x2.
    x2 = recommend_x2(results, gameweek=data.get("gameweek"),
                      total_gameweeks=data.get("total_gameweeks", 9),
                      holding=data.get("x2_holding", True))
    out = {"posture": posture, "matches": results, "x2": x2}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
