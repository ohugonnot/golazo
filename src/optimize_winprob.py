#!/usr/bin/env python3
"""
Optimiseur Monte-Carlo "maximiser P(finir 1er)" pour un pool de pronostics.

Au lieu d'optimiser l'EV match par match, on maximise directement la PROBABILITÉ
DE FINIR 1er de la journée sur l'ensemble des joueurs — l'objectif correct pour
gagner un contest (maximiser l'espérance ≠ maximiser P(gagner)).

Méthode :
  1. Par match : matrice Dixon-Coles des scores (vraies cotes si dispo, sinon quotations).
  2. Modèle d'adversaire : chaque rival tire un score ~ popularité du field
     (résultat via la popularité réelle ; score exact via prior conditionnel). Override
     possible par des profils si connus.
  3. Simulation : N tirages des vrais scores + des pronos adverses -> points de chacun
     (barème du pool : points cote si bon résultat + bonus rareté si score exact).
  4. Mes pronos : ascension de coordonnées — pour chaque match, on teste tous les scores
     plausibles et on garde celui qui maximise P(je finis 1er), en gelant les autres. 2-3 passes.
  5. x2 : on teste de doubler chaque match, on garde le meilleur pour P(1er).

Sortie : pronos optimisés, P(1er) avant/après, reco x2, et comparaison au prono courant.
Usage : python optimize_winprob.py < input.json  [--sims 4000] [--current current.json]
Le nombre d'adversaires est lu dans l'entrée JSON (clé "n_opponents", défaut 11).
Stdlib uniquement.
"""
import json, sys, random, math
import model as M

if hasattr(sys.stdout, 'reconfigure') and (sys.stdout.encoding or '').lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

random.seed(12345)  # reproductible

DEFAULT_N_OPP = 11  # adversaires par défaut (surchargé par la clé "n_opponents" de l'entrée)
MAXG = 10           # must match model.py score_matrix default, else 1X2 parity breaks

def build_match(m, popularity_override=None):
    """Retourne (matrix, points, field_result, score_pop_conditional)."""
    points = {"H": m["points_home"], "D": m["points_draw"], "A": m["points_away"]}
    rho = m.get("rho", -0.13)
    if m.get("odds_home") and m.get("odds_draw") and m.get("odds_away"):
        p = M.demargin(m["odds_home"], m["odds_draw"], m["odds_away"], method="shin")
        src = "REAL"
    else:
        # Mirror model.py fallback — single source of truth for the power constant.
        p = M.demargin(points["H"], points["D"], points["A"], method="power", k=M.FALLBACK_POWER_K)
        src = "pool"
    lh, la = M.solve_lambdas(p[0], p[1], p[2], m.get("total_hint"), rho)
    matrix = M.score_matrix(lh, la, rho, max_goals=MAXG)
    matrix = M.calibrate_to_outcomes(matrix, p[0], p[1], p[2])
    # résultat field : stats.bets réel si dispo, sinon proba modèle.
    # Single source of truth: delegate to model.fill_field_bets (same logic as evaluate_match).
    fb = m.get("field_bets") or {}
    ph, pd, pa = M.outcome_probs(matrix)
    fr = M.fill_field_bets(
        {"H": fb.get("home"), "D": fb.get("draw"), "A": fb.get("away")},
        {"H": ph, "D": pd, "A": pa},
    )
    share, capped_bonus_fn = M.conditional_popularity(matrix.keys(), popularity_override)
    return {"matrix": matrix, "points": points, "field_result": fr,
            "share": share, "capped_bonus_fn": capped_bonus_fn, "src": src, "lh": lh, "la": la}

def cells_and_cum(prob_by_cell):
    cells = list(prob_by_cell.keys())
    w = [prob_by_cell[c] for c in cells]
    tot = sum(w); w = [x/tot for x in w]
    cum = []
    acc = 0.0
    for x in w:
        acc += x; cum.append(acc)
    return cells, cum

def draw(cells, cum):
    r = random.random()
    lo, hi = 0, len(cum)-1
    while lo < hi:
        mid = (lo+hi)//2
        if r <= cum[mid]: hi = mid
        else: lo = mid+1
    return cells[lo]

def opp_pick_dist(mi):
    """Distribution de prono d'un adversaire pour ce match : P(score) = P(résultat field)
    * popularité conditionnelle du score. Renvoie (cells, cum)."""
    fr = mi["field_result"]; share = mi["share"]
    dist = {}
    for (i, j) in mi["matrix"].keys():
        r = M.result_of(i, j)
        dist[(i, j)] = max(fr.get(r, 0.0), 0.0) * share(i, j)
    return cells_and_cum(dist)

def pts(points, capped_bonus_fn, pick, actual):
    """Points pool d'un prono `pick` face au vrai score `actual`."""
    pi, pj = pick; ai, aj = actual
    r_pick = M.result_of(pi, pj); r_act = M.result_of(ai, aj)
    if r_pick != r_act:
        return 0.0
    base = points[r_pick]
    if pi == ai and pj == aj:
        return base + capped_bonus_fn(pi, pj)
    return base

def candidate_scores(mi, cap=0.012):
    """Scores plausibles à optimiser (proba cellule >= cap)."""
    cs = [c for c, p in mi["matrix"].items() if p >= cap]
    # garantir présence de 0-0, 1-0, 0-1, 1-1, 2-0, 0-2, 2-1, 1-2
    for c in [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2)]:
        if c not in cs: cs.append(c)
    return cs

def main():
    raw = sys.stdin.read()
    data = json.loads(raw)
    sims = 4000
    current_path = None
    objective = "ev"   # "ev" (esp. de points, défaut sain) | "p1" (P(1er) 1 manche, max agressif)
    kappa = 0.0        # blend: objectif = E[points] + kappa * P(1er) * 100 (si objective='blend')
    args = sys.argv[1:]
    for k in range(len(args)):
        if args[k] == "--sims": sims = int(args[k+1])
        if args[k] == "--current": current_path = args[k+1]
        if args[k] == "--objective": objective = args[k+1]
        if args[k] == "--kappa": kappa = float(args[k+1])
    matches = data["matches"]
    pop_override = data.get("popularity_override")
    n_opp = int(data.get("n_opponents", DEFAULT_N_OPP))
    mis = [build_match(m, pop_override) for m in matches]
    nm = len(matches)

    # pré-tirage : vrais scores + pronos adverses, et points adverses par sim
    actual = [[None]*nm for _ in range(sims)]
    opp_max = [0.0]*sims
    true_cc = [cells_and_cum(mi["matrix"]) for mi in mis]
    opp_cc = [opp_pick_dist(mi) for mi in mis]
    for s in range(sims):
        # vrais scores
        for mm in range(nm):
            actual[s][mm] = draw(*true_cc[mm])
        # adversaires
        best = 0.0
        for _ in range(n_opp):
            tot = 0.0
            for mm in range(nm):
                pick = draw(*opp_cc[mm])
                tot += pts(mis[mm]["points"], mis[mm]["capped_bonus_fn"], pick, actual[s][mm])
            if tot > best: best = tot
        opp_max[s] = best

    # pronos de départ : prono courant si fourni, sinon mode de la matrice
    start = {}
    if current_path:
        try:
            with open(current_path, encoding="utf-8-sig") as f:
                cur = json.load(f)
        except FileNotFoundError:
            raise SystemExit(f"--current file not found: {current_path}")
        except json.JSONDecodeError as e:
            raise SystemExit(f"--current file is not valid JSON: {e}")
        if not isinstance(cur, list) or (cur and not isinstance(cur[0], dict)):
            raise SystemExit('--current file must be a JSON array of {"id","h","a"} objects')
        try:
            by_id = {c["id"]: (c["h"], c["a"]) for c in cur}
        except KeyError as e:
            raise SystemExit(f'--current entries must have keys "id", "h", "a" — missing key {e}')
        for mm in range(nm):
            start[mm] = by_id.get(matches[mm]["id"]) or max(mis[mm]["matrix"], key=mis[mm]["matrix"].get)
    else:
        for mm in range(nm):
            start[mm] = max(mis[mm]["matrix"], key=mis[mm]["matrix"].get)

    mypick = dict(start)

    def my_total_array(pick_override=None):
        arr = [0.0]*sims
        for s in range(sims):
            t = 0.0
            for mm in range(nm):
                pk = pick_override[mm] if pick_override else mypick[mm]
                t += pts(mis[mm]["points"], mis[mm]["capped_bonus_fn"], pk, actual[s][mm])
            arr[s] = t
        return arr

    def winprob(my_total):
        w = 0.0
        for s in range(sims):
            if my_total[s] > opp_max[s]: w += 1.0
            elif my_total[s] == opp_max[s]: w += 0.5
        return w / sims

    def evaluate(base_wo, mm, cand):
        """Renvoie (mean_total, winprob) si on joue `cand` sur le match mm."""
        ssum = 0.0; w = 0.0
        pts_m = mis[mm]["points"]; bonus_fn = mis[mm]["capped_bonus_fn"]
        for s in range(sims):
            t = base_wo[s] + pts(pts_m, bonus_fn, cand, actual[s][mm])
            ssum += t
            if t > opp_max[s]: w += 1.0
            elif t == opp_max[s]: w += 0.5
        return ssum / sims, w / sims

    def obj(mean_t, wp):
        if objective == "p1": return wp
        if objective == "blend": return mean_t + kappa * wp * 100.0
        return mean_t  # "ev"

    base_total = my_total_array(start)
    wp_start = winprob(base_total)
    ev_start = sum(base_total) / sims

    # ascension de coordonnées selon l'objectif choisi
    my_total = list(base_total)
    for _ in range(3):
        improved = False
        for mm in range(nm):
            cur_pick = mypick[mm]
            contrib_cur = [pts(mis[mm]["points"], mis[mm]["capped_bonus_fn"], cur_pick, actual[s][mm]) for s in range(sims)]
            base_wo = [my_total[s] - contrib_cur[s] for s in range(sims)]
            cands = candidate_scores(mis[mm])
            if cur_pick not in cands: cands.append(cur_pick)
            best_pick, best_obj, best_arr = None, -1e18, None
            for cand in cands:
                mt, wp = evaluate(base_wo, mm, cand)
                o = obj(mt, wp)
                if o > best_obj + 1e-12:
                    best_obj, best_pick = o, cand
            if best_pick != cur_pick:
                mypick[mm] = best_pick; improved = True
                my_total = [base_wo[s] + pts(mis[mm]["points"], mis[mm]["capped_bonus_fn"], best_pick, actual[s][mm]) for s in range(sims)]
        if not improved:
            break
    wp_opt = winprob(my_total); ev_opt = sum(my_total) / sims

    # placement x2 : double la contribution d'un match -> maximise l'objectif
    best_x2, best_x2_obj, best_x2_wp = None, obj(ev_opt, wp_opt), wp_opt
    for mm in range(nm):
        contrib = [pts(mis[mm]["points"], mis[mm]["capped_bonus_fn"], mypick[mm], actual[s][mm]) for s in range(sims)]
        ssum = 0.0; w = 0.0
        for s in range(sims):
            t = my_total[s] + contrib[s]
            ssum += t
            if t > opp_max[s]: w += 1.0
            elif t == opp_max[s]: w += 0.5
        o = obj(ssum / sims, w / sims)
        if o > best_x2_obj + 1e-9:
            best_x2_obj, best_x2, best_x2_wp = o, mm, w / sims

    out = {"sims": sims, "objective": objective,
           "ev_start": round(ev_start, 1), "ev_opt": round(ev_opt, 1),
           "winprob_start": round(wp_start, 4), "winprob_opt": round(wp_opt, 4),
           "winprob_with_x2": round(best_x2_wp, 4),
           "x2_match": (f'{matches[best_x2]["home"]}-{matches[best_x2]["away"]}' if best_x2 is not None else None),
           "x2_id": (matches[best_x2]["id"] if best_x2 is not None else None),
           "matches": []}
    for mm in range(nm):
        i, j = mypick[mm]; si, sj = start[mm]
        out["matches"].append({
            "id": matches[mm]["id"], "home": matches[mm]["home"], "away": matches[mm]["away"],
            "src": mis[mm]["src"], "opt_pick": f"{i}-{j}", "start_pick": f"{si}-{sj}",
            "changed": (mypick[mm] != start[mm]),
            "expected_total": round(mis[mm]["lh"] + mis[mm]["la"], 2),
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
