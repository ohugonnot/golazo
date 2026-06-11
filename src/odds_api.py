#!/usr/bin/env python3
"""
Client The Odds API (the-odds-api.com) — récupère les vraies cotes 1X2 + over/under
et les fusionne dans un fichier d'entrée de l'engine (matching par nom d'équipe).

Supprime la saisie manuelle des cotes. Stdlib pure.

Clé API : variable d'environnement ODDS_API_KEY, sinon fichier local .odds_api_key
(gitignoré). Tier gratuit 500 crédits/mois (h2h+totals × région eu = 2 crédits par appel,
1 appel = tous les matchs à venir). Sport key configurable via ODDS_API_SPORT.

Sous-commandes :
  fetch                       -> JSON brut de l'API (matchs à venir, 1X2 + O/U)
  enrich <input.json>         -> fusionne les cotes fraîches DANS ce fichier (par nom d'équipe),
                                 backup préalable, n'écrase QUE odds_* / total_hint / odds_source
  quota                       -> crédits restants (lus dans les headers de la dernière réponse)

Timing : appeler juste avant le coup d'envoi = closing odds, supérieures aux cotes J-2
(les compositions tombent ~24-48h avant le match).
"""
import json, os, sys, time, unicodedata, urllib.request, urllib.error, urllib.parse

if hasattr(sys.stdout, 'reconfigure') and (sys.stdout.encoding or '').lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "..", ".odds_api_key")   # fallback local, gitignoré
BASE = "https://api.the-odds-api.com/v4"
SPORT = os.environ.get("ODDS_API_SPORT", "soccer_fifa_world_cup")
# region eu = bookmakers européens (Pinnacle, bet365…) ; markets h2h=1X2, totals=O/U.
REGIONS = "eu"
MARKETS = "h2h,totals"
# Dernier header quota vu (rempli par _get), pour la sous-commande `quota`.
_last_headers = {}


def api_key():
    k = os.environ.get("ODDS_API_KEY", "").strip()
    if k:
        return k
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            k = f.read().strip()
    except FileNotFoundError:
        raise SystemExit(
            "Clé The Odds API introuvable.\n"
            "Créer une clé gratuite sur https://the-odds-api.com/#get-access (email only), puis :\n"
            "  export ODDS_API_KEY=la_cle   (ou : echo 'la_cle' > .odds_api_key)")
    if not k:
        raise SystemExit(f"{KEY_FILE} est vide — y coller la clé The Odds API.")
    return k


def _get(path, params):
    """GET JSON avec gestion 401/429/quota. Retourne (data, headers)."""
    global _last_headers
    q = urllib.parse.urlencode(params)
    url = f"{BASE}{path}?{q}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "golazo/1.0"})
            r = urllib.request.urlopen(req, timeout=20)
            _last_headers = {k.lower(): v for k, v in r.headers.items()}
            raw = r.read().decode("utf-8", "replace")
            return (json.loads(raw) if raw.strip() else None), _last_headers
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 429 and attempt < 3:
                wait = (5, 15, 30)[attempt]
                print(f"HTTP 429 (quota/rate) — attente {wait}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 401:
                raise SystemExit("HTTP 401 — clé The Odds API invalide ou expirée.")
            if e.code == 422:
                raise SystemExit(f"HTTP 422 — paramètres refusés: {body}")
            raise SystemExit(f"HTTP {e.code} sur {path}: {body}")
        except urllib.error.URLError as e:
            raise SystemExit(f"Erreur réseau: {e.reason}")
    raise SystemExit("429 persistant — quota épuisé, réessayer plus tard.")


def fetch():
    """Renvoie la liste brute des matchs CdM à venir avec cotes 1X2 + O/U."""
    data, _ = _get(f"/sports/{SPORT}/odds", {
        "apiKey": api_key(), "regions": REGIONS, "markets": MARKETS,
        "oddsFormat": "decimal", "dateFormat": "iso",
    })
    return data or []


# Nos fichiers utilisent les noms FR, l'API les noms EN. Table d'alias FR -> nom canonique
# (= nom EN de l'API). Couvre les 48 équipes CdM 2026. Clé = nom FR brut (sera normalisé).
TEAM_ALIASES_FR = {
    "Afrique du Sud": "South Africa", "Algérie": "Algeria", "Allemagne": "Germany",
    "Angleterre": "England", "Arabie saoudite": "Saudi Arabia", "Argentine": "Argentina",
    "Australie": "Australia", "Autriche": "Austria", "Belgique": "Belgium",
    "Bosnie": "Bosnia & Herzegovina", "Brésil": "Brazil", "Cap-Vert": "Cape Verde",
    "Colombie": "Colombia", "Corée du Sud": "South Korea", "Croatie": "Croatia",
    "Côte d’Ivoire": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast", "Espagne": "Spain",
    "Haïti": "Haiti", "Irak": "Iraq", "Japon": "Japan", "Jordanie": "Jordan",
    "Maroc": "Morocco", "Mexique": "Mexico", "Norvège": "Norway",
    "Nouvelle-Zélande": "New Zealand", "Ouzbékistan": "Uzbekistan", "Pays-Bas": "Netherlands",
    "RD Congo": "DR Congo", "Suisse": "Switzerland", "Suède": "Sweden", "Sénégal": "Senegal",
    "Tchéquie": "Czech Republic", "Tunisie": "Tunisia", "Turquie": "Turkey",
    "Écosse": "Scotland", "Égypte": "Egypt", "Équateur": "Ecuador", "États-Unis": "USA",
}


def _norm(name):
    """Normalise un nom d'équipe pour le matching (sans accents, minuscule, alnum).
    Applique d'abord la table d'alias FR->EN pour que 'Corée du Sud' == 'South Korea'."""
    if not name:
        return ""
    canonical = TEAM_ALIASES_FR.get(name.strip(), name)
    s = unicodedata.normalize("NFKD", canonical).encode("ascii", "ignore").decode("ascii")
    return "".join(c for c in s.lower() if c.isalnum())


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def consensus_odds(event):
    """Agrège les bookmakers d'un événement -> dict {home,draw,away,total_hint} (médianes).
    home/away/draw = cotes 1X2 ; total_hint = ligne over/under la plus cotée (point pivot)."""
    h, d, a = [], [], []
    totals_points = {}  # point -> liste des cotes 'Over' (pour choisir la ligne pivot ~50/50)
    home_name, away_name = event.get("home_team"), event.get("away_team")
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = mk.get("key")
            if key == "h2h":
                for oc in mk.get("outcomes", []):
                    nm = oc.get("name"); price = oc.get("price")
                    if nm == home_name:
                        h.append(price)
                    elif nm == away_name:
                        a.append(price)
                    elif nm and _norm(nm) in ("draw", "tie"):
                        d.append(price)
            elif key == "totals":
                for oc in mk.get("outcomes", []):
                    if oc.get("name") == "Over" and oc.get("point") is not None:
                        totals_points.setdefault(oc["point"], []).append(oc.get("price"))
    # ligne over/under pivot = celle dont la cote Over médiane est la plus proche de 2.0 (~50/50)
    total_hint = None
    if totals_points:
        best_pt, best_dist = None, 1e9
        for pt, prices in totals_points.items():
            med = _median(prices)
            if med is None:
                continue
            dist = abs(med - 2.0)
            if dist < best_dist:
                best_dist, best_pt = dist, pt
        total_hint = best_pt
    return {
        "home": _median(h), "draw": _median(d), "away": _median(a),
        "total_hint": total_hint,
        "n_books": len(event.get("bookmakers", [])),
    }


def enrich(enriched_path):
    """Fusionne les cotes fraîches dans enriched_path (matching par nom d'équipe).
    Écrit un backup, ne touche QUE odds_home/draw/away, total_hint, odds_source.
    Retourne un rapport (matchs mis à jour / non trouvés)."""
    with open(enriched_path, encoding="utf-8") as f:
        doc = json.load(f)
    matches = doc.get("matches", doc if isinstance(doc, list) else [])
    events = fetch()
    # index des events par paire de noms normalisés (ordre indépendant)
    idx = {}
    for ev in events:
        key = frozenset((_norm(ev.get("home_team")), _norm(ev.get("away_team"))))
        idx[key] = ev
    report = {"updated": [], "not_found": [], "events_seen": len(events)}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for m in matches:
        key = frozenset((_norm(m.get("home")), _norm(m.get("away"))))
        ev = idx.get(key)
        if not ev:
            report["not_found"].append(f'{m.get("home")}-{m.get("away")}')
            continue
        co = consensus_odds(ev)
        if not all(co[k] and co[k] > 1.0 for k in ("home", "draw", "away")):
            report["not_found"].append(f'{m.get("home")}-{m.get("away")} (cotes 1X2 incomplètes)')
            continue
        m["odds_home"], m["odds_draw"], m["odds_away"] = co["home"], co["draw"], co["away"]
        if co["total_hint"]:
            m["total_hint"] = co["total_hint"]
        m["odds_source"] = f'the-odds-api/{co["n_books"]}books/{stamp}'
        report["updated"].append(f'{m.get("home")}-{m.get("away")}')
    if report["updated"]:
        backup = f"{enriched_path}.bak_{stamp}"
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        report["backup"] = backup
    return report


def quota():
    """Crédits restants/usés d'après les headers de la dernière réponse (déclenche un fetch léger)."""
    fetch()
    return {
        "remaining": _last_headers.get("x-requests-remaining"),
        "used": _last_headers.get("x-requests-used"),
        "last_cost": _last_headers.get("x-requests-last"),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "fetch":
        print(json.dumps(fetch(), ensure_ascii=False, indent=2))
    elif cmd == "enrich":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: odds_api.py enrich <enriched.json>")
        print(json.dumps(enrich(sys.argv[2]), ensure_ascii=False, indent=2))
    elif cmd == "quota":
        print(json.dumps(quota(), ensure_ascii=False, indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
