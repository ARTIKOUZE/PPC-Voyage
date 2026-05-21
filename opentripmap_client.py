"""Vérification OpenTripMap des activités générées par le LLM.

Le LLM propose la liste (il connaît la pertinence touristique). Pour chaque
activité, on interroge OTM `autosuggest` autour du centre-ville et on garde le
meilleur match par nom. Si trouvé : on remplace GPS + adresse par les valeurs
OTM (Wikidata/OSM, vérifiées). Sinon : on garde le résultat LLM et on tag
`verified=False`."""
from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_KEY = os.environ.get("OPENTRIPMAP_API_KEY", "").strip()
_BASE = "https://api.opentripmap.com/0.1/en"
_TIMEOUT = 12
_MATCH_THRESHOLD = 0.70

# OTM kinds → nos 5 catégories. Sert à éviter qu'un musée LLM matche un hôtel OTM.
_KIND_TO_CATEGORY = {
    "museums": "culture", "national_museums": "culture", "historic": "culture",
    "historic_architecture": "culture", "monuments_and_memorials": "culture",
    "monuments": "culture", "religion": "culture", "monasteries": "culture",
    "architecture": "culture", "palaces": "culture", "castles": "culture",
    "fortifications": "culture", "cultural": "culture", "archaeology": "culture",
    "theatres_and_entertainments": "culture", "towers": "culture",
    "gardens_and_parks": "nature", "natural": "nature", "beaches": "nature",
    "view_points": "nature", "water": "nature",
    "foods": "gastro", "restaurants": "gastro",
    "shops": "shopping", "marketplaces": "shopping",
    "amusements": "nightlife",
}

# Kinds qui invalident un match (LLM dit musée, OTM dit hôtel).
_DISQUALIFYING_KINDS = {
    "accomodations", "other_hotels", "skyscrapers", "fast_food",
    "tourist_facilities",
}

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _match_score(llm_name: str, otm_name: str) -> float:
    a, b = _norm(llm_name), _norm(otm_name)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    base = SequenceMatcher(None, a, b).ratio()
    a_tokens, b_tokens = set(a.split()), set(b.split())
    if not a_tokens:
        return base
    if a_tokens.issubset(b_tokens):
        return max(base, 0.88)
    overlap = len(a_tokens & b_tokens) / len(a_tokens)
    return base * (0.4 + 0.6 * overlap)

def _classify_otm(kinds_csv: str) -> Optional[str]:
    for k in (kinds_csv or "").split(","):
        cat = _KIND_TO_CATEGORY.get(k.strip())
        if cat:
            return cat
    return None

def _categories_compatible(llm_cat: str, otm_kinds: str) -> bool:
    """Les catégories sont compatibles si OTM n'a pas de kind disqualifiant ET que la catégorie déduite n'est pas radicalement différente."""
    kinds = set(k.strip() for k in (otm_kinds or "").split(","))
    if kinds & _DISQUALIFYING_KINDS:
        return False
    otm_cat = _classify_otm(otm_kinds)
    if otm_cat is None:
        return True  # OTM ne sait pas → on accepte
    # culture et nature sont parfois interchangeables (jardin historique, etc.)
    interchangeable = {("culture", "nature"), ("nature", "culture")}
    if otm_cat == llm_cat:
        return True
    if (llm_cat, otm_cat) in interchangeable:
        return True
    return False

def _get_with_retry(url: str, params: dict, max_tries: int = 4) -> Optional[requests.Response]:
    delay = 0.4
    for _ in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT)
            if r.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            return r
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
    return None

def _geoname(city: str) -> Optional[tuple[float, float, str]]:
    if not _API_KEY:
        return None
    r = _get_with_retry(f"{_BASE}/places/geoname",
                        {"name": city, "apikey": _API_KEY})
    if not r or r.status_code != 200:
        return None
    d = r.json()
    if d.get("status") != "OK":
        return None
    return d.get("lat"), d.get("lon"), d.get("country", "")

def _autosuggest(name: str, lat: float, lon: float,
                 radius_m: int = 20000) -> list[dict]:
    r = _get_with_retry(f"{_BASE}/places/autosuggest", {
        "name": name, "radius": radius_m, "lat": lat, "lon": lon,
        "limit": 6, "apikey": _API_KEY,
    })
    if not r or r.status_code != 200:
        return []
    feats = (r.json() or {}).get("features") or []
    out = []
    for f in feats:
        props = f.get("properties") or {}
        geom = (f.get("geometry") or {}).get("coordinates") or [None, None]
        if not props.get("name") or geom[0] is None:
            continue
        out.append({
            "name": props["name"],
            "xid": props.get("xid"),
            "kinds": props.get("kinds", ""),
            "lat": geom[1],
            "lon": geom[0],
            "rate": props.get("rate", 1),
            "wikidata": props.get("wikidata", ""),
        })
    return out

def _xid_detail(xid: str) -> Optional[dict]:
    r = _get_with_retry(f"{_BASE}/places/xid/{xid}",
                        {"apikey": _API_KEY})
    if not r or r.status_code != 200:
        return None
    return r.json()

def _format_address(addr: dict) -> str:
    if not addr:
        return ""
    parts = []
    if addr.get("house_number") and addr.get("road"):
        parts.append(f"{addr['house_number']} {addr['road']}")
    elif addr.get("road"):
        parts.append(addr["road"])
    if addr.get("postcode") and addr.get("city"):
        parts.append(f"{addr['postcode']} {addr['city']}")
    elif addr.get("city"):
        parts.append(addr["city"])
    if addr.get("country"):
        parts.append(addr["country"])
    return ", ".join(parts)

def _verify_one(act: dict, city_lat: float, city_lon: float) -> dict:
    """Verifie une activite LLM contre OTM. Renvoie une copie eventuellement enrichie."""
    out = dict(act)
    out["verified"] = False
    candidates = _autosuggest(act["name"], city_lat, city_lon)
    if not candidates:
        return out

    llm_cat = act.get("category", "culture")
    best = None
    best_score = 0.0
    for c in candidates:
        if not _categories_compatible(llm_cat, c.get("kinds", "")):
            continue
        score = _match_score(act["name"], c["name"])
        if score > best_score:
            best, best_score = c, score

    if not best or best_score < _MATCH_THRESHOLD:
        return out

    out["verified"] = True
    out["latitude"] = float(best["lat"])
    out["longitude"] = float(best["lon"])
    out["otm_xid"] = best["xid"]
    out["match_score"] = round(best_score, 2)
    if best.get("wikidata"):
        out["wikidata"] = best["wikidata"]

    if best.get("xid"):
        detail = _xid_detail(best["xid"])
        if detail:
            addr = _format_address(detail.get("address") or {})
            if addr:
                out["address"] = addr
            if detail.get("wikipedia"):
                out["wikipedia"] = detail["wikipedia"]
            preview = (detail.get("preview") or {}).get("source")
            if preview:
                out["image"] = preview
    return out

def verify_activities(
    activities: list[dict],
    city_name: str,
    city_lat: Optional[float] = None,
    city_lon: Optional[float] = None,
) -> tuple[list[dict], dict]:
    """Verifie chaque activite LLM contre OTM. Renvoie (activities_enrichies, stats)."""
    if not _API_KEY:
        logger.info("[OTM] cle absente, verification desactivee")
        return activities, {"verified": 0, "total": len(activities), "skipped": True}

    if city_lat is None or city_lon is None:
        geo = _geoname(city_name)
        if not geo:
            logger.warning("[OTM] geoname KO pour '%s' — verification sautee", city_name)
            return activities, {"verified": 0, "total": len(activities), "skipped": True}
        city_lat, city_lon, _ = geo

    verified_list: list[dict] = [None] * len(activities)  # type: ignore
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_verify_one, act, city_lat, city_lon): i
            for i, act in enumerate(activities)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                verified_list[i] = fut.result()
            except Exception as e:
                logger.debug("[OTM] verif failed for #%d: %s", i, e)
                verified_list[i] = dict(activities[i], verified=False)

    n_verified = sum(1 for a in verified_list if a.get("verified"))
    stats = {
        "verified": n_verified,
        "total": len(activities),
        "ratio": round(n_verified / len(activities), 2) if activities else 0,
        "skipped": False,
    }
    logger.info("[OTM] %s : %d/%d activites verifiees (%.0f%%)",
                city_name, n_verified, len(activities), stats["ratio"] * 100)
    return verified_list, stats
