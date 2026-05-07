"""
Fournisseur de données dynamique pour le planificateur de voyage.

Pipeline d'enrichissement (par ordre de priorité) :
  1. Cache local                → données déjà calculées (TTL 7 jours)
  2. KNOWN_ACTIVITIES (dataset) → données curatées pour les grands sites (conf ≥ 0.90)
  3. Overpass / OSM             → horaires et tarifs réels (conf ≈ 0.85)
  4. OpenTripMap                → POI de base avec coordonnées
  5. Estimations intelligentes  → templates par type/catégorie/popularité (conf ≈ 0.55)
  6. Defaults sécurisés         → jamais de valeur None dans la sortie (conf ≈ 0.20)

Chaque activité expose un champ `confidence` (0.0–1.0) utilisé par le solveur
pour ajuster le priority_score.

APIs utilisées :
  - OpenTripMap (POIs)      — clé OPENTRIPMAP_API_KEY dans .env
  - OSRM (matrice trajets)  — gratuit, sans clé
  - Overpass/OSM (horaires) — gratuit, sans clé
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OSRM_BASE_URL = "http://router.project-osrm.org"
OPENTRIPMAP_API_KEY = os.environ.get("OPENTRIPMAP_API_KEY", "")
OPENTRIPMAP_BASE_URL = "https://api.opentripmap.com/0.1"

# Mapping catégories OpenTripMap → nos 5 catégories
OTM_CATEGORY_MAP = {
    "museums": "culture",
    "theatres_and_entertainments": "culture",
    "historic": "culture",
    "architecture": "culture",
    "monuments_and_memorials": "culture",
    "churches": "culture",
    "archaeological_sites": "culture",
    "cultural": "culture",
    "religion": "culture",
    "foods": "gastro",
    "shops": "shopping",
    "amusements": "nightlife",
    "sport": "nature",
    "natural": "nature",
    "gardens_and_parks": "nature",
    "beaches": "nature",
}

# ─────────────────────────────────────────────────────────────────────────────
# Import des modules d'enrichissement
# ─────────────────────────────────────────────────────────────────────────────

try:
    from cache.cache_manager import CacheManager, TTL_CITY_DATA, TTL_OSM_DATA
    _cache = CacheManager()
    _CACHE_AVAILABLE = True
except ImportError:
    logger.warning("cache.cache_manager indisponible – pas de cache TTL")
    _cache = None  # type: ignore
    _CACHE_AVAILABLE = False

try:
    from datasets.activity_templates import (
        get_duration_estimate,
        get_cost_estimate,
        get_hours_estimate,
        lookup_known,
        priority_from_confidence,
    )
    _TEMPLATES_AVAILABLE = True
except ImportError:
    logger.warning("datasets.activity_templates indisponible – estimation basique")
    _TEMPLATES_AVAILABLE = False

try:
    from api_clients.overpass_client import fetch_osm_pois, enrich_activity
    _OVERPASS_AVAILABLE = True
except ImportError:
    logger.warning("api_clients.overpass_client indisponible – pas d'enrichissement OSM")
    _OVERPASS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass RawPlace (inchangée)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawPlace:
    """Lieu brut retourné par OpenTripMap."""
    id: str
    name: str
    latitude: float
    longitude: float
    kinds: str
    description: str = ""
    wikipedia_url: str = ""
    image_url: str = ""
    rating: int = 0  # 1–7 (7 = plus populaire)


# ─────────────────────────────────────────────────────────────────────────────
# OpenTripMap : Points d'intérêt
# ─────────────────────────────────────────────────────────────────────────────

def geocode_city(city_name: str, lang: str = "fr") -> Optional[dict]:  # noqa: ARG001
    """Géocode une ville via OpenTripMap. Retourne {name, country, lat, lon, ...}."""
    url = f"{OPENTRIPMAP_BASE_URL}/en/places/geoname"
    params = {"name": city_name}
    if OPENTRIPMAP_API_KEY:
        params["apikey"] = OPENTRIPMAP_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "lat" in data and "lon" in data:
            return data
    except Exception as e:
        logger.error("[OpenTripMap] Geocoding error: %s", e)
    return None


def fetch_places(
    lat: float,
    lon: float,
    radius_m: int = 10_000,
    kinds: str = "interesting_places,museums,historic,foods,gardens_and_parks,architecture",
    limit: int = 50,
    min_rating: int = 2,
    lang: str = "fr",  # noqa: ARG001  (passé à l'API future, non utilisé sur l'endpoint radius)
) -> list[RawPlace]:
    """Récupère les POI autour d'une position GPS via OpenTripMap."""
    url = f"{OPENTRIPMAP_BASE_URL}/en/places/radius"
    params = {
        "radius": radius_m,
        "lon": lon,
        "lat": lat,
        "kinds": kinds,
        "rate": min_rating,
        "limit": limit,
        "format": "json",
    }
    if OPENTRIPMAP_API_KEY:
        params["apikey"] = OPENTRIPMAP_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        places = []
        for item in data:
            name = item.get("name", "").strip()
            if not name or len(name) < 3:
                continue
            places.append(RawPlace(
                id=item.get("xid", ""),
                name=name,
                latitude=item["point"]["lat"],
                longitude=item["point"]["lon"],
                kinds=item.get("kinds", ""),
                rating=item.get("rate", 0),
            ))
        places.sort(key=lambda p: p.rating, reverse=True)
        return places
    except Exception as e:
        logger.error("[OpenTripMap] Fetch error: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# OSRM : Matrice de temps de trajet
# ─────────────────────────────────────────────────────────────────────────────

def osrm_travel_matrix(
    coordinates: list[tuple[float, float]],
    mode: str = "foot",
) -> Optional[list[list[int]]]:
    """Matrice NxN de temps de trajet en minutes via OSRM Table API."""
    if len(coordinates) < 2:
        return None
    servers = {
        "foot": "https://routing.openstreetmap.de/routed-foot",
        "car": "http://router.project-osrm.org",
        "bike": "https://routing.openstreetmap.de/routed-bike",
    }
    base = servers.get(mode, servers["foot"])
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    url = f"{base}/table/v1/driving/{coords_str}"
    try:
        resp = requests.get(url, params={"annotations": "duration"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok" and data.get("durations"):
            return [
                [max(1, round(cell / 60)) if cell is not None else 30 for cell in row]
                for row in data["durations"]
            ]
    except Exception as e:
        logger.error("[OSRM] Matrix error: %s", e)
    return None


def osrm_travel_time(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    mode: str = "foot",
) -> Optional[int]:
    """Temps de trajet point-à-point en minutes via OSRM."""
    servers = {
        "foot": "https://routing.openstreetmap.de/routed-foot",
        "car": "http://router.project-osrm.org",
        "bike": "https://routing.openstreetmap.de/routed-bike",
    }
    base = servers.get(mode, servers["foot"])
    url = f"{base}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    try:
        resp = requests.get(url, params={"overview": "false"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return max(1, round(data["routes"][0]["duration"] / 60))
    except Exception as e:
        logger.error("[OSRM] Route error: %s", e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Estimation intelligente (remplace les 3 fonctions hardcodées)
# ─────────────────────────────────────────────────────────────────────────────

def classify_category(otm_kinds: str) -> str:
    """Convertit les catégories OpenTripMap vers nos 5 catégories."""
    for kind in otm_kinds.lower().split(","):
        kind = kind.strip()
        for otm_key, our_cat in OTM_CATEGORY_MAP.items():
            if otm_key in kind:
                return our_cat
    return "culture"


def estimate_duration(
    category: str,
    kinds: str = "",
    rating: int = 5,
    name: str = "",
) -> tuple[float, float]:
    """
    Estime la durée de visite. Retourne (hours, confidence).

    Priorité : known_activities > kind_template > category_default.
    Le rating OSM (1–7) est converti sur l'échelle 1–10 des templates.
    """
    if _TEMPLATES_AVAILABLE:
        r10 = min(10, max(1, round(rating * 10 / 7)))
        return get_duration_estimate(category, kinds, r10, name)
    # Fallback basique
    base = {"culture": 2.0, "gastro": 2.0, "nature": 2.5, "shopping": 2.0, "nightlife": 3.0}
    return base.get(category, 2.0), 0.25


def estimate_cost(
    category: str,
    kinds: str = "",
    rating: int = 5,
    name: str = "",
    student: bool = False,
) -> tuple[int, float]:
    """
    Estime le coût d'entrée. Retourne (euros, confidence).

    Intègre :
    - tarif étudiant si student=True et donnée disponible
    - gratuité détectée dans KNOWN_ACTIVITIES
    - interpolation par rating sinon
    """
    if _TEMPLATES_AVAILABLE:
        r10 = min(10, max(1, round(rating * 10 / 7)))
        return get_cost_estimate(category, kinds, r10, name, student=student)
    # Fallback basique
    if category == "nature":
        return 0, 0.25
    if category == "gastro":
        return 30 + rating * 3, 0.25
    if category == "nightlife":
        return 20 + rating * 3, 0.25
    if category == "shopping":
        return 30, 0.20
    return max(0, 8 + rating * 2), 0.25


def estimate_hours(
    category: str,
    kinds: str = "",
    name: str = "",
) -> tuple[int, int, float]:
    """
    Estime les horaires d'ouverture/fermeture. Retourne (open, close, confidence).
    """
    if _TEMPLATES_AVAILABLE:
        return get_hours_estimate(category, kinds, name)
    # Fallback basique
    hours = {
        "culture": (9, 18),
        "gastro": (11, 22),
        "nature": (7, 20),
        "shopping": (10, 20),
        "nightlife": (19, 24),
    }
    o, c = hours.get(category, (9, 18))
    return o, c, 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Enrichissement OSM (Overpass)
# ─────────────────────────────────────────────────────────────────────────────

def _load_osm_index(
    city_name: str,
    lat: float,
    lon: float,
    radius_m: int = 10_000,
) -> dict:
    """
    Charge l'index OSM pour une ville.
    Utilise le cache (TTL 7 jours) avant d'interroger Overpass.
    """
    if not _OVERPASS_AVAILABLE:
        return {}

    cache_key = f"osm_{city_name.lower().replace(' ', '_')}"

    if _CACHE_AVAILABLE and _cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info("[OSM] Données chargées depuis le cache (%d entrées)", len(cached))
            # Reconstruire les objets OSMPlace depuis le dict sérialisé
            try:
                from api_clients.overpass_client import OSMPlace
                return {
                    k: OSMPlace(**v) for k, v in cached.items()
                    if isinstance(v, dict)
                }
            except Exception:
                return cached  # garde le dict brut si reconstruction échoue

    logger.info("[OSM] Interrogation Overpass pour %s…", city_name)
    osm_index = fetch_osm_pois(lat, lon, radius_m)

    if _CACHE_AVAILABLE and _cache and osm_index:
        # Sérialiser les OSMPlace en dicts pour le JSON
        try:
            from dataclasses import asdict
            serializable = {k: asdict(v) for k, v in osm_index.items()}
            _cache.set(cache_key, serializable, ttl_hours=TTL_OSM_DATA)
        except Exception as e:
            logger.warning("[OSM] Impossible de mettre en cache : %s", e)

    return osm_index


# ─────────────────────────────────────────────────────────────────────────────
# Conversion RawPlace → activité enrichie
# ─────────────────────────────────────────────────────────────────────────────

def _make_activity(
    place: RawPlace,
    osm_index: dict,
) -> dict:
    """
    Construit un dict d'activité prêt pour le solveur, en passant
    par les 4 niveaux d'enrichissement.
    """
    category = classify_category(place.kinds)
    name = place.name.strip()

    # ── Niveau 1 : KNOWN_ACTIVITIES (confidence ≥ 0.90) ─────────────────
    known = lookup_known(name) if _TEMPLATES_AVAILABLE else None

    # ── Niveau 2-3 : estimation intelligente (kind templates + known) ────
    duration, dur_conf = estimate_duration(category, place.kinds, place.rating, name)
    cost, cost_conf = estimate_cost(category, place.kinds, place.rating, name)
    open_h, close_h, hours_conf = estimate_hours(category, place.kinds, name)

    # Confiance globale = max des trois sources
    confidence = max(dur_conf, cost_conf, hours_conf)

    activity_id = place.id or name.lower().replace(" ", "_")[:30]

    activity = {
        "id": activity_id,
        "name": name,
        "category": category,
        "duration_hours": duration,
        "cost_euros": cost,
        "opening_hour": open_h,
        "closing_hour": close_h,
        "latitude": place.latitude,
        "longitude": place.longitude,
        "priority_score": _compute_priority(place.rating, confidence),
        "zone": "",
        "confidence": round(confidence, 2),
        "data_source_detail": "known_activities" if known else "template",
    }

    # ── Niveau 3 : enrichissement Overpass (override si meilleure conf) ──
    if osm_index:
        try:
            activity = enrich_activity(activity, osm_index)
            # Recalculer la priorité après enrichissement
            activity["priority_score"] = _compute_priority(
                place.rating, activity.get("confidence", confidence)
            )
        except Exception as e:
            logger.warning("[Enrich] Erreur pour '%s' : %s", name, e)

    return activity


def _compute_priority(rating: int, confidence: float) -> int:
    """
    Calcule le priority_score (1–10) combinant popularité et fiabilité des données.
    """
    # rating OSM est sur 1–7, convertir sur 1–10
    base = min(10, max(1, round(rating * 10 / 7)))
    if _TEMPLATES_AVAILABLE:
        return priority_from_confidence(base, confidence)
    return base


def raw_places_to_activities(
    places: list[RawPlace],
    osm_index: Optional[dict] = None,
) -> list[dict]:
    """
    Convertit les lieux OpenTripMap en activités pour le solveur.
    Élimine les doublons par nom.
    """
    activities: list[dict] = []
    seen_names: set[str] = set()
    osm = osm_index or {}

    for place in places:
        clean_name = place.name.strip()
        if clean_name in seen_names:
            continue
        seen_names.add(clean_name)

        try:
            act = _make_activity(place, osm)
            activities.append(act)
        except Exception as e:
            logger.warning("[Activity] Erreur pour '%s' : %s", clean_name, e)

    return activities


# ─────────────────────────────────────────────────────────────────────────────
# Assignation des zones (clustering géographique)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_zones(activities: list[dict], city_lat: float, city_lon: float) -> None:
    """Assigne une zone géographique par quadrant à chaque activité (in-place)."""
    for act in activities:
        dlat = act["latitude"] - city_lat
        dlon = act["longitude"] - city_lon
        if dlat >= 0 and dlon >= 0:
            act["zone"] = "nord-est"
        elif dlat >= 0 and dlon < 0:
            act["zone"] = "nord-ouest"
        elif dlat < 0 and dlon >= 0:
            act["zone"] = "sud-est"
        else:
            act["zone"] = "sud-ouest"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal : ville → données prêtes pour le solveur
# ─────────────────────────────────────────────────────────────────────────────

def build_city_data(
    city_name: str,
    radius_m: int = 8_000,
    max_activities: int = 30,
    transport_mode: str = "foot",
    lang: str = "fr",
    enrich_osm: bool = True,
) -> Optional[dict]:
    """
    Pipeline complet :
      1. Géocode la ville (OpenTripMap)
      2. Récupère les POI (OpenTripMap)
      3. Enrichit avec Overpass/OSM (horaires & prix réels)
      4. Convertit en activités avec estimations intelligentes
      5. Calcule la matrice de trajets (OSRM)
      6. Assigne les zones géographiques

    Returns:
        {"city":{…}, "activities":[…], "travel_matrix":[[…]], "data_source":"…"}
        ou None si une étape critique échoue.
    """
    logger.info("[Pipeline] Recherche de données pour %s…", city_name)

    # 1. Géocoding
    geo = geocode_city(city_name, lang)
    if not geo:
        logger.error("[Pipeline] Ville '%s' non trouvée", city_name)
        return None

    lat, lon = float(geo["lat"]), float(geo["lon"])
    logger.info("[Pipeline] %s trouvée : (%.4f, %.4f)", city_name, lat, lon)

    # 2. Récupération des POI
    kinds = ",".join([
        "interesting_places", "museums", "historic", "architecture",
        "monuments_and_memorials", "churches",
        "foods", "gardens_and_parks", "natural", "amusements", "shops",
    ])
    places = fetch_places(lat, lon, radius_m, kinds, max_activities * 2, min_rating=2, lang=lang)
    logger.info("[Pipeline] %d lieux trouvés via OpenTripMap", len(places))

    if not places:
        logger.error("[Pipeline] Aucun lieu trouvé pour %s", city_name)
        return None

    # 3. Chargement de l'index OSM (enrichissement horaires/prix)
    osm_index: dict = {}
    if enrich_osm:
        osm_index = _load_osm_index(city_name, lat, lon, radius_m)
        logger.info("[Pipeline] %d entrées OSM disponibles", len(osm_index))

    # 4. Conversion + enrichissement
    activities = raw_places_to_activities(places[:max_activities], osm_index)
    logger.info("[Pipeline] %d activités après enrichissement", len(activities))

    # 5. Matrice de trajets OSRM
    coords = [(a["latitude"], a["longitude"]) for a in activities]
    logger.info("[Pipeline] Calcul matrice de temps via OSRM (%d×%d)…",
                len(coords), len(coords))
    travel_matrix = osrm_travel_matrix(coords, transport_mode)
    if not travel_matrix:
        logger.error("[Pipeline] OSRM indisponible")
        return None
    logger.info("[Pipeline] Matrice OSRM %d×%d calculée", len(travel_matrix), len(travel_matrix))

    # 6. Zones géographiques
    _assign_zones(activities, lat, lon)

    # Statistiques de confiance
    confs = [a.get("confidence", 0.0) for a in activities]
    avg_conf = sum(confs) / len(confs) if confs else 0
    high_conf = sum(1 for c in confs if c >= 0.80)
    logger.info("[Pipeline] Confiance moyenne : %.2f | %d activités haute confiance / %d",
                avg_conf, high_conf, len(activities))

    return {
        "city": {
            "name": geo.get("name", city_name),
            "country": geo.get("country", ""),
            "latitude": lat,
            "longitude": lon,
            "population": geo.get("population", 0),
        },
        "activities": activities,
        "travel_matrix": travel_matrix,
        "data_source": _data_source_label(bool(osm_index), bool(OPENTRIPMAP_API_KEY)),
        "confidence_stats": {
            "average": round(avg_conf, 2),
            "high_confidence_count": high_conf,
            "total": len(activities),
        },
    }


def _data_source_label(osm: bool, otm_key: bool) -> str:
    parts = ["opentripmap" if otm_key else "opentripmap_no_key"]
    if osm:
        parts.append("overpass_osm")
    parts.append("osrm")
    return "+".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Cache ville (remplace les fonctions save/load_city_cache)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key_city(city_name: str) -> str:
    return f"city_{city_name.lower().replace(' ', '_')}"


def _save_city(city_name: str, data: dict) -> None:
    if _CACHE_AVAILABLE and _cache:
        _cache.set(_cache_key_city(city_name), data, ttl_hours=TTL_CITY_DATA)
    else:
        # Fallback : fichier JSON direct dans cache/
        _dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(_dir, exist_ok=True)
        path = os.path.join(_dir, f"{city_name.lower().replace(' ', '_')}.json")
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _load_city(city_name: str) -> Optional[dict]:
    if _CACHE_AVAILABLE and _cache:
        return _cache.get(_cache_key_city(city_name))
    # Fallback : fichier JSON direct
    import json
    _dir = os.path.join(os.path.dirname(__file__), "cache")
    path = os.path.join(_dir, f"{city_name.lower().replace(' ', '_')}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def get_city_data(city_name: str, **kwargs) -> Optional[dict]:
    """
    Point d'entrée : charge du cache (TTL 7 jours) ou appelle le pipeline.
    Le cache expiré est automatiquement rafraîchi.
    """
    cached = _load_city(city_name)
    if cached:
        logger.info("[Cache] Données de %s chargées depuis le cache", city_name)
        return cached

    data = build_city_data(city_name, **kwargs)
    if data:
        _save_city(city_name, data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Données pré-curatées Rome (fallback instantané)
# ─────────────────────────────────────────────────────────────────────────────

ROME_PRECACHED = {
    "city": {
        "name": "Roma", "country": "IT",
        "latitude": 41.8933, "longitude": 12.4829,
        "population": 2_873_000,
    },
    "activities": [
        {"id": "colosseum",     "name": "Colisée",                "category": "culture",
         "duration_hours": 2.5, "cost_euros": 18, "opening_hour": 8,  "closing_hour": 19,
         "latitude": 41.8902,  "longitude": 12.4922, "priority_score": 10, "zone": "sud-est",
         "confidence": 0.92},
        {"id": "vatican",       "name": "Musées du Vatican",       "category": "culture",
         "duration_hours": 3.5, "cost_euros": 17, "opening_hour": 8,  "closing_hour": 18,
         "latitude": 41.9065,  "longitude": 12.4536, "priority_score": 10, "zone": "nord-ouest",
         "confidence": 0.92},
        {"id": "pantheon",      "name": "Panthéon",                "category": "culture",
         "duration_hours": 1.0, "cost_euros": 5,  "opening_hour": 9,  "closing_hour": 19,
         "latitude": 41.8986,  "longitude": 12.4769, "priority_score": 8,  "zone": "nord-ouest",
         "confidence": 0.90},
        {"id": "forum",         "name": "Forum Romain",            "category": "culture",
         "duration_hours": 2.0, "cost_euros": 16, "opening_hour": 8,  "closing_hour": 19,
         "latitude": 41.8925,  "longitude": 12.4853, "priority_score": 9,  "zone": "sud-est",
         "confidence": 0.92},
        {"id": "trevi",         "name": "Fontaine de Trevi",       "category": "culture",
         "duration_hours": 0.75,"cost_euros": 0,  "opening_hour": 7,  "closing_hour": 23,
         "latitude": 41.9009,  "longitude": 12.4833, "priority_score": 9,  "zone": "nord-est",
         "confidence": 0.92},
        {"id": "borghese",      "name": "Galerie Borghèse",        "category": "culture",
         "duration_hours": 2.0, "cost_euros": 15, "opening_hour": 9,  "closing_hour": 19,
         "latitude": 41.9142,  "longitude": 12.4921, "priority_score": 9,  "zone": "nord-est",
         "confidence": 0.92},
        {"id": "trastevere",    "name": "Trastevere",              "category": "nature",
         "duration_hours": 2.0, "cost_euros": 0,  "opening_hour": 9,  "closing_hour": 23,
         "latitude": 41.8840,  "longitude": 12.4700, "priority_score": 7,  "zone": "sud-ouest",
         "confidence": 0.85},
        {"id": "piazza_navona", "name": "Piazza Navona",           "category": "culture",
         "duration_hours": 1.0, "cost_euros": 0,  "opening_hour": 0,  "closing_hour": 24,
         "latitude": 41.8992,  "longitude": 12.4731, "priority_score": 8,  "zone": "nord-ouest",
         "confidence": 0.90},
        {"id": "campo_fiori",   "name": "Campo de' Fiori",         "category": "gastro",
         "duration_hours": 1.5, "cost_euros": 15, "opening_hour": 8,  "closing_hour": 22,
         "latitude": 41.8956,  "longitude": 12.4722, "priority_score": 7,  "zone": "sud-ouest",
         "confidence": 0.80},
        {"id": "catacombs",     "name": "Catacombes de San Callisto","category": "culture",
         "duration_hours": 1.5, "cost_euros": 10, "opening_hour": 9,  "closing_hour": 17,
         "latitude": 41.8582,  "longitude": 12.5137, "priority_score": 6,  "zone": "sud-est",
         "confidence": 0.88},
        {"id": "castel_angelo", "name": "Château Saint-Ange",      "category": "culture",
         "duration_hours": 2.0, "cost_euros": 15, "opening_hour": 9,  "closing_hour": 19,
         "latitude": 41.9031,  "longitude": 12.4663, "priority_score": 8,  "zone": "nord-ouest",
         "confidence": 0.92},
        {"id": "villa_borghese","name": "Parc Villa Borghèse",     "category": "nature",
         "duration_hours": 2.0, "cost_euros": 0,  "opening_hour": 6,  "closing_hour": 20,
         "latitude": 41.9145,  "longitude": 12.4855, "priority_score": 7,  "zone": "nord-est",
         "confidence": 0.88},
        {"id": "testaccio",     "name": "Food tour Testaccio",     "category": "gastro",
         "duration_hours": 3.0, "cost_euros": 45, "opening_hour": 11, "closing_hour": 15,
         "latitude": 41.8761,  "longitude": 12.4756, "priority_score": 8,  "zone": "sud-ouest",
         "confidence": 0.85},
        {"id": "appian_way",    "name": "Via Appia Antica",        "category": "nature",
         "duration_hours": 3.0, "cost_euros": 15, "opening_hour": 8,  "closing_hour": 18,
         "latitude": 41.8554,  "longitude": 12.5245, "priority_score": 7,  "zone": "sud-est",
         "confidence": 0.90},
        {"id": "ostia",         "name": "Ostia Antica",            "category": "culture",
         "duration_hours": 4.0, "cost_euros": 12, "opening_hour": 8,  "closing_hour": 17,
         "latitude": 41.7558,  "longitude": 12.2916, "priority_score": 6,  "zone": "sud-ouest",
         "confidence": 0.90},
        {"id": "cooking_class", "name": "Cours de cuisine",        "category": "gastro",
         "duration_hours": 3.0, "cost_euros": 65, "opening_hour": 10, "closing_hour": 14,
         "latitude": 41.8840,  "longitude": 12.4690, "priority_score": 7,  "zone": "sud-ouest",
         "confidence": 0.80},
        {"id": "piazza_spagna", "name": "Place d'Espagne",         "category": "culture",
         "duration_hours": 1.0, "cost_euros": 0,  "opening_hour": 7,  "closing_hour": 23,
         "latitude": 41.9060,  "longitude": 12.4828, "priority_score": 8,  "zone": "nord-est",
         "confidence": 0.88},
        {"id": "wine_tasting",  "name": "Dégustation de vins",     "category": "gastro",
         "duration_hours": 2.0, "cost_euros": 35, "opening_hour": 16, "closing_hour": 21,
         "latitude": 41.8965,  "longitude": 12.4800, "priority_score": 6,  "zone": "sud-est",
         "confidence": 0.80},
        {"id": "trastevere_night","name": "Soirée Trastevere",     "category": "nightlife",
         "duration_hours": 3.0, "cost_euros": 40, "opening_hour": 19, "closing_hour": 24,
         "latitude": 41.8845,  "longitude": 12.4695, "priority_score": 7,  "zone": "sud-ouest",
         "confidence": 0.80},
        {"id": "pincio_sunset", "name": "Coucher de soleil Pincio","category": "nature",
         "duration_hours": 1.5, "cost_euros": 0,  "opening_hour": 17, "closing_hour": 21,
         "latitude": 41.9115,  "longitude": 12.4785, "priority_score": 8,  "zone": "nord-ouest",
         "confidence": 0.85},
    ],
    "travel_matrix": None,
    "data_source": "precached+known_activities",
    "confidence_stats": {"average": 0.88, "high_confidence_count": 20, "total": 20},
}


def get_rome_data() -> Optional[dict]:
    """
    Retourne les données pré-curatées de Rome avec matrice OSRM.
    Enrichissement OSM optionnel (si Overpass disponible et cache absent).
    """
    data = {k: v for k, v in ROME_PRECACHED.items()}
    data["activities"] = [dict(a) for a in ROME_PRECACHED["activities"]]

    if data["travel_matrix"] is None:
        coords = [(a["latitude"], a["longitude"]) for a in data["activities"]]
        matrix = osrm_travel_matrix(coords, "foot")
        if not matrix:
            logger.error("[get_rome_data] OSRM indisponible")
            return None
        data["travel_matrix"] = matrix
        data["data_source"] = "precached+osrm"

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Données pré-curatées Paris (fallback instantané)
# ─────────────────────────────────────────────────────────────────────────────

PARIS_PRECACHED = {
    "city": {
        "name": "Paris", "country": "FR",
        "latitude": 48.8566, "longitude": 2.3522,
        "population": 2_161_000,
    },
    "activities": [
        {"id": "louvre",            "name": "Musée du Louvre",          "category": "culture",
         "duration_hours": 3.5, "cost_euros": 17, "opening_hour": 9,  "closing_hour": 18,
         "latitude": 48.8606,  "longitude": 2.3376, "priority_score": 10, "zone": "nord-ouest",
         "confidence": 0.95},
        {"id": "eiffel_tower",      "name": "Tour Eiffel",              "category": "culture",
         "duration_hours": 2.0, "cost_euros": 26, "opening_hour": 9,  "closing_hour": 23,
         "latitude": 48.8584,  "longitude": 2.2945, "priority_score": 10, "zone": "nord-ouest",
         "confidence": 0.95},
        {"id": "notre_dame",        "name": "Cathédrale Notre-Dame",    "category": "culture",
         "duration_hours": 1.5, "cost_euros": 0,  "opening_hour": 8,  "closing_hour": 18,
         "latitude": 48.8530,  "longitude": 2.3499, "priority_score": 9,  "zone": "sud-est",
         "confidence": 0.92},
        {"id": "musee_orsay",       "name": "Musée d'Orsay",            "category": "culture",
         "duration_hours": 3.0, "cost_euros": 16, "opening_hour": 9,  "closing_hour": 18,
         "latitude": 48.8600,  "longitude": 2.3266, "priority_score": 9,  "zone": "nord-ouest",
         "confidence": 0.92},
        {"id": "arc_triomphe",      "name": "Arc de Triomphe",          "category": "culture",
         "duration_hours": 1.0, "cost_euros": 13, "opening_hour": 10, "closing_hour": 23,
         "latitude": 48.8738,  "longitude": 2.2950, "priority_score": 9,  "zone": "nord-ouest",
         "confidence": 0.92},
        {"id": "sacre_coeur",       "name": "Sacré-Cœur",               "category": "culture",
         "duration_hours": 1.0, "cost_euros": 0,  "opening_hour": 6,  "closing_hour": 22,
         "latitude": 48.8867,  "longitude": 2.3431, "priority_score": 8,  "zone": "nord-est",
         "confidence": 0.90},
        {"id": "versailles",        "name": "Château de Versailles",    "category": "culture",
         "duration_hours": 5.0, "cost_euros": 20, "opening_hour": 9,  "closing_hour": 18,
         "latitude": 48.8049,  "longitude": 2.1204, "priority_score": 10, "zone": "sud-ouest",
         "confidence": 0.95},
        {"id": "pompidou",          "name": "Centre Pompidou",          "category": "culture",
         "duration_hours": 2.5, "cost_euros": 15, "opening_hour": 11, "closing_hour": 21,
         "latitude": 48.8606,  "longitude": 2.3522, "priority_score": 8,  "zone": "nord-est",
         "confidence": 0.90},
        {"id": "marais",            "name": "Le Marais",                "category": "culture",
         "duration_hours": 2.5, "cost_euros": 0,  "opening_hour": 9,  "closing_hour": 22,
         "latitude": 48.8566,  "longitude": 2.3620, "priority_score": 8,  "zone": "nord-est",
         "confidence": 0.85},
        {"id": "sainte_chapelle",   "name": "Sainte-Chapelle",          "category": "culture",
         "duration_hours": 1.0, "cost_euros": 13, "opening_hour": 9,  "closing_hour": 17,
         "latitude": 48.8554,  "longitude": 2.3450, "priority_score": 8,  "zone": "sud-est",
         "confidence": 0.92},
        {"id": "palais_royal",      "name": "Jardin du Palais-Royal",   "category": "nature",
         "duration_hours": 1.0, "cost_euros": 0,  "opening_hour": 8,  "closing_hour": 22,
         "latitude": 48.8638,  "longitude": 2.3370, "priority_score": 7,  "zone": "nord-ouest",
         "confidence": 0.88},
        {"id": "tuileries",         "name": "Jardin des Tuileries",     "category": "nature",
         "duration_hours": 1.5, "cost_euros": 0,  "opening_hour": 7,  "closing_hour": 21,
         "latitude": 48.8635,  "longitude": 2.3323, "priority_score": 7,  "zone": "nord-ouest",
         "confidence": 0.88},
        {"id": "luxembourg",        "name": "Jardin du Luxembourg",     "category": "nature",
         "duration_hours": 1.5, "cost_euros": 0,  "opening_hour": 7,  "closing_hour": 21,
         "latitude": 48.8462,  "longitude": 2.3372, "priority_score": 7,  "zone": "sud-ouest",
         "confidence": 0.88},
        {"id": "montmartre",        "name": "Montmartre",               "category": "culture",
         "duration_hours": 2.5, "cost_euros": 0,  "opening_hour": 8,  "closing_hour": 23,
         "latitude": 48.8867,  "longitude": 2.3431, "priority_score": 8,  "zone": "nord-est",
         "confidence": 0.85},
        {"id": "seine_cruise",      "name": "Croisière sur la Seine",   "category": "nature",
         "duration_hours": 1.5, "cost_euros": 15, "opening_hour": 10, "closing_hour": 22,
         "latitude": 48.8566,  "longitude": 2.3417, "priority_score": 7,  "zone": "sud-est",
         "confidence": 0.85},
        {"id": "latin_quarter",     "name": "Quartier Latin",           "category": "gastro",
         "duration_hours": 2.0, "cost_euros": 25, "opening_hour": 11, "closing_hour": 23,
         "latitude": 48.8510,  "longitude": 2.3463, "priority_score": 7,  "zone": "sud-est",
         "confidence": 0.85},
        {"id": "champs_elysees",    "name": "Champs-Élysées",           "category": "shopping",
         "duration_hours": 2.0, "cost_euros": 0,  "opening_hour": 9,  "closing_hour": 22,
         "latitude": 48.8698,  "longitude": 2.3078, "priority_score": 7,  "zone": "nord-ouest",
         "confidence": 0.85},
        {"id": "moulin_rouge",      "name": "Moulin Rouge",             "category": "nightlife",
         "duration_hours": 2.5, "cost_euros": 87, "opening_hour": 19, "closing_hour": 23,
         "latitude": 48.8842,  "longitude": 2.3323, "priority_score": 8,  "zone": "nord-ouest",
         "confidence": 0.90},
        {"id": "musee_rodin",       "name": "Musée Rodin",              "category": "culture",
         "duration_hours": 2.0, "cost_euros": 14, "opening_hour": 10, "closing_hour": 18,
         "latitude": 48.8555,  "longitude": 2.3161, "priority_score": 7,  "zone": "nord-ouest",
         "confidence": 0.90},
        {"id": "ile_saint_louis",   "name": "Île Saint-Louis",          "category": "gastro",
         "duration_hours": 1.5, "cost_euros": 10, "opening_hour": 10, "closing_hour": 22,
         "latitude": 48.8521,  "longitude": 2.3564, "priority_score": 7,  "zone": "sud-est",
         "confidence": 0.85},
        {"id": "opera_garnier",     "name": "Opéra Garnier",            "category": "culture",
         "duration_hours": 1.5, "cost_euros": 15, "opening_hour": 10, "closing_hour": 17,
         "latitude": 48.8720,  "longitude": 2.3316, "priority_score": 8,  "zone": "nord-ouest",
         "confidence": 0.90},
        {"id": "wine_bar_paris",    "name": "Bar à vin parisien",       "category": "gastro",
         "duration_hours": 2.0, "cost_euros": 35, "opening_hour": 17, "closing_hour": 23,
         "latitude": 48.8560,  "longitude": 2.3530, "priority_score": 6,  "zone": "sud-est",
         "confidence": 0.80},
    ],
    "travel_matrix": None,
    "data_source": "precached+known_activities",
    "confidence_stats": {"average": 0.89, "high_confidence_count": 22, "total": 22},
}


def get_paris_data() -> Optional[dict]:
    """
    Retourne les données pré-curatées de Paris avec matrice OSRM.
    """
    data = {k: v for k, v in PARIS_PRECACHED.items()}
    data["activities"] = [dict(a) for a in PARIS_PRECACHED["activities"]]

    if data["travel_matrix"] is None:
        coords = [(a["latitude"], a["longitude"]) for a in data["activities"]]
        matrix = osrm_travel_matrix(coords, "foot")
        if not matrix:
            logger.error("[get_paris_data] OSRM indisponible")
            return None
        data["travel_matrix"] = matrix
        data["data_source"] = "precached+osrm"

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Compatibilité legacy (utilisée par orchestrator.py)
# ─────────────────────────────────────────────────────────────────────────────

# Ces fonctions sont conservées pour ne pas casser l'interface existante.
# Elles délèguent désormais aux nouvelles fonctions d'estimation intelligente.

def _legacy_estimate_duration(category: str, rating: int) -> float:
    dur, _ = estimate_duration(category, "", rating)
    return dur

def _legacy_estimate_cost(category: str, rating: int) -> int:
    cost, _ = estimate_cost(category, "", rating)
    return cost

def _legacy_estimate_hours(category: str) -> tuple[int, int]:
    o, c, _ = estimate_hours(category)
    return o, c


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 60)
    print("Test estimations intelligentes")
    print("=" * 60)

    test_cases = [
        ("Colisée",          "culture", "museums,historic", 7),
        ("Tour Eiffel",      "culture", "architecture",     7),
        ("Galerie inconnue", "culture", "museums",          4),
        ("Parc inconnu",     "nature",  "gardens_and_parks",3),
        ("Restaurant",       "gastro",  "foods",            5),
    ]
    for name, cat, kinds, rating in test_cases:
        dur, dc = estimate_duration(cat, kinds, rating, name)
        cost, cc = estimate_cost(cat, kinds, rating, name)
        oh, ch, hc = estimate_hours(cat, kinds, name)
        print(f"\n  {name} ({cat}, rating={rating})")
        print(f"    durée   : {dur}h  (conf {dc:.2f})")
        print(f"    coût    : {cost}€  (conf {cc:.2f})")
        print(f"    horaires: {oh}h–{ch}h (conf {hc:.2f})")

    print("\n" + "=" * 60)
    print("Test données pré-curatées Rome (matrice via OSRM)")
    print("=" * 60)
    rome = get_rome_data()
    if rome:
        print(f"Ville  : {rome['city']['name']} ({rome['city']['country']})")
        print(f"Source : {rome['data_source']}")
        print(f"Activ. : {len(rome['activities'])}")
        print(f"Conf.  : {rome.get('confidence_stats', {})}")
        print(f"Matrice: {len(rome['travel_matrix'])}×{len(rome['travel_matrix'])}")
        print("\nTop 5 :")
        for a in sorted(rome["activities"], key=lambda x: x["priority_score"], reverse=True)[:5]:
            print(f"  {a['priority_score']}/10 | {a['name']} ({a['category']}) "
                  f"— {a['cost_euros']}€, {a['duration_hours']}h, "
                  f"conf={a.get('confidence', '?')}")
