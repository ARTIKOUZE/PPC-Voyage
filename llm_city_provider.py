"""
Génération dynamique de données de ville via le LLM.

Le LLM connaît toutes les attractions touristiques du monde.
On lui demande de produire un JSON structuré pour n'importe quelle ville,
puis on calcule la matrice de trajets via OSRM.

Pipeline :
  1. Cache local (TTL 30 jours) — évite de re-générer à chaque session
  2. LLM → JSON des activités + coordonnées GPS de la ville
  3. OSRM → matrice de temps de trajet à pied
  4. Assignation des zones géographiques (quadrants)
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from llm_client import (
    chat_with_fallback, QWEN_NO_THINK, _strip_thinking, _extract_json_blob,
)

logger = logging.getLogger(__name__)


def _norm(name: str) -> str:
    """Normalise un nom d'activité pour la déduplication : minuscules, sans accents, sans ponctuation."""
    import unicodedata
    name = name.lower().strip()
    name = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

# Cache désactivé : chaque requête appelle le LLM (données toujours fraîches,
# fonctionne pour n'importe quelle ville sans pré-cache).
_cache = None
_CACHE_AVAILABLE = False
TTL_LLM_CITY = 0


# ─────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"culture", "gastro", "nature", "shopping", "nightlife"}


class LLMActivity(BaseModel):
    id: str
    name: str
    category: str
    duration_hours: float = Field(ge=0.25, le=8.0)
    cost_euros: int = Field(ge=0)
    opening_hour: int = Field(ge=0, le=23)
    closing_hour: int = Field(ge=1, le=24)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    priority_score: int = Field(ge=1, le=10)
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)

    def clean_category(self) -> str:
        return self.category if self.category in VALID_CATEGORIES else "culture"


class LLMHotel(BaseModel):
    name: str
    address: str = ""
    price_per_night: int = Field(ge=0)
    stars: int = Field(ge=0, le=5, default=3)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    description: str = ""


class LLMCityData(BaseModel):
    city: dict
    activities: list[LLMActivity]
    hotels: list[LLMHotel] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

CITY_ACTIVITIES_PROMPT = """\
You are a travel data expert. Generate tourist data for: {city_name}

Return ONLY a valid JSON object. No markdown, no explanations.

Structure:
{{
  "city": {{"name":"...","country":"<2-letter ISO>","latitude":<lat>,"longitude":<lon>,"population":<int>}},
  "activities": [ {{"id":"slug","name":"<French>","category":"<culture|gastro|nature|shopping|nightlife>","duration_hours":<float>,"cost_euros":<int>,"opening_hour":<int>,"closing_hour":<int>,"latitude":<lat>,"longitude":<lon>,"priority_score":<1-10>,"confidence":<0.80-0.95>}}, ... ],
  "hotels": [ {{"name":"<real hotel>","address":"<full address>","price_per_night":<int>,"stars":<1-5>,"latitude":<lat>,"longitude":<lon>,"description":"<French, <12 words>"}}, ... ]
}}

RULES:
- Exactly 15 activities, diverse across the 5 categories.
- Mix: 5 iconic (priority 9-10), 3 gastro, 2-3 nature, 1-2 shopping/nightlife if relevant.
- Activity names in French. id = unique ASCII slug. GPS 4+ decimals. Prices in euros.

REALISTIC duration_hours (use these typical visit times from real tourist averages):
- Major museum (Louvre, Met, British Museum, Prado): 3.0–4.0
- Standard museum / gallery: 1.5–2.5
- Iconic monument with visit (Eiffel Tower up, Colosseum inside): 2.0–3.0
- Quick photo-stop monument (Trevi, Brandenburg Gate): 0.5–1.0
- Cathedral / church visit: 0.5–1.5 (1.5 only for major ones like Sagrada Familia)
- Restaurant / food tour / cooking class: 1.5–2.5
- Market / food market visit: 1.0–1.5
- Park / garden stroll: 1.0–2.0
- Day trip outside city (e.g., Versailles, Pompeii): 4.0–6.0
- Neighborhood walk: 1.5–2.5
- Bar / nightclub session: 2.0–3.0
- Shopping district: 1.5–2.5
Pick the appropriate value for each specific activity — do NOT default to 1.5.

REALISTIC opening_hour / closing_hour (use real opening hours, not 8-22 by default):
- Museums: typically open 9-10, close 17-18
- Restaurants: open 12 or 19, close 14 or 23 (lunch + dinner)
- Bars / clubs: open 18-21, close 24
- Parks: open 6-8, close 20-22
- Markets: open 6-9, close 14-16
- Outdoor monuments (Eiffel base, Trevi): open 0, close 24 (always accessible)

- Exactly 3 REAL hotels in the city: 1 budget (50-90€, 2-3★), 1 mid-range (100-180€, 3-4★), 1 premium (200-400€, 4-5★). Real names + addresses.
- Be CONCISE: no extra fields, no commentary. Output the JSON and stop.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Appel LLM
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm_for_city(city_name: str, max_retries: int = 1) -> Optional[dict]:
    """Appelle le LLM (primaire + fallback) et retourne le dict JSON brut,
    ou None en cas d'échec total.
    Timeout par appel : 180 s (génération de 15 activités + 3 hôtels = lourd)."""
    prompt = CITY_ACTIVITIES_PROMPT.format(city_name=city_name)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = chat_with_fallback(
                timeout=180.0,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3500,
                response_format={"type": "json_object"},
                extra_body=QWEN_NO_THINK,
            )
            raw = resp.choices[0].message.content or ""
            blob = _extract_json_blob(raw)
            data = json.loads(blob)
            return data
        except (json.JSONDecodeError, TypeError) as e:
            last_err = f"parse error (attempt {attempt}): {e}"
            logger.warning("[LLMCity] %s", last_err)
        except Exception as e:
            last_err = f"api error (attempt {attempt}): {e}"
            logger.warning("[LLMCity] %s", last_err)

    logger.error("[LLMCity] Échec après %d tentatives pour '%s': %s",
                 max_retries + 1, city_name, last_err)
    return None


def _validate_and_clean(raw: dict, city_name: str) -> Optional[dict]:
    """Valide le JSON LLM et retourne un dict propre pour le solveur."""
    try:
        city_info = raw.get("city", {})
        if not city_info.get("latitude") or not city_info.get("longitude"):
            logger.error("[LLMCity] Coordonnées ville manquantes pour '%s'", city_name)
            return None

        raw_acts = raw.get("activities", [])
        if not raw_acts:
            logger.error("[LLMCity] Aucune activité générée pour '%s'", city_name)
            return None

        activities = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()

        for item in raw_acts:
            try:
                act = LLMActivity(**item)

                # Dédupliquer par ID
                act_id = act.id
                if act_id in seen_ids:
                    act_id = f"{act_id}_{len(seen_ids)}"
                seen_ids.add(act_id)

                # Dédupliquer par nom normalisé (évite "Tour Eiffel" et "Tour Eiffel - sommet")
                norm_name = _norm(act.name)
                if norm_name in seen_names:
                    logger.debug("[LLMCity] Doublon de nom ignoré: '%s'", act.name)
                    continue
                # Vérifier aussi si un nom existant est un sous-ensemble (préfixe ≥ 6 chars)
                if any(norm_name.startswith(s[:6]) and abs(len(norm_name) - len(s)) < 15
                       for s in seen_names):
                    logger.debug("[LLMCity] Nom trop proche d'un existant, ignoré: '%s'", act.name)
                    continue
                seen_names.add(norm_name)

                activities.append({
                    "id": act_id,
                    "name": act.name,
                    "category": act.clean_category(),
                    "duration_hours": act.duration_hours,
                    "cost_euros": act.cost_euros,
                    "opening_hour": act.opening_hour,
                    "closing_hour": act.closing_hour,
                    "latitude": act.latitude,
                    "longitude": act.longitude,
                    "priority_score": act.priority_score,
                    "zone": "",
                    "confidence": act.confidence,
                    "data_source_detail": "llm",
                })
            except (ValidationError, TypeError) as e:
                logger.debug("[LLMCity] Activité invalide ignorée: %s — %s", item.get("name", "?"), e)

        if len(activities) < 5:
            logger.error("[LLMCity] Trop peu d'activités valides (%d) pour '%s'",
                         len(activities), city_name)
            return None

        # Hôtels (optionnel)
        hotels = []
        for item in raw.get("hotels", []) or []:
            try:
                h = LLMHotel(**item)
                hotels.append({
                    "name": h.name,
                    "address": h.address,
                    "price_per_night": h.price_per_night,
                    "stars": h.stars,
                    "latitude": h.latitude,
                    "longitude": h.longitude,
                    "description": h.description,
                })
            except (ValidationError, TypeError) as e:
                logger.debug("[LLMCity] Hôtel invalide ignoré: %s — %s",
                             item.get("name", "?"), e)

        return {
            "city": {
                "name": city_info.get("name", city_name),
                "country": city_info.get("country", ""),
                "latitude": float(city_info["latitude"]),
                "longitude": float(city_info["longitude"]),
                "population": int(city_info.get("population", 0)),
            },
            "activities": activities,
            "hotels": hotels,
        }

    except Exception as e:
        logger.error("[LLMCity] Validation échouée pour '%s': %s", city_name, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OSRM + Zones
# ─────────────────────────────────────────────────────────────────────────────

def _assign_zones(activities: list[dict], city_lat: float, city_lon: float) -> None:
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


_TRANSPORT_SPEEDS_MPH = {"foot": 4000, "bike": 15000, "car": 30000}  # mètres/heure


def _compute_travel_matrix(
    activities: list[dict],
    transport_mode: str = "foot",
) -> Optional[list[list[int]]]:
    """Calcule la matrice OSRM, ou une matrice haversine de fallback."""
    try:
        from data_provider import osrm_travel_matrix
        coords = [(a["latitude"], a["longitude"]) for a in activities]
        matrix = osrm_travel_matrix(coords, transport_mode)
        if matrix:
            return matrix
    except Exception as e:
        logger.warning("[LLMCity] OSRM error: %s — fallback haversine", e)

    # Fallback : estimation haversine selon mode de transport
    logger.warning("[LLMCity] Matrice haversine utilisée (OSRM indisponible)")
    speed_mh = _TRANSPORT_SPEEDS_MPH.get(transport_mode, 4000)
    coords = [(a["latitude"], a["longitude"]) for a in activities]
    n = len(coords)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                R = 6_371_000
                phi1, phi2 = math.radians(lat1), math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlam = math.radians(lon2 - lon1)
                a = (math.sin(dphi / 2) ** 2
                     + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
                dist_m = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                minutes = max(1, round(dist_m / (speed_mh / 60)))
                row.append(minutes)
        matrix.append(row)
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ─────────────────────────────────────────────────────────────────────────────

def generate_city_data(
    city_name: str,
    transport_mode: str = "foot",
) -> Optional[dict]:
    """
    Génère (ou charge depuis le cache) les données complètes d'une ville.

    Retourne un dict compatible avec le solveur :
        {"city": {...}, "activities": [...], "travel_matrix": [[...]],
         "hotels": [...], "transport_mode": "foot", ...}
    ou None en cas d'échec.
    """
    # Pour rester compatible avec les anciens caches : pas de suffixe pour "foot"
    base_key = f"llm_city_{city_name.lower().replace(' ', '_')}"
    cache_key = base_key if transport_mode == "foot" else f"{base_key}_{transport_mode}"

    # 1. Cache
    if _CACHE_AVAILABLE and _cache:
        cached = _cache.get(cache_key)
        if cached:
            logger.info("[LLMCity] '%s' chargée depuis le cache (%d activités)",
                        city_name, len(cached.get("activities", [])))
            return cached

    logger.info("[LLMCity] Génération LLM pour '%s'…", city_name)

    # 2. LLM
    raw = _call_llm_for_city(city_name)
    if not raw:
        return None

    # 3. Validation
    data = _validate_and_clean(raw, city_name)
    if not data:
        return None

    city_lat = data["city"]["latitude"]
    city_lon = data["city"]["longitude"]

    # 4. Zones
    _assign_zones(data["activities"], city_lat, city_lon)

    # 5. Matrice de trajets selon le mode de transport
    matrix = _compute_travel_matrix(data["activities"], transport_mode)
    if not matrix:
        return None

    data["travel_matrix"] = matrix
    data["transport_mode"] = transport_mode
    data["data_source"] = "llm+osrm"

    confs = [a.get("confidence", 0.0) for a in data["activities"]]
    data["confidence_stats"] = {
        "average": round(sum(confs) / len(confs), 2) if confs else 0.0,
        "high_confidence_count": sum(1 for c in confs if c >= 0.80),
        "total": len(data["activities"]),
    }

    # 6. Mise en cache (30 jours)
    if _CACHE_AVAILABLE and _cache:
        _cache.set(cache_key, data, ttl_hours=TTL_LLM_CITY)
        logger.info("[LLMCity] '%s' mise en cache (%d activités)", city_name, len(data["activities"]))

    return data
