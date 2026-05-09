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

from llm_client import get_client, LLM_MODEL, QWEN_NO_THINK, _strip_thinking, _extract_json_blob

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

try:
    from cache.cache_manager import CacheManager
    _cache = CacheManager()
    _CACHE_AVAILABLE = True
except ImportError:
    _cache = None  # type: ignore
    _CACHE_AVAILABLE = False

TTL_LLM_CITY = 720  # 30 jours — la géographie ne change pas


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


class LLMCityData(BaseModel):
    city: dict
    activities: list[LLMActivity]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

CITY_ACTIVITIES_PROMPT = """\
You are a travel data expert with complete knowledge of tourist attractions worldwide.

Generate a list of the best tourist activities for the city: {city_name}

Return ONLY a valid JSON object. No markdown, no ```, no explanation.

Required structure:
{{
  "city": {{
    "name": "<official city name>",
    "country": "<ISO 2-letter country code>",
    "latitude": <city center lat>,
    "longitude": <city center lon>,
    "population": <approximate population as integer>
  }},
  "activities": [
    {{
      "id": "<slug_lowercase_no_spaces>",
      "name": "<Activity name in French>",
      "category": "<culture|gastro|nature|shopping|nightlife>",
      "duration_hours": <float, typical visit time>,
      "cost_euros": <integer, entrance fee in euros, 0 if free>,
      "opening_hour": <integer 0-23>,
      "closing_hour": <integer 1-24>,
      "latitude": <accurate GPS latitude>,
      "longitude": <accurate GPS longitude>,
      "priority_score": <integer 1-10, 10=iconic must-see>,
      "confidence": <float 0.80-0.95>
    }},
    ...
  ]
}}

RULES:
1. Generate exactly 22 activities, diverse across categories.
2. Include top-5 iconic landmarks (priority_score 9-10).
3. Include 3-4 gastro experiences (local restaurants, food markets, cooking classes).
4. Include 2-3 nature/parks activities.
5. Include 1-2 shopping/nightlife activities if relevant to the city.
6. GPS coordinates must be accurate (4+ decimal places) for each specific location.
7. Prices in euros — convert from local currency if needed, use 0 for free sites.
8. All activity names in French (e.g., "Musée du Louvre", "Tour Eiffel").
9. id must be a unique lowercase ASCII slug (e.g., "louvre", "eiffel_tower").
10. confidence: 0.90-0.95 for world-famous sites, 0.80-0.88 for well-known local spots.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Appel LLM
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm_for_city(city_name: str, max_retries: int = 2) -> Optional[dict]:
    """Appelle le LLM et retourne le dict JSON brut, ou None en cas d'échec."""
    client = get_client()
    prompt = CITY_ACTIVITIES_PROMPT.format(city_name=city_name)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
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

        return {
            "city": {
                "name": city_info.get("name", city_name),
                "country": city_info.get("country", ""),
                "latitude": float(city_info["latitude"]),
                "longitude": float(city_info["longitude"]),
                "population": int(city_info.get("population", 0)),
            },
            "activities": activities,
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


def _compute_travel_matrix(activities: list[dict]) -> Optional[list[list[int]]]:
    """Calcule la matrice OSRM, ou une matrice haversine de fallback."""
    try:
        from data_provider import osrm_travel_matrix
        coords = [(a["latitude"], a["longitude"]) for a in activities]
        matrix = osrm_travel_matrix(coords, "foot")
        if matrix:
            return matrix
    except Exception as e:
        logger.warning("[LLMCity] OSRM error: %s — fallback haversine", e)

    # Fallback : estimation haversine (vitesse pied ~4 km/h)
    logger.warning("[LLMCity] Matrice haversine utilisée (OSRM indisponible)")
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
                minutes = max(1, round(dist_m / (4000 / 60)))
                row.append(minutes)
        matrix.append(row)
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ─────────────────────────────────────────────────────────────────────────────

def generate_city_data(city_name: str) -> Optional[dict]:
    """
    Génère (ou charge depuis le cache) les données complètes d'une ville.

    Retourne un dict compatible avec le solveur :
        {"city": {...}, "activities": [...], "travel_matrix": [[...]], ...}
    ou None en cas d'échec.
    """
    cache_key = f"llm_city_{city_name.lower().replace(' ', '_')}"

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

    # 5. Matrice de trajets
    matrix = _compute_travel_matrix(data["activities"])
    if not matrix:
        return None

    data["travel_matrix"] = matrix
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
