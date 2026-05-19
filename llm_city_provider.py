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
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from llm_client import (
    chat_with_fallback, QWEN_NO_THINK, _extract_json_blob, parse_json_salvage,
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
    address: str = ""
    nearest_stop: str = ""          # nom de l'arrêt le plus proche
    transit_options: list = Field(default_factory=list)  # [{type, line}, ...]
    transit_exit: str = ""

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

The user is planning a trip of {num_days} days, so generate {n_activities} activities
to give the planner a comfortable buffer (≥ 4 per day + extras for refinement).

Return ONLY a valid JSON object. No markdown, no explanations.

Structure:
{{
  "city": {{"name":"...","country":"<2-letter ISO>","latitude":<lat>,"longitude":<lon>,"population":<int>}},
  "activities": [ {{"id":"slug","name":"<French>","category":"<culture|gastro|nature|shopping|nightlife>","duration_hours":<float>,"cost_euros":<int>,"opening_hour":<int>,"closing_hour":<int>,"latitude":<lat>,"longitude":<lon>,"priority_score":<1-10>,"confidence":<0.80-0.95>,"address":"<exact street address>","nearest_stop":"<official stop name>","transit_options":[{{"type":"<metro|bus|tram|rer|train|funiculaire>","line":"<number or letter>"}}],"transit_exit":"<exit name or empty>"}}, ... ],
  "hotels": [ {{"name":"<real hotel>","address":"<full address>","price_per_night":<int>,"stars":<1-5>,"latitude":<lat>,"longitude":<lon>,"description":"<French, <12 words>"}}, ... ]
}}

RULES:
- Generate exactly {n_activities} activities, diverse across the 5 categories — at least 4 per planned day plus a buffer for refinement.
- Mix proportions: ~40% culture (including a few iconic priority 9-10), ~20% gastro, ~20% nature, ~10% shopping, ~10% nightlife (adjust to what the city actually offers).
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

- Exactly 3 REAL hotels in the city, covering 3 price tiers. Use ACTUAL booking prices (what Booking.com / Hotels.com shows today):
  * Budget hotel (2-3★): real price typically 60-130€/night for major European cities
  * Mid-range hotel (3-4★): real price typically 140-280€/night
  * Premium hotel (4-5★): real price typically 290-600€/night
  CRITICAL: Do NOT include ultra-luxury palaces (Ritz, Plaza Athénée, George V, Four Seasons, etc.) — their real price is 1000-5000€/night and would bust any normal budget. Use good 4-5★ hotels at realistic prices instead.
  CRITICAL: price_per_night must reflect the actual nightly rate for 1 room, not total stay cost.
- address: FULL address with street number (e.g. "5 Avenue Anatole France, 75007 Paris" or "99 Rue de Rivoli, 75001 Paris"). MANDATORY street number. Never just a street name without number.
- nearest_stop: official name of the nearest public transport stop (any mode: metro, bus, tram, RER, train, funiculaire). Empty string if truly none.
- transit_options: ALL transport options at nearest_stop. Each entry: {{"type": "<metro|bus|tram|rer|train|funiculaire>", "line": "<number or letter>"}}. Examples: [{{"type":"metro","line":"6"}},{{"type":"metro","line":"9"}}] for Trocadéro. Empty list only if no public transport within 800m.
- transit_exit: specific exit/direction if helpful (e.g. "Sortie Tour Eiffel"), else empty string.
- Be CONCISE: no extra fields, no commentary. Output the JSON and stop.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Appel LLM
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm_for_city(
    city_name: str,
    num_days: int = 5,
    max_retries: int = 1,
) -> Optional[dict]:
    """Appelle le LLM (primaire + fallback) et retourne le dict JSON brut,
    ou None en cas d'échec total.
    Dimensionne le pool selon num_days : ≥ 4 activités/jour + buffer."""
    # Pool : 4 activités/jour + 4 de buffer, plancher 16, plafond 32 (limite LLM)
    n_activities = max(16, min(32, num_days * 4 + 4))
    prompt = CITY_ACTIVITIES_PROMPT.format(
        city_name=city_name,
        num_days=num_days,
        n_activities=n_activities,
    )

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = chat_with_fallback(
                timeout=240.0,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=6000,
                response_format={"type": "json_object"},
                extra_body=QWEN_NO_THINK,
            )
            raw = resp.choices[0].message.content or ""
            blob = _extract_json_blob(raw)
            # parse_json_salvage : récupère le préfixe valide si le LLM coupe
            # un objet en plein milieu (typique sur réponses très longues).
            data = parse_json_salvage(blob)
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
            # Filet : LLM peut occasionnellement renvoyer une string au lieu d'un dict.
            if not isinstance(item, dict):
                logger.debug("[LLMCity] Item non-dict ignoré: %r", item)
                continue
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
                    "address": act.address or "",
                    "nearest_stop": act.nearest_stop or "",
                    "transit_options": [
                        {"type": str(o.get("type","")).lower(), "line": str(o.get("line",""))}
                        for o in (act.transit_options or [])
                        if isinstance(o, dict) and o.get("type")
                    ],
                    "transit_exit": act.transit_exit or "",
                    "data_source_detail": "llm",
                })
            except (ValidationError, TypeError, AttributeError) as e:
                name = item.get("name", "?") if isinstance(item, dict) else repr(item)[:40]
                logger.debug("[LLMCity] Activité invalide ignorée: %s — %s", name, e)

        if len(activities) < 5:
            logger.error("[LLMCity] Trop peu d'activités valides (%d) pour '%s'",
                         len(activities), city_name)
            return None

        # Hôtels (optionnel)
        hotels = []
        for item in raw.get("hotels", []) or []:
            if not isinstance(item, dict):
                logger.debug("[LLMCity] Hôtel non-dict ignoré: %r", item)
                continue
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
            except (ValidationError, TypeError, AttributeError) as e:
                name = item.get("name", "?") if isinstance(item, dict) else repr(item)[:40]
                logger.debug("[LLMCity] Hôtel invalide ignoré: %s — %s", name, e)

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


def _compute_travel_matrix(
    activities: list[dict],
    transport_mode: str = "foot",
) -> Optional[list[list[int]]]:
    """Calcule la matrice OSRM. Pas de fallback : si OSRM échoue, on renvoie
    None et l'appelant abandonne la génération de la ville."""
    from data_provider import osrm_travel_matrix
    coords = [(a["latitude"], a["longitude"]) for a in activities]
    matrix = osrm_travel_matrix(coords, transport_mode)
    if not matrix:
        logger.error("[LLMCity] OSRM indisponible — pas de fallback, ville rejetée")
        return None
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée public
# ─────────────────────────────────────────────────────────────────────────────

_ENRICHMENT_FIELDS = (
    "opening_hours", "flexible_hours", "closed_days",
    "price_info", "student_discount", "student_cost_euros",
    "last_entry_before_close_minutes",
    "address", "nearest_stop", "transit_options", "transit_exit",
    "opening_hour", "closing_hour", "cost_euros",
    "data_confidence", "data_source",
)


def _apply_enrichment_cache(data: dict, city_name: str) -> bool:
    """Merge le cache d'enrichissement activity_info dans city_data. Non-bloquant."""
    try:
        import re as _re
        from activity_info_service import _load_cache as _ai_cache
        city_key = _re.sub(r"[^\w]", "_", city_name.lower())
        cached = _ai_cache(city_key)
        if not cached:
            return False
        enriched_by_id = {k: v for k, v in cached.items() if not k.startswith("_")}
        if not enriched_by_id:
            return False
        applied = 0
        for act in data.get("activities", []):
            e = enriched_by_id.get(act["id"])
            if e:
                for f in _ENRICHMENT_FIELDS:
                    if f in e:
                        act[f] = e[f]
                applied += 1
        if applied:
            data["data_source"] = data.get("data_source", "llm+osrm") + "+enriched"
            logger.info("[LLMCity] '%s' — %d activités enrichies depuis cache", city_name, applied)
        return applied > 0
    except Exception as e:
        logger.debug("[LLMCity] _apply_enrichment_cache: %s", e)
        return False


def _enrich_in_background(activities: list, city_name: str, city_country: str) -> None:
    """Lance l'enrichissement (horaires, prix, metro…) dans un thread daemon."""
    import threading, copy

    def _run():
        try:
            from activity_info_service import enrich_activities_batch
            enrich_activities_batch(copy.deepcopy(activities), city_name, city_country)
            logger.info("[LLMCity] Enrichissement background terminé pour '%s'", city_name)
        except Exception as exc:
            logger.warning("[LLMCity] Enrichissement background échoué pour '%s': %s", city_name, exc)

    threading.Thread(target=_run, daemon=True, name=f"enrich-{city_name}").start()


def generate_city_data(
    city_name: str,
    transport_mode: str = "foot",
    num_days: int = 5,
) -> Optional[dict]:
    """
    Génère (ou charge depuis le cache) les données complètes d'une ville.

    Retourne un dict compatible avec le solveur :
        {"city": {...}, "activities": [...], "travel_matrix": [[...]],
         "hotels": [...], "transport_mode": "foot", ...}
    ou None en cas d'échec.

    L'enrichissement (horaires réels, prix, stations de métro…) tourne en background
    et est appliqué dès le prochain appel via le cache activity_info.
    """
    base_key = f"llm_city_{city_name.lower().replace(' ', '_')}"
    cache_key = base_key if transport_mode == "foot" else f"{base_key}_{transport_mode}"
    city_country = ""

    # 1. Cache LLM ville
    if _CACHE_AVAILABLE and _cache:
        cached = _cache.get(cache_key)
        if cached:
            logger.info("[LLMCity] '%s' chargée depuis le cache (%d activités)",
                        city_name, len(cached.get("activities", [])))
            city_country = cached.get("city", {}).get("country", "")
            # Appliquer l'enrichissement depuis son propre cache si disponible
            enriched = _apply_enrichment_cache(cached, city_name)
            if enriched and _CACHE_AVAILABLE and _cache:
                # Persister la version enrichie pour éviter de re-merger à chaque appel
                _cache.set(cache_key, cached, ttl_hours=TTL_LLM_CITY)
            elif not enriched:
                # Pas encore de cache enrichissement → lancer en background
                _enrich_in_background(cached.get("activities", []), city_name, city_country)
            return cached

    logger.info("[LLMCity] Génération LLM pour '%s'…", city_name)

    # 2. LLM
    raw = _call_llm_for_city(city_name, num_days=num_days)
    if not raw:
        return None

    # 3. Validation
    data = _validate_and_clean(raw, city_name)
    if not data:
        return None

    city_lat = data["city"]["latitude"]
    city_lon = data["city"]["longitude"]
    city_country = data["city"].get("country", "")

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

    # 6. Cache immédiat (sans enrichissement)
    if _CACHE_AVAILABLE and _cache:
        _cache.set(cache_key, data, ttl_hours=TTL_LLM_CITY)
        logger.info("[LLMCity] '%s' mise en cache (%d activités)", city_name, len(data["activities"]))

    # 7. Enrichissement en background (station métro, horaires, prix)
    _enrich_in_background(data["activities"], city_name, city_country)

    return data
