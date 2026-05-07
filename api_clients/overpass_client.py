"""
Client Overpass API (OpenStreetMap) pour enrichir les données d'activités.

Données récupérées SANS clé API, complètement gratuites :
  - opening_hours : horaires d'ouverture réels
  - fee / charge / admission : tarifs
  - tourism / historic / leisure : type de lieu

Fonctionnement :
  1. fetch_osm_pois(lat, lon, radius_m) → dict[name_normalized → OSMPlace]
  2. enrich_activity(activity_dict, osm_index) → activity_dict enrichi

Le résultat est mis en cache (7 jours) par l'appelant (data_provider).
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
REQUEST_TIMEOUT = 35  # secondes
RETRY_AFTER = 3       # secondes entre deux tentatives

# Distance maximale pour le matching par coordonnées (en mètres)
MAX_COORD_MATCH_DIST = 80


# ─────────────────────────────────────────────────────────────────────────────
# Structure de données OSM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OSMPlace:
    """Données brutes issues d'un élément OpenStreetMap."""
    name: str
    latitude: float
    longitude: float
    opening_hours_raw: Optional[str] = None
    open_hour: Optional[int] = None
    close_hour: Optional[int] = None
    is_free: Optional[bool] = None      # True = gratuit, False = payant
    price_euros: Optional[int] = None
    tourism_type: Optional[str] = None
    historic_type: Optional[str] = None
    leisure_type: Optional[str] = None
    confidence: float = 0.0             # 0.0 si aucune donnée utile, jusqu'à 0.85

    def has_hours(self) -> bool:
        return self.open_hour is not None and self.close_hour is not None

    def has_cost(self) -> bool:
        return self.is_free is not None or self.price_euros is not None


# ─────────────────────────────────────────────────────────────────────────────
# Parsing des horaires OpenStreetMap
# ─────────────────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})")
_CLOSED_RE = re.compile(r"\boff\b|\bclosed\b|\bfermé\b", re.IGNORECASE)


def parse_opening_hours(oh: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse une chaîne OpenStreetMap opening_hours.

    Gère les cas courants :
      "09:00-18:00"
      "Mo-Su 09:00-18:00"
      "Mo-Fr 09:00-17:00; Sa-Su 10:00-16:00"
      "24/7"
      "off" / "closed"
      Formats avec mois : "Apr-Oct: 09:00-19:00"

    Retourne (open_hour_int, close_hour_int) ou (None, None) si non parsable.
    Prend le min des heures d'ouverture et le max des heures de fermeture
    pour donner la plage la plus large sur la semaine.
    """
    if not oh or not oh.strip():
        return None, None

    oh = oh.strip()

    # Cas spéciaux
    if "24/7" in oh:
        return 0, 24
    if _CLOSED_RE.search(oh) and not _TIME_RE.search(oh):
        return None, None  # fermé (pas de plage horaire utile)

    matches = _TIME_RE.findall(oh)
    if not matches:
        return None, None

    opens: list[float] = []
    closes: list[float] = []

    for h_open, m_open, h_close, m_close in matches:
        t_open = int(h_open) + int(m_open) / 60
        t_close = int(h_close) + int(m_close) / 60
        # Minuit = 00:00 → on interprète comme 24h si c'est une heure de fermeture
        if t_close <= t_open:
            t_close = 24.0
        opens.append(t_open)
        closes.append(t_close)

    if not opens:
        return None, None

    return int(min(opens)), int(max(closes))


# ─────────────────────────────────────────────────────────────────────────────
# Parsing du coût
# ─────────────────────────────────────────────────────────────────────────────

_PRICE_RE = re.compile(r"(\d+(?:[.,]\d+)?)")


def parse_cost(
    fee_tag: str,
    charge_tag: str,
    admission_tag: str,
) -> tuple[Optional[bool], Optional[int]]:
    """
    Parse les tags OSM fee/charge/admission.

    Retourne (is_free: bool|None, price_euros: int|None).
    """
    # fee=no → gratuit
    if fee_tag.lower().strip() in ("no", "free", "gratuit", "0", "false"):
        return True, 0

    # Extraire un prix depuis charge ou admission
    raw_price = charge_tag or admission_tag
    if raw_price:
        numbers = _PRICE_RE.findall(raw_price.replace(",", "."))
        if numbers:
            values = [float(n) for n in numbers]
            avg = sum(values) / len(values)
            return False, max(1, int(round(avg)))

    # fee=yes sans montant précis → payant, montant inconnu
    if fee_tag.lower().strip() in ("yes", "oui", "true", "1"):
        return False, None

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Requête Overpass
# ─────────────────────────────────────────────────────────────────────────────

def _build_query(lat: float, lon: float, radius_m: int) -> str:
    """Construit la requête Overpass QL."""
    return f"""
[out:json][timeout:30];
(
  node(around:{radius_m},{lat},{lon})
    [~"^(tourism|historic|leisure|amenity)$"~"."]
    ["name"];
  way(around:{radius_m},{lat},{lon})
    [~"^(tourism|historic|leisure|amenity)$"~"."]
    ["name"];
);
out center tags;
""".strip()


def _normalize_name(name: str) -> str:
    """Normalise un nom pour le matching : minuscules, sans accents, sans ponctuation."""
    import unicodedata
    name = name.lower().strip()
    # Supprimer les accents
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    # Supprimer la ponctuation sauf les espaces et tirets
    name = re.sub(r"[^\w\s-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_osm_pois(
    lat: float,
    lon: float,
    radius_m: int = 10_000,
    max_retries: int = 2,
) -> dict[str, OSMPlace]:
    """
    Interroge l'Overpass API et retourne un index des POIs enrichis.

    Returns:
        {nom_normalisé: OSMPlace}
        La clé est le nom normalisé (minuscules, sans accents).
        Chaque entrée peut aussi être retrouvée par coordonnées via find_by_coords().
    """
    query = _build_query(lat, lon, radius_m)
    result: dict[str, OSMPlace] = {}

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=REQUEST_TIMEOUT,
                headers={"Accept-Language": "fr,en"},
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.Timeout:
            logger.warning("Overpass timeout (tentative %d/%d)", attempt + 1, max_retries + 1)
            if attempt < max_retries:
                time.sleep(RETRY_AFTER)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning("Overpass rate limit – attente %ds", RETRY_AFTER * 2)
                time.sleep(RETRY_AFTER * 2)
            else:
                logger.warning("Overpass HTTP error: %s", e)
                return {}
        except Exception as e:
            logger.warning("Overpass error: %s", e)
            return {}
    else:
        logger.error("Overpass inaccessible après %d tentatives", max_retries + 1)
        return {}

    elements = data.get("elements", [])
    logger.info("[Overpass] %d éléments reçus", len(elements))

    for elem in elements:
        tags = elem.get("tags", {})
        name = tags.get("name", "").strip()
        if not name or len(name) < 3:
            continue

        # Coordonnées (way → center, node → direct)
        if elem["type"] == "node":
            e_lat, e_lon = float(elem["lat"]), float(elem["lon"])
        else:
            center = elem.get("center", {})
            if not center:
                continue
            e_lat, e_lon = float(center["lat"]), float(center["lon"])

        # Parsing des horaires
        oh_raw = tags.get("opening_hours", "")
        open_h, close_h = parse_opening_hours(oh_raw)

        # Parsing du coût
        is_free, price = parse_cost(
            tags.get("fee", ""),
            tags.get("charge", ""),
            tags.get("admission", ""),
        )

        # Calcul du score de confiance
        conf = 0.0
        if open_h is not None:
            conf += 0.45
        if is_free is not None or price is not None:
            conf += 0.30
        if conf > 0:
            conf = min(0.85, conf + 0.10)

        place = OSMPlace(
            name=name,
            latitude=e_lat,
            longitude=e_lon,
            opening_hours_raw=oh_raw or None,
            open_hour=open_h,
            close_hour=close_h,
            is_free=is_free,
            price_euros=price,
            tourism_type=tags.get("tourism"),
            historic_type=tags.get("historic"),
            leisure_type=tags.get("leisure"),
            confidence=conf,
        )

        # Indexer par nom normalisé (plusieurs alias)
        for key in _name_keys(name):
            if key not in result or place.confidence > result[key].confidence:
                result[key] = place

    logger.info("[Overpass] %d POIs indexés (confidence > 0 : %d)",
                len(result),
                sum(1 for p in result.values() if p.confidence > 0))
    return result


def _name_keys(name: str) -> list[str]:
    """Génère plusieurs variantes de clé pour un nom."""
    keys = [_normalize_name(name)]
    # Supprimer les articles courants au début
    prefixes = ["the ", "le ", "la ", "les ", "l'", "un ", "une "]
    norm = keys[0]
    for p in prefixes:
        if norm.startswith(p):
            keys.append(norm[len(p):])
    return list(dict.fromkeys(keys))  # dédupliquer en gardant l'ordre


def find_by_coords(
    osm_index: dict[str, OSMPlace],
    lat: float,
    lon: float,
    max_dist_m: float = MAX_COORD_MATCH_DIST,
) -> Optional[OSMPlace]:
    """
    Cherche un POI OSM par coordonnées GPS (distance ≤ max_dist_m).
    Utile quand le matching par nom échoue.
    """
    best: Optional[OSMPlace] = None
    best_dist = float("inf")

    for place in osm_index.values():
        d = _haversine(lat, lon, place.latitude, place.longitude)
        if d < max_dist_m and d < best_dist:
            best_dist = d
            best = place

    return best


def enrich_activity(
    activity: dict,
    osm_index: dict[str, OSMPlace],
) -> dict:
    """
    Enrichit un dict d'activité avec les données OSM disponibles.

    Stratégie :
      1. Matching par nom (normalisé)
      2. Matching par coordonnées GPS
      3. Si aucun match : l'activité reste inchangée

    Champs mis à jour si la donnée OSM est plus fiable que l'estimation :
      - opening_hour, closing_hour (si has_hours())
      - cost_euros (si has_cost() ET plus précis)
      - confidence (champ ajouté)
      - data_source_detail (champ ajouté pour diagnostic)
    """
    name = activity.get("name", "")
    lat = activity.get("latitude", 0.0)
    lon = activity.get("longitude", 0.0)

    # Recherche OSM
    osm: Optional[OSMPlace] = None

    norm_name = _normalize_name(name)
    if norm_name in osm_index:
        osm = osm_index[norm_name]
    else:
        # Essai par clés alternatives
        for key in _name_keys(name)[1:]:
            if key in osm_index:
                osm = osm_index[key]
                break

    if osm is None and lat and lon:
        osm = find_by_coords(osm_index, lat, lon)

    if osm is None or osm.confidence == 0:
        return activity

    enriched = dict(activity)
    enriched["data_source_detail"] = "osm"

    # Horaires : priorité aux données OSM si disponibles
    if osm.has_hours():
        enriched["opening_hour"] = osm.open_hour
        enriched["closing_hour"] = osm.close_hour
        enriched["confidence"] = max(
            enriched.get("confidence", 0.0), osm.confidence
        )
        logger.debug("[OSM] %s → horaires %dh-%dh (conf %.2f)",
                     name, osm.open_hour, osm.close_hour, osm.confidence)

    # Coût : OSM est prioritaire si la donnée est explicite
    if osm.has_cost():
        current_conf = enriched.get("confidence", 0.0)
        if osm.confidence > current_conf or enriched.get("cost_euros") == 0:
            if osm.is_free:
                enriched["cost_euros"] = 0
            elif osm.price_euros is not None:
                enriched["cost_euros"] = osm.price_euros
            enriched["confidence"] = max(current_conf, osm.confidence)
            logger.debug("[OSM] %s → coût %s€ (conf %.2f)",
                         name, enriched["cost_euros"], osm.confidence)

    return enriched
