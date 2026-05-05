"""
Fournisseur de données dynamique pour le planificateur de voyage.

Utilise deux API :
  - OpenTripMap : points d'intérêt touristiques (clé requise, gratuite)
  - OSRM : temps de trajet réels (Open Source Routing Machine, sans clé)

Pour exécuter ce module :
  - pip install requests
  - clé OpenTripMap : https://opentripmap.io/product
  - OSRM : aucune clé
"""

import requests
import json
import os
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# OSRM public demo server (gratuit, pas de clé)
OSRM_BASE_URL = "http://router.project-osrm.org"

# OpenTripMap (clé OBLIGATOIRE même pour géocoder)
# Obtiens une clé gratuite sur https://opentripmap.io/product
OPENTRIPMAP_API_KEY = os.environ.get("OPENTRIPMAP_API_KEY", "")
OPENTRIPMAP_BASE_URL = "https://api.opentripmap.com/0.1"

# Mapping catégories OpenTripMap → nos catégories
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


@dataclass
class RawPlace:
    """Lieu brut retourné par l'API."""
    id: str
    name: str
    latitude: float
    longitude: float
    kinds: str  # catégories OpenTripMap, séparées par des virgules
    description: str = ""
    wikipedia_url: str = ""
    image_url: str = ""
    rating: int = 0  # 1-7 selon OpenTripMap (7 = le plus populaire)


# ─────────────────────────────────────────────
# OpenTripMap : Points d'intérêt
# ─────────────────────────────────────────────

def geocode_city(city_name: str, lang: str = "fr") -> Optional[dict]:
    """
    Géocode une ville via OpenTripMap.
    Retourne {name, country, lat, lon, population} ou None.
    """
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
        print(f"[OpenTripMap] Geocoding error: {e}")
    return None


def fetch_places(
    lat: float, lon: float,
    radius_m: int = 10000,
    kinds: str = "interesting_places,museums,historic,foods,gardens_and_parks,architecture",
    limit: int = 50,
    min_rating: int = 3,
    lang: str = "fr"
) -> list[RawPlace]:
    """
    Récupère les points d'intérêt autour d'une position GPS.
    
    Args:
        lat, lon : coordonnées du centre (la ville)
        radius_m : rayon de recherche en mètres (défaut 10km)
        kinds : types de lieux (voir doc OpenTripMap)
        limit : nombre max de résultats
        min_rating : filtre sur la popularité (1-7, 3 = assez connu)
        lang : langue des descriptions
    
    Returns:
        Liste de RawPlace triés par rating décroissant
    """
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

        # Trier par popularité
        places.sort(key=lambda p: p.rating, reverse=True)
        return places

    except Exception as e:
        print(f"[OpenTripMap] Fetch error: {e}")
        return []


def fetch_place_details(xid: str, lang: str = "fr") -> Optional[dict]:
    """
    Récupère les détails d'un lieu (description, image, Wikipedia, etc.).
    Attention : rate limité, appeler avec parcimonie.
    """
    url = f"{OPENTRIPMAP_BASE_URL}/{lang}/places/xid/{xid}"
    params = {}
    if OPENTRIPMAP_API_KEY:
        params["apikey"] = OPENTRIPMAP_API_KEY

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[OpenTripMap] Details error for {xid}: {e}")
        return None


# ─────────────────────────────────────────────
# OSRM : Temps de trajet (100% gratuit, pas de clé)
# ─────────────────────────────────────────────

def osrm_travel_time(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    mode: str = "foot"
) -> Optional[int]:
    """
    Temps de trajet en minutes via OSRM (serveur public gratuit).
    
    Modes : "foot" (marche), "car" (voiture), "bike" (vélo)
    Serveurs publics OSRM :
      - foot:  https://routing.openstreetmap.de/routed-foot
      - car:   http://router.project-osrm.org (ou routed-car)
      - bike:  https://routing.openstreetmap.de/routed-bike
    """
    servers = {
        "foot": "https://routing.openstreetmap.de/routed-foot",
        "car": "http://router.project-osrm.org",
        "bike": "https://routing.openstreetmap.de/routed-bike",
    }
    base = servers.get(mode, servers["foot"])

    # OSRM format : lon,lat (pas lat,lon !)
    coords = f"{lon1},{lat1};{lon2},{lat2}"
    url = f"{base}/route/v1/{mode if mode != 'foot' else 'driving'}/{coords}"
    params = {"overview": "false"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") == "Ok" and data.get("routes"):
            duration_sec = data["routes"][0]["duration"]
            return max(1, round(duration_sec / 60))
    except Exception as e:
        print(f"[OSRM] Route error: {e}")
    return None


def osrm_travel_matrix(
    coordinates: list[tuple[float, float]],
    mode: str = "foot"
) -> Optional[list[list[int]]]:
    """
    Matrice de temps de trajet NxN via OSRM Table API.
    Beaucoup plus efficace que N² appels individuels.
    
    Args:
        coordinates : liste de (lat, lon)
        mode : "foot", "car", "bike"
    
    Returns:
        Matrice NxN de durées en minutes, ou None si erreur
    """
    if len(coordinates) < 2:
        return None

    servers = {
        "foot": "https://routing.openstreetmap.de/routed-foot",
        "car": "http://router.project-osrm.org",
        "bike": "https://routing.openstreetmap.de/routed-bike",
    }
    base = servers.get(mode, servers["foot"])

    # OSRM format : lon,lat;lon,lat;...
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    url = f"{base}/table/v1/{mode if mode != 'foot' else 'driving'}/{coords_str}"
    params = {"annotations": "duration"}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") == "Ok" and data.get("durations"):
            # Convertir secondes → minutes
            return [
                [max(1, round(cell / 60)) if cell is not None else 30
                 for cell in row]
                for row in data["durations"]
            ]
    except Exception as e:
        print(f"[OSRM] Matrix error: {e}")
    return None


# ─────────────────────────────────────────────
# Conversion OpenTripMap → format Activity du solveur
# ─────────────────────────────────────────────

def classify_category(otm_kinds: str) -> str:
    """Convertit les catégories OpenTripMap vers nos 5 catégories."""
    kinds = otm_kinds.lower().split(",")
    for kind in kinds:
        kind = kind.strip()
        for otm_key, our_cat in OTM_CATEGORY_MAP.items():
            if otm_key in kind:
                return our_cat
    return "culture"  # défaut


#pas vraiment fonctionnel, il faut trouver un autre moyen d'estimer la durée de visite
def estimate_duration(category: str, rating: int) -> float:
    """Estime la durée de visite selon la catégorie et la popularité."""
    base = {"culture": 1.5, "gastro": 2.0, "nature": 2.0, "shopping": 1.5, "nightlife": 2.5}
    duration = base.get(category, 1.5)
    if rating >= 6:
        duration += 0.5
    return duration


#pas vraiment fonctionnel, il faut trouver un autre moyen d'estimer le coût d'entrée
def estimate_cost(category: str, rating: int) -> int:
    """Estime le coût d'entrée selon la catégorie et la popularité."""
    if category == "nature":
        return 0
    if category == "gastro":
        return 30 + rating * 3
    if category == "nightlife":
        return 20 + rating * 3
    if category == "shopping":
        return 30
    if rating >= 5:
        return 10 + rating * 2
    return 0


#pas vraiment fonctionnel, il faut trouver un autre moyen d'estimer les horaires d'ouverture/fermeture
def estimate_hours(category: str) -> tuple[int, int]:
    """Estime les horaires d'ouverture/fermeture."""
    hours = {
        "culture": (9, 18),
        "gastro": (11, 22),
        "nature": (7, 20),
        "shopping": (10, 20),
        "nightlife": (19, 24),
    }
    return hours.get(category, (9, 18))


def raw_places_to_activities(places: list[RawPlace]) -> list[dict]:
    """
    Convertit les lieux OpenTripMap en activités pour le solveur CP-SAT.
    """
    activities = []
    seen_names = set()

    for place in places:
        # Dédupliquer par nom
        clean_name = place.name.strip()
        if clean_name in seen_names:
            continue
        seen_names.add(clean_name)

        category = classify_category(place.kinds)
        open_h, close_h = estimate_hours(category)

        # Créer un ID propre
        activity_id = place.id or clean_name.lower().replace(" ", "_")[:30]

        activities.append({
            "id": activity_id,
            "name": clean_name,
            "category": category,
            "duration_hours": estimate_duration(category, place.rating),
            "cost_euros": estimate_cost(category, place.rating),
            "opening_hour": open_h,
            "closing_hour": close_h,
            "latitude": place.latitude,
            "longitude": place.longitude,
            "priority_score": min(10, max(1, place.rating + 3)),
            "zone": "",  # sera rempli après clustering
        })

    return activities


# ─────────────────────────────────────────────
# Pipeline complet : ville → données prêtes pour le solveur
# ─────────────────────────────────────────────

def build_city_data(
    city_name: str,
    radius_m: int = 8000,
    max_activities: int = 30,
    transport_mode: str = "foot",
    lang: str = "fr"
) -> Optional[dict]:
    """
    Pipeline complet :
      1. Géocode la ville (OpenTripMap)
      2. Récupère les POI (OpenTripMap)
      3. Convertit en activités
      4. Matrice de trajets via OSRM (échoue si OSRM ne répond pas)

    Returns:
        {"city":{...}, "activities":[...], "travel_matrix":[[...]], "data_source":"..."}
        ou None si une étape critique échoue.
    """
    print(f"[Pipeline] Recherche de données pour {city_name}...")

    geo = geocode_city(city_name, lang)
    if not geo:
        print(f"[Pipeline] Ville '{city_name}' non trouvée")
        return None

    lat, lon = geo["lat"], geo["lon"]
    print(f"[Pipeline] {city_name} trouvée : ({lat}, {lon})")

    # Liste de catégories valides chez OpenTripMap (août 2025).
    # 'archaeological_sites' a été retiré car il provoque un 400 sur l'API actuelle.
    kinds = ",".join([
        "interesting_places", "museums", "historic", "architecture",
        "monuments_and_memorials", "churches",
        "foods", "gardens_and_parks", "natural", "amusements", "shops"
    ])
    places = fetch_places(lat, lon, radius_m, kinds, max_activities * 2, min_rating=2, lang=lang)
    print(f"[Pipeline] {len(places)} lieux trouvés via OpenTripMap")

    if not places:
        print("[Pipeline] Aucun lieu trouvé")
        return None

    # 3. Convertir en activités
    activities = raw_places_to_activities(places)[:max_activities]
    print(f"[Pipeline] {len(activities)} activités après filtrage")

    coords = [(a["latitude"], a["longitude"]) for a in activities]
    print("[Pipeline] Calcul de la matrice de temps via OSRM...")
    travel_matrix = osrm_travel_matrix(coords, transport_mode)
    if not travel_matrix:
        print("[Pipeline] OSRM indisponible — abandon (pas de fallback)")
        return None
    data_source = "opentripmap+osrm"
    print(f"[Pipeline] Matrice OSRM {len(travel_matrix)}x{len(travel_matrix)} calculée")

    # Assigner des zones par clustering simple (par quadrant géographique)
    for act in activities:
        dlat = act["latitude"] - lat
        dlon = act["longitude"] - lon
        if dlat >= 0 and dlon >= 0:
            act["zone"] = "nord-est"
        elif dlat >= 0 and dlon < 0:
            act["zone"] = "nord-ouest"
        elif dlat < 0 and dlon >= 0:
            act["zone"] = "sud-est"
        else:
            act["zone"] = "sud-ouest"

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
        "data_source": data_source,
    }


# ─────────────────────────────────────────────
# Cache local (pour ne pas re-appeler les API)
# ─────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def save_city_cache(city_name: str, data: dict):
    """Sauvegarde les données d'une ville en cache local."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{city_name.lower().replace(' ', '_')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Cache] Données sauvegardées dans {path}")


def load_city_cache(city_name: str) -> Optional[dict]:
    """Charge les données d'une ville depuis le cache."""
    path = os.path.join(CACHE_DIR, f"{city_name.lower().replace(' ', '_')}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Cache] Données chargées depuis {path}")
        return data
    return None


def get_city_data(city_name: str, **kwargs) -> Optional[dict]:
    """
    Point d'entrée : charge du cache ou appelle les API.
    """
    # 1. Vérifier le cache
    cached = load_city_cache(city_name)
    if cached:
        return cached

    # 2. Appeler les API
    data = build_city_data(city_name, **kwargs)
    if data:
        save_city_cache(city_name, data)
        return data

    return None


# ─────────────────────────────────────────────
# Données pré-cachées pour Rome (démo / fallback)
# ─────────────────────────────────────────────

ROME_PRECACHED = {
    "city": {
        "name": "Roma", "country": "IT",
        "latitude": 41.8933, "longitude": 12.4829,
        "population": 2873000,
    },
    "activities": [
        {"id": "colosseum", "name": "Colisée", "category": "culture", "duration_hours": 2.5, "cost_euros": 18, "opening_hour": 8, "closing_hour": 19, "latitude": 41.8902, "longitude": 12.4922, "priority_score": 10, "zone": "sud-est"},
        {"id": "vatican", "name": "Musées du Vatican", "category": "culture", "duration_hours": 3.5, "cost_euros": 17, "opening_hour": 8, "closing_hour": 18, "latitude": 41.9065, "longitude": 12.4536, "priority_score": 10, "zone": "nord-ouest"},
        {"id": "pantheon", "name": "Panthéon", "category": "culture", "duration_hours": 1.0, "cost_euros": 0, "opening_hour": 8, "closing_hour": 19, "latitude": 41.8986, "longitude": 12.4769, "priority_score": 8, "zone": "nord-ouest"},
        {"id": "forum", "name": "Forum Romain", "category": "culture", "duration_hours": 2.0, "cost_euros": 16, "opening_hour": 8, "closing_hour": 19, "latitude": 41.8925, "longitude": 12.4853, "priority_score": 9, "zone": "sud-est"},
        {"id": "trevi", "name": "Fontaine de Trevi", "category": "culture", "duration_hours": 1.0, "cost_euros": 0, "opening_hour": 7, "closing_hour": 23, "latitude": 41.9009, "longitude": 12.4833, "priority_score": 9, "zone": "nord-est"},
        {"id": "borghese", "name": "Galerie Borghèse", "category": "culture", "duration_hours": 2.0, "cost_euros": 15, "opening_hour": 9, "closing_hour": 19, "latitude": 41.9142, "longitude": 12.4921, "priority_score": 9, "zone": "nord-est"},
        {"id": "trastevere", "name": "Trastevere", "category": "nature", "duration_hours": 2.0, "cost_euros": 0, "opening_hour": 9, "closing_hour": 23, "latitude": 41.8840, "longitude": 12.4700, "priority_score": 7, "zone": "sud-ouest"},
        {"id": "piazza_navona", "name": "Piazza Navona", "category": "culture", "duration_hours": 1.0, "cost_euros": 0, "opening_hour": 7, "closing_hour": 23, "latitude": 41.8992, "longitude": 12.4731, "priority_score": 8, "zone": "nord-ouest"},
        {"id": "campo_fiori", "name": "Campo de' Fiori", "category": "gastro", "duration_hours": 1.5, "cost_euros": 15, "opening_hour": 8, "closing_hour": 22, "latitude": 41.8956, "longitude": 12.4722, "priority_score": 7, "zone": "sud-ouest"},
        {"id": "catacombs", "name": "Catacombes de San Callisto", "category": "culture", "duration_hours": 1.5, "cost_euros": 10, "opening_hour": 9, "closing_hour": 17, "latitude": 41.8582, "longitude": 12.5137, "priority_score": 6, "zone": "sud-est"},
        {"id": "castel_angelo", "name": "Château Saint-Ange", "category": "culture", "duration_hours": 2.0, "cost_euros": 15, "opening_hour": 9, "closing_hour": 19, "latitude": 41.9031, "longitude": 12.4663, "priority_score": 8, "zone": "nord-ouest"},
        {"id": "villa_borghese", "name": "Parc Villa Borghèse", "category": "nature", "duration_hours": 2.0, "cost_euros": 0, "opening_hour": 7, "closing_hour": 20, "latitude": 41.9145, "longitude": 12.4855, "priority_score": 7, "zone": "nord-est"},
        {"id": "testaccio", "name": "Food tour Testaccio", "category": "gastro", "duration_hours": 3.0, "cost_euros": 45, "opening_hour": 11, "closing_hour": 15, "latitude": 41.8761, "longitude": 12.4756, "priority_score": 8, "zone": "sud-ouest"},
        {"id": "appian_way", "name": "Via Appia Antica", "category": "nature", "duration_hours": 3.0, "cost_euros": 15, "opening_hour": 8, "closing_hour": 18, "latitude": 41.8554, "longitude": 12.5245, "priority_score": 7, "zone": "sud-est"},
        {"id": "ostia", "name": "Ostia Antica", "category": "culture", "duration_hours": 4.0, "cost_euros": 12, "opening_hour": 8, "closing_hour": 17, "latitude": 41.7558, "longitude": 12.2916, "priority_score": 6, "zone": "sud-ouest"},
        {"id": "cooking_class", "name": "Cours de cuisine", "category": "gastro", "duration_hours": 3.0, "cost_euros": 65, "opening_hour": 10, "closing_hour": 14, "latitude": 41.8840, "longitude": 12.4690, "priority_score": 7, "zone": "sud-ouest"},
        {"id": "piazza_spagna", "name": "Place d'Espagne", "category": "culture", "duration_hours": 1.0, "cost_euros": 0, "opening_hour": 7, "closing_hour": 23, "latitude": 41.9060, "longitude": 12.4828, "priority_score": 8, "zone": "nord-est"},
        {"id": "wine_tasting", "name": "Dégustation de vins", "category": "gastro", "duration_hours": 2.0, "cost_euros": 35, "opening_hour": 16, "closing_hour": 21, "latitude": 41.8965, "longitude": 12.4800, "priority_score": 6, "zone": "sud-est"},
        {"id": "trastevere_night", "name": "Soirée Trastevere", "category": "nightlife", "duration_hours": 3.0, "cost_euros": 40, "opening_hour": 19, "closing_hour": 24, "latitude": 41.8845, "longitude": 12.4695, "priority_score": 7, "zone": "sud-ouest"},
        {"id": "pincio_sunset", "name": "Coucher de soleil Pincio", "category": "nature", "duration_hours": 1.5, "cost_euros": 0, "opening_hour": 17, "closing_hour": 21, "latitude": 41.9115, "longitude": 12.4785, "priority_score": 8, "zone": "nord-ouest"},
    ],
    "travel_matrix": None,
    "data_source": "precached",
}


def get_rome_data() -> Optional[dict]:
    """Retourne les données pré-cachées de Rome. La matrice de trajets est
    calculée à la demande via OSRM (pas de fallback : retourne None si OSRM HS)."""
    data = {k: v for k, v in ROME_PRECACHED.items()}
    data["activities"] = [dict(a) for a in ROME_PRECACHED["activities"]]
    if data["travel_matrix"] is None:
        coords = [(a["latitude"], a["longitude"]) for a in data["activities"]]
        matrix = osrm_travel_matrix(coords, "foot")
        if not matrix:
            print("[get_rome_data] OSRM indisponible — abandon")
            return None
        data["travel_matrix"] = matrix
        data["data_source"] = "precached+osrm"
    return data


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Test avec données pré-cachées de Rome (matrice via OSRM)")
    print("=" * 50)

    rome = get_rome_data()
    if rome:
        print(f"\nVille : {rome['city']['name']} ({rome['city']['country']})")
        print(f"Source : {rome['data_source']}")
        print(f"Activités : {len(rome['activities'])}")
        print(f"Matrice : {len(rome['travel_matrix'])}x{len(rome['travel_matrix'])}")
        print("\nTop 5 activités :")
        for a in sorted(rome["activities"], key=lambda x: x["priority_score"], reverse=True)[:5]:
            print(f"  {a['priority_score']}/10 | {a['name']} ({a['category']}) - {a['cost_euros']}€, {a['duration_hours']}h")

    print("\n" + "=" * 50)
    print("Test avec API réelles (OpenTripMap + OSRM)")
    print("=" * 50)
    data = build_city_data("Barcelona", max_activities=10)
    if data:
        print(f"\nVille : {data['city']['name']}")
        print(f"Source : {data['data_source']}")
        print(f"Activités : {len(data['activities'])}")
        for a in data["activities"][:5]:
            print(f"  {a['name']} ({a['category']}) - {a['cost_euros']}€")
