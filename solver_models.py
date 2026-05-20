"""Modèles de domaine + utilitaires pour le solveur. Séparé de solver.py pour garder ce dernier focalisé sur la modélisation CP-SAT"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Activity:
    id: str
    name: str
    category: str
    duration_hours: float
    cost_euros: int
    opening_hour: int
    closing_hour: int
    zone: str
    priority_score: int
    available_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    latitude: float = 0.0
    longitude: float = 0.0
    nearest_stop: str = ""
    transit_options: list = field(default_factory=list)
    transit_exit: str = ""
    closed_days: list[str] = field(default_factory=list)

@dataclass
class TravelConstraints:
    """Contraintes consolidées (LLM-extraites + valeurs par défaut)."""
    destination: str = "Rome"
    num_days: int = 5
    total_budget: int = 2000
    daily_food_budget: int = 60
    num_travelers: int = 1
    hotel_per_night: int = 100

    preferred_categories: list[str] = field(default_factory=list)
    avoided_categories: list[str] = field(default_factory=list)
    preferred_pace: str = "moderate"
    morning_preference: str = "culture"

    must_visit: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)
    must_visit_on_day: dict[str, int] = field(default_factory=dict)
    incompatible_pairs: list[tuple[str, str]] = field(default_factory=list)
    prerequisites: dict[str, str] = field(default_factory=dict)

    max_activities_per_day: int = 6
    min_activities_per_day: int = 1
    max_per_category: dict[str, int] = field(default_factory=dict)
    min_per_category: dict[str, int] = field(default_factory=dict)

    day_start_hour: Optional[int] = None
    day_end_hour: Optional[int] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trip_weekdays: list[int] = field(default_factory=list)

def dict_to_activity(d: dict) -> Activity:
    """Convertit un dict (LLM city provider / data_provider) en Activity."""
    options = d.get("transit_options") or []
    closed = d.get("closed_days") or []
    return Activity(
        id=d["id"],
        name=d["name"],
        category=d.get("category", "culture"),
        duration_hours=float(d.get("duration_hours", 1.5)),
        cost_euros=int(d.get("cost_euros", 0)),
        opening_hour=int(d.get("opening_hour", 9)),
        closing_hour=int(d.get("closing_hour", 18)),
        zone=d.get("zone", ""),
        priority_score=int(d.get("priority_score", 5)),
        available_days=d.get("available_days", [0, 1, 2, 3, 4, 5, 6]),
        latitude=float(d.get("latitude", 0.0)),
        longitude=float(d.get("longitude", 0.0)),
        nearest_stop=str(d.get("nearest_stop") or ""),
        transit_options=[o for o in options if isinstance(o, dict) and o.get("type")],
        transit_exit=str(d.get("transit_exit") or ""),
        closed_days=[str(c).lower() for c in closed if isinstance(c, str)],
    )

def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS. Utilisée uniquement pour l'affichage (jamais comme fallback du temps de trajet OSRM)."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
