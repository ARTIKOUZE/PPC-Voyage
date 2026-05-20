"""Génère des liens Google Maps pour les transitions en transports en commun. Utilise l'adresse ou le nom de l'activité plutôt que les coordonnées GPS brutes,"""
from __future__ import annotations

import logging
import urllib.parse

logger = logging.getLogger(__name__)

_FOOT_MODES = {"foot", "bike", "car"}

def _maps_url(origin: str, destination: str, mode: str = "transit") -> str:
    if mode == "foot":
        travelmode = "walking"
    elif mode == "bike":
        travelmode = "bicycling"
    elif mode == "car":
        travelmode = "driving"
    else:
        travelmode = "transit"
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={urllib.parse.quote(origin)}"
        f"&destination={urllib.parse.quote(destination)}"
        f"&travelmode={travelmode}"
    )

def _has_street_number(address: str) -> bool:
    """Vérifie qu'une adresse contient un numéro de rue."""
    import re
    return bool(re.search(r"\b\d+\b", address))

def _best_label(act: dict) -> str:
    '''Retourne la désignation la plus précise possible pour Google Maps. Format cible : "Nom du lieu, 40 Bd Haussmann, 75009 Paris"'''
    name = (act.get("name") or "").strip()
    address = (act.get("address") or "").strip()
    city = (act.get("city") or "").strip()

    if name and address and _has_street_number(address):
        return f"{name}, {address}"

    if name and city:
        return f"{name}, {city}"

    return name or address

def _find_act(plan: dict, act_id: str) -> dict:
    for day in plan.get("days", []):
        for act in day.get("activities", []):
            if act.get("id") == act_id:
                return act
    return {}

_TRANSIT_MODES = {"metro", "bus", "rer", "tram", "train", "funiculaire", "ferry", "navette", "transit"}

def enrich_transitions_with_routing(plan: dict, _city_name: str = "") -> dict:
    """Ajoute un lien Google Maps sur chaque transition en transports en commun. Utilise l'adresse exacte de l'activité pour un résultat précis."""
    city_name = (plan.get("city") or {}).get("name", _city_name)

    for day in plan.get("days", []):
        for trans in day.get("transitions", []):

            from_act = _find_act(plan, trans.get("from_id", ""))
            to_act = _find_act(plan, trans.get("to_id", ""))

            if not from_act or not to_act:
                continue

            from_act.setdefault("city", city_name)
            to_act.setdefault("city", city_name)

            origin = _best_label(from_act)
            destination = _best_label(to_act)

            if origin and destination:
                trans["maps_url"] = _maps_url(origin, destination, trans.get("mode", "transit"))
    return plan
