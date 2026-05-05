"""
Orchestrateur du pipeline complet :

  User message (NL)
    → LLM extraction (llm_client.extract_constraints)
    → merge avec l'état de la session
    → chargement données ville (data_provider.get_city_data)
    → CP-SAT (solver.solve_with_city_data)
    → LLM narration (llm_client.narrate_plan)
    → { reply, constraints, plan }

Garde l'état des contraintes côté serveur (par session_id).
C'est la couche "métier" invoquée par api_server.py.
"""

from __future__ import annotations
import threading
from typing import Optional

from llm_client import extract_constraints, narrate_plan
from data_provider import get_city_data, get_rome_data
from solver import solve_with_city_data


# ─────────────────────────────────────────────
# État par session
# ─────────────────────────────────────────────

DEFAULT_CONSTRAINTS = {
    "destination": "Rome",
    "num_days": 5,
    "total_budget": 2000,
    "num_travelers": 1,
    "hotel_per_night": 100,
    "daily_food_budget": 60,
    "preferred_categories": [],
    "avoided_categories": [],
    "preferred_pace": "moderate",
    "must_visit": [],
    "must_avoid": [],
    "max_activities_per_day": 4,
    "min_activities_per_day": 1,
}


class SessionStore:
    """Stockage en mémoire des contraintes par session_id."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = dict(DEFAULT_CONSTRAINTS)
            return dict(self._sessions[session_id])

    def set(self, session_id: str, constraints: dict):
        with self._lock:
            self._sessions[session_id] = dict(constraints)

    def reset(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)


_store = SessionStore()


# ─────────────────────────────────────────────
# Merge des contraintes (arrays unionnés, scalaires remplacés)
# ─────────────────────────────────────────────

ARRAY_FIELDS = {
    "preferred_categories", "avoided_categories",
    "must_visit", "must_avoid",
}

INCOMPATIBLE_PAIRS = [
    ("preferred_categories", "avoided_categories"),
    ("must_visit", "must_avoid"),
]


def merge_constraints(current: dict, update: dict) -> dict:
    """
    Fusionne les contraintes.
    - Arrays : union, en retirant les doublons.
    - Scalaires : remplacement.
    - Résolution de conflits : si une catégorie apparaît dans preferred et avoided,
      le champ modifié dans `update` gagne.
    """
    merged = dict(current)

    for key, value in update.items():
        if value is None:
            continue
        if key in ARRAY_FIELDS and isinstance(value, list):
            existing = merged.get(key, []) or []
            merged[key] = list(dict.fromkeys([*existing, *value]))
        else:
            merged[key] = value

    # Résoudre les conflits preferred ↔ avoided etc.
    for a, b in INCOMPATIBLE_PAIRS:
        if a in update and b in merged:
            # Les éléments fraîchement ajoutés à `a` sortent de `b`
            merged[b] = [x for x in merged[b] if x not in update[a]]
        if b in update and a in merged:
            merged[a] = [x for x in merged[a] if x not in update[b]]

    return merged


# ─────────────────────────────────────────────
# Cache des données ville (un seul fetch par ville par run)
# ─────────────────────────────────────────────

_city_cache: dict[str, dict] = {}
_city_cache_lock = threading.Lock()


def load_city_data(city_name: str) -> Optional[dict]:
    """Charge les données d'une ville (avec mémoïsation en mémoire)."""
    key = city_name.strip().lower()
    with _city_cache_lock:
        if key in _city_cache:
            return _city_cache[key]

    # Rome : données pré-cachées toujours disponibles, évite un round-trip API
    if key in ("rome", "roma"):
        data = get_rome_data()
    else:
        data = get_city_data(city_name)

    if data:
        with _city_cache_lock:
            _city_cache[key] = data
    return data


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def handle_turn(
    session_id: str,
    user_message: str,
    solve_timeout: int = 10,
) -> dict:
    """
    Traite un tour de conversation.

    Returns:
        {
          "reply": str,                  # texte à afficher dans le chat
          "extracted": dict,             # contraintes extraites de CE message
          "constraints": dict,           # état consolidé après merge
          "plan": dict | None,           # plan CP-SAT ou None si INFEASIBLE/ville inconnue
          "city": dict,                  # infos ville (name, country, lat, lon)
          "errors": list[str],           # erreurs non bloquantes (extraction, etc.)
        }
    """
    errors: list[str] = []
    current = _store.get(session_id)

    # 1. Extraction LLM
    extracted, extract_err = extract_constraints(user_message, current)
    if extract_err:
        errors.append(f"llm_extract: {extract_err}")

    # 2. Merge
    merged = merge_constraints(current, extracted)
    _store.set(session_id, merged)

    # 3. Données ville (destination peut avoir changé)
    destination = merged.get("destination", "Rome")
    city_data = load_city_data(destination)

    if not city_data:
        errors.append(f"city_not_found: {destination}")
        reply = f"Je n'ai pas trouvé de données pour '{destination}'. Essaye une autre ville ou reste sur Rome."
        return {
            "reply": reply,
            "extracted": extracted,
            "constraints": merged,
            "plan": None,
            "city": {"name": destination},
            "errors": errors,
        }

    # 4. CP-SAT
    plan = solve_with_city_data(merged, city_data, time_limit_seconds=solve_timeout)

    # 5. Narration
    reply = narrate_plan(user_message, plan, merged, extracted)

    return {
        "reply": reply,
        "extracted": extracted,
        "constraints": merged,
        "plan": plan,
        "city": city_data.get("city", {}),
        "errors": errors,
    }


def reset_session(session_id: str):
    _store.reset(session_id)


def get_session_state(session_id: str) -> dict:
    return _store.get(session_id)


# ─────────────────────────────────────────────
# Test manuel (sans LLM) : vérifie juste le câblage data → solver
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Test du merge
    cur = {"preferred_categories": ["culture"], "avoided_categories": ["shopping"]}
    upd = {"preferred_categories": ["gastro"], "num_days": 7}
    print("merge test:", merge_constraints(cur, upd))

    # Test du pipeline (mockera l'extraction si le LLM est injoignable)
    print("\n--- Tour 1 ---")
    result = handle_turn("test-session", "5 jours à Rome, budget 1500€, j'aime la culture")
    print("extracted:", result["extracted"])
    print("reply:", result["reply"])
    if result["plan"]:
        print("plan status:", result["plan"].get("status"))
        print("activities:", result["plan"].get("summary", {}).get("total_activities"))

    print("\n--- Tour 2 ---")
    result = handle_turn("test-session", "ajoute de la gastro et rythme tranquille")
    print("extracted:", result["extracted"])
    print("constraints:", json.dumps(result["constraints"], ensure_ascii=False))
