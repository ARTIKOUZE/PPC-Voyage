"""Orchestrateur du pipeline complet :
User message (NL)"""

from __future__ import annotations
import threading
from typing import Optional

from llm_client import extract_constraints, narrate_plan
from llm_city_provider import generate_city_data
from solver import solve_with_city_data, explain_solution
from dialog_manager import next_question, format_missing_summary, get_missing_critical, CRITICAL_FIELDS
from constraint_extractor import detect_vague_fields

DEFAULT_CONSTRAINTS = {
    "destination": None,
    "num_days": None,
    "total_budget": None,
    "day_start_hour": 9,
    "day_end_hour": 19,
    "num_travelers": 1,
    "hotel_per_night": None,
    "daily_food_budget": 60,
    "preferred_categories": [],
    "avoided_categories": [],
    "preferred_pace": "moderate",
    "must_visit": [],
    "must_avoid": [],
    "must_visit_on_day": {},
    "max_activities_per_day": 6,
    "min_activities_per_day": 2,
    "transport_mode": None,
    "start_date": None,
    "end_date": None,
}

class SessionStore:
    """Stockage en mémoire des contraintes par session_id, plus métadonnées
    de dialogue (dernier champ demandé) et données ville scoped session."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._meta: dict[str, dict] = {}
        self._city_data: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = dict(DEFAULT_CONSTRAINTS)
            return dict(self._sessions[session_id])

    def set(self, session_id: str, constraints: dict):
        with self._lock:
            self._sessions[session_id] = dict(constraints)

    def get_meta(self, session_id: str) -> dict:
        with self._lock:
            return dict(self._meta.get(session_id, {}))

    def set_meta(self, session_id: str, meta: dict):
        with self._lock:
            self._meta[session_id] = dict(meta)

    def get_city_data(
        self, session_id: str, destination: str, transport_mode: str,
    ) -> Optional[dict]:
        """Récupère les données ville cachées pour cette session, si elles
        correspondent à (destination, transport_mode). Sinon None."""
        key = f"{(destination or '').lower().strip()}|{transport_mode or 'foot'}"
        with self._lock:
            store = self._city_data.get(session_id, {})
            return store.get(key)

    def set_city_data(
        self, session_id: str, destination: str, transport_mode: str, data: dict,
    ):
        key = f"{(destination or '').lower().strip()}|{transport_mode or 'foot'}"
        with self._lock:
            self._city_data.setdefault(session_id, {})[key] = data

    def reset(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._meta.pop(session_id, None)
            self._city_data.pop(session_id, None)

_store = SessionStore()

ARRAY_FIELDS = {
    "preferred_categories", "avoided_categories",
    "must_visit", "must_avoid",
}

_STRUCTURAL_FIELDS = {
    "num_days", "total_budget", "preferred_pace",
    "preferred_categories", "avoided_categories",
    "day_start_hour", "day_end_hour",
    "max_activities_per_day", "min_activities_per_day",
    "hotel_per_night", "daily_food_budget", "num_travelers",
    "transport_mode", "morning_preference", "destination",
}

def _determine_touched_days(extracted: dict, previous_plan: dict) -> Optional[set]:
    """Décide quels jours (0-indexed) l'utilisateur a explicitement modifiés
    ce tour-ci. Les autres jours seront hard-pinned dans le solveur."""
    if not previous_plan:
        return None

    if _STRUCTURAL_FIELDS & set(extracted.keys()):
        return None

    touched: set[int] = set()

    for act_id, day_1based in (extracted.get("must_visit_on_day") or {}).items():
        try:
            touched.add(int(day_1based) - 1)
        except (TypeError, ValueError):
            continue
        for d, entries in previous_plan.items():
            for entry in entries:
                aid = entry[0] if isinstance(entry, (list, tuple)) else entry
                if aid == act_id:
                    touched.add(d)

    for act_id in (extracted.get("must_avoid") or []):
        for d, entries in previous_plan.items():
            for entry in entries:
                aid = entry[0] if isinstance(entry, (list, tuple)) else entry
                if aid == act_id:
                    touched.add(d)

    existing = set()
    for entries in previous_plan.values():
        for entry in entries:
            aid = entry[0] if isinstance(entry, (list, tuple)) else entry
            existing.add(aid)
    for act_id in (extracted.get("must_visit") or []):
        if act_id not in existing:
            return None

    return touched

DICT_FIELDS = {
    "must_visit_on_day",
    "min_per_category",
    "max_per_category",
}

INCOMPATIBLE_PAIRS = [
    ("preferred_categories", "avoided_categories"),
    ("must_visit", "must_avoid"),
]

def _strip_default_reemissions(extracted: dict, current: dict) -> dict:
    """Filtre défensif contre les ré-émissions parasites du LLM.
    Problème : quand l'utilisateur fait une édition activity-level (ex:"""
    activity_level_keys = {"must_visit", "must_visit_on_day", "must_avoid"}
    is_activity_edit = bool(activity_level_keys & set(extracted.keys()))
    if not is_activity_edit:
        return extracted

    suspect_defaults = {
        "num_travelers": 1,
        "hotel_per_night": 100,
        "daily_food_budget": 60,
        "min_activities_per_day": 1,
        "max_activities_per_day": 6,
    }
    cleaned = dict(extracted)
    for field, suspect_val in suspect_defaults.items():
        if field not in cleaned:
            continue
        extracted_val = cleaned[field]
        current_val = current.get(field)
        if extracted_val == suspect_val and current_val not in (None, suspect_val):
            del cleaned[field]
    return cleaned

def _is_invalid_critical_update(key: str, value) -> bool:
    """Détecte les valeurs invalides pour les champs critiques.
    Évite que le LLM, en ré-émettant accidentellement un champ avec une"""
    if key not in CRITICAL_FIELDS:
        return False
    if key == "destination":
        return not isinstance(value, str) or len(value.strip()) < 2
    if key in ("total_budget", "num_days"):
        return not isinstance(value, (int, float)) or value <= 0
    if key == "start_date":
        import re as _re
        return not isinstance(value, str) or not _re.match(r"^\d{4}-\d{2}-\d{2}$", value)
    return False

def merge_constraints(current: dict, update: dict) -> dict:
    """Fusionne les contraintes.
    - Arrays : union, en retirant les doublons."""
    merged = dict(current)

    for key, value in update.items():
        if value is None:
            continue
        if _is_invalid_critical_update(key, value) and not _is_invalid_critical_update(key, merged.get(key)):
            continue
        if key in ARRAY_FIELDS and isinstance(value, list):
            existing = merged.get(key, []) or []
            merged[key] = list(dict.fromkeys([*existing, *value]))
        elif key in DICT_FIELDS and isinstance(value, dict):
            existing = merged.get(key, {}) or {}
            merged[key] = {**existing, **value}
        else:
            merged[key] = value

    for a, b in INCOMPATIBLE_PAIRS:
        if a in update and b in merged:
            merged[b] = [x for x in merged[b] if x not in update[a]]
        if b in update and a in merged:
            merged[a] = [x for x in merged[a] if x not in update[b]]

    return merged

def load_city_data(
    city_name: str, transport_mode: str = "foot", num_days: int = 5,
) -> Optional[dict]:
    """Génère les données d'une ville via le LLM (pas de cache global).
    `num_days` permet de dimensionner le pool d'activités (≥ 4/jour + buffer)."""
    return generate_city_data(
        city_name, transport_mode=transport_mode, num_days=num_days,
    )

def _resolve_hotel_budget(constraints: dict) -> dict:
    """Si l'utilisateur n'a pas explicitement fixé hotel_per_night (None),
    on en calcule un comme 40 % du budget total / num_days. Ça permet aux"""
    if constraints.get("hotel_per_night") is not None:
        return constraints
    total = constraints.get("total_budget") or 0
    days = constraints.get("num_days") or 1
    if total > 0 and days > 0:
        budget_per_night = max(50, int(total * 0.4 / days))
        out = dict(constraints)
        out["hotel_per_night"] = budget_per_night
        return out
    return constraints

def _build_plan_summary(session_id: str, current: dict) -> Optional[str]:
    """Construit une description compacte du plan actuel pour aider le LLM à
    interpréter des requêtes relatives (« plus de culture », « moins d'activités »)."""
    meta = _store.get_meta(session_id)
    last_plan = meta.get("last_plan")
    if not last_plan:
        return None

    flat_ids: list[str] = []
    for entries in last_plan.values():
        for entry in entries:
            aid = entry[0] if isinstance(entry, (list, tuple)) else entry
            flat_ids.append(aid)

    total = len(flat_ids)
    num_days = len(last_plan)
    if total == 0 or num_days == 0:
        return None

    destination = current.get("destination") or ""
    transport = current.get("transport_mode") or "foot"
    city_data = _store.get_city_data(session_id, destination, transport)
    by_category: dict[str, int] = {}
    if city_data:
        acts_by_id = {a["id"]: a for a in city_data.get("activities", [])}
        for aid in flat_ids:
            cat = (acts_by_id.get(aid) or {}).get("category")
            if cat:
                by_category[cat] = by_category.get(cat, 0) + 1

    cat_str = ", ".join(f"{n} {c}" for c, n in sorted(by_category.items(), key=lambda x: -x[1]))
    avg = total / num_days
    summary = f"{total} activités sur {num_days} jours (~{avg:.1f}/jour)"
    if cat_str:
        summary += f", dont {cat_str}"
    return summary

def handle_turn(
    session_id: str,
    user_message: str,
    solve_timeout: int = 10,
    mode: str = "flexible",
    transport_mode: Optional[str] = None,
) -> dict:
    """Traite un tour de conversation.
    Returns:"""
    import time
    t_start = time.time()
    errors: list[str] = []
    current = _store.get(session_id)
    meta = _store.get_meta(session_id)
    pending_field = meta.get("pending_field")

    plan_summary = _build_plan_summary(session_id, current)

    t_ex = time.time()
    extracted, extract_err = extract_constraints(
        user_message, current,
        pending_field=pending_field,
        plan_summary=plan_summary,
    )
    extraction_ms = int((time.time() - t_ex) * 1000)
    extracted = _strip_default_reemissions(extracted, current)
    if extract_err:
        errors.append(f"llm_extract: {extract_err}")

    if extract_err and "api error" in extract_err and not extracted:
        reply = (
            "⚠️ Le service de langage (LLM) est injoignable pour le moment "
            "— probablement temporaire. Tu peux soit :\n"
            "• réessayer dans quelques instants,\n"
            "• ou m'écrire directement les contraintes dans un format simple "
            "(ex: \"Rome\", \"5\", \"2000\", \"9h-18h\"), je les comprendrai "
            "même sans le LLM."
        )
        return {
            "reply": reply,
            "extracted": {},
            "constraints": current,
            "plan": None,
            "city": {"name": current.get("destination") or ""},
            "errors": errors,
            "needs_info": reply,
            "explanation": None,
            "llm_unreachable": True,
        }

    merged = merge_constraints(current, extracted)
    merged = _resolve_hotel_budget(merged)
    _store.set(session_id, merged)

    vague_fields = detect_vague_fields(user_message)
    pending_question = next_question(merged, vague_fields)

    if pending_question:
        next_missing = get_missing_critical(merged)
        next_field = next_missing[0] if next_missing else None
        if next_field is None:
            for f in CRITICAL_FIELDS:
                if vague_fields.get(f):
                    next_field = f
                    break
        _store.set_meta(session_id, {"pending_field": next_field})

        missing_info = format_missing_summary(merged)
        errors.append(f"incomplete_constraints: {missing_info}")
        return {
            "reply": pending_question,
            "extracted": extracted,
            "constraints": merged,
            "plan": None,
            "city": {"name": merged.get("destination", "")},
            "errors": errors,
            "needs_info": pending_question,
            "explanation": None,
        }

    _store.set_meta(session_id, {"pending_field": None})

    destination = merged.get("destination", "Rome")
    effective_transport = transport_mode or merged.get("transport_mode") or "foot"
    if transport_mode:
        merged["transport_mode"] = transport_mode
        _store.set(session_id, merged)
    city_data = _store.get_city_data(session_id, destination, effective_transport)
    if city_data is None:
        city_data = load_city_data(destination, transport_mode=effective_transport, num_days=int(merged.get("num_days") or 5))
        if city_data:
            _store.set_city_data(session_id, destination, effective_transport, city_data)

    if not city_data:
        errors.append(f"city_not_found: {destination}")
        reply = (
            f"⚠️ Le LLM n'a pas pu générer les données pour '{destination}' "
            "(timeout ou service indisponible). "
            "Réessaye dans quelques instants, ou tente une ville différente."
        )
        return {
            "reply": reply,
            "extracted": extracted,
            "constraints": merged,
            "plan": None,
            "city": {"name": destination},
            "errors": errors,
            "needs_info": None,
            "explanation": None,
            "llm_unreachable": True,
        }

    prev_meta = _store.get_meta(session_id)
    previous_plan = None
    touched_days = None
    if (prev_meta.get("last_destination") == destination
            and prev_meta.get("last_plan")):
        previous_plan = prev_meta["last_plan"]
        touched_days = _determine_touched_days(extracted, previous_plan)

    plan = solve_with_city_data(
        merged, city_data, time_limit_seconds=solve_timeout, mode=mode,
        previous_plan=previous_plan, touched_days=touched_days,
    )

    if plan and plan.get("status") in ("OPTIMAL", "FEASIBLE"):
        last_plan_map = {
            d["day"] - 1: [
                (a["id"], a.get("start_slot", 0))
                for a in d.get("activities", [])
            ]
            for d in plan.get("days", [])
        }
        new_meta = _store.get_meta(session_id)
        new_meta["last_plan"] = last_plan_map
        new_meta["last_destination"] = destination
        _store.set_meta(session_id, new_meta)

    explanation = explain_solution(plan, merged)

    previous_count = None
    if previous_plan:
        previous_count = sum(len(v) for v in previous_plan.values())
    reply = narrate_plan(user_message, plan, merged, extracted,
                          previous_count=previous_count)

    return {
        "reply": reply,
        "extracted": extracted,
        "constraints": merged,
        "plan": plan,
        "city": city_data.get("city", {}),
        "errors": errors,
        "needs_info": None,
        "explanation": explanation,
        "source": "chat",
        "extraction_ms": extraction_ms,
        "total_pipeline_ms": int((time.time() - t_start) * 1000),
    }

def reset_session(session_id: str):
    _store.reset(session_id)

def get_session_state(session_id: str) -> dict:
    return _store.get(session_id)

if __name__ == "__main__":
    import json

    cur = {"preferred_categories": ["culture"], "avoided_categories": ["shopping"]}
    upd = {"preferred_categories": ["gastro"], "num_days": 7}
    print("merge test:", merge_constraints(cur, upd))

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
