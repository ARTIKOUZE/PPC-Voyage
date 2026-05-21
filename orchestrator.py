"""Orchestrateur du pipeline complet :
User message (NL)"""

from __future__ import annotations
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

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
    "day_specific_start_hour": {},
    "day_specific_end_hour": {},
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

# Champs structurels "soft" : si le LLM les ré-émet en même temps qu'une édition
# activity-level (must_avoid / must_visit_on_day), on suppose que c'est une
# hallucination et on ignore pour le calcul de touched_days. Si l'utilisateur
# veut VRAIMENT changer ces champs, il devra le faire dans un tour séparé.
_STRUCTURAL_SOFT = {
    "preferred_categories", "avoided_categories",
    "day_start_hour", "day_end_hour",
    "max_activities_per_day", "min_activities_per_day",
    "hotel_per_night", "daily_food_budget", "num_travelers",
    "morning_preference", "preferred_pace",
}

def _determine_touched_days(extracted: dict, previous_plan: dict) -> Optional[set]:
    """Jours (0-indexed) à laisser libres au solveur. Les autres sont pinned.

    Règles :
    - Pas de plan précédent ou changement structurel → None (pas de pin).
    - must_visit_on_day = {X: D} → touche le jour D (la cible). On NE touche PAS
      le jour où X était avant ; il reste pinné avec ses autres activités, juste
      sans X (qui est forcée ailleurs via la contrainte hard must_visit_on_day).
    - must_avoid = [Y] → touche RIEN. La journée où Y était reste pinned avec
      les autres activités ; Y est juste exclue (selected=0 hard, pas de pin
      pour Y donc pas de conflit).
    - must_visit nouveau (sans jour) → touche RIEN. block_new_on_pinned passera
      à False côté solveur pour que la nouvelle activité puisse atterrir
      sur n'importe quel jour pinned ayant de la place.
    """
    if not previous_plan:
        return None

    # Si l'utilisateur fait une édition activity-level (must_avoid / must_visit_on_day),
    # on tolère les ré-émissions structurelles "soft" (probables hallucinations LLM).
    is_activity_edit = bool({"must_avoid", "must_visit_on_day", "must_visit"}
                            & set(extracted.keys()))
    if is_activity_edit:
        hard_structural = _STRUCTURAL_FIELDS - _STRUCTURAL_SOFT
        if hard_structural & set(extracted.keys()):
            return None
    else:
        if _STRUCTURAL_FIELDS & set(extracted.keys()):
            return None

    touched: set[int] = set()
    for _act_id, day_1based in (extracted.get("must_visit_on_day") or {}).items():
        try:
            touched.add(int(day_1based) - 1)
        except (TypeError, ValueError):
            continue
    return touched


def _has_new_must_visit(extracted: dict, previous_plan: dict) -> bool:
    """Vrai si l'utilisateur ajoute une nouvelle activité sans préciser de jour.
    Dans ce cas, block_new_on_pinned doit passer à False pour permettre au
    solveur de placer la nouvelle activité sur un jour pinned."""
    if not previous_plan:
        return False
    existing = set()
    for entries in previous_plan.values():
        for entry in entries:
            aid = entry[0] if isinstance(entry, (list, tuple)) else entry
            existing.add(aid)
    # Activités ciblées par must_visit_on_day : la cible est déjà touched →
    # l'activité atterrira sur un jour libre, pas besoin de relâcher le block.
    pinned_target = set((extracted.get("must_visit_on_day") or {}).keys())
    for act_id in (extracted.get("must_visit") or []):
        if act_id not in existing and act_id not in pinned_target:
            return True
    return False

DICT_FIELDS = {
    "must_visit_on_day",
    "min_per_category",
    "max_per_category",
    "day_specific_start_hour",
    "day_specific_end_hour",
}

INCOMPATIBLE_PAIRS = [
    ("preferred_categories", "avoided_categories"),
    ("must_visit", "must_avoid"),
]

def _is_noop_update(value, current_value) -> bool:
    """Vrai si l'update n'apporte aucun changement par rapport à l'état courant.
    Tient compte des types : sets/lists comparées comme sous-ensemble,
    dicts comparées clé-par-clé."""
    if isinstance(value, list) and isinstance(current_value, list):
        return set(value).issubset(set(current_value))
    if isinstance(value, dict) and isinstance(current_value, dict):
        return all(current_value.get(k) == v for k, v in value.items())
    return current_value == value


_INTENT_KEYWORDS = {
    "preferred_pace": ("rythme", "intense", "tranquille", "relaxed", "modere", "moderate", "calme", "rapide"),
    "preferred_categories": ("culture", "gastro", "nature", "shopping", "nightlife", "musee", "musée", "restau", "parc"),
    "avoided_categories": ("evite", "évite", "pas de", "sans"),
    "day_start_hour": ("commence", "debut", "début", "commencer", "matin"),
    "day_end_hour": ("finir", "fin", "soir", "arrete", "arrête"),
    "max_activities_per_day": ("max", "maximum", "pas plus", "moins d", "fatigu"),
    "min_activities_per_day": ("min", "minimum", "au moins", "plus d"),
    "num_travelers": ("voyageur", "personne", "couple", "famille", "ami", "duo", "trio"),
    "hotel_per_night": ("hotel", "hôtel", "nuit"),
    "daily_food_budget": ("nourriture", "repas", "manger", "deli"),
    "morning_preference": ("matin", "morning"),
    "num_days": ("jour", "day", "semaine", "weekend", "week-end"),
    "total_budget": ("budget", "euros", "€"),
    "transport_mode": ("pied", "marche", "velo", "vélo", "voiture", "transport", "bus"),
}

def _user_mentions(key: str, user_message: str) -> bool:
    if not user_message:
        return False
    msg = user_message.lower()
    return any(kw in msg for kw in _INTENT_KEYWORDS.get(key, ()))

def _strip_default_reemissions(extracted: dict, current: dict,
                                user_message: str = "") -> dict:
    """Filtre défensif contre les ré-émissions parasites du LLM.

    Pendant une édition activity-level (must_visit*, must_avoid), un champ
    structurel n'est conservé que si :
      - sa nouvelle valeur diffère vraiment de l'état actuel, ET
      - l'utilisateur l'a mentionné explicitement (mot-clé associé).
    Sinon on l'écarte — c'est une hallucination LLM qui casserait le pinning.
    """
    activity_level_keys = {"must_visit", "must_visit_on_day", "must_avoid"}
    is_activity_edit = bool(activity_level_keys & set(extracted.keys()))
    if not is_activity_edit:
        return extracted

    cleaned: dict = {}
    for key, value in extracted.items():
        if key in activity_level_keys:
            cleaned[key] = value
            continue
        cur = current.get(key)
        if _is_noop_update(value, cur):
            continue
        # Si l'utilisateur n'a pas mentionné le champ (mots-clés associés
        # absents du message), on suppose hallucination LLM et on strip.
        if key in _INTENT_KEYWORDS and not _user_mentions(key, user_message):
            continue
        cleaned[key] = value
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

    def _norm_for_diff(s):
        if not isinstance(s, str):
            return s
        import re as _re, unicodedata as _ud
        x = s.lower().strip().replace("œ", "oe").replace("æ", "ae")
        x = "".join(c for c in _ud.normalize("NFD", x) if _ud.category(c) != "Mn")
        return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", " ", x)).strip()

    for a, b in INCOMPATIBLE_PAIRS:
        if a in update and b in merged:
            forbidden = {_norm_for_diff(x) for x in update[a]}
            merged[b] = [x for x in merged[b] if _norm_for_diff(x) not in forbidden]
        if b in update and a in merged:
            forbidden = {_norm_for_diff(x) for x in update[b]}
            merged[a] = [x for x in merged[a] if _norm_for_diff(x) not in forbidden]

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

def _unknown_must_visit_names(extracted: dict, city_data: dict) -> list[str]:
    """Renvoie les noms must_visit / must_visit_on_day qui ne matchent
    aucune activite du pool (apres normalisation tokens + fuzzy).
    """
    import re as _re
    import unicodedata as _ud
    from difflib import SequenceMatcher

    def _norm(s):
        s = (s or "").lower().strip().replace("œ", "oe").replace("æ", "ae")
        s = "".join(c for c in _ud.normalize("NFD", s) if _ud.category(c) != "Mn")
        return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", " ", s)).strip()

    pool_names = [a.get("name", "") for a in (city_data or {}).get("activities", [])]
    pool_ids = {a.get("id", "") for a in (city_data or {}).get("activities", [])}
    pool_norms = [_norm(n) for n in pool_names]

    def _matches_pool(query: str) -> bool:
        if query in pool_ids:
            return True
        q = _norm(query)
        if not q:
            return True
        if any(q == p for p in pool_norms):
            return True
        # contenance ou tokens
        q_tokens = set(q.split())
        for p in pool_norms:
            if q in p or p in q:
                return True
            p_tokens = set(p.split())
            if q_tokens and q_tokens.issubset(p_tokens):
                return True
        # fuzzy
        return any(SequenceMatcher(None, q, p).ratio() >= 0.75 for p in pool_norms)

    candidates: list[str] = []
    for name in (extracted.get("must_visit") or []):
        if isinstance(name, str) and not _matches_pool(name):
            candidates.append(name)
    for name in (extracted.get("must_visit_on_day") or {}).keys():
        if isinstance(name, str) and not _matches_pool(name) and name not in candidates:
            candidates.append(name)
    return candidates

def _compute_plan_diff(previous_plan: dict, new_plan: dict, city_data: dict) -> dict:
    """Diff entre l'ancien et le nouveau plan : activités déplacées, ajoutées, retirées.
    Sert à fournir au LLM le contexte exact des compromis pour qu'il les justifie."""
    acts_by_id = {a["id"]: a for a in (city_data or {}).get("activities", [])}

    def _name(aid: str) -> str:
        return (acts_by_id.get(aid) or {}).get("name", aid)

    # ancien jour de chaque activité
    prev_day_of: dict[str, int] = {}
    for d_idx, entries in previous_plan.items():
        for entry in entries:
            aid = entry[0] if isinstance(entry, (list, tuple)) else entry
            prev_day_of[aid] = d_idx

    # nouveau jour de chaque activité
    new_day_of: dict[str, int] = {}
    for d in new_plan.get("days", []):
        d_idx = d["day"] - 1
        for a in d.get("activities", []):
            new_day_of[a["id"]] = d_idx

    moved: list[dict] = []
    added: list[dict] = []
    removed: list[dict] = []
    for aid, d_new in new_day_of.items():
        if aid not in prev_day_of:
            added.append({"name": _name(aid), "day": d_new + 1})
        elif prev_day_of[aid] != d_new:
            moved.append({"name": _name(aid),
                          "from_day": prev_day_of[aid] + 1,
                          "to_day": d_new + 1})
    for aid, d_old in prev_day_of.items():
        if aid not in new_day_of:
            removed.append({"name": _name(aid), "day": d_old + 1})

    return {"moved": moved, "added": added, "removed": removed}

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

    # Mapping jour-de-séjour ↔ date ↔ jour-de-semaine pour interpréter
    # « le samedi », « le dernier jour », etc.
    start_date = current.get("start_date")
    if start_date:
        try:
            from datetime import date, timedelta
            y, m, dd = map(int, start_date.split("-"))
            d0 = date(y, m, dd)
            jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
            mapping = []
            for i in range(num_days):
                d_i = d0 + timedelta(days=i)
                mapping.append(f"jour {i+1}={jours_fr[d_i.weekday()]} {d_i.strftime('%d/%m')}")
            summary += ". Calendrier : " + " ; ".join(mapping)
        except (ValueError, TypeError):
            pass

    # Liste explicite des noms : le LLM doit les ré-utiliser tels quels dans must_avoid/must_visit
    if city_data:
        names = [
            (acts_by_id.get(aid) or {}).get("name")
            for aid in flat_ids
        ]
        names = [n for n in names if n]
        if names:
            summary += ". Activités actuellement planifiées : " + " | ".join(names)
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
    extracted = _strip_default_reemissions(extracted, current, user_message)
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
        meta_upd = _store.get_meta(session_id)
        meta_upd["pending_field"] = next_field
        _store.set_meta(session_id, meta_upd)

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

    new_meta = _store.get_meta(session_id)
    new_meta["pending_field"] = None
    _store.set_meta(session_id, new_meta)

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

    if city_data:
        unknown = _unknown_must_visit_names(extracted, city_data)
        if unknown:
            from llm_city_provider import extend_city_data_with_activities
            logger.info("[Orch] %d activite(s) absente(s) du pool, extension via LLM : %s",
                        len(unknown), unknown)
            city_data, added_names = extend_city_data_with_activities(
                city_data, unknown, transport_mode=effective_transport,
            )
            if added_names:
                _store.set_city_data(session_id, destination, effective_transport, city_data)
                errors.append(f"pool_extended: {added_names}")

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
    block_new_on_pinned = True
    if (prev_meta.get("last_destination") == destination
            and prev_meta.get("last_plan")):
        previous_plan = prev_meta["last_plan"]
        touched_days = _determine_touched_days(extracted, previous_plan)
        if _has_new_must_visit(extracted, previous_plan):
            block_new_on_pinned = False

    logger.info(
        "[Orch] user=%r | extracted=%s | touched_days=%s | block_new=%s | has_prev_plan=%s",
        user_message, extracted, touched_days, block_new_on_pinned,
        bool(previous_plan),
    )

    plan = solve_with_city_data(
        merged, city_data, time_limit_seconds=solve_timeout, mode=mode,
        previous_plan=previous_plan, touched_days=touched_days,
        block_new_on_pinned=block_new_on_pinned,
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
    plan_diff = None
    if previous_plan and plan and plan.get("status") in ("OPTIMAL", "FEASIBLE"):
        previous_count = sum(len(v) for v in previous_plan.values())
        plan_diff = _compute_plan_diff(previous_plan, plan, city_data)
    reply = narrate_plan(user_message, plan, merged, extracted,
                          previous_count=previous_count, plan_diff=plan_diff)

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
