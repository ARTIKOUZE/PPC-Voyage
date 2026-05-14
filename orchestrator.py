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
from llm_city_provider import generate_city_data
from solver import solve_with_city_data, explain_solution
from dialog_manager import next_question, format_missing_summary, get_missing_critical, CRITICAL_FIELDS
from constraint_extractor import detect_vague_fields


# ─────────────────────────────────────────────
# État par session
# ─────────────────────────────────────────────

DEFAULT_CONSTRAINTS = {
    # Contraintes critiques : None au démarrage → dialog_manager demandera
    "destination": None,
    "num_days": None,
    "total_budget": None,
    # Plage horaire par défaut : 9h-19h (l'utilisateur peut surcharger via NL :
    # "je commence à 8h", "on veut finir à 22h"…)
    "day_start_hour": 9,
    "day_end_hour": 19,
    # Valeurs par défaut raisonnables pour les champs optionnels
    "num_travelers": 1,
    "hotel_per_night": 100,
    "daily_food_budget": 60,
    "preferred_categories": [],
    "avoided_categories": [],
    "preferred_pace": "moderate",
    "must_visit": [],
    "must_avoid": [],
    "must_visit_on_day": {},
    "max_activities_per_day": 6,
    "min_activities_per_day": 1,
    "transport_mode": None,
}


class SessionStore:
    """Stockage en mémoire des contraintes par session_id, plus métadonnées
    de dialogue (dernier champ demandé)."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._meta: dict[str, dict] = {}
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

    def reset(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._meta.pop(session_id, None)


_store = SessionStore()


# ─────────────────────────────────────────────
# Merge des contraintes (arrays unionnés, scalaires remplacés)
# ─────────────────────────────────────────────

ARRAY_FIELDS = {
    "preferred_categories", "avoided_categories",
    "must_visit", "must_avoid",
}

# Champs dont la modification affecte tous les jours → pas de pinning.
_STRUCTURAL_FIELDS = {
    "num_days", "total_budget", "preferred_pace",
    "preferred_categories", "avoided_categories",
    "day_start_hour", "day_end_hour",
    "max_activities_per_day", "min_activities_per_day",
    "hotel_per_night", "daily_food_budget", "num_travelers",
    "transport_mode", "morning_preference", "destination",
}


def _determine_touched_days(extracted: dict, previous_plan: dict) -> Optional[set]:
    """
    Décide quels jours (0-indexed) l'utilisateur a explicitement modifiés
    ce tour-ci. Les autres jours seront hard-pinned dans le solveur.

    Returns:
      - None : pas de pinning (changement structurel, premier tour, ou
        ajout d'activité sans jour précisé — le solveur doit pouvoir
        choisir librement où la placer).
      - set() : aucun jour touché → tout est pinned (l'utilisateur a juste
        discuté sans modifier de contraintes).
      - set(...): jours touchés ; les autres sont pinned.
    """
    if not previous_plan:
        return None

    # Changement structurel : impossible d'isoler localement
    if _STRUCTURAL_FIELDS & set(extracted.keys()):
        return None

    touched: set[int] = set()

    # must_visit_on_day : jour ciblé + jour où l'activité était avant
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

    # must_avoid : jour où l'activité était
    for act_id in (extracted.get("must_avoid") or []):
        for d, entries in previous_plan.items():
            for entry in entries:
                aid = entry[0] if isinstance(entry, (list, tuple)) else entry
                if aid == act_id:
                    touched.add(d)

    # must_visit ajouté sans précision de jour : le solveur doit pouvoir
    # placer la nouvelle activité n'importe où → pas de pinning
    existing = set()
    for entries in previous_plan.values():
        for entry in entries:
            aid = entry[0] if isinstance(entry, (list, tuple)) else entry
            existing.add(aid)
    for act_id in (extracted.get("must_visit") or []):
        if act_id not in existing:
            return None

    return touched

# Champs dictionnaire : les clés sont unionnées (la nouvelle valeur écrase l'ancienne)
DICT_FIELDS = {
    "must_visit_on_day",
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
        elif key in DICT_FIELDS and isinstance(value, dict):
            existing = merged.get(key, {}) or {}
            merged[key] = {**existing, **value}  # nouvelles valeurs écrasent les anciennes
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
# Chargement des données ville (sans cache : LLM appelé à chaque requête)
# ─────────────────────────────────────────────

def load_city_data(city_name: str, transport_mode: str = "foot") -> Optional[dict]:
    """Génère les données d'une ville via le LLM. Pas de cache : toujours frais."""
    return generate_city_data(city_name, transport_mode=transport_mode)


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def handle_turn(
    session_id: str,
    user_message: str,
    solve_timeout: int = 10,
    mode: str = "flexible",
    transport_mode: Optional[str] = None,
) -> dict:
    """
    Traite un tour de conversation.

    Returns:
        {
          "reply": str,                  # texte à afficher dans le chat
          "extracted": dict,             # contraintes extraites de CE message
          "constraints": dict,           # état consolidé après merge
          "plan": dict | None,           # plan CP-SAT ou None si incomplet/INFEASIBLE
          "city": dict,                  # infos ville (name, country, lat, lon)
          "errors": list[str],           # erreurs non bloquantes
          "needs_info": str | None,      # question à poser si contraintes critiques manquantes
          "explanation": str | None,     # explication des compromis (si plan produit)
        }
    """
    import time
    t_start = time.time()
    errors: list[str] = []
    current = _store.get(session_id)
    meta = _store.get_meta(session_id)
    pending_field = meta.get("pending_field")

    # 1. Extraction LLM (avec hint sur le champ en attente pour les réponses courtes)
    t_ex = time.time()
    extracted, extract_err = extract_constraints(
        user_message, current, pending_field=pending_field
    )
    extraction_ms = int((time.time() - t_ex) * 1000)
    if extract_err:
        errors.append(f"llm_extract: {extract_err}")

    # Si l'extraction a fait une erreur "API/network" (LLM injoignable) ET
    # qu'on n'a rien pu extraire ni via regex fallback, on prévient l'utilisateur
    # plutôt que de boucler.
    if extract_err and "api error" in extract_err and not extracted:
        # Notre regex fallback est déjà tenté en interne par extract_constraints
        # quand pending_field est connu. Si on arrive ici sans rien, c'est qu'il
        # n'a rien trouvé non plus.
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

    # 2. Merge
    merged = merge_constraints(current, extracted)
    _store.set(session_id, merged)

    # 3. Vérifier les contraintes critiques manquantes (dialog_manager)
    vague_fields = detect_vague_fields(user_message)
    pending_question = next_question(merged, vague_fields)

    if pending_question:
        # Identifier le champ qu'on va demander pour le prochain tour
        next_missing = get_missing_critical(merged)
        next_field = next_missing[0] if next_missing else None
        if next_field is None:
            # Fallback : trouver le premier champ vague
            for f in CRITICAL_FIELDS:
                if vague_fields.get(f):
                    next_field = f
                    break
        _store.set_meta(session_id, {"pending_field": next_field})

        # Contraintes critiques incomplètes : ne pas lancer le solveur
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

    # Toutes les contraintes critiques sont présentes → nettoyer pending_field
    _store.set_meta(session_id, {"pending_field": None})

    # 4. Données ville (toutes les contraintes critiques sont présentes)
    destination = merged.get("destination", "Rome")
    # Priorité au paramètre d'API (sélecteur UI), puis au LLM extract, puis "foot"
    effective_transport = transport_mode or merged.get("transport_mode") or "foot"
    if transport_mode:
        merged["transport_mode"] = transport_mode
        _store.set(session_id, merged)
    city_data = load_city_data(destination, transport_mode=effective_transport)

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

    # 5. CP-SAT — multi-tours : pin hard des jours non touchés + bonus stabilité.
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

    # Sauvegarder le plan pour le prochain tour (avec start_slot pour pin précis)
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

    # 6. Explication des compromis
    explanation = explain_solution(plan, merged)

    # 7. Narration LLM
    reply = narrate_plan(user_message, plan, merged, extracted)

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


# ─────────────────────────────────────────────
# Mode formulaire : contraintes structurées directes, sans LLM d'extraction.
# Sert à comparer NL vs formulaire (objectif 5 du sujet).
# ─────────────────────────────────────────────

def handle_form(
    session_id: str,
    form_constraints: dict,
    solve_timeout: int = 10,
    mode: str = "flexible",
    transport_mode: Optional[str] = None,
) -> dict:
    """
    Pipeline alternatif : reçoit des contraintes déjà structurées (depuis un
    formulaire UI), saute l'extraction LLM, lance directement
    data ville → CP-SAT → narration courte (sans LLM si possible).

    Returns le même shape que handle_turn pour rester échangeable côté front.
    """
    import time
    t_start = time.time()
    errors: list[str] = []
    current = _store.get(session_id)

    # Merge : les valeurs du formulaire écrasent l'état courant
    merged = merge_constraints(current, form_constraints)
    _store.set(session_id, merged)

    # Validation : champs critiques
    missing = get_missing_critical(merged)
    if missing:
        return {
            "reply": ("Formulaire incomplet : "
                      + format_missing_summary(merged)),
            "extracted": form_constraints,
            "constraints": merged,
            "plan": None,
            "city": {"name": merged.get("destination") or ""},
            "errors": [f"form_incomplete: {missing}"],
            "needs_info": format_missing_summary(merged),
            "explanation": None,
            "source": "form",
            "extraction_ms": 0,
        }

    destination = merged.get("destination", "Rome")
    effective_transport = transport_mode or merged.get("transport_mode") or "foot"
    if transport_mode:
        merged["transport_mode"] = transport_mode
        _store.set(session_id, merged)

    city_data = load_city_data(destination, transport_mode=effective_transport)
    if not city_data:
        return {
            "reply": (f"⚠️ Le LLM n'a pas pu générer les données pour "
                      f"'{destination}'."),
            "extracted": form_constraints,
            "constraints": merged,
            "plan": None,
            "city": {"name": destination},
            "errors": [f"city_not_found: {destination}"],
            "needs_info": None,
            "explanation": None,
            "source": "form",
            "extraction_ms": 0,
            "llm_unreachable": True,
        }

    prev_meta = _store.get_meta(session_id)
    previous_plan = None
    touched_days = None
    if (prev_meta.get("last_destination") == destination
            and prev_meta.get("last_plan")):
        previous_plan = prev_meta["last_plan"]
        # Le formulaire ne supporte pas les contraintes activity-level
        # (must_visit_on_day, etc.) — donc structural toujours. Pas de pin.
        touched_days = _determine_touched_days(form_constraints, previous_plan)

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

    # En mode formulaire on génère une narration courte déterministe (pas de LLM
    # pour rester strictement comparable à un formulaire instantané).
    summary = plan.get("summary", {}) if plan else {}
    reply = (
        f"Plan généré pour {destination} : "
        f"{summary.get('total_activities', 0)} activités sur "
        f"{merged.get('num_days')} jours, "
        f"{summary.get('total_cost', 0)}€ / {summary.get('budget', 0)}€."
    )

    return {
        "reply": reply,
        "extracted": form_constraints,
        "constraints": merged,
        "plan": plan,
        "city": city_data.get("city", {}),
        "errors": errors,
        "needs_info": None,
        "explanation": explanation,
        "source": "form",
        "extraction_ms": 0,  # bypass LLM = 0ms d'extraction
        "total_pipeline_ms": int((time.time() - t_start) * 1000),
    }


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
