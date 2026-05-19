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
    # hotel_per_night : None → auto-calculé à 40 % du budget total / num_days
    # (permet à un budget plus élevé de proposer naturellement des hôtels haut de gamme).
    # Sera surchargé si l'utilisateur précise explicitement "hôtel max X€/nuit".
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
    # Dates de séjour (ISO YYYY-MM-DD) — critique pour vérifier les ouvertures
    "start_date": None,
    "end_date": None,
}


class SessionStore:
    """Stockage en mémoire des contraintes par session_id, plus métadonnées
    de dialogue (dernier champ demandé) et données ville scoped session.

    Le cache ville par session est CRITIQUE pour la stabilité multi-tours :
    chaque appel LLM génère un set d'activités différent (IDs aléatoires),
    donc sans ce cache le plan précédent ne peut pas être préservé (bonus
    de stabilité ne matche rien, pinning impossible). Le cache est invalidé
    par reset() ou par changement de destination/transport.
    """

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._meta: dict[str, dict] = {}
        self._city_data: dict[str, dict] = {}  # session_id → {key: city_data}
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
    "min_per_category",
    "max_per_category",
}

INCOMPATIBLE_PAIRS = [
    ("preferred_categories", "avoided_categories"),
    ("must_visit", "must_avoid"),
]


def _strip_default_reemissions(extracted: dict, current: dict) -> dict:
    """
    Filtre défensif contre les ré-émissions parasites du LLM.

    Problème : quand l'utilisateur fait une édition activity-level (ex:
    "déplace le Louvre au jour 2"), le LLM peut accidentellement ré-émettre
    un champ scalaire avec sa valeur par défaut (ex: num_travelers=1) au
    lieu de l'omettre. Ça écrase la valeur courante de l'utilisateur (qui
    pouvait être 3 voyageurs).

    Stratégie sûre : on ne supprime QUE quand
      (a) le message contient une modif activity-level (must_visit*, must_avoid),
          signe que c'est une édition pas un changement scalaire,
      (b) ET le champ extrait a une valeur "default suspecte",
      (c) ET la valeur courante de ce champ diffère du défaut.

    Sous ces 3 conditions, c'est presque certainement une ré-émission parasite.
    Si l'utilisateur dit "on est 1 maintenant" sans édition d'activité,
    la condition (a) n'est pas remplie → la valeur est préservée.
    """
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
    """
    Détecte les valeurs invalides pour les champs critiques.
    Évite que le LLM, en ré-émettant accidentellement un champ avec une
    valeur vide/zéro, écrase une valeur valide existante dans l'état.
    """
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
    """
    Fusionne les contraintes.
    - Arrays : union, en retirant les doublons.
    - Scalaires : remplacement.
    - Résolution de conflits : si une catégorie apparaît dans preferred et avoided,
      le champ modifié dans `update` gagne.
    - Protection : un update qui invaliderait un champ critique déjà valide
      est ignoré (le LLM ré-émet parfois `destination: ""` par erreur).
    """
    merged = dict(current)

    for key, value in update.items():
        if value is None:
            continue
        # Garde anti-écrasement : ne pas invalider un champ critique déjà valide
        if _is_invalid_critical_update(key, value) and not _is_invalid_critical_update(key, merged.get(key)):
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

def load_city_data(
    city_name: str, transport_mode: str = "foot", num_days: int = 5,
) -> Optional[dict]:
    """Génère les données d'une ville via le LLM (pas de cache global).
    `num_days` permet de dimensionner le pool d'activités (≥ 4/jour + buffer)."""
    return generate_city_data(
        city_name, transport_mode=transport_mode, num_days=num_days,
    )


# ─────────────────────────────────────────────
# Résumé du plan pour le LLM (interpréter les requêtes relatives)
# ─────────────────────────────────────────────

def _resolve_hotel_budget(constraints: dict) -> dict:
    """
    Si l'utilisateur n'a pas explicitement fixé hotel_per_night (None),
    on en calcule un comme 40 % du budget total / num_days. Ça permet aux
    voyageurs avec un gros budget d'avoir naturellement accès aux hôtels
    plus haut de gamme, sans qu'ils aient à le dire explicitement.

    Sémantique : hotel_per_night = prix de CHAMBRE par nuit (pas par tête).
    Plancher : 50 € (en dessous, aucun hôtel ne rentre généralement).
    """
    if constraints.get("hotel_per_night") is not None:
        return constraints  # explicitly set by user
    total = constraints.get("total_budget") or 0
    days = constraints.get("num_days") or 1
    if total > 0 and days > 0:
        budget_per_night = max(50, int(total * 0.4 / days))
        out = dict(constraints)
        out["hotel_per_night"] = budget_per_night
        return out
    return constraints


def _build_plan_summary(session_id: str, current: dict) -> Optional[str]:
    """
    Construit une description compacte du plan actuel pour aider le LLM à
    interpréter des requêtes relatives (« plus de culture », « moins d'activités »).

    Exemple de sortie :
        "14 activités sur 6 jours (~2.3/jour), dont 5 culture, 4 gastro, 3 nature, 2 nightlife"
    """
    meta = _store.get_meta(session_id)
    last_plan = meta.get("last_plan")
    if not last_plan:
        return None

    # Reconstituer la liste d'IDs d'activités du plan
    flat_ids: list[str] = []
    for entries in last_plan.values():
        for entry in entries:
            aid = entry[0] if isinstance(entry, (list, tuple)) else entry
            flat_ids.append(aid)

    total = len(flat_ids)
    num_days = len(last_plan)
    if total == 0 or num_days == 0:
        return None

    # Récupérer les catégories depuis le city_data caché
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

    # Calculer un résumé du plan actuel à passer au LLM pour interpréter
    # les requêtes relatives ("plus d'activités", "plus de culture").
    plan_summary = _build_plan_summary(session_id, current)

    # 1. Extraction LLM (avec hint sur le champ en attente pour les réponses courtes)
    t_ex = time.time()
    extracted, extract_err = extract_constraints(
        user_message, current,
        pending_field=pending_field,
        plan_summary=plan_summary,
    )
    extraction_ms = int((time.time() - t_ex) * 1000)
    # Filtrer les ré-émissions parasites du LLM (ex: num_travelers=1 ré-écrasé)
    extracted = _strip_default_reemissions(extracted, current)
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
    # Si l'utilisateur n'a pas explicité de budget hôtel, on prend 40 % du total / jour
    merged = _resolve_hotel_budget(merged)
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
    # Réutiliser le pool d'activités si on l'a déjà généré pour cette session
    # (sinon chaque tour multi-tours redonne des IDs différents → la stabilité
    # ne peut pas matcher l'ancien plan).
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

    # 7. Narration LLM (avec compte précédent pour ne pas halluciner le delta)
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
