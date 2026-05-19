"""
Client LLM pour l'assistant de planification de voyage.

Utilise un endpoint OpenAI-compatible (qwen3-35b via text-generation-webui).
Deux responsabilités :
  1. Extraction structurée des contraintes depuis du langage naturel (JSON mode)
  2. Narration du plan CP-SAT résolu (texte libre concis)

Stratégie d'extraction :
  - Pydantic schéma strict pour valider la sortie
  - Prompt système avec peu d'exemples (few-shot léger)
  - JSON mode si supporté, sinon parse manuel avec fallback
  - Retry 1x en cas d'erreur de parsing
"""

from __future__ import annotations
import os
import json
import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.medium.text-generation-webui.myia.io/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")

# Fallback : modèle plus léger (omnicoder-9b) sur un autre endpoint.
# Utilisé automatiquement si l'endpoint principal échoue (timeout, 5xx, etc.).
LLM_BASE_URL_FALLBACK = os.environ.get(
    "LLM_BASE_URL_FALLBACK", "https://api.mini.text-generation-webui.myia.io/v1"
)
LLM_API_KEY_FALLBACK = os.environ.get(
    "LLM_API_KEY_FALLBACK", "FEECE4DF2224BF0A5E28A1A4378BD20B"
)
LLM_MODEL_FALLBACK = os.environ.get("LLM_MODEL_FALLBACK", "omnicoder-9b")

_client: Optional[OpenAI] = None
_client_fallback: Optional[OpenAI] = None

# Hard-disable du mode "thinking" de qwen3 : sans ça le modèle consomme tous les
# max_tokens en raisonnement interne (tokens cachés mais comptés) et renvoie un
# message vide. Cette option est passée via extra_body et propagée par le backend
# text-generation-webui au chat template du modèle.
QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        # Timeout explicite pour éviter de bloquer l'UI si l'endpoint est lent
        _client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY or "dummy",
            timeout=60.0,
            max_retries=1,
        )
    return _client


def get_fallback_client() -> OpenAI:
    global _client_fallback
    if _client_fallback is None:
        _client_fallback = OpenAI(
            base_url=LLM_BASE_URL_FALLBACK,
            api_key=LLM_API_KEY_FALLBACK or "dummy",
            timeout=60.0,
            max_retries=1,
        )
    return _client_fallback


def chat_with_fallback(timeout: Optional[float] = None, **kwargs):
    """
    Lance un chat.completions.create sur l'endpoint principal et bascule
    automatiquement sur le fallback en cas d'échec (timeout, 5xx, JSON vide).

    Args:
        timeout: override le timeout par défaut du client (pour appels lourds)
        **kwargs: paramètres OpenAI standards (messages, max_tokens, etc.)
            `model` est automatiquement remplacé par le modèle correspondant
            à chaque endpoint, sauf s'il est explicitement passé.

    Returns:
        L'objet ChatCompletion (peut venir du primaire ou du fallback).

    Raises:
        La dernière exception si les DEUX endpoints échouent.
    """
    import logging
    logger = logging.getLogger(__name__)
    last_err: Optional[Exception] = None

    for label, client, default_model in [
        ("primary", get_client(), LLM_MODEL),
        ("fallback", get_fallback_client(), LLM_MODEL_FALLBACK),
    ]:
        try:
            call_kwargs = dict(kwargs)
            # Le modèle dépend de l'endpoint : si l'appelant n'a pas forcé,
            # on prend le modèle par défaut de cet endpoint.
            call_kwargs.setdefault("model", default_model)
            if label == "fallback":
                # Forcer le modèle du fallback (ignore le model du primaire)
                call_kwargs["model"] = default_model
            target = client.with_options(timeout=timeout) if timeout else client
            return target.chat.completions.create(**call_kwargs)
        except Exception as e:
            last_err = e
            logger.warning("[LLM/%s] échec : %s — tentative suivante", label, e)

    # Les deux endpoints ont échoué
    assert last_err is not None
    raise last_err


# ─────────────────────────────────────────────
# Schéma des contraintes (aligné avec solver.TravelConstraints)
# ─────────────────────────────────────────────

VALID_CATEGORIES = ["culture", "gastro", "nature", "shopping", "nightlife"]
VALID_PACES = ["relaxed", "moderate", "intense"]
VALID_TRANSPORT = ["foot", "bike", "car"]


class ExtractedConstraints(BaseModel):
    """Sous-ensemble des contraintes modifiables par tour utilisateur.
    Tous les champs sont optionnels : on ne renvoie que ce que le message modifie."""

    destination: Optional[str] = None
    num_days: Optional[int] = Field(None, ge=1, le=21)
    total_budget: Optional[int] = Field(None, ge=0)
    num_travelers: Optional[int] = Field(None, ge=1, le=20)
    hotel_per_night: Optional[int] = Field(None, ge=0)
    daily_food_budget: Optional[int] = Field(None, ge=0)

    preferred_categories: Optional[list[str]] = None
    avoided_categories: Optional[list[str]] = None
    preferred_pace: Optional[str] = None
    morning_preference: Optional[str] = None

    must_visit: Optional[list[str]] = None
    must_avoid: Optional[list[str]] = None
    must_visit_on_day: Optional[dict[str, int]] = None   # {"louvre": 3}  (jours 1-indexés)

    max_activities_per_day: Optional[int] = Field(None, ge=1, le=8)
    min_activities_per_day: Optional[int] = Field(None, ge=0, le=8)

    # Cibles par catégorie sur tout le voyage (utiles pour "plus de culture", etc.)
    min_per_category: Optional[dict[str, int]] = None
    max_per_category: Optional[dict[str, int]] = None

    day_start_hour: Optional[int] = Field(None, ge=0, le=23)
    day_end_hour: Optional[int] = Field(None, ge=1, le=24)

    transport_mode: Optional[str] = None  # "foot", "bike", "car"

    # Dates de séjour
    start_date: Optional[str] = None     # ISO YYYY-MM-DD
    end_date: Optional[str] = None       # ISO YYYY-MM-DD

    def clean(self) -> dict:
        """Retourne un dict ne contenant que les champs renseignés et validés."""
        out = {}
        for k, v in self.model_dump(exclude_none=True).items():
            if k in ("preferred_categories", "avoided_categories"):
                v = [c for c in v if c in VALID_CATEGORIES]
                if not v:
                    continue
            if k == "preferred_pace" and v not in VALID_PACES:
                continue
            if k == "morning_preference" and v not in VALID_CATEGORIES:
                continue
            if k == "transport_mode" and v not in VALID_TRANSPORT:
                continue
            if k == "must_visit_on_day":
                v = {
                    str(act): int(day)
                    for act, day in v.items()
                    if isinstance(day, (int, float)) and int(day) >= 1
                }
                if not v:
                    continue
            if k in ("min_per_category", "max_per_category") and isinstance(v, dict):
                v = {
                    str(cat).lower(): int(n)
                    for cat, n in v.items()
                    if str(cat).lower() in VALID_CATEGORIES and isinstance(n, (int, float)) and int(n) >= 0
                }
                if not v:
                    continue
            if k in ("start_date", "end_date"):
                import re as _re
                if not _re.match(r"^\d{4}-\d{2}-\d{2}$", str(v)):
                    continue
            out[k] = v
        return out


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a constraint extractor for a travel planner based on a CP-SAT solver.

From the user's message, you must extract ONLY the modified constraints and return them in strict JSON.

AUTHORIZED FIELDS:
- destination (string): city name
- num_days (int, 1-21)
- total_budget (int, euros)
- num_travelers (int, 1-20)
- hotel_per_night (int, euros/night)
- daily_food_budget (int, euros/day/person for meals)
- preferred_categories (array of {"culture","gastro","nature","shopping","nightlife"})
- avoided_categories (array, same values)
- preferred_pace: "relaxed" (2 activities/day), "moderate" (3), "intense" (4)
- morning_preference (a category to favor in the morning)
- must_visit (array of activity names/IDs: use when user wants an activity, without specifying a day)
- must_visit_on_day (object mapping activity name to day number 1-indexed: use ONLY when user specifies a particular day)
- must_avoid (array of activity names/IDs)
- max_activities_per_day (int)
- min_activities_per_day (int)
- day_start_hour (int, 0-23): hour when the user wants to start activities each day (e.g. 9 for 9h)
- day_end_hour (int, 1-24): hour when the user wants to stop activities each day (e.g. 18 for 18h, 24 for midnight)
- transport_mode (string): "foot" (walking, default), "bike" (vélo), or "car" (voiture). Detect from words like "à pied", "marche", "vélo", "voiture", "en bus" (→ car).
- start_date (string ISO YYYY-MM-DD): exact arrival date. Always normalize to ISO format.
- end_date (string ISO YYYY-MM-DD): exact departure date. Compute it if user gives start_date + num_days (end = start + num_days - 1).

DATE HANDLING:
- Convert ANY date format the user provides to ISO YYYY-MM-DD.
- "12/06/2026" → "2026-06-12"  (DD/MM/YYYY is French standard)
- "14 août 2026" → "2026-08-14"
- "week-end du 14 juin" → start_date="2026-06-13" (Saturday), end_date="2026-06-14" (Sunday), num_days=2
- If user says "j'arrive le 12/06/2026 pour 3 jours" → start_date="2026-06-12", num_days=3, end_date="2026-06-14"
- If user says "du 5 au 8 septembre" (year omitted) → use current/next year accordingly.

STRICT RULES:
1. Respond ONLY with valid JSON (no markdown, no ```, no comments).
2. Return only the fields that the message explicitly mentions or modifies. If the message contains no usable constraint, return {}.
3. Do NOT guess default values. A missing field = not modified.
4. Avoided categories go into avoided_categories, not into must_avoid.
5. For must_visit_on_day: ALWAYS also add the activity to must_visit.
6. Use the activity's common name in English or French as the key (e.g. "louvre", "eiffel tower", "colosseum").

EXAMPLES:

User: "5-day trip to Rome with €2000, we love culture"
→ {"destination":"Rome","num_days":5,"total_budget":2000,"preferred_categories":["culture"]}

User: "We are 2, relaxed pace and no shopping"
→ {"num_travelers":2,"preferred_pace":"relaxed","avoided_categories":["shopping"]}

User: "Budget €1500, max 3 activities per day, I like gastronomy"
→ {"total_budget":1500,"max_activities_per_day":3,"preferred_categories":["gastro"]}

User: "Je veux commencer à 10h et finir à 18h"
→ {"day_start_hour":10,"day_end_hour":18}

User: "On commence tôt vers 8h du matin et on arrête à 22h le soir"
→ {"day_start_hour":8,"day_end_hour":22}

User: "Hello!"
→ {}

User: "I absolutely want to see the Colosseum"
→ {"must_visit":["colosseum"]}

User: "Je veux faire le Louvre le jour 3"
→ {"must_visit":["louvre"],"must_visit_on_day":{"louvre":3}}

User: "Can you add the Eiffel Tower on day 1?"
→ {"must_visit":["eiffel tower"],"must_visit_on_day":{"eiffel tower":1}}

User: "Inclure Notre-Dame le jour 2 et le Louvre le jour 4"
→ {"must_visit":["notre-dame","louvre"],"must_visit_on_day":{"notre-dame":2,"louvre":4}}

User: "du lundi 12/06/2026 au jeudi 15/06/2026"
→ {"start_date":"2026-06-12","end_date":"2026-06-15","num_days":4}

User: "j'arrive le 12/06/2026 et je reste 3 jours"
→ {"start_date":"2026-06-12","end_date":"2026-06-14","num_days":3}

User: "week-end du 14 août 2026"
→ {"start_date":"2026-08-15","end_date":"2026-08-16","num_days":2}

User: "du 5 au 8 septembre 2026"
→ {"start_date":"2026-09-05","end_date":"2026-09-08","num_days":4}

RELATIVE REQUESTS — when "Plan actuel" is provided, use the CURRENT AVG/day
to compute concrete deltas. CRITICAL: the new constraint must ACTUALLY
constrain compared to current avg, otherwise nothing changes.

For "plus / encore plus" — bump pace first; emit min_activities_per_day
ONLY if pace is already intense (it's now soft, no infeasibility risk).

User (Plan actuel : 14/6j ~2.3/jour, pace=relaxed) "plus d'activités"
→ {"preferred_pace":"moderate"}

User (Plan actuel : 12/3j ~4/jour, pace=moderate) "plus d'activités"
→ {"preferred_pace":"intense"}

User (Plan actuel : 13/5j ~2.6/jour, pace=intense) "encore plus d'activités"
→ {"min_activities_per_day":4}
(pace already at intense → push the soft floor; round up from current avg)

For "moins / encore moins" — max_activities_per_day MUST be STRICTLY BELOW
current avg/day, otherwise it doesn't reduce anything. Round DOWN aggressively.
Never emit max ≥ current avg.

User (Plan actuel : 24/4j ~6/jour, pace=intense) "moins d'activités"
→ {"max_activities_per_day":4,"preferred_pace":"moderate"}
(6 → 4, real cut)

User (Plan actuel : 13/5j ~2.6/jour, pace=intense) "moins d'activités"
→ {"max_activities_per_day":2,"preferred_pace":"moderate"}
(2.6 → 2. Do NOT emit max=3 — that would NOT constrain since 2.6 < 3.)

User (Plan actuel : 10/5j ~2.0/jour, pace=moderate) "encore moins"
→ {"max_activities_per_day":1,"preferred_pace":"relaxed"}
(2 → 1, real cut)

User (Plan actuel : 16/4j ~4/jour, pace=moderate) "moins d'activités on est fatigués"
→ {"max_activities_per_day":3,"preferred_pace":"relaxed"}

PER-CATEGORY REQUESTS — when the user wants more/less of a specific category:

User (Plan actuel : 12 activités, dont 3 culture, 4 gastro) "je veux plus de culture"
→ {"min_per_category":{"culture":5}}
(3 culture → bump to 5, leaves other categories alone)

User (Plan actuel : 14 activités, dont 5 culture, 2 gastro) "plus de gastronomie"
→ {"min_per_category":{"gastro":4}}

User (Plan actuel : 10 activités, dont 6 culture) "trop de culture"
→ {"max_per_category":{"culture":3}}

CRITICAL — do not re-emit fields that are NOT mentioned in the user's message,
even if "Contraintes actuelles" shows a value for them. Re-emitting unchanged
fields corrupts the multi-turn merge. Example:

User (current num_travelers=3, num_days=5, total_budget=2500): "Déplace le Louvre au jour 2"
→ {"must_visit":["louvre"],"must_visit_on_day":{"louvre":2}}
(Notice: num_travelers, num_days, total_budget are NOT re-emitted because the
user didn't mention them.)
"""


NARRATION_SYSTEM_PROMPT = """You are the conversational interface of a travel planner.
The CP-SAT solver has already produced an optimal plan that the user sees in a timeline next to the chat.

YOUR ROLE:
- Respond in French, 2–3 sentences maximum, friendly and concise tone.
- Acknowledge the user's request and mention ONE highlight of the plan (e.g., "I scheduled the Colosseum on day 1" or "Your budget is respected with €120 margin").
- DO NOT recite the plan in text (the user already sees it).
- If the plan is INFEASIBLE, briefly explain which constraint is likely too tight and suggest a concrete relaxation.
- If the message was an explanation question ("why this activity on that day?"), give the CP-SAT reason in natural language (e.g., "Because it opens at 9am and it's in the same area as the next activity").
"""


# ─────────────────────────────────────────────
# Extraction de contraintes
# ─────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Retire les blocs <think>…</think> émis par qwen3/deepseek en mode reasoning."""
    return _THINK_RE.sub("", text)


def _extract_json_blob(text: str) -> str:
    """Extrait le premier objet JSON présent dans le texte, après suppression
    des blocs de thinking et des fences markdown."""
    text = _strip_thinking(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = _JSON_RE.search(text)
    return m.group(0) if m else text


def parse_json_salvage(blob: str):
    """
    Parse JSON ; si une erreur de syntaxe survient au milieu du JSON
    (typique avec les LLM longs : virgule manquante, troncature), tronque
    au dernier item de tableau bien fermé AVANT l'erreur et ferme proprement
    la structure pour récupérer ce qui est parseable.

    Returns: dict/list parsé. Raise la JSONDecodeError d'origine si non récupérable.
    """
    try:
        return json.loads(blob)
    except json.JSONDecodeError as initial_err:
        err_pos = initial_err.pos

    # Tronquer juste avant l'erreur.
    prefix = blob[:err_pos]
    # Le dernier item complet d'un tableau se termine par "}," suivi du début
    # d'un nouvel item, ou par "}]" en fin de tableau. On cherche la dernière
    # occurrence de "}," dans le prefix : c'est le dernier item séparé.
    candidates = []
    last_comma_after_brace = prefix.rfind("},")
    if last_comma_after_brace >= 0:
        candidates.append(last_comma_after_brace + 1)  # position après '}'
    # Cas où l'erreur survient dès le 2ᵉ item : on cherche le '}' qui
    # précède l'erreur, suivi seulement d'espaces.
    last_brace = prefix.rfind("}")
    if last_brace >= 0:
        candidates.append(last_brace + 1)

    for truncate_at in candidates:
        truncated = blob[:truncate_at]
        # Refermer le tableau + l'objet racine
        for closing in ("]}", "}", ""):
            try:
                return json.loads(truncated + closing)
            except json.JSONDecodeError:
                continue

    raise initial_err


_PENDING_FIELD_HINTS: dict[str, str] = {
    "destination": "destination (city name)",
    "total_budget": "total_budget (integer, euros)",
    "num_days": "num_days (integer, number of days)",
    "day_start_hour": "day_start_hour (integer 0-23, hour to start activities)",
    "day_end_hour": "day_end_hour (integer 1-24, hour to stop activities)",
}


def extract_constraints(
    user_message: str,
    current_constraints: Optional[dict] = None,
    max_retries: int = 1,
    pending_field: Optional[str] = None,
    plan_summary: Optional[str] = None,
) -> tuple[dict, Optional[str]]:
    """
    Extrait les contraintes d'un message utilisateur.

    Args:
        pending_field: si le bot vient d'asker pour un champ donné, on l'indique
            au LLM pour qu'il interprète correctement une réponse courte
            (ex: "5" → num_days=5 si pending_field="num_days").

    Returns:
        (constraints_dict, error_message_or_None)
        constraints_dict ne contient que les champs à modifier.
    """
    current_constraints = current_constraints or {}

    hint = ""
    if pending_field and pending_field in _PENDING_FIELD_HINTS:
        hint = (
            f"\n\nIMPORTANT: The assistant just asked the user for the field "
            f"\"{_PENDING_FIELD_HINTS[pending_field]}\". "
            f"Even if the reply is very short (a bare number, an hour, a single word), "
            f"interpret it as the value of that field."
        )

    plan_block = f"\nPlan actuel : {plan_summary}\n" if plan_summary else ""
    user_content = (
        f"Contraintes actuelles : {json.dumps(current_constraints, ensure_ascii=False)}\n"
        f"{plan_block}"
        f"\nMessage utilisateur : \"{user_message}\"{hint}\n\n"
        "Extrais les contraintes modifiées en JSON strict."
    )

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = chat_with_fallback(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
                extra_body=QWEN_NO_THINK,
            )
            raw = resp.choices[0].message.content or ""
            blob = _extract_json_blob(raw)
            data = json.loads(blob)
            extracted = ExtractedConstraints(**data).clean()
            return extracted, None

        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            last_err = f"parse error (attempt {attempt}): {e}"
        except Exception as e:
            last_err = f"api error (attempt {attempt}): {e}"

    return {}, last_err


# ─────────────────────────────────────────────
# Narration du plan
# ─────────────────────────────────────────────

def _summarize_plan(plan: dict, constraints: dict) -> str:
    """Résumé compact du plan pour l'injecter dans le prompt."""
    if not plan:
        return "aucun plan encore"
    if plan.get("status") == "INFEASIBLE":
        return f"INFEASIBLE — {plan.get('message', '')}"

    summary = plan.get("summary", {})
    days = plan.get("days", [])
    highlights = []
    for d in days[:3]:
        acts = d.get("activities", [])
        if acts:
            top = max(acts, key=lambda a: a.get("priority_score", a.get("cost", 0)))
            highlights.append(f"J{d['day']}: {top['name']}")

    return (
        f"{summary.get('total_activities', 0)} activités sur {constraints.get('num_days', '?')} jours, "
        f"coût {summary.get('total_cost', 0)}€/{summary.get('budget', 0)}€. "
        f"Temps forts : {'; '.join(highlights)}."
    )


def narrate_plan(
    user_message: str,
    plan: dict,
    constraints: dict,
    extracted_changes: dict,
    previous_count: Optional[int] = None,
) -> str:
    """Génère la réponse conversationnelle à afficher dans le chat.
    `previous_count` (si fourni) = nombre d'activités du plan précédent,
    pour que le LLM puisse être honnête sur le delta réel et ne pas
    halluciner « j'ai ajouté X activités » alors que rien n'a bougé.
    Aucun fallback : si le LLM échoue, l'exception remonte à l'appelant."""
    plan_summary = _summarize_plan(plan, constraints)
    changes_str = json.dumps(extracted_changes, ensure_ascii=False) if extracted_changes else "aucune"

    # Delta concret pour empêcher l'hallucination du LLM narrateur
    delta_block = ""
    if previous_count is not None and plan and plan.get("summary"):
        new_count = plan["summary"].get("total_activities", 0)
        delta = new_count - previous_count
        if delta > 0:
            delta_block = (
                f"\nChangement effectif : +{delta} activités "
                f"({previous_count} → {new_count})."
            )
        elif delta < 0:
            delta_block = (
                f"\nChangement effectif : {delta} activités "
                f"({previous_count} → {new_count})."
            )
        else:
            delta_block = (
                f"\nChangement effectif : AUCUN ({previous_count} activités, identique). "
                "Mentionne-le honnêtement au lieu de prétendre avoir ajouté/retiré quelque chose. "
                "Suggère plutôt une autre piste (augmenter l'amplitude horaire, allonger le séjour…)."
            )

    user_content = (
        f"Message utilisateur : \"{user_message}\"\n"
        f"Contraintes modifiées par ce message : {changes_str}\n"
        f"Résumé du plan : {plan_summary}"
        f"{delta_block}\n\n"
        "Réponds en 2-3 phrases max."
    )

    resp = chat_with_fallback(
        messages=[
            {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=400,
        extra_body=QWEN_NO_THINK,
    )
    raw = resp.choices[0].message.content or ""
    return _strip_thinking(raw).strip()


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"LLM endpoint : {LLM_BASE_URL}")
    print(f"Modèle       : {LLM_MODEL}")
    print("=" * 60)

    tests = [
        "Voyage de 5 jours à Rome avec 2000€, on adore la culture et la gastro",
        "On est 2, rythme tranquille, pas de shopping",
        "Budget 1500€, max 3 activités/jour",
        "Salut !",
    ]
    for msg in tests:
        print(f"\n> {msg}")
        extracted, err = extract_constraints(msg)
        if err:
            print(f"  [err] {err}")
        print(f"  extracted: {extracted}")
