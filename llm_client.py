"""Client LLM pour l'assistant de planification de voyage. Utilise un endpoint OpenAI-compatible (qwen3-35b via text-generation-webui)."""

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

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.medium.text-generation-webui.myia.io/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")

LLM_BASE_URL_FALLBACK = os.environ.get(
    "LLM_BASE_URL_FALLBACK", "https://api.mini.text-generation-webui.myia.io/v1"
)
LLM_API_KEY_FALLBACK = os.environ.get(
    "LLM_API_KEY_FALLBACK", "FEECE4DF2224BF0A5E28A1A4378BD20B"
)
LLM_MODEL_FALLBACK = os.environ.get("LLM_MODEL_FALLBACK", "omnicoder-9b")

_client: Optional[OpenAI] = None
_client_fallback: Optional[OpenAI] = None

QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

def get_client() -> OpenAI:
    global _client
    if _client is None:
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
    """Lance un chat.completions.create sur l'endpoint principal et bascule automatiquement sur le fallback en cas d'échec (timeout, 5xx, JSON vide)."""
    import logging
    logger = logging.getLogger(__name__)
    last_err: Optional[Exception] = None

    for label, client, default_model in [
        ("primary", get_client(), LLM_MODEL),
        ("fallback", get_fallback_client(), LLM_MODEL_FALLBACK),
    ]:
        try:
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("model", default_model)
            if label == "fallback":
                call_kwargs["model"] = default_model
            target = client.with_options(timeout=timeout) if timeout else client
            return target.chat.completions.create(**call_kwargs)
        except Exception as e:
            last_err = e
            logger.warning("[LLM/%s] échec : %s — tentative suivante", label, e)

    assert last_err is not None
    raise last_err

VALID_CATEGORIES = ["culture", "gastro", "nature", "shopping", "nightlife"]
VALID_PACES = ["relaxed", "moderate", "intense"]
VALID_TRANSPORT = ["foot", "bike", "car"]

class ExtractedConstraints(BaseModel):
    """Sous-ensemble des contraintes modifiables par tour utilisateur. Tous les champs sont optionnels : on ne renvoie que ce que le message modifie."""

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
    must_visit_on_day: Optional[dict[str, int]] = None

    max_activities_per_day: Optional[int] = Field(None, ge=1, le=8)
    min_activities_per_day: Optional[int] = Field(None, ge=0, le=8)

    min_per_category: Optional[dict[str, int]] = None
    max_per_category: Optional[dict[str, int]] = None

    day_start_hour: Optional[int] = Field(None, ge=0, le=23)
    day_end_hour: Optional[int] = Field(None, ge=1, le=24)

    transport_mode: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None

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
- day_start_hour (int, 0-23): GLOBAL hour when activities start each day (every day)
- day_end_hour (int, 1-24): GLOBAL hour when activities stop each day (every day)
- day_specific_start_hour (object {day_num: hour}): override start hour for a SPECIFIC trip day (1-indexed)
- day_specific_end_hour (object {day_num: hour}): override end hour for a SPECIFIC trip day (1-indexed)
  CRITICAL: when the user mentions a SPECIFIC day or weekday ("le samedi", "le jour 3", "le dernier jour"), use the day-specific field — NEVER day_end_hour (which applies to ALL days).
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

User: "j'aimerais finir à midi le samedi" (Plan : ...jour 2=samedi...)
→ {"day_specific_end_hour":{"2":12}}

User: "le jour 3 on commence à 11h"
→ {"day_specific_start_hour":{"3":11}}

User: "le dernier jour on finit à 14h" (Plan : 4 jours)
→ {"day_specific_end_hour":{"4":14}}

User: "le jour 1 et 2 on commence à 10h"
→ {"day_specific_start_hour":{"1":10,"2":10}}

User: "Hello!"
→ {}

User: "I absolutely want to see the Colosseum"
→ {"must_visit":["colosseum"]}

User: "Je veux faire le Louvre le jour 3"
→ {"must_visit":["louvre"],"must_visit_on_day":{"louvre":3}}

User: "retire la Tour Eiffel"
→ {"must_avoid":["tour eiffel"]}

User: "enlève le Louvre du séjour"
→ {"must_avoid":["louvre"]}

User: "je ne veux plus aller à Notre-Dame"
→ {"must_avoid":["notre-dame"]}

User: "supprime l'activité du musée d'Orsay"
→ {"must_avoid":["musée d'orsay"]}

RE-AJOUT (« remet », « remets », « rajoute », « ajoute de nouveau ») :
ALWAYS emit must_visit with the activity name. NEVER emit must_avoid for re-adding.

User: "remet la Tour Eiffel"
→ {"must_visit":["tour eiffel"]}

User: "remets le Louvre"
→ {"must_visit":["louvre"]}

User: "rajoute la cathédrale St-Patrick"
→ {"must_visit":["cathédrale st-patrick"]}

User: "ajoute de nouveau le musée d'Orsay"
→ {"must_visit":["musée d'orsay"]}

User: "ajoute le Sacré-Cœur le jour 2"
→ {"must_visit":["sacré-cœur"],"must_visit_on_day":{"sacré-cœur":2}}

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
- Acknowledge the user's request and mention ONE highlight of the plan (e.g., "j'ai placé le Colisée le jour 1" or "Votre budget est respecté avec une marge de 120 €").
- DO NOT recite the plan in text (the user already sees it).
- If the plan is INFEASIBLE, briefly explain which constraint is likely too tight and suggest a concrete relaxation.
- If the message was an explanation question ("why this activity on that day?"), give the CP-SAT reason in natural language (e.g., "Parce que ça ouvre à 9h et c'est dans la même zone que la prochaine activité").

RÈGLES STRICTES sur le compte d'activités :
- Le bloc « Changement effectif » te donne le delta RÉEL (avant → après).
- Si delta = +N, dis "j'ai ajouté N activité(s)" — JAMAIS le total.
- Si delta = -N, dis "j'ai retiré N activité(s)".
- Si delta = 0 ou AUCUN, tu DOIS dire que le plan reste inchangé (mêmes activités, même nombre).
  Ne JAMAIS écrire "j'ai ajouté/intégré X activités" quand AUCUN changement n'a eu lieu.
  Explique pourquoi (créneaux saturés, contraintes repas, durées) ET suggère une vraie piste
  concrète : étendre l'amplitude horaire (ex: "essaye 8h-22h au lieu de 9h-19h"),
  allonger le séjour, augmenter le budget, ou relâcher une contrainte spécifique.
- Pour mentionner le total, dis "le plan compte X activités au total" (pas "j'ai ajouté X").
"""

_JSON_RE = re.compile(r"\{[\s\S]*\}")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def _strip_thinking(text: str) -> str:
    """Retire les blocs <think>…</think> émis par qwen3/deepseek en mode reasoning."""
    return _THINK_RE.sub("", text)

def _extract_json_blob(text: str) -> str:
    """Extrait le premier objet JSON présent dans le texte, après suppression des blocs de thinking et des fences markdown."""
    text = _strip_thinking(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = _JSON_RE.search(text)
    return m.group(0) if m else text

def parse_json_salvage(blob: str):
    """Parse JSON ; si une erreur de syntaxe survient au milieu du JSON (typique avec les LLM longs : virgule manquante, troncature), tronque"""
    try:
        return json.loads(blob)
    except json.JSONDecodeError as initial_err:
        err_pos = initial_err.pos

    prefix = blob[:err_pos]
    candidates = []
    last_comma_after_brace = prefix.rfind("},")
    if last_comma_after_brace >= 0:
        candidates.append(last_comma_after_brace + 1)
    last_brace = prefix.rfind("}")
    if last_brace >= 0:
        candidates.append(last_brace + 1)

    for truncate_at in candidates:
        truncated = blob[:truncate_at]
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
    """Extrait les contraintes d'un message utilisateur. Args:"""
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
    plan_diff: Optional[dict] = None,
) -> str:
    """Génère la réponse conversationnelle à afficher dans le chat. `previous_count` (si fourni) = nombre d'activités du plan précédent,"""
    plan_summary = _summarize_plan(plan, constraints)
    changes_str = json.dumps(extracted_changes, ensure_ascii=False) if extracted_changes else "aucune"

    delta_block = ""
    if previous_count is not None and plan and plan.get("summary"):
        new_count = plan["summary"].get("total_activities", 0)
        delta = new_count - previous_count
        if delta > 0:
            delta_block = (
                f"\n\n=== CHANGEMENT EFFECTIF ===\n"
                f"Avant : {previous_count} activités. Après : {new_count} activités.\n"
                f"DELTA : +{delta} activité(s) ajoutée(s).\n"
                f"Tu DOIS dire 'j'ai ajouté {delta} activité(s)', PAS 'j'ai ajouté {new_count}'.\n"
                f"Total du plan : {new_count} activités."
            )
        elif delta < 0:
            delta_block = (
                f"\n\n=== CHANGEMENT EFFECTIF ===\n"
                f"Avant : {previous_count} activités. Après : {new_count} activités.\n"
                f"DELTA : {delta} activité(s) retirée(s).\n"
                f"Tu DOIS dire 'j'ai retiré {abs(delta)} activité(s)'.\n"
                f"Total du plan : {new_count} activités."
            )
        else:
            delta_block = (
                f"\n\n=== CHANGEMENT EFFECTIF : AUCUN ===\n"
                f"Avant : {previous_count} activités. Après : {previous_count} activités. DELTA : 0.\n"
                f"INTERDICTION ABSOLUE de dire 'j'ai ajouté/intégré/retiré X activités'.\n"
                f"Tu DOIS dire que le plan reste inchangé à {previous_count} activités.\n"
                f"NE PRESUME PAS de la cause (n'invente pas « horaires saturés »,\n"
                f"« créneaux pleins », « contraintes repas »…). Dis simplement que la\n"
                f"demande n'a pas pu être appliquée et propose à l'utilisateur de\n"
                f"reformuler ou de préciser ce qu'il souhaite (jour visé, activité"
                f" précise, plage horaire à étendre s'il connaît la contrainte qui bloque)."
            )

    pool_block = ""
    if plan and plan.get("pool_stats"):
        ps = plan["pool_stats"]
        if ps.get("pool_exhausted"):
            pool_block = (
                f"\n[SATURATION] Toutes les {ps['pool_size']} activités du pool sont "
                f"déjà incluses ; impossible d'en ajouter sans changer la durée ou les contraintes."
            )

    diff_block = ""
    if plan_diff:
        moved = plan_diff.get("moved", [])
        added = plan_diff.get("added", [])
        removed = plan_diff.get("removed", [])
        if moved or added or removed:
            lines = ["\n\n=== DIFF DETAILLE (par rapport au tour precedent) ==="]
            if added:
                lines.append("Ajoutees :")
                for x in added:
                    lines.append(f"  + '{x['name']}' (jour {x['day']})")
            if removed:
                lines.append("Retirees :")
                for x in removed:
                    lines.append(f"  - '{x['name']}' (etait jour {x['day']})")
            if moved:
                lines.append("Deplacees :")
                for x in moved:
                    lines.append(f"  ~ '{x['name']}' : jour {x['from_day']} -> jour {x['to_day']}")
            if moved and added:
                lines.append(
                    "\nIMPORTANT : il y a a la fois des ajouts ET des deplacements. "
                    "Tu DOIS justifier les deplacements : ce sont des compromis que le "
                    "solveur a faits pour caser les ajouts dans les fenetres horaires "
                    "et les contraintes de capacite. Exemple : 'pour caser X au jour 2, "
                    "j'ai du deplacer Y du jour 2 au jour 3 ou il y avait de la place'."
                )
            diff_block = "\n".join(lines)

    user_content = (
        f"Message utilisateur : \"{user_message}\"\n"
        f"Contraintes modifiées par ce message : {changes_str}\n"
        f"Résumé du plan : {plan_summary}"
        f"{delta_block}"
        f"{diff_block}"
        f"{pool_block}\n\n"
        "Réponds en 2-3 phrases max (4-5 si tu dois justifier des déplacements)."
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
