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

_client: Optional[OpenAI] = None

# Hard-disable du mode "thinking" de qwen3 : sans ça le modèle consomme tous les
# max_tokens en raisonnement interne (tokens cachés mais comptés) et renvoie un
# message vide. Cette option est passée via extra_body et propagée par le backend
# text-generation-webui au chat template du modèle.
QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "dummy")
    return _client


# ─────────────────────────────────────────────
# Schéma des contraintes (aligné avec solver.TravelConstraints)
# ─────────────────────────────────────────────

VALID_CATEGORIES = ["culture", "gastro", "nature", "shopping", "nightlife"]
VALID_PACES = ["relaxed", "moderate", "intense"]


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

    max_activities_per_day: Optional[int] = Field(None, ge=1, le=8)
    min_activities_per_day: Optional[int] = Field(None, ge=0, le=8)

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
- must_visit (array of known activity IDs — leave empty if you do not know the exact IDs)
- must_avoid (array of IDs)
- max_activities_per_day (int)
- min_activities_per_day (int)

STRICT RULES:
1. Respond ONLY with valid JSON (no markdown, no ```, no comments).
2. Return only the fields that the message explicitly mentions or modifies. If the message contains no usable constraint, return {}.
3. Do NOT guess default values. A missing field = not modified.
4. Avoided categories go into avoided_categories, not into must_avoid.

EXAMPLES:

User: "5-day trip to Rome with €2000, we love culture"
→ {"destination":"Rome","num_days":5,"total_budget":2000,"preferred_categories":["culture"]}

User: "We are 2, relaxed pace and no shopping"
→ {"num_travelers":2,"preferred_pace":"relaxed","avoided_categories":["shopping"]}

User: "Budget €1500, max 3 activities per day, I like gastronomy"
→ {"total_budget":1500,"max_activities_per_day":3,"preferred_categories":["gastro"]}

User: "Hello!"
→ {}

User: "I absolutely want to see the Colosseum"
→ {"must_visit":["colosseum"]}
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


def extract_constraints(
    user_message: str,
    current_constraints: Optional[dict] = None,
    max_retries: int = 1,
) -> tuple[dict, Optional[str]]:
    """
    Extrait les contraintes d'un message utilisateur.

    Returns:
        (constraints_dict, error_message_or_None)
        constraints_dict ne contient que les champs à modifier.
    """
    current_constraints = current_constraints or {}
    client = get_client()

    user_content = (
        f"Contraintes actuelles : {json.dumps(current_constraints, ensure_ascii=False)}\n\n"
        f"Message utilisateur : \"{user_message}\"\n\n"
        "Extrais les contraintes modifiées en JSON strict."
    )

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
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
            return ExtractedConstraints(**data).clean(), None

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
) -> str:
    """Génère la réponse conversationnelle à afficher dans le chat."""
    client = get_client()

    plan_summary = _summarize_plan(plan, constraints)
    changes_str = json.dumps(extracted_changes, ensure_ascii=False) if extracted_changes else "aucune"

    user_content = (
        f"Message utilisateur : \"{user_message}\"\n"
        f"Contraintes modifiées par ce message : {changes_str}\n"
        f"Résumé du plan : {plan_summary}\n\n"
        "Réponds en 2-3 phrases max."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
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
