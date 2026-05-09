"""
Extracteur de contraintes enrichi avec niveau d'obligation (required/value).

Prend en entrée un texte utilisateur et retourne un JSON structuré :
{
    "destination": {"value": "Rome", "required": True},
    "total_budget": {"value": 2000, "required": True},
    "num_days": {"value": 5, "required": True},
    "preferred_categories": {"value": ["culture"], "required": False},
    ...
}

Règles :
- Contraintes critiques (destination, budget, durée) → required = True
- Préférences et options → required = False
- Détection d'expressions vagues → signale les champs à clarifier
- Module d'évaluation intégré pour mesurer la qualité de l'extraction
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from llm_client import extract_constraints as _llm_extract


# ─── Classification des champs ─────────────────────────────────────────────

# Champs critiques → required = True automatiquement
CRITICAL_FIELDS: frozenset[str] = frozenset({
    "destination",
    "num_days",
    "total_budget",
})

# Champs optionnels → required = False
OPTIONAL_FIELDS: frozenset[str] = frozenset({
    "num_travelers",
    "hotel_per_night",
    "daily_food_budget",
    "preferred_categories",
    "avoided_categories",
    "preferred_pace",
    "morning_preference",
    "must_visit",
    "must_avoid",
    "max_activities_per_day",
    "min_activities_per_day",
})

# Valeurs par défaut raisonnables pour les champs optionnels
REASONABLE_DEFAULTS: dict[str, Any] = {
    "num_travelers": 1,
    "hotel_per_night": 100,
    "daily_food_budget": 60,
    "preferred_pace": "moderate",
    "max_activities_per_day": 3,
    "min_activities_per_day": 1,
}

# ─── Détection des expressions vagues ──────────────────────────────────────

_VAGUE_PATTERNS: dict[str, list[str]] = {
    "total_budget": [
        r"\bconfortable\b",
        r"\braisonnable\b",
        r"\bpas trop cher\b",
        r"\babordable\b",
        r"\bmod[eé]r[eé]\b",
        r"\baccessible\b",
        r"\bmoyen\b",
        r"\bpas cher\b",
        r"\bbien\b",
        r"\bpas trop\b",
        r"\bpetit budget\b",
        r"\bgros budget\b",
    ],
    "num_days": [
        r"\bquelques jours\b",
        r"\bplusieurs jours\b",
        r"\bun moment\b",
        r"\bun s[eé]jour\b",
        r"\bune semaine environ\b",
    ],
    "destination": [
        r"\bquelque part\b",
        r"\nun peu partout\b",
        r"\bn'importe o[uù]\b",
    ],
}


def detect_vague_fields(message: str) -> dict[str, bool]:
    """
    Détecte les expressions vagues dans le message utilisateur.

    Returns:
        {field: True} pour chaque champ mentionné de façon floue.
    """
    msg_lower = message.lower()
    vague: dict[str, bool] = {}
    for field, patterns in _VAGUE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, msg_lower):
                vague[field] = True
                break
    return vague


# ─── Extraction principale ─────────────────────────────────────────────────

@dataclass
class ConstraintItem:
    """Contrainte avec son niveau d'obligation."""
    value: Any
    required: bool

    def to_dict(self) -> dict:
        return {"value": self.value, "required": self.required}


def extract_structured_constraints(
    user_message: str,
    current_constraints: Optional[dict] = None,
) -> tuple[dict[str, dict], Optional[str], dict[str, bool]]:
    """
    Extrait les contraintes depuis un message utilisateur en les structurant
    avec un niveau d'obligation.

    Args:
        user_message: texte brut de l'utilisateur
        current_constraints: état courant (pour le contexte LLM)

    Returns:
        (structured, error, vague_fields)

        structured: {field: {"value": ..., "required": bool}}
        error: message d'erreur ou None
        vague_fields: {field: True} pour les champs flous détectés
    """
    flat_constraints, error = _llm_extract(user_message, current_constraints)
    vague_fields = detect_vague_fields(user_message)

    structured: dict[str, dict] = {}

    for field, value in flat_constraints.items():
        required = field in CRITICAL_FIELDS
        structured[field] = {"value": value, "required": required}

    # Appliquer les défauts raisonnables pour les champs non encore connus
    if current_constraints is not None:
        for field, default_val in REASONABLE_DEFAULTS.items():
            if field not in structured and field not in current_constraints:
                structured[field] = {"value": default_val, "required": False}

    return structured, error, vague_fields


def to_flat_dict(structured: dict[str, dict]) -> dict:
    """Convertit le format structuré vers le dict plat attendu par le solveur."""
    return {field: item["value"] for field, item in structured.items()}


def structured_from_flat(flat: dict) -> dict[str, dict]:
    """
    Construit le format structuré depuis un dict plat (pour l'état de session).
    Utile pour afficher l'état courant avec les niveaux d'obligation.
    """
    return {
        field: {"value": value, "required": field in CRITICAL_FIELDS}
        for field, value in flat.items()
    }


# ─── Module d'évaluation ───────────────────────────────────────────────────

def evaluate_extraction(extracted: dict, expected: dict) -> dict:
    """
    Évalue la qualité de l'extraction LLM en comparant avec les valeurs attendues.

    Gestion des ambiguïtés :
    - Comparaison insensible à la casse pour les strings
    - Tolérance ±10% pour les valeurs numériques
    - Correspondance partielle pour les listes

    Args:
        extracted: dict plat des contraintes extraites
        expected: dict plat des contraintes attendues (ground truth)

    Returns:
        {
            "precision": float,
            "recall": float,
            "f1": float,
            "correct": int,
            "false_positives": int,
            "false_negatives": int,
            "details": {field: {"extracted": ..., "expected": ..., "match": bool}}
        }
    """
    tp = fp = fn = 0
    details: dict[str, dict] = {}

    all_fields = set(extracted.keys()) | set(expected.keys())

    for field in all_fields:
        ext_val = extracted.get(field)
        exp_val = expected.get(field)

        if ext_val is not None and exp_val is not None:
            match = _values_match(ext_val, exp_val)
            if match:
                tp += 1
            else:
                fp += 1
            details[field] = {
                "extracted": ext_val,
                "expected": exp_val,
                "match": match,
            }
        elif ext_val is not None:
            fp += 1
            details[field] = {
                "extracted": ext_val,
                "expected": None,
                "match": False,
                "note": "false positive (not expected)",
            }
        else:
            fn += 1
            details[field] = {
                "extracted": None,
                "expected": exp_val,
                "match": False,
                "note": "false negative (missed)",
            }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "correct": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "details": details,
    }


def _values_match(a: Any, b: Any) -> bool:
    """Comparaison tolérante entre deux valeurs extraites."""
    # Listes : intersection partielle (au moins 50% de correspondance)
    if isinstance(a, list) and isinstance(b, list):
        if not a and not b:
            return True
        if not a or not b:
            return False
        a_set = {str(x).lower() for x in a}
        b_set = {str(x).lower() for x in b}
        overlap = len(a_set & b_set)
        return overlap / max(len(a_set), len(b_set)) >= 0.5

    # Numériques : tolérance ±10%
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if b == 0:
            return a == 0
        return abs(a - b) / abs(b) <= 0.10

    # Strings : insensible à la casse
    return str(a).lower().strip() == str(b).lower().strip()


# ─── Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test constraint_extractor ===\n")

    test_cases = [
        "Je veux partir 5 jours à Rome avec un budget de 2000 euros",
        "Budget confortable, pas trop cher",
        "On est 2, rythme tranquille, j'aime la culture et la gastronomie",
        "Quelques jours quelque part en Europe",
        "Salut !",
    ]

    for msg in test_cases:
        print(f"Message : {msg!r}")
        structured, err, vague = extract_structured_constraints(msg, {})
        if err:
            print(f"  [erreur] {err}")
        print(f"  Structuré  : {structured}")
        print(f"  Flou       : {vague}")
        print()

    # Test évaluation
    print("=== Test évaluation ===")
    extracted = {"destination": "Rome", "num_days": 5, "total_budget": 2100}
    expected = {"destination": "Rome", "num_days": 5, "total_budget": 2000}
    result = evaluate_extraction(extracted, expected)
    print(f"Précision={result['precision']}  Rappel={result['recall']}  F1={result['f1']}")
    for field, det in result["details"].items():
        status = "✓" if det["match"] else "✗"
        print(f"  {status} {field}: extrait={det['extracted']} attendu={det['expected']}")
