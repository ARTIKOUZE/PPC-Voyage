"""Gestionnaire de dialogue multi-tour pour la complétion des contraintes.
Responsabilités :"""
from __future__ import annotations

from typing import Optional

CRITICAL_FIELDS: dict[str, str] = {
    "destination": "Tu veux partir où ?",
    "total_budget": "Quel est ton budget maximum (en euros) ?",
    "num_days": "Combien de jours dure ton voyage ?",
    "start_date": (
        "Quelles sont tes dates exactes de séjour ? "
        "(ex: \"du 12/06/2026 au 15/06/2026\" ou \"j'arrive le 12/06/2026 pour 3 jours\") "
        "— sans ça je ne peux pas vérifier les jours d'ouverture des activités."
    ),
}

_VAGUE_CLARIFICATIONS: dict[str, str] = {
    "total_budget": (
        "Tu m'as parlé de budget, mais j'ai besoin d'un montant précis. "
        "Quel est ton budget maximum en euros ?"
    ),
    "num_days": (
        "Peux-tu me préciser le nombre exact de jours ?"
    ),
    "destination": (
        "Tu as mentionné une destination, mais peux-tu me donner "
        "le nom d'une ville précise ?"
    ),
    "day_start_hour": (
        "À quelle heure précise veux-tu commencer tes journées ? (ex: 9h, 10h)"
    ),
    "day_end_hour": (
        "Jusqu'à quelle heure veux-tu faire des activités ? (ex: 18h, 20h, 22h)"
    ),
    "start_date": (
        "Pour vérifier les jours d'ouverture des activités, j'ai besoin de tes dates "
        "exactes de séjour. Format : \"12/06/2026 au 15/06/2026\" ou "
        "\"j'arrive le 12/06/2026 pour 3 jours\"."
    ),
}

_FIELD_LABELS: dict[str, str] = {
    "destination": "la destination",
    "total_budget": "le budget",
    "num_days": "la durée du séjour",
    "start_date": "les dates exactes de séjour",
    "day_start_hour": "l'heure de début de journée",
    "day_end_hour": "l'heure de fin de journée",
}

def _is_missing(field: str, value) -> bool:
    """Retourne True si la valeur est absente ou invalide pour un champ critique."""
    if value is None:
        return True

    if field == "destination":
        return not isinstance(value, str) or len(value.strip()) < 2

    if field in ("total_budget", "num_days"):
        if not isinstance(value, (int, float)):
            return True
        return value <= 0

    if field == "day_start_hour":
        if not isinstance(value, (int, float)):
            return True
        return not (0 <= value <= 23)

    if field == "day_end_hour":
        if not isinstance(value, (int, float)):
            return True
        return not (1 <= value <= 24)

    if field == "start_date":
        if not isinstance(value, str):
            return True
        import re as _re
        return not _re.match(r"^\d{4}-\d{2}-\d{2}$", value)

    return False

def get_missing_critical(constraints: dict) -> list[str]:
    """Retourne la liste des champs critiques manquants ou invalides.
    Args:"""
    return [
        field
        for field in CRITICAL_FIELDS
        if _is_missing(field, constraints.get(field))
    ]

def is_complete(constraints: dict) -> bool:
    """Retourne True si toutes les contraintes critiques sont présentes et valides.
    Le solveur peut être lancé."""
    return len(get_missing_critical(constraints)) == 0

def generate_question(field: str, vague: bool = False) -> str:
    """Génère la question à poser pour obtenir la valeur d'un champ critique.
    Args:"""
    if vague and field in _VAGUE_CLARIFICATIONS:
        return _VAGUE_CLARIFICATIONS[field]
    return CRITICAL_FIELDS.get(field, f"Peux-tu préciser {field} ?")

def next_question(
    constraints: dict,
    vague_fields: Optional[dict[str, bool]] = None,
) -> Optional[str]:
    """Retourne la prochaine question à poser, ou None si tout est complet.
    Priorité :"""
    missing = get_missing_critical(constraints)
    if missing:
        return generate_question(missing[0])

    if vague_fields:
        for field in CRITICAL_FIELDS:
            if vague_fields.get(field) and _is_missing(field, constraints.get(field)):
                return generate_question(field, vague=True)

    return None

def format_missing_summary(constraints: dict) -> str:
    """Retourne un message récapitulatif des informations manquantes.
    Utile pour le débogage ou l'affichage dans les logs."""
    missing = get_missing_critical(constraints)
    if not missing:
        return "Toutes les contraintes critiques sont présentes."

    labels = [_FIELD_LABELS.get(f, f) for f in missing]
    return f"Informations manquantes : {', '.join(labels)}."

def format_constraints_status(constraints: dict) -> str:
    """Affiche l'état détaillé de chaque contrainte critique."""
    lines = ["État des contraintes critiques :"]
    for field, question in CRITICAL_FIELDS.items():
        value = constraints.get(field)
        if _is_missing(field, value):
            lines.append(f"  ✗ {_FIELD_LABELS.get(field, field)} : manquant")
        else:
            lines.append(f"  ✓ {_FIELD_LABELS.get(field, field)} : {value!r}")
    return "\n".join(lines)

if __name__ == "__main__":
    print("=== Test dialog_manager ===\n")

    test_states = [
        ({}, None),
        ({"destination": "Rome"}, None),
        ({"destination": "Rome", "total_budget": 2000}, None),
        ({"destination": "Rome", "total_budget": 2000, "num_days": 5}, None),
        ({"destination": "Rome", "num_days": 5}, {"total_budget": True}),
    ]

    for constraints, vague in test_states:
        complete = is_complete(constraints)
        question = next_question(constraints, vague)
        print(f"Contraintes : {constraints}")
        print(f"  Complet    : {complete}")
        print(f"  Manquant   : {get_missing_critical(constraints)}")
        print(f"  Question   : {question!r}")
        print()
