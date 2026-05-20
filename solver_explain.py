"""Génération de l'explication des compromis d'un plan CP-SAT.
Séparé de solver.py : c'est de la présentation utilisateur, pas de la"""
from __future__ import annotations

_PACE_TARGET = {"relaxed": 2, "moderate": 3, "intense": 4}

def explain_solution(solution: dict, constraints: dict) -> str:
    """Texte d'explication des compromis. Vide-friendly et INFEASIBLE-aware."""
    if not solution:
        return "Aucune solution disponible."

    if solution.get("status") == "INFEASIBLE":
        lines = ["Aucun plan ne satisfait toutes les contraintes."]
        if solution.get("message"):
            lines.append(solution["message"])
        lines.append(
            f"Suggestions : augmenter le budget (actuel : "
            f"{constraints.get('total_budget', 0)} €), allonger le séjour "
            f"(actuel : {constraints.get('num_days', 0)} j), ou assouplir "
            f"les catégories."
        )
        return "\n".join(lines)

    lines: list[str] = []
    summary = solution.get("summary", {})
    days = solution.get("days", [])

    if solution.get("mode") == "strict":
        lines.append(
            "Mode strict : seules les activités des catégories préférées "
            "ont été sélectionnées."
        )

    remaining = summary.get("remaining_budget", 0)
    total_cost = summary.get("total_cost", 0)
    budget = summary.get("budget", 0)
    if remaining < 0:
        lines.append(f"⚠ Budget dépassé de {abs(remaining)} € (coût total : {total_cost} €).")
    elif remaining < 50:
        lines.append(f"Budget quasi épuisé : {total_cost} € / {budget} € ({remaining} € restants).")
    else:
        lines.append(f"Budget respecté : {total_cost} € / {budget} € ({remaining} € de marge).")

    num_days = len(days)
    total_acts = sum(len(d.get("activities", [])) for d in days)
    avg = total_acts / num_days if num_days else 0
    pace = constraints.get("preferred_pace", "moderate")
    target = _PACE_TARGET.get(pace, 3)
    if abs(avg - target) > 0.9:
        direction = "moins" if avg < target else "plus"
        lines.append(
            f"Rythme ajusté : {avg:.1f} activité(s)/jour en moyenne ({direction} "
            f"que le rythme '{pace}' souhaité de {target}/jour). "
            f"Cause probable : créneaux ou budget."
        )

    selected_cats = [a["category"] for d in days for a in d.get("activities", [])]
    for cat in constraints.get("preferred_categories", []):
        count = selected_cats.count(cat)
        if count == 0:
            lines.append(
                f"⚠ Catégorie préférée « {cat} » absente : budget insuffisant, "
                "horaires incompatibles ou quota journalier atteint."
            )
        else:
            lines.append(f"✓ {count} activité(s) « {cat} » planifiée(s).")

    for cat in constraints.get("avoided_categories", []):
        count = selected_cats.count(cat)
        if count > 0:
            lines.append(
                f"ℹ {count} activité(s) « {cat} » incluse(s) malgré l'évitement "
                "(nécessaires pour le minimum par jour)."
            )

    selected_ids = {a["id"] for d in days for a in d.get("activities", [])}
    for act_id in constraints.get("must_visit", []):
        if act_id not in selected_ids:
            lines.append(
                f"⚠ Activité obligatoire « {act_id} » absente : vérifier coût "
                "et horaires."
            )

    for v in solution.get("violated_soft_constraints", []):
        if not any(v in line for line in lines):
            lines.append(f"⚠ {v}")

    return "\n".join(lines) if lines else "✓ Toutes les contraintes et préférences sont respectées."
