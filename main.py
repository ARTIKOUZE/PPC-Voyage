"""Interface conversationnelle CLI du planificateur de voyage (LLM + CP-SAT). Usage :"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import uuid

from orchestrator import handle_turn, reset_session, get_session_state
from dialog_manager import format_constraints_status, is_complete
from constraint_extractor import extract_structured_constraints, evaluate_extraction
from solver import explain_solution

WIDTH = 80

def _hr(char: str = "─") -> str:
    return char * WIDTH

def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix)

def print_banner():
    print(_hr("═"))
    print("  Assistant de planification de voyage — LLM + CP-SAT")
    print("  Tapez votre demande en langage naturel, ou /help pour l'aide.")
    print(_hr("═"))
    print()

def print_help():
    print(_hr())
    print("Commandes disponibles :")
    print("  /reset      Nouvelle conversation (efface les contraintes)")
    print("  /state      Affiche les contraintes actuelles de la session")
    print("  /explain    Ré-affiche l'explication du dernier plan")
    print("  /mode       Bascule entre strict et flexible")
    print("  /eval <msg> Évalue l'extraction LLM sur <msg> (debug)")
    print("  /help       Affiche cette aide")
    print("  /quit       Quitte le programme")
    print(_hr())

def print_plan_summary(plan: dict):
    """Affiche un résumé concis du plan."""
    if not plan or plan.get("status") == "INFEASIBLE":
        return

    summary = plan.get("summary", {})
    days = plan.get("days", [])
    mode = plan.get("mode", "flexible")
    fallback = plan.get("mode_fallback", "")

    print()
    print(_hr())
    mode_str = f"{mode}" + (f" → {fallback}" if fallback else "")
    print(f"  PLAN OPTIMISÉ  [mode: {mode_str}]")
    print(_hr())

    for day in days:
        acts = day.get("activities", [])
        day_label = f"Jour {day['day']}"
        if not acts:
            print(f"  {day_label}: (aucune activité)")
            continue
        print(f"\n  {day_label}:")
        for act in acts:
            cost_str = f"{act['cost']}€" if act["cost"] > 0 else "gratuit"
            print(
                f"    {act['start_time']}–{act['end_time']}  "
                f"{act['name']} ({act['category']}) — {cost_str}"
            )

    print()
    print(_hr("·"))
    print(
        f"  Coût total    : {summary.get('total_cost', 0)}€ / {summary.get('budget', 0)}€"
        f"  (marge : {summary.get('remaining_budget', 0)}€)"
    )
    print(f"  Hôtel         : {summary.get('hotel_cost', 0)}€")
    print(f"  Repas         : {summary.get('food_cost', 0)}€")
    print(f"  Activités     : {summary.get('activity_cost', 0)}€")
    print(f"  Nb activités  : {summary.get('total_activities', 0)}")

    stats = plan.get("stats", {})
    print(
        f"  Solve         : {stats.get('solve_time_ms', 0)} ms  "
        f"({stats.get('status_name', '')})"
    )
    print(_hr())

def print_constraints_respected(plan: dict):
    respected = plan.get("respected_constraints", [])
    violated = plan.get("violated_soft_constraints", [])

    if respected or violated:
        print()
        print("  Contraintes :")
        for r in respected[:5]:
            print(f"    ✓ {r}")
        for v in violated:
            print(f"    ✗ {v}")

def print_assistant(text: str):
    print()
    print(f"  Assistant : {text}")
    print()

def run(session_id: str, initial_mode: str = "flexible"):
    mode = initial_mode
    last_plan: dict | None = None
    last_explanation: str | None = None

    print_banner()
    print(f"  Session : {session_id}   Mode : {mode}")
    print()
    print_assistant(
        "Bonjour ! Décrivez votre voyage (ex: 'Je veux partir 5 jours à Rome "
        "avec un budget de 2000€')."
    )

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split(maxsplit=1)
            command = cmd[0].lower()

            if command in ("/quit", "/exit", "/q"):
                print("Au revoir !")
                break

            elif command == "/help":
                print_help()

            elif command == "/reset":
                reset_session(session_id)
                last_plan = None
                last_explanation = None
                print_assistant("Session réinitialisée. Décrivez votre nouveau voyage.")

            elif command == "/state":
                state = get_session_state(session_id)
                print()
                print(format_constraints_status(state))
                print()
                print("  Toutes les contraintes :")
                for k, v in state.items():
                    if v and v != [] and v != {}:
                        print(f"    {k} = {v!r}")
                print()

            elif command == "/explain":
                if last_plan and last_explanation:
                    print()
                    print(_hr("·"))
                    for line in last_explanation.split("\n"):
                        print(f"  {line}")
                    print(_hr("·"))
                    print()
                else:
                    print_assistant("Aucun plan disponible. Décrivez votre voyage d'abord.")

            elif command == "/mode":
                mode = "strict" if mode == "flexible" else "flexible"
                print_assistant(
                    f"Mode basculé vers '{mode}'. "
                    + (
                        "Les catégories préférées seront imposées comme contraintes dures."
                        if mode == "strict"
                        else "Les préférences sont traitées comme des objectifs souples."
                    )
                )

            elif command == "/eval":
                if len(cmd) < 2:
                    print_assistant("Usage : /eval <message à tester>")
                else:
                    test_msg = cmd[1]
                    structured, err, vague = extract_structured_constraints(
                        test_msg, get_session_state(session_id)
                    )
                    print()
                    print(f"  Extraction pour : {test_msg!r}")
                    print(f"  Résultat structuré :")
                    for field, item in structured.items():
                        req_label = "REQUIS" if item["required"] else "optionnel"
                        print(f"    {field}: {item['value']!r}  [{req_label}]")
                    if vague:
                        print(f"  Expressions vagues détectées : {list(vague.keys())}")
                    if err:
                        print(f"  Erreur : {err}")
                    print()

            else:
                print_assistant(f"Commande inconnue : {command}. Tapez /help.")

            continue

        print("  [Traitement en cours…]")

        result = handle_turn(
            session_id=session_id,
            user_message=user_input,
            solve_timeout=15,
            mode=mode,
        )

        reply = result.get("reply", "")
        plan = result.get("plan")
        explanation = result.get("explanation")
        needs_info = result.get("needs_info")
        errors = result.get("errors", [])

        print_assistant(reply)

        if needs_info:
            continue

        if plan and plan.get("status") != "INFEASIBLE":
            last_plan = plan
            last_explanation = explanation
            print_plan_summary(plan)
            print_constraints_respected(plan)

            if explanation:
                print()
                print(_hr("·"))
                print("  Explication des compromis :")
                for line in explanation.split("\n"):
                    print(f"    {line}")
                print(_hr("·"))
                print()

        elif plan and plan.get("status") == "INFEASIBLE":
            last_plan = plan
            last_explanation = explanation
            if explanation:
                print()
                for line in explanation.split("\n"):
                    print(f"  {line}")
                print()

        non_critical_errors = [e for e in errors if not e.startswith("incomplete_constraints")]
        if non_critical_errors:
            print(f"  [debug] {'; '.join(non_critical_errors)}")

def main():
    parser = argparse.ArgumentParser(
        description="Assistant de planification conversationnel (LLM + CP-SAT)"
    )
    parser.add_argument(
        "--mode",
        choices=["flexible", "strict"],
        default="flexible",
        help="Mode du solveur : 'flexible' (compromis acceptés) ou 'strict' (préférences imposées)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="ID de session (généré automatiquement si absent)",
    )
    args = parser.parse_args()

    session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"
    run(session_id=session_id, initial_mode=args.mode)

if __name__ == "__main__":
    main()
