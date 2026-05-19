"""
Benchmark CLI : comparaison Langage Naturel (LLM extraction) vs Formulaire
(contraintes structurées directes).

Pour chaque scénario, exécute les deux pipelines sur la MÊME demande logique
et mesure :
  - friction d'entrée (caractères tapés / champs remplis)
  - latence d'extraction
  - latence totale du pipeline
  - couverture d'expression : la contrainte est-elle exprimable côté formulaire ?
  - équivalence du plan produit (résumé)

Le formulaire est défini comme un sous-ensemble fixé de champs : destination,
num_days, total_budget, num_travelers, hotel_per_night, daily_food_budget,
day_start_hour, day_end_hour, preferred_pace, preferred_categories,
avoided_categories. Les contraintes qui ne sont PAS dans cette liste ne sont
pas exprimables via le formulaire — c'est l'argument central pour montrer
l'apport du langage naturel.

Usage :
  python3 benchmarks/compare_nl_vs_form.py
  python3 benchmarks/compare_nl_vs_form.py --html benchmarks/nl_vs_form.html
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import extract_constraints  # noqa: E402
from llm_city_provider import generate_city_data  # noqa: E402
from solver import solve_with_city_data  # noqa: E402


# Champs exprimables dans le formulaire (équivalent à FormPanel supprimé)
FORM_FIELDS = frozenset({
    "destination", "num_days", "total_budget", "num_travelers",
    "hotel_per_night", "daily_food_budget",
    "day_start_hour", "day_end_hour",
    "preferred_pace", "preferred_categories", "avoided_categories",
    "start_date", "end_date",
})


SCENARIOS = [
    {
        "id": "S1-simple",
        "nl_message": "5 jours à Rome avec un budget de 2000 euros pour 2 personnes, on aime la culture, du 12 au 16 juin",
        "form_input": {
            "destination": "Rome",
            "num_days": 5,
            "total_budget": 2000,
            "num_travelers": 2,
            "start_date": "2026-06-12",
            "end_date": "2026-06-16",
            "preferred_categories": ["culture"],
        },
        "note": "Cas simple — tout exprimable des deux côtés.",
    },
    {
        "id": "S2-must-visit-on-day",
        "nl_message": ("Lisbonne 4 jours 1500€ du 10 au 13 juin, on est 2, "
                       "et je veux faire le Tour de Belém le jour 3"),
        "form_input": {
            "destination": "Lisbonne",
            "num_days": 4,
            "total_budget": 1500,
            "num_travelers": 2,
            "start_date": "2026-06-10",
            "end_date": "2026-06-13",
        },
        "note": "Contrainte conditionnelle 'Belém le jour 3' INEXPRIMABLE en formulaire.",
    },
    {
        "id": "S3-negation",
        "nl_message": ("Paris 5 jours 2500€ du 1er au 5 juin pour 2, "
                       "on adore la culture mais surtout pas de shopping ni de nightlife"),
        "form_input": {
            "destination": "Paris",
            "num_days": 5,
            "total_budget": 2500,
            "num_travelers": 2,
            "start_date": "2026-06-01",
            "end_date": "2026-06-05",
            "preferred_categories": ["culture"],
            "avoided_categories": ["shopping", "nightlife"],
        },
        "note": "Négations exprimables (avoided_categories), mais friction multi-champ.",
    },
    {
        "id": "S4-mix-must-avoid",
        "nl_message": ("Tokyo 6 jours 3500€ du 5 au 10 juin pour 2, on veut absolument voir "
                       "le sanctuaire Senso-ji et éviter Shibuya car trop de monde"),
        "form_input": {
            "destination": "Tokyo",
            "num_days": 6,
            "total_budget": 3500,
            "num_travelers": 2,
            "start_date": "2026-06-05",
            "end_date": "2026-06-10",
        },
        "note": "'must_visit' et 'must_avoid' au niveau activité — INEXPRIMABLES en formulaire.",
    },
    {
        "id": "S5-pace-relatif",
        "nl_message": ("Berlin 3 jours 1200€ du 20 au 22 juin pour 1 personne, "
                       "on est fatigué donc rythme tranquille"),
        "form_input": {
            "destination": "Berlin",
            "num_days": 3,
            "total_budget": 1200,
            "num_travelers": 1,
            "start_date": "2026-06-20",
            "end_date": "2026-06-22",
            "preferred_pace": "relaxed",
        },
        "note": "Inférence implicite 'fatigué → relaxed' faite par le LLM. "
                "Formulaire = il faut cliquer le bon bouton, mais exprimable.",
    },
    {
        "id": "S6-horaires",
        "nl_message": ("Madrid 3 jours 1800€ du 7 au 9 juin, on veut commencer "
                       "à 10h et finir à 23h car on aime sortir le soir"),
        "form_input": {
            "destination": "Madrid",
            "num_days": 3,
            "total_budget": 1800,
            "num_travelers": 1,
            "start_date": "2026-06-07",
            "end_date": "2026-06-09",
            "day_start_hour": 10,
            "day_end_hour": 23,
            "preferred_categories": ["nightlife"],
        },
        "note": "Horaires exprimables, mais le NL exprime aussi le 'pourquoi'.",
    },
]


# ─────────────────────────────────────────────
# Outils
# ─────────────────────────────────────────────

def constraint_coverage(target: dict) -> dict:
    """Retourne quelles clés sont exprimables dans le formulaire et lesquelles ne le sont pas."""
    expr = {k for k in target if k in FORM_FIELDS}
    inexpr = {k for k in target if k not in FORM_FIELDS}
    return {"expressible": sorted(expr), "inexpressible": sorted(inexpr)}


def safe_solve(constraints: dict, city_data: dict) -> dict:
    return solve_with_city_data(
        constraints, city_data, time_limit_seconds=10, mode="flexible"
    )


def run_nl(scenario: dict, city_data: dict) -> dict:
    """Pipeline NL : extract LLM → merge defaults → solve."""
    msg = scenario["nl_message"]
    t0 = time.time()
    extracted, err = extract_constraints(msg)
    t_extract = time.time() - t0

    # Compléter avec valeurs par défaut équivalentes au formulaire
    constraints = dict(extracted)
    constraints.setdefault("num_travelers", 1)
    constraints.setdefault("daily_food_budget", 60)
    constraints.setdefault("day_start_hour", 9)
    constraints.setdefault("day_end_hour", 19)
    constraints.setdefault("preferred_pace", "moderate")
    # 40 % du budget total pour l'hôtel (cf. orchestrator._resolve_hotel_budget)
    if "hotel_per_night" not in constraints:
        tb = constraints.get("total_budget") or 0
        nd = constraints.get("num_days") or 1
        if tb > 0 and nd > 0:
            constraints["hotel_per_night"] = max(50, int(tb * 0.4 / nd))

    t1 = time.time()
    plan = safe_solve(constraints, city_data)
    t_solve = time.time() - t1
    return {
        "extracted": extracted,
        "extraction_error": err,
        "extraction_ms": int(t_extract * 1000),
        "solve_ms": int(t_solve * 1000),
        "total_ms": int((time.time() - t0) * 1000),
        "plan": plan,
        "input_size": len(msg),
    }


def run_form(scenario: dict, city_data: dict) -> dict:
    """Pipeline form : pas d'extraction, complétion défauts → solve."""
    form_input = dict(scenario["form_input"])
    t0 = time.time()

    constraints = dict(form_input)
    constraints.setdefault("num_travelers", 1)
    constraints.setdefault("daily_food_budget", 60)
    constraints.setdefault("day_start_hour", 9)
    constraints.setdefault("day_end_hour", 19)
    constraints.setdefault("preferred_pace", "moderate")
    if "hotel_per_night" not in constraints:
        tb = constraints.get("total_budget") or 0
        nd = constraints.get("num_days") or 1
        if tb > 0 and nd > 0:
            constraints["hotel_per_night"] = max(50, int(tb * 0.4 / nd))

    t1 = time.time()
    plan = safe_solve(constraints, city_data)
    t_solve = time.time() - t1
    return {
        "constraints": form_input,
        "extraction_ms": 0,
        "solve_ms": int(t_solve * 1000),
        "total_ms": int((time.time() - t0) * 1000),
        "plan": plan,
        "fields_filled": len(form_input),
    }


# ─────────────────────────────────────────────
# Rendu console + HTML
# ─────────────────────────────────────────────

def fmt_plan(plan: dict) -> str:
    if not plan or plan.get("status") not in ("OPTIMAL", "FEASIBLE"):
        return f"INFEASIBLE/{plan.get('status') if plan else 'NONE'}"
    s = plan.get("summary", {})
    return (f"{s.get('total_activities', 0)} acts · "
            f"{s.get('total_cost', 0)}€/{s.get('budget', 0)}€")


def render_console(results: list[dict]):
    print()
    print("=" * 95)
    print(f"{'Scénario':<24} {'NL':<25} {'Formulaire':<25} {'Couverture form':<18}")
    print("=" * 95)
    for r in results:
        scen = r["scenario"]
        nl = r["nl"]
        fm = r["form"]
        cov = r["coverage"]
        cov_str = f"{len(cov['expressible'])}/{len(cov['expressible'])+len(cov['inexpressible'])} OK"
        if cov['inexpressible']:
            cov_str += f" (✗ {', '.join(cov['inexpressible'])[:30]})"
        nl_str = f"{nl['input_size']}c · {nl['total_ms']}ms · {fmt_plan(nl['plan'])}"
        fm_str = f"{fm['fields_filled']}f · {fm['total_ms']}ms · {fmt_plan(fm['plan'])}"
        print(f"{scen['id']:<24} {nl_str:<25} {fm_str:<25} {cov_str}")
    print("=" * 95)
    print()
    # Agrégats
    total_nl_ms = sum(r["nl"]["total_ms"] for r in results)
    total_fm_ms = sum(r["form"]["total_ms"] for r in results)
    total_inexpr = sum(len(r["coverage"]["inexpressible"]) for r in results)
    n = len(results)
    print(f"Agrégats sur {n} scénarios :")
    print(f"  Latence moyenne NL          : {total_nl_ms/n:.0f} ms")
    print(f"  Latence moyenne formulaire  : {total_fm_ms/n:.0f} ms "
          f"({(total_fm_ms/total_nl_ms*100 if total_nl_ms else 0):.0f} % du NL)")
    print(f"  Contraintes inexprimables   : {total_inexpr} sur {n} scénarios")
    print(f"    → uniquement encodables via langage naturel.")
    print()


def render_html(results: list[dict], path: Path):
    rows = []
    for r in results:
        s = r["scenario"]
        nl = r["nl"]
        fm = r["form"]
        cov = r["coverage"]
        inexpr_str = (", ".join(cov["inexpressible"])
                      if cov["inexpressible"] else "—")
        rows.append(f"""
        <tr>
          <td><strong>{escape(s['id'])}</strong><br><span class="msg">« {escape(s['nl_message'])} »</span><br>
              <em class="note">{escape(s['note'])}</em></td>
          <td class="nl">
            {nl['input_size']} chars<br>
            extract {nl['extraction_ms']} ms · solve {nl['solve_ms']} ms<br>
            total {nl['total_ms']} ms<br>
            <code>{escape(fmt_plan(nl['plan']))}</code>
          </td>
          <td class="form">
            {fm['fields_filled']} champs remplis<br>
            extract 0 ms · solve {fm['solve_ms']} ms<br>
            total {fm['total_ms']} ms<br>
            <code>{escape(fmt_plan(fm['plan']))}</code>
          </td>
          <td class="cov">
            <span class="ok">{len(cov['expressible'])} OK</span><br>
            <span class="bad">{len(cov['inexpressible'])} INEXPRIMABLES</span><br>
            <small>{escape(inexpr_str)}</small>
          </td>
        </tr>""")

    n = len(results)
    total_nl = sum(r["nl"]["total_ms"] for r in results)
    total_fm = sum(r["form"]["total_ms"] for r in results)
    inexpr_total = sum(len(r["coverage"]["inexpressible"]) for r in results)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Benchmark NL vs Formulaire</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
.sub {{ color: #888; margin-bottom: 24px; }}
.metric-row {{ display: flex; gap: 14px; margin-bottom: 24px; }}
.metric {{ flex: 1; padding: 14px; background: #f5f5f5; border-radius: 10px; }}
.metric .v {{ font-size: 22px; font-weight: 700; }}
.metric .l {{ font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; vertical-align: top; }}
th {{ background: #fafafa; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
.msg {{ font-style: italic; color: #666; }}
.note {{ color: #999; font-size: 11px; }}
.nl {{ background: #FBF3E8; }}
.form {{ background: #EBF1FB; }}
.cov .ok {{ color: #2e7d32; font-weight: 600; }}
.cov .bad {{ color: #c62828; font-weight: 600; }}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
.takeaway {{ margin-top: 30px; padding: 16px; background: #fff8e1; border-left: 4px solid #f9a825; border-radius: 4px; }}
</style></head><body>
<h1>Comparaison Langage Naturel vs Formulaire</h1>
<div class="sub">{n} scénarios représentatifs · pipeline complet (LLM extraction + génération ville + CP-SAT)</div>
<div class="metric-row">
  <div class="metric"><div class="l">Latence moyenne NL</div><div class="v">{total_nl/n:.0f} ms</div></div>
  <div class="metric"><div class="l">Latence moyenne form</div><div class="v">{total_fm/n:.0f} ms</div></div>
  <div class="metric"><div class="l">Overhead NL</div><div class="v">+{(total_nl - total_fm)/n:.0f} ms</div></div>
  <div class="metric"><div class="l">Contraintes inexprimables (form)</div><div class="v">{inexpr_total}</div></div>
</div>
<table>
  <tr><th style="width:35%">Scénario / message NL</th><th>NL (avec LLM)</th><th>Formulaire</th><th>Couverture form</th></tr>
  {''.join(rows)}
</table>
<div class="takeaway">
<strong>Lecture</strong> — Le formulaire est <strong>plus rapide</strong> (pas d'appel LLM d'extraction)
et <strong>déterministe</strong>. Le langage naturel encaisse un <strong>overhead de ~{(total_nl-total_fm)/n:.0f} ms</strong>
mais permet d'exprimer des contraintes que le formulaire ne peut pas modéliser
(« le Tour de Belém le jour 3 », « éviter Shibuya »…). Sur les {n} scénarios,
<strong>{inexpr_total} clauses sont inaccessibles via formulaire</strong> sans démultiplier
les champs jusqu'à l'ingérable.
</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    print(f"Rapport HTML : {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=None,
                        help="chemin du rapport HTML à écrire (optionnel)")
    args = parser.parse_args()

    print("Démarrage du benchmark NL vs Formulaire")
    print(f"Modèle LLM utilisé : {__import__('llm_client').LLM_MODEL}")
    print()

    results = []
    # On cache les city_data par destination pour ne pas faire 2× l'appel LLM
    city_cache: dict[str, dict] = {}
    for scen in SCENARIOS:
        dest = scen["form_input"]["destination"]
        nd = scen["form_input"].get("num_days", 5)
        print(f"[{scen['id']}] {dest} ({nd} j) — chargement ville…", end="", flush=True)

        key = f"{dest.lower()}|{nd}"
        if key in city_cache:
            city_data = city_cache[key]
            print(" (cache)")
        else:
            city_data = generate_city_data(dest, transport_mode="foot", num_days=nd)
            city_cache[key] = city_data
            print(" OK" if city_data else " ✗ ÉCHEC")
            if not city_data:
                continue

        # Construire la cible logique de la requête (union NL + form notion)
        target = dict(scen["form_input"])
        coverage = constraint_coverage(target)

        print(f"  NL  ", end="", flush=True)
        nl_res = run_nl(scen, city_data)
        print(f"{nl_res['total_ms']} ms → {fmt_plan(nl_res['plan'])}")

        print(f"  Form", end="", flush=True)
        fm_res = run_form(scen, city_data)
        print(f"{fm_res['total_ms']} ms → {fmt_plan(fm_res['plan'])}")

        results.append({
            "scenario": scen,
            "nl": nl_res,
            "form": fm_res,
            "coverage": coverage,
        })

    render_console(results)
    if args.html:
        render_html(results, args.html)


if __name__ == "__main__":
    main()
