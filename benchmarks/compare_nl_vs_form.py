"""Benchmark expressivite NL vs Formulaire : diff JSON pur, sans solveur.

Pour chaque scenario, on compare :
  - le `form_input` : ce qu'un formulaire structure (raisonnable) peut capturer
  - l'`extracted` : ce que l'extraction LLM tire du meme message NL

Pas d'appel solveur, pas de generation de ville. On veut juste montrer
combien d'ambiguites le NL resout qu'un formulaire ne peut pas modeliser.

Lancer :
  python3 benchmarks/compare_nl_vs_form.py
  python3 benchmarks/compare_nl_vs_form.py --html benchmarks/report_nl_vs_form.html
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import extract_constraints  # noqa: E402

FORM_FIELDS = frozenset({
    "destination", "num_days", "total_budget", "num_travelers",
    "hotel_per_night", "daily_food_budget",
    "day_start_hour", "day_end_hour",
    "preferred_pace", "preferred_categories", "avoided_categories",
    "start_date", "end_date", "transport_mode",
})

NL_ONLY_FIELDS = frozenset({
    "must_visit", "must_avoid", "must_visit_on_day",
    "day_specific_start_hour", "day_specific_end_hour",
    "max_activities_per_day", "min_activities_per_day",
    "morning_preference",
})

@dataclass
class Scenario:
    id: str
    ambiguity_type: str  # categorie d'ambiguite : implicite, conditionnelle, etc.
    nl_message: str
    form_input: dict  # ce qu'un formulaire raisonnable capturerait
    expected_nl_fields: list[str] = field(default_factory=list)  # champs attendus en NL-only
    explanation: str = ""  # phrase pedagogique pour la slide

SCENARIOS: list[Scenario] = [
    Scenario(
        id="S1-baseline",
        ambiguity_type="aucune",
        nl_message="5 jours a Rome avec 2000€ pour 2 personnes, on aime la culture, du 12 au 16 juin 2026",
        form_input={
            "destination": "Rome", "num_days": 5, "total_budget": 2000,
            "num_travelers": 2, "start_date": "2026-06-12",
            "end_date": "2026-06-16", "preferred_categories": ["culture"],
        },
        expected_nl_fields=[],
        explanation="Cas trivial : tout est exprimable des deux cotes. Sert de baseline.",
    ),
    Scenario(
        id="S2-activite-au-jour",
        ambiguity_type="objet imbrique activite -> jour",
        nl_message=("Lisbonne 4 jours 1500€ du 10 au 13 juin 2026 pour 2, "
                    "je veux faire la Tour de Belem le jour 3"),
        form_input={
            "destination": "Lisbonne", "num_days": 4, "total_budget": 1500,
            "num_travelers": 2, "start_date": "2026-06-10",
            "end_date": "2026-06-13",
        },
        expected_nl_fields=["must_visit", "must_visit_on_day"],
        explanation=("L'utilisateur EPINGLE une activite a un jour precis. "
                     "Le formulaire devrait offrir un widget 'activite -> jour' "
                     "dont la liste d'activites n'est connue qu'apres generation."),
    ),
    Scenario(
        id="S3-inclusion-exclusion",
        ambiguity_type="inclusion + exclusion nommees",
        nl_message=("Tokyo 6 jours 3500€ du 5 au 10 juin 2026 pour 2, "
                    "on veut voir le sanctuaire Senso-ji et eviter Shibuya"),
        form_input={
            "destination": "Tokyo", "num_days": 6, "total_budget": 3500,
            "num_travelers": 2, "start_date": "2026-06-05",
            "end_date": "2026-06-10",
        },
        expected_nl_fields=["must_visit", "must_avoid"],
        explanation=("Inclusion ET exclusion d'activites nommees. "
                     "Formulaire = il faudrait deux text-areas libres "
                     "sans validation, sans completion, sans coherence."),
    ),
    Scenario(
        id="S4-negation-categorie",
        ambiguity_type="negations en chaine",
        nl_message=("Madrid 4 jours 1800€ du 1 au 4 juin 2026 pour 2, "
                    "on aime la gastro mais pas le shopping ni le nightlife"),
        form_input={
            "destination": "Madrid", "num_days": 4, "total_budget": 1800,
            "num_travelers": 2, "start_date": "2026-06-01",
            "end_date": "2026-06-04",
            "preferred_categories": ["gastro"],
            "avoided_categories": ["shopping", "nightlife"],
        },
        expected_nl_fields=[],
        explanation=("Cas ou les deux gagnent. Le formulaire peut le faire avec "
                     "2 multi-select. Important pour montrer qu'on ne pretend pas "
                     "que le NL est superieur partout."),
    ),
    Scenario(
        id="S5-date-implicite",
        ambiguity_type="duree implicite via expression de date",
        nl_message="Amsterdam le week-end du 14 juin 2026, 600€, 2 personnes, on aime la culture",
        form_input={
            "destination": "Amsterdam", "total_budget": 600,
            "num_travelers": 2, "preferred_categories": ["culture"],
        },
        expected_nl_fields=[],  # ces 3 champs sont en FORM_FIELDS mais le user ne les a pas tapes
        explanation=("'week-end du 14 juin' = num_days=2 + start_date=samedi 14 juin "
                     "+ end_date=dimanche 15 juin. Le LLM doit calculer 3 valeurs depuis "
                     "1 expression. Formulaire = 3 champs a remplir manuellement."),
    ),
    Scenario(
        id="S6-compound-multi-axe",
        ambiguity_type="compose multi-contraintes en 1 phrase",
        nl_message=("4 jours a Vienne du 18 au 21 juin 2026 pour 2 personnes, 2200€, "
                    "on veut absolument l'opera de Vienne le jour 1 au soir, "
                    "on adore la musique et la gastro, on evite les musees, "
                    "et on commence pas avant 10h"),
        form_input={
            "destination": "Vienne", "num_days": 4, "total_budget": 2200,
            "num_travelers": 2, "start_date": "2026-06-18",
            "end_date": "2026-06-21",
            "preferred_categories": ["culture", "gastro"],
            "avoided_categories": [],
            "day_start_hour": 10,
        },
        expected_nl_fields=["must_visit", "must_visit_on_day"],
        explanation=("UNE phrase NL = 7 contraintes distinctes a remplir manuellement "
                     "dans 7 champs differents du formulaire. Le NL traite tout en "
                     "une seule extraction."),
    ),
]

def diff_json(extracted: dict, form_input: dict) -> dict:
    """Compare les deux JSON et categorise les differences."""
    nl_keys = set(extracted.keys())
    form_keys = set(form_input.keys())

    only_nl_strict = sorted(nl_keys & NL_ONLY_FIELDS)            # NL-only par nature
    only_nl_soft = sorted((nl_keys - form_keys) - NL_ONLY_FIELDS)  # NL a capture, form n'avait pas
    only_form = sorted(form_keys - nl_keys)                       # form avait, NL n'a pas
    both = sorted(nl_keys & form_keys)
    value_mismatches = []
    for k in both:
        if extracted[k] != form_input[k]:
            value_mismatches.append((k, form_input[k], extracted[k]))

    return {
        "only_nl_strict": only_nl_strict,
        "only_nl_soft": only_nl_soft,
        "only_form": only_form,
        "both": both,
        "value_mismatches": value_mismatches,
    }

def render_console(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print(f"{'Scenario':<22} {'Ambiguite':<32} {'NL-only strict':<22} {'NL-only soft':<20}")
    print("=" * 100)
    for r in results:
        s: Scenario = r["scenario"]
        d = r["diff"]
        nl_strict = ", ".join(d["only_nl_strict"]) or "(aucune)"
        nl_soft = ", ".join(d["only_nl_soft"]) or "(aucune)"
        print(f"{s.id:<22} {s.ambiguity_type[:31]:<32} {nl_strict[:21]:<22} {nl_soft[:19]:<20}")
    print("=" * 100)

    print()
    n = len(results)
    nl_only_total = sum(len(r["diff"]["only_nl_strict"]) for r in results)
    nl_soft_total = sum(len(r["diff"]["only_nl_soft"]) for r in results)
    avg_extract = sum(r["extraction_ms"] for r in results) / n
    print(f"Agreges sur {n} scenarios :")
    print(f"  Latence moyenne d'extraction LLM : {avg_extract:.0f} ms")
    print(f"  Total champs NL-only stricts     : {nl_only_total} (inexpressibles en formulaire)")
    print(f"  Total champs NL-only doux        : {nl_soft_total} (le user n'aurait pas pense remplir)")
    print()

def render_html(results: list[dict], path: Path) -> None:
    rows = []
    for r in results:
        s: Scenario = r["scenario"]
        ex = r["extracted"]
        fm = s.form_input
        d = r["diff"]

        # Construction du JSON formate avec marquage des differences
        def _fmt_json(data: dict, mark_keys: set[str], mark_class: str) -> str:
            lines = ["{"]
            items = list(data.items())
            for i, (k, v) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                val = json.dumps(v, ensure_ascii=False)
                if len(val) > 80:
                    val = val[:77] + "..."
                key_class = f" class='{mark_class}'" if k in mark_keys else ""
                lines.append(f"  <span{key_class}><strong>\"{escape(k)}\"</strong>: "
                             f"{escape(val)}{comma}</span>")
            lines.append("}")
            return "<br>".join(lines)

        nl_marks = set(d["only_nl_strict"]) | set(d["only_nl_soft"])
        form_marks = set(d["only_form"])

        nl_html = _fmt_json(ex, nl_marks, "diff-nl")
        form_html = _fmt_json(fm, form_marks, "diff-form")

        # Badges
        if d["only_nl_strict"]:
            verdict = (f"<strong class='badge advantage'>NL capture "
                       f"{len(d['only_nl_strict'])} contrainte(s) inexprimables</strong>")
        elif d["only_nl_soft"]:
            verdict = (f"<strong class='badge soft'>NL capture "
                       f"{len(d['only_nl_soft'])} champ(s) que le user n'a pas saisi</strong>")
        else:
            verdict = "<strong class='badge neutral'>Egalite — exprimable des deux cotes</strong>"

        rows.append(f"""
        <div class='scenario'>
          <div class='hdr'>
            <span class='id'>{escape(s.id)}</span>
            <span class='tag'>{escape(s.ambiguity_type)}</span>
            {verdict}
          </div>
          <div class='msg'>« {escape(s.nl_message)} »</div>
          <div class='explanation'>{escape(s.explanation)}</div>
          <div class='cols'>
            <div class='col col-form'>
              <div class='col-title'>📋 Formulaire (form_input)</div>
              <pre>{form_html}</pre>
            </div>
            <div class='col col-nl'>
              <div class='col-title'>💬 NL extraction (LLM) — {r['extraction_ms']} ms</div>
              <pre>{nl_html}</pre>
            </div>
          </div>
        </div>""")

    n = len(results)
    nl_only_total = sum(len(r["diff"]["only_nl_strict"]) for r in results)
    nl_soft_total = sum(len(r["diff"]["only_nl_soft"]) for r in results)
    avg_extract = sum(r["extraction_ms"] for r in results) / n
    advantage_count = sum(1 for r in results if r["diff"]["only_nl_strict"])

    html = f"""<!doctype html><html lang='fr'><head><meta charset='utf-8'>
<title>Expressivite NL vs Formulaire — diff JSON</title>
<style>
  :root {{
    --bg: #fbfaf7; --bg-form: #EBF1FB; --bg-nl: #FBF3E8;
    --txt: #2b2b2b; --muted: #8a8a8a; --line: #e8e3db;
    --advantage: #b8860b; --soft: #2e7d32; --neutral: #6b6b6b;
    --diff-nl: #fff8e1; --diff-form: #ffebee;
  }}
  body {{ font-family: -apple-system, sans-serif; max-width: 1300px;
         margin: 24px auto; padding: 0 20px; color: var(--txt); background: var(--bg); }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); margin-bottom: 24px; }}
  .metric-row {{ display: grid; grid-template-columns: repeat(4, 1fr);
                 gap: 12px; margin-bottom: 28px; }}
  .metric {{ padding: 16px; background: white; border: 1px solid var(--line);
             border-radius: 12px; }}
  .metric .v {{ font-size: 28px; font-weight: 700; }}
  .metric .l {{ font-size: 10px; color: var(--muted); text-transform: uppercase;
                letter-spacing: 0.8px; margin-bottom: 4px; }}
  .scenario {{ background: white; border: 1px solid var(--line);
               border-radius: 12px; padding: 20px; margin-bottom: 18px; }}
  .hdr {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
  .id {{ font-family: 'DM Mono', monospace; font-weight: 700; font-size: 14px; }}
  .tag {{ font-size: 10px; padding: 2px 8px; background: #f1ede5;
          border-radius: 10px; text-transform: uppercase;
          letter-spacing: 0.5px; color: #555; }}
  .badge {{ font-size: 11px; padding: 3px 10px; border-radius: 12px; margin-left: auto; }}
  .badge.advantage {{ color: var(--advantage); background: #fff8e1; }}
  .badge.soft {{ color: var(--soft); background: #e8f5e9; }}
  .badge.neutral {{ color: var(--neutral); background: #f5f5f5; }}
  .msg {{ font-style: italic; color: #555; padding: 10px 14px;
          background: #faf7f2; border-radius: 8px; margin-bottom: 8px; }}
  .explanation {{ font-size: 12px; color: #777; margin-bottom: 14px;
                  padding-left: 10px; border-left: 3px solid var(--line); }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .col {{ padding: 14px; border-radius: 8px; }}
  .col-form {{ background: var(--bg-form); }}
  .col-nl {{ background: var(--bg-nl); }}
  .col-title {{ font-weight: 600; font-size: 12px; margin-bottom: 8px;
                text-transform: uppercase; letter-spacing: 0.5px; }}
  pre {{ font-family: 'DM Mono', 'Courier New', monospace; font-size: 11px;
         margin: 0; white-space: pre-wrap; word-break: break-word; }}
  .diff-nl {{ background: var(--diff-nl); padding: 1px 4px; border-radius: 3px;
              border-left: 3px solid var(--advantage); display: inline-block; }}
  .diff-form {{ background: var(--diff-form); padding: 1px 4px; border-radius: 3px;
                border-left: 3px solid #c62828; display: inline-block; }}
  .takeaway {{ margin-top: 30px; padding: 20px; background: white;
               border-left: 4px solid var(--advantage); border-radius: 8px; }}
  .takeaway li {{ margin: 6px 0; }}
</style></head><body>

<h1>Expressivite Langage Naturel vs Formulaire</h1>
<div class='sub'>{n} scenarios — comparaison purement structurelle (extraction LLM seule, pas de solveur)</div>

<div class='metric-row'>
  <div class='metric'><div class='l'>Scenarios testes</div><div class='v'>{n}</div></div>
  <div class='metric'><div class='l'>Scenarios avec avantage NL</div><div class='v'>{advantage_count}/{n}</div></div>
  <div class='metric'><div class='l'>Champs NL-only stricts</div><div class='v'>{nl_only_total}</div></div>
  <div class='metric'><div class='l'>Latence extraction LLM</div><div class='v'>{avg_extract:.0f} ms</div></div>
</div>

<p style='color:#777; font-size:13px;'>
<strong>Comment lire :</strong> chaque carte montre un scenario avec son message NL, ce qu'un
<strong>formulaire</strong> permettrait de capturer (gauche), et ce que le <strong>LLM extrait</strong>
du message (droite). Les cles surlignees en <span class='diff-nl'>jaune</span> sont des champs que
le NL capture mais pas le formulaire — ces contraintes seraient <strong>perdues</strong> avec un
formulaire classique.
</p>

{''.join(rows)}

<div class='takeaway'>
<strong>Lecture des resultats</strong>
<ul>
  <li><strong>{advantage_count}/{n} scenarios</strong> contiennent au moins une contrainte
      <em>strictement</em> inexprimable en formulaire (must_visit, must_visit_on_day,
      day_specific_*_hour, must_avoid…).</li>
  <li>Au total, <strong>{nl_only_total} champs NL-only stricts</strong> sont captures par
      le LLM dans ces messages, contre <strong>0</strong> par le formulaire.</li>
  <li>L'extraction LLM coute en moyenne <strong>{avg_extract:.0f} ms</strong> par message —
      negligeable face au temps de remplissage manuel d'un formulaire equivalent (15-60s).</li>
  <li>Le formulaire reste valide pour le sous-ensemble des contraintes structurees
      (destination, dates, budget, categories), mais montre ses limites des qu'on touche
      aux contraintes <strong>relationnelles</strong> (activite-jour) ou <strong>conditionnelles</strong>
      (override par jour de semaine).</li>
</ul>
</div>

</body></html>"""
    path.write_text(html, encoding="utf-8")
    print(f"Rapport HTML : {path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path,
                        default=Path(__file__).parent / "report_nl_vs_form.html",
                        help="Chemin du rapport HTML")
    args = parser.parse_args()

    print("Benchmark expressivite NL vs Formulaire (diff JSON, sans solveur)")
    print(f"Modele LLM : {__import__('llm_client').LLM_MODEL}")
    print()

    results = []
    for scen in SCENARIOS:
        print(f"[{scen.id}] {scen.ambiguity_type}")
        t0 = time.time()
        extracted, err = extract_constraints(scen.nl_message)
        extraction_ms = int((time.time() - t0) * 1000)
        if err:
            print(f"  ⚠ {err}")
        print(f"  NL extracted en {extraction_ms} ms : "
              f"{json.dumps(extracted, ensure_ascii=False)[:160]}")
        diff = diff_json(extracted, scen.form_input)
        if diff["only_nl_strict"]:
            print(f"  ✓ NL-only strict : {diff['only_nl_strict']}")
        if diff["only_nl_soft"]:
            print(f"  + NL-only soft   : {diff['only_nl_soft']}")
        if diff["value_mismatches"]:
            print(f"  ≠ valeurs differentes : {diff['value_mismatches']}")
        print()
        results.append({
            "scenario": scen, "extracted": extracted,
            "extraction_ms": extraction_ms, "diff": diff,
            "error": err,
        })

    render_console(results)
    if args.html:
        render_html(results, args.html)

if __name__ == "__main__":
    main()
