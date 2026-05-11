"""
Benchmark de l'extraction LLM des contraintes (objectif 4 du sujet).

Charge benchmarks/extraction_dataset.json, lance chaque cas à travers
llm_client.extract_constraints, compare avec la vérité-terrain via
constraint_extractor.evaluate_extraction, et produit :

  - un résumé console (F1 global + par champ + par catégorie d'ambiguïté),
  - un rapport HTML détaillé avec cas par cas (benchmarks/report.html).

Usage :
  python3 benchmarks/run_extraction_benchmark.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from html import escape
from pathlib import Path

# Permettre l'import depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import extract_constraints  # noqa: E402
from constraint_extractor import evaluate_extraction  # noqa: E402


DATASET_PATH = Path(__file__).parent / "extraction_dataset.json"
REPORT_PATH = Path(__file__).parent / "report.html"


def aggregate(results: list[dict]) -> dict:
    """Agrège TP/FP/FN sur tous les cas pour produire P/R/F1 global et par champ."""
    tp_total = fp_total = fn_total = 0
    by_field: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in results:
        ev = r["eval"]
        tp_total += ev["correct"]
        fp_total += ev["false_positives"]
        fn_total += ev["false_negatives"]
        for field, det in ev["details"].items():
            slot = by_field[field]
            cat_slot = by_category[r["case"]["category"]]
            if det["match"]:
                slot["tp"] += 1
                cat_slot["tp"] += 1
            elif det.get("extracted") is None and det.get("expected") is not None:
                slot["fn"] += 1
                cat_slot["fn"] += 1
            else:
                slot["fp"] += 1
                cat_slot["fp"] += 1

    def f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 3), round(r, 3), round(f, 3)

    p, r, f = f1(tp_total, fp_total, fn_total)
    return {
        "global": {"precision": p, "recall": r, "f1": f,
                   "tp": tp_total, "fp": fp_total, "fn": fn_total},
        "by_field": {k: {**v, "metrics": f1(v["tp"], v["fp"], v["fn"])}
                     for k, v in sorted(by_field.items())},
        "by_category": {k: {**v, "metrics": f1(v["tp"], v["fp"], v["fn"])}
                        for k, v in sorted(by_category.items())},
    }


def render_html(results: list[dict], agg: dict, elapsed: float) -> str:
    g = agg["global"]
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }
    h1 { font-size: 22px; margin: 0 0 4px; }
    .sub { color: #888; font-size: 13px; margin-bottom: 24px; }
    .metric-row { display: flex; gap: 14px; margin-bottom: 24px; }
    .metric { flex: 1; padding: 14px; border-radius: 10px; background: #f5f5f5; }
    .metric .v { font-size: 26px; font-weight: 700; }
    .metric .l { font-size: 11px; color: #777; letter-spacing: 0.5px; text-transform: uppercase; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }
    th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #eee; }
    th { background: #fafafa; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #555; }
    .case { border: 1px solid #eee; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
    .case.fail { border-color: #f4cccc; background: #fff7f7; }
    .case.pass { border-color: #d6e6c8; background: #f8fbf3; }
    .case h3 { margin: 0 0 6px; font-size: 14px; }
    .case .tag { font-size: 10px; padding: 1px 7px; border-radius: 10px; background: #eee; margin-right: 6px; vertical-align: middle; }
    .case .msg { font-style: italic; color: #444; margin: 6px 0; }
    pre { background: #f7f7f7; padding: 8px 10px; border-radius: 6px; font-size: 12px; margin: 4px 0; overflow-x: auto; }
    .match { color: #2e7d32; }
    .miss { color: #c62828; font-weight: 600; }
    .small { font-size: 11px; color: #999; }
    .f1-good { color: #2e7d32; }
    .f1-ok { color: #f9a825; }
    .f1-bad { color: #c62828; }
    """

    def f1_class(v: float) -> str:
        return "f1-good" if v >= 0.9 else "f1-ok" if v >= 0.7 else "f1-bad"

    # Header + metrics globales
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>Bench LLM Extraction</title><style>{css}</style></head><body>",
        "<h1>Benchmark d'extraction LLM — Travel Planner</h1>",
        f"<div class='sub'>{len(results)} cas testés en {elapsed:.1f}s — modèle : <code>{os.environ.get('LLM_MODEL', 'qwen3.6-35b-a3b')}</code></div>",
        "<div class='metric-row'>",
        f"<div class='metric'><div class='l'>Précision</div><div class='v {f1_class(g['precision'])}'>{g['precision']:.2f}</div></div>",
        f"<div class='metric'><div class='l'>Rappel</div><div class='v {f1_class(g['recall'])}'>{g['recall']:.2f}</div></div>",
        f"<div class='metric'><div class='l'>F1</div><div class='v {f1_class(g['f1'])}'>{g['f1']:.2f}</div></div>",
        f"<div class='metric'><div class='l'>Slots OK / FP / FN</div><div class='v'>{g['tp']}/{g['fp']}/{g['fn']}</div></div>",
        "</div>",
    ]

    # Tableau par catégorie d'ambiguïté
    parts.append("<h2>F1 par catégorie d'ambiguïté</h2><table><tr><th>Catégorie</th><th>TP</th><th>FP</th><th>FN</th><th>Précision</th><th>Rappel</th><th>F1</th></tr>")
    for cat, v in agg["by_category"].items():
        p, r, f = v["metrics"]
        parts.append(f"<tr><td>{cat}</td><td>{v['tp']}</td><td>{v['fp']}</td><td>{v['fn']}</td>"
                     f"<td>{p:.2f}</td><td>{r:.2f}</td><td class='{f1_class(f)}'>{f:.2f}</td></tr>")
    parts.append("</table>")

    # Tableau par champ extrait
    parts.append("<h2>F1 par champ extrait</h2><table><tr><th>Champ</th><th>TP</th><th>FP</th><th>FN</th><th>Précision</th><th>Rappel</th><th>F1</th></tr>")
    for field, v in agg["by_field"].items():
        p, r, f = v["metrics"]
        parts.append(f"<tr><td><code>{field}</code></td><td>{v['tp']}</td><td>{v['fp']}</td><td>{v['fn']}</td>"
                     f"<td>{p:.2f}</td><td>{r:.2f}</td><td class='{f1_class(f)}'>{f:.2f}</td></tr>")
    parts.append("</table>")

    # Détail cas par cas
    parts.append("<h2>Cas par cas</h2>")
    for r in results:
        case = r["case"]
        ev = r["eval"]
        all_match = ev["false_positives"] == 0 and ev["false_negatives"] == 0
        klass = "pass" if all_match else "fail"
        parts.append(f"<div class='case {klass}'>")
        parts.append(f"<h3>{case['id']} <span class='tag'>{case['category']}</span> "
                     f"<span class='small'>F1={ev['f1']:.2f} · ⏱ {r.get('elapsed_ms',0):.0f}ms</span></h3>")
        parts.append(f"<div class='msg'>« {escape(case['message'])} »</div>")
        # Mismatch summary
        if not all_match:
            mismatches = []
            for field, det in ev["details"].items():
                if not det["match"]:
                    mismatches.append(
                        f"<li><code>{field}</code> : attendu <span class='match'>{json.dumps(det['expected'], ensure_ascii=False)}</span>, "
                        f"extrait <span class='miss'>{json.dumps(det['extracted'], ensure_ascii=False)}</span></li>"
                    )
            if mismatches:
                parts.append("<ul>" + "".join(mismatches) + "</ul>")
        if r.get("error"):
            parts.append(f"<div class='miss small'>⚠ Erreur LLM : {escape(str(r['error']))}</div>")
        parts.append(f"<pre>extracted = {json.dumps(r['extracted'], ensure_ascii=False, indent=2)}</pre>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)


def main():
    with DATASET_PATH.open() as f:
        dataset = json.load(f)
    cases = dataset["cases"]
    print(f"Chargé {len(cases)} cas depuis {DATASET_PATH}")
    print("Modèle :", os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b"))
    print()

    results = []
    t0 = time.time()
    for case in cases:
        t_case = time.time()
        try:
            extracted, err = extract_constraints(case["message"])
        except Exception as e:
            extracted, err = {}, f"exception: {e}"
        ev = evaluate_extraction(extracted, case["expected"])
        elapsed_ms = (time.time() - t_case) * 1000

        all_match = ev["false_positives"] == 0 and ev["false_negatives"] == 0
        flag = "✓" if all_match else "✗"
        print(f"{flag} [{case['category']:11}] {case['id']:14} "
              f"F1={ev['f1']:.2f} ({elapsed_ms:5.0f}ms)  « {case['message'][:50]}{'…' if len(case['message'])>50 else ''} »")
        if err:
            print(f"  ⚠ {err}")

        results.append({
            "case": case,
            "extracted": extracted,
            "eval": ev,
            "error": err,
            "elapsed_ms": elapsed_ms,
        })

    elapsed = time.time() - t0
    agg = aggregate(results)

    print()
    g = agg["global"]
    print(f"=== Résultat global ===")
    print(f"Précision : {g['precision']:.2f}")
    print(f"Rappel    : {g['recall']:.2f}")
    print(f"F1        : {g['f1']:.2f}")
    print(f"Slots TP={g['tp']}  FP={g['fp']}  FN={g['fn']}  "
          f"({len(results)} cas en {elapsed:.1f}s)")
    print()
    print("=== F1 par catégorie ===")
    for cat, v in agg["by_category"].items():
        p, r, f = v["metrics"]
        print(f"  {cat:11} F1={f:.2f}  (P={p:.2f}  R={r:.2f}  n={v['tp']+v['fp']+v['fn']})")

    REPORT_PATH.write_text(render_html(results, agg, elapsed), encoding="utf-8")
    print(f"\nRapport HTML : {REPORT_PATH}")


if __name__ == "__main__":
    main()
