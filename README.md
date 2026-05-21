# PPC-Voyage — Assistant de planification de séjour

Projet réalisé dans le cadre du cours de Programmation Par Contraintes (SCIA).
Sujet I1 : un assistant conversationnel où l'utilisateur décrit son voyage en
langage naturel, un LLM extrait les contraintes typées, un solveur CP-SAT
construit le plan optimal, et le LLM narre le résultat.

## En 1 phrase

> Tu écris « Lisbonne 4 jours 1500€ pour 2, je veux la Tour de Belém le jour 3 et finir à midi le samedi », et l'app te sort un planning concret jour-par-jour qui respecte budget, horaires d'ouverture, temps de trajet et tes contraintes spécifiques.

## Lancer le projet

```bash
# 1) Backend Python
cp .env.example .env             # remplir les clés LLM + OpenTripMap
pip install -r requirements.txt
uvicorn api_server:app --reload --port 8000

# 2) Frontend React (dans un autre terminal)
cd planner-ui
cp .env.example .env             # adresse du backend
npm install
npm run dev
# → http://localhost:5173
```

## Comment ça marche : le pipeline

```
User (français libre)
        │
        ▼
┌─────────────────────┐
│  extract_constraints│  llm_client.py
│  (LLM zero-shot)    │  → JSON typé via Pydantic
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ merge_constraints   │  orchestrator.py
│ + session state     │  (multi-tour : pinning, diff)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ city_data           │  llm_city_provider.py
│ LLM génère POIs +   │  + opentripmap_client.py
│ OTM vérifie         │  → GPS Wikipédia, adresses réelles
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ solve CP-SAT        │  solver.py
│ (~5-10s)            │  → maximize bonus − penalty
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ narrate_plan        │  llm_client.py
│ (justifie compromis)│  → bulle chat
└─────────────────────┘
```

## Le cœur du sujet : le modèle CP-SAT

C'est ce qui compte vraiment dans le projet. Le LLM n'est qu'une interface.

### Variables de décision (`solver.py`)

| Variable | Type | Sens |
|---|---|---|
| `selected[a]` | `BoolVar` | activité `a` retenue dans le voyage ? |
| `assign[a, d]` | `BoolVar` | `a` planifiée le jour `d` ? |
| `start[a, d]` | `IntVar` | créneau de début (1 slot = 30 min depuis 7h) |
| `intervals[a, d]` | `OptionalFixedSizeIntervalVar` | enveloppe scheduling, activée par `assign` |

### 8 familles de contraintes HARD (inviolables, `model.add(...)`)

1. **Cohérence sélection ↔ assignment** — un seul jour par activité, cohérence selected/assign via `add_at_most_one`, `add_bool_or`, `add_implication`
2. **Budget** — `Σ selected[a]·cost[a]·travelers ≤ activity_budget` + plafond food par jour
3. **Temporel / scheduling** — fenêtres journalières + horaires d'ouverture par activité ; supporte override par jour (`day_specific_end_hour: {2: 12}`)
4. **Logique activité-level** — `must_visit`, `must_avoid`, `must_visit_on_day`, incompatibilités, prérequis, fermetures hebdomadaires
5. **Capacité / no-overlap** — `add_no_overlap` sur les intervalles d'un jour + temps de trajet inter-activités via matrice OSRM
6. **Cardinalité** — min/max activités par jour (dynamique selon fenêtre dispo) + min/max par catégorie
7. **Repas (disjonctifs)** — pour les restaurants : `is_lunch ∨ is_dinner` (12h-14h ou 19h30-21h30)
8. **Stabilité multi-tour (pinning)** — activités des jours non touchés forcées à leur slot précédent

### 3 BONUS soft (ajoutés à l'objectif)

| Bonus | Valeur | Description |
|---|---|---|
| Priorité d'activité | 5–13 | score Wikipédia + 3/5 si catégorie préférée − 5 si évitée |
| Bonus matin | +2 | activité de `morning_preference` qui démarre avant 12h |
| Stabilité multi-tour | +8 ou +20 | activité préservée d'un tour à l'autre |

### 5 PÉNALITÉS soft (soustraites de l'objectif)

| Pénalité | Poids | Description |
|---|---|---|
| Sous-dépense budget | × 1/20€ | activity_cost < 70% du budget activités |
| Shortfall pace | × 15/20/25 | day_count < target (relaxed=3, moderate=4, intense=5) |
| Overflow pace | × 10/4/1 | day_count > target (asymétrique selon pace) |
| Plancher min/jour | × 20 | day_count < min_activities_per_day |
| Trajet | × 0–8 par paire | `(travel_min − 15) // 10`, 0 si < 15 min |

### Objectif

```
maximize(  Σ bonus  −  Σ penalty  )
```

C'est exactement la formulation **WCSP** (Weighted Constraint Satisfaction Problem,
Schiex & Verfaillie 1995) : contraintes hard inviolables + somme pondérée des soft.

→ Référence détaillée : [CONSTRAINTS_REFERENCE.md](CONSTRAINTS_REFERENCE.md)

## Multi-tour : modifier sans tout casser

Le défi : *« je retire la Tour Eiffel »* ne doit pas reshuffler tout le séjour.

**Mécanisme** : `SessionStore` garde le dernier plan. Pour chaque nouveau tour,
on détecte les jours « touchés » par la requête (`_determine_touched_days`).
Les autres jours sont **pin-hard** : `assign[a, d] == 1` + `start[a, d] == slot_précédent`.
Les activités explicitement déplacées (`must_visit_on_day`) ou supprimées
(`must_avoid`) sont exemptées du pin pour éviter les conflits.

Un bonus de stabilité (+8 ou +20) renforce la continuité sur les jours touchés.

→ Détails : [orchestrator.py](orchestrator.py), `_determine_touched_days` et
  `_compute_plan_diff` (qui calcule les compromis du solveur à expliquer
  ensuite au LLM pour la narration).

## Robustesse de l'extraction LLM

Pour mesurer la fiabilité du « pont NL → contraintes CSP », on a annoté un
dataset de **40 messages d'utilisateur** en 10 catégories d'ambiguïté :

```
simple, vague, négation, conditionnelle, implicite, préférence,
minimal, complex, adversarial, tricky
```

Le benchmark calcule P/R/F1 par champ et par catégorie :

```bash
python3 benchmarks/run_extraction_benchmark.py
# → benchmarks/report.html
```

**Catégories difficiles** : `adversarial` (chiffres en lettres, contradictions,
abréviations) et `tricky` (sarcasme, contraintes inhabituelles, restaurants
nommés). Ce sont nos cas d'échec assumés.

## Comparaison NL vs Formulaire

Pour valider que la couche NL n'est pas juste de la latence en plus, on compare
le JSON extrait par le LLM vs ce qu'un formulaire raisonnable pourrait capturer :

```bash
python3 benchmarks/compare_nl_vs_form.py
# → benchmarks/report_nl_vs_form.html
```

Résultat : 4/6 scénarios contiennent au moins une contrainte **strictement
inexprimable** en formulaire (`must_visit_on_day`, `day_specific_end_hour`,
`must_avoid` activité-level, compounds multi-axe).

## Fichiers clés

| Fichier | Rôle |
|---|---|
| [api_server.py](api_server.py) | FastAPI : `/chat`, `/state`, `/reset`, `/health` |
| [orchestrator.py](orchestrator.py) | Pipeline complet, `SessionStore`, merge, multi-tour |
| [solver.py](solver.py) | Modèle CP-SAT (8 hard + 3 bonus + 5 pénalités) |
| [solver_models.py](solver_models.py) | `Activity`, `TravelConstraints`, helpers GPS |
| [solver_explain.py](solver_explain.py) | Explication textuelle du plan |
| [llm_client.py](llm_client.py) | Extraction (Pydantic + few-shots) + narration + fallback endpoint |
| [llm_city_provider.py](llm_city_provider.py) | Génération LLM des POIs + extension dynamique du pool |
| [opentripmap_client.py](opentripmap_client.py) | Vérification OTM (GPS Wikipédia + adresses réelles) |
| [dialog_manager.py](dialog_manager.py) | Détection contraintes critiques manquantes |
| [constraint_extractor.py](constraint_extractor.py) | `evaluate_extraction` pour le calcul F1 |
| [planner-ui/src/App.jsx](planner-ui/src/App.jsx) | UI React (chat + timeline + budget bar) |

## Tests

```bash
# Tests offline du solveur sur 8 villes mockées (sans LLM)
python3 test_solver_scenarios.py

# Benchmark d'extraction LLM (F1 sur 40 cas annotés)
python3 benchmarks/run_extraction_benchmark.py

# Comparaison NL vs Formulaire
python3 benchmarks/compare_nl_vs_form.py
```

## APIs externes utilisées

| Service | Usage | Clé |
|---|---|---|
| LLM (qwen3-35b OpenAI-compat) | extraction + narration | oui (LLM_API_KEY) |
| LLM fallback (omnicoder-9b) | bascule automatique si timeout/5xx | oui (LLM_API_KEY_FALLBACK) |
| OpenTripMap | vérification POIs (GPS Wikipédia) | oui (gratuite, OPENTRIPMAP_API_KEY) |
| OSRM | matrice de trajets piéton/vélo/voiture | non (public) |

## Limites assumées

- **Extraction LLM** non-déterministe. F1 ≈ 0.95 global ; cas adversarial / tricky
  font descendre la métrique. Documenté dans le rapport.
- **Pool de POIs** généré par LLM, donc dépendant de sa culture touristique.
  Vérification OTM autosuggest pour les POIs majeurs ; extension dynamique
  quand le user nomme une activité absente du pool initial.
- **Temps de trajet** OSRM uniquement (piéton / vélo / voiture). Pas de transport
  public natif, mais le solveur choisit `transit` par segment quand la marche
  dépasse 25 min / 1.5 km.
- **Multi-tour** : on garde 1 plan précédent en mémoire par session.
  Pas de versioning au-delà (pas d'« undo » sur N tours).

## Cadre théorique

Le modèle CP-SAT utilisé est un **WCSP** (Weighted CSP, Schiex & Verfaillie 1995) :
contraintes dures inviolables (`model.add(...)`) + somme pondérée de soft
(`maximize(Σ bonus − Σ penalty)`).

Le solveur de Google OR-Tools utilise du **lazy clause generation** hérité du
solveur SAT (CDCL — Conflict-Driven Clause Learning), avec présolveur, recherche
LNS (Large Neighborhood Search), et propagateurs spécialisés pour les contraintes
globales (`no_overlap` = cumulative, `add_at_most_one` = AMO, etc.). Sur nos
problèmes (~20 activités × 5 jours), il termine en 1-5s en mode OPTIMAL.

L'extraction LLM relève du **slot-filling** en NLU (protocole d'évaluation
classique : ATIS, MultiATIS++), avec une particularité : on n'a pas fine-tuné
le modèle, on s'appuie uniquement sur des few-shots dans le prompt + une
validation typée Pydantic en post-traitement.
