# Assistant de planification de voyage — LLM + CP-SAT

Assistant conversationnel qui combine un LLM (qwen3-35b via API OpenAI-compat)
pour l'extraction de contraintes en langage naturel, et un solveur CP-SAT
(Google OR-Tools) pour la résolution de plans de voyage optimaux.

## Architecture

```
┌────────────────────┐  HTTP    ┌────────────────────────────┐
│  Frontend React    │ ───────► │  FastAPI (api_server.py)   │
│  travel_planner.   │          │                            │
│       jsx          │ ◄─────── │  Orchestrator              │
└────────────────────┘          │  (orchestrator.py)         │
                                │   1. extract (LLM)         │
                                │   2. merge contraintes     │
                                │   3. fetch ville (POIs)    │
                                │   4. solve (CP-SAT)        │
                                │   5. narrate (LLM)         │
                                └──────┬──────┬──────┬───────┘
                                       │      │      │
                       ┌───────────────▼─┐  ┌─▼─────┐  ┌▼──────────┐
                       │ llm_client.py   │  │solver │  │data_provid│
                       │ qwen3 (OpenAI-  │  │.py    │  │er.py      │
                       │  compat)        │  │CP-SAT │  │OpenTripMap│
                       │                 │  │6 types│  │+ OSRM     │
                       │                 │  │contr. │  │+ haversine│
                       └─────────────────┘  └───────┘  └───────────┘
```

## Types de contraintes CP-SAT (6, dans [solver.py](solver.py))

1. **Budget** — total et par catégorie (gastro plafonné par jour)
2. **Temporelles** — `IntervalVar` + `NoOverlap`, respect des horaires d'ouverture
3. **Logiques** — must-visit, must-avoid, paires incompatibles, prérequis (A avant B)
4. **Capacité / ressource** — temps de trajet entre activités via matrice OSRM
5. **Préférences soft** — bonus/malus catégories préférées/évitées, pénalité écart au rythme
6. **Cardinalité** — min/max activités par jour, min/max par catégorie

## Lancer le projet

### 1. Backend Python

```bash
cp .env.example .env
pip install -r requirements.txt

# (optionnel) test des modules indépendamment
python3 solver.py          # CP-SAT seul
python3 data_provider.py   # OpenTripMap + OSRM
python3 llm_client.py      # extraction LLM seule

# lancer le serveur HTTP
python3 api_server.py
# ou
uvicorn api_server:app --reload --port 8000
```

API disponible sur `http://127.0.0.1:8000` :
- `GET /health`
- `POST /chat` — `{"session_id":"...","message":"..."}` → `{reply, extracted, constraints, plan, city, errors}`
- `GET /state?session_id=...`
- `POST /reset` — `{"session_id":"..."}`

### 2. Frontend React

[travel_planner.jsx](travel_planner.jsx) est un composant React autonome.
Pour le tester, monte un projet Vite minimal :

```bash
npm create vite@latest planner-ui -- --template react
cd planner-ui
# remplace src/App.jsx par travel_planner.jsx
npm install
npm run dev
```

Le frontend lit `VITE_API_BASE` (par défaut `http://127.0.0.1:8000`).

## Ce que fait chaque fichier

| Fichier | Rôle |
|---|---|
| [llm_client.py](llm_client.py) | Client OpenAI-compat vers qwen3. Extraction JSON validée par Pydantic + narration. Désactive proprement le mode reasoning de qwen3 via `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`. |
| [data_provider.py](data_provider.py) | OpenTripMap (POIs) + OSRM (matrice de trajets) + fallback haversine + cache disque. Données pré-cachées pour Rome. |
| [solver.py](solver.py) | CP-SAT : 6 types de contraintes, scheduling avec `IntervalVar`, accepte une matrice de trajets dynamique. |
| [orchestrator.py](orchestrator.py) | Pipeline complet, état de session multi-tours, merge intelligent des contraintes. |
| [api_server.py](api_server.py) | FastAPI minimal qui expose `/chat`, `/state`, `/reset`. |
| [travel_planner.jsx](travel_planner.jsx) | UI React (chat + timeline + budget bar). Tape sur le backend, plus aucune API LLM côté navigateur. |

## APIs externes utilisées (toutes gratuites)

- **OpenTripMap** — POIs touristiques (clé optionnelle dans `.env`)
- **OSRM** — temps de trajet réel à pied/voiture/vélo (sans clé)
- **LLM OpenAI-compat** — qwen3-35b (clé fournie dans `.env`)



trouver des solutions pour les technique de  estimate_duration, de estimate_cost et de estimate_hours dans le fichier data_provider.py
elles sont pour le moment codés en dur, trouver un moyen d'avoir des vraies infos ou de creer quelque chose de plus exhaustif et réel au vu des contraintes.