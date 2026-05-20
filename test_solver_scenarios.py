"""Tests offline du solveur : multi-villes, multi-budgets, multi-durées. N'utilise pas le LLM (qui est actuellement down). Charge des données"""
import json
import sys
import math

from solver import solve_with_city_data

def build_fake_city(name: str, lat: float, lon: float, country: str = "XX",
                    n_acts: int = 18) -> dict:
    """Génère un city_data plausible pour tester le solveur."""
    cats = ["culture", "gastro", "nature", "shopping", "nightlife"]
    activities = []
    import random
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    for i in range(n_acts):
        cat = cats[i % len(cats)]
        d_lat = (rng.random() - 0.5) * 0.08
        d_lon = (rng.random() - 0.5) * 0.08
        cost = rng.choice([0, 0, 5, 10, 15, 25, 40])
        dur = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
        activities.append({
            "id": f"{name.lower().replace(' ', '_')}_act_{i}",
            "name": f"Activité {i+1} ({cat}) à {name}",
            "category": cat,
            "duration_hours": dur,
            "cost_euros": cost,
            "opening_hour": rng.choice([8, 9, 10]),
            "closing_hour": rng.choice([18, 19, 20, 21, 22]),
            "latitude": lat + d_lat,
            "longitude": lon + d_lon,
            "priority_score": rng.randint(4, 10),
            "zone": "",
            "confidence": 0.85,
        })

    n = len(activities)
    matrix = [[0] * n for _ in range(n)]
    R = 6_371_000
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = activities[i]["latitude"], activities[i]["longitude"]
            lat2, lon2 = activities[j]["latitude"], activities[j]["longitude"]
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)
            a = (math.sin(dphi/2)**2
                 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2)
            dist_m = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            matrix[i][j] = max(1, round(dist_m / (4000 / 60)))

    hotels = [
        {"name": f"Hostel {name}", "address": f"1 rue principale, {name}",
         "price_per_night": 70, "stars": 2,
         "latitude": lat, "longitude": lon,
         "description": f"Auberge sympa au centre de {name}."},
        {"name": f"Hotel Central {name}", "address": f"10 avenue centrale, {name}",
         "price_per_night": 130, "stars": 3,
         "latitude": lat + 0.005, "longitude": lon - 0.005,
         "description": f"Hôtel confortable au cœur de {name}."},
        {"name": f"Grand Hotel {name}", "address": f"50 boulevard premium, {name}",
         "price_per_night": 280, "stars": 5,
         "latitude": lat - 0.003, "longitude": lon + 0.003,
         "description": f"Établissement haut de gamme à {name}."},
    ]

    return {
        "city": {"name": name, "country": country, "latitude": lat,
                 "longitude": lon, "population": 1_000_000},
        "activities": activities,
        "travel_matrix": matrix,
        "transport_mode": "foot",
        "data_source": "fake-offline",
        "hotels": hotels,
    }

def make_constraints(city, days, budget, start=9, end=19, pace="moderate",
                     prefs=None, travelers=1):
    return {
        "destination": city,
        "num_days": days,
        "total_budget": budget,
        "num_travelers": travelers,
        "hotel_per_night": 100,
        "daily_food_budget": 50,
        "day_start_hour": start,
        "day_end_hour": end,
        "preferred_categories": prefs or [],
        "avoided_categories": [],
        "preferred_pace": pace,
        "must_visit": [],
        "must_avoid": [],
        "must_visit_on_day": {},
        "max_activities_per_day": 5,
        "min_activities_per_day": 1,
    }

SCENARIOS = [
    ("Paris", 48.8566, 2.3522, 4, 2500, "moderate", ["culture"]),
    ("Lisbonne", 38.7223, -9.1393, 3, 1200, "relaxed", ["gastro", "culture"]),
    ("Marrakech", 31.6295, -7.9811, 5, 1800, "intense", ["culture", "shopping"]),
    ("Berlin", 52.5200, 13.4050, 2, 800, "moderate", ["nightlife"]),
    ("Kyoto", 35.0116, 135.7681, 7, 3500, "moderate", ["culture", "nature"]),
    ("Reykjavik", 64.1466, -21.9426, 3, 1500, "moderate", ["nature"]),
    ("Petite ville test", 45.0, 5.0, 2, 600, "relaxed", []),
    ("Mégalopole", 40.0, -74.0, 6, 4000, "intense", []),
]

def run_one(case):
    city, lat, lon, days, budget, pace, prefs = case
    print("=" * 78)
    print(f"Scénario : {city}  | {days} jours  | budget {budget}€  | pace={pace}  | prefs={prefs}")
    city_data = build_fake_city(city, lat, lon)
    constraints = make_constraints(city, days, budget, pace=pace, prefs=prefs)

    try:
        plan = solve_with_city_data(constraints, city_data, time_limit_seconds=8)
    except Exception as e:
        print(f"  ❌ EXCEPTION: {type(e).__name__}: {e}")
        return False

    status = plan.get("status")
    print(f"  status: {status}")
    if status not in ("OPTIMAL", "FEASIBLE"):
        print(f"  message: {plan.get('message')}")
        return False

    summary = plan.get("summary", {})
    total = summary.get("total_activities", 0)
    cost = summary.get("total_cost", 0)
    print(f"  → {total} activités, coût total {cost}€/{budget}€  "
          f"(reste {summary.get('remaining_budget')}€)")

    hotel = plan.get("hotel")
    if hotel:
        print(f"  hôtel: {hotel['name']} ({hotel['price_per_night']}€/nuit, "
              f"{hotel['stars']}★)")
    else:
        print("  ⚠ aucun hôtel sélectionné")

    transport = plan.get("transport_mode")
    print(f"  transport: {transport}")

    issues = []
    for d in plan.get("days", []):
        n = d.get("activity_count", 0)
        tt = d.get("total_travel_minutes")
        transitions = d.get("transitions", [])
        print(f"  Jour {d['day']}: {n} acts, trajet total {tt} min, "
              f"{len(transitions)} transitions")
        for act in d["activities"]:
            print(f"    {act['start_time']}–{act['end_time']}  {act['name']}")
        for t in transitions:
            mode = t.get("mode")
            dist = t.get("distance_m")
            print(f"      → {t['minutes']} min · {dist}m · {mode}")

        if len(transitions) != max(0, n - 1):
            issues.append(f"Jour {d['day']} : {len(transitions)} transitions pour {n} activités")
        if n >= 2 and (tt is None or tt <= 0):
            issues.append(f"Jour {d['day']} : {n} activités mais travel total = {tt}")

    if issues:
        print(f"  ⚠ Anomalies : {issues}")
        return False
    return True

def main():
    print("\n=== Tests du solveur sur 8 scénarios ===\n")
    results = []
    for case in SCENARIOS:
        ok = run_one(case)
        results.append((case[0], ok))
        print()

    print("\n=== Résumé ===")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    failed = [n for n, ok in results if not ok]
    sys.exit(0 if not failed else 1)

if __name__ == "__main__":
    main()
