"""
Solveur CP-SAT pour la planification de voyage.
Modélise un voyage multi-jours avec activités, repas, transport et hébergement.

Types de contraintes implémentés :
  1. Budget (globales et par catégorie)
  2. Temporelles / Scheduling (IntervalVar, durées, horaires d'ouverture)
  3. Incompatibilités et prérequis logiques
  4. Capacité / ressource (1 activité par créneau, temps de trajet)
  5. Préférences soft (priorités utilisateur, pondérées dans l'objectif)
  6. Cardinalité (min/max d'activités par catégorie, par jour)
"""

from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import Optional
import json


# ─────────────────────────────────────────────
# Données du domaine
# ─────────────────────────────────────────────

@dataclass
class Activity:
    id: str
    name: str
    category: str            # "culture", "gastro", "nature", "shopping", "nightlife"
    duration_hours: float     # durée en heures
    cost_euros: int           # coût par personne
    opening_hour: int         # heure d'ouverture (0-23)
    closing_hour: int         # heure de fermeture (0-23)
    zone: str                # zone géographique dans la ville
    priority_score: int       # 1-10, score d'intérêt par défaut
    available_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    latitude: float = 0.0
    longitude: float = 0.0


def dict_to_activity(d: dict) -> Activity:
    """Convertit un dict venant de data_provider.build_city_data() en Activity."""
    return Activity(
        id=d["id"],
        name=d["name"],
        category=d.get("category", "culture"),
        duration_hours=float(d.get("duration_hours", 1.5)),
        cost_euros=int(d.get("cost_euros", 0)),
        opening_hour=int(d.get("opening_hour", 9)),
        closing_hour=int(d.get("closing_hour", 18)),
        zone=d.get("zone", ""),
        priority_score=int(d.get("priority_score", 5)),
        available_days=d.get("available_days", [0, 1, 2, 3, 4, 5, 6]),
        latitude=float(d.get("latitude", 0.0)),
        longitude=float(d.get("longitude", 0.0)),
    )


@dataclass
class TravelConstraints:
    """Contraintes extraites du langage naturel par le LLM."""
    destination: str = "Rome"
    num_days: int = 5
    total_budget: int = 2000
    daily_food_budget: int = 60
    num_travelers: int = 1
    hotel_per_night: int = 100

    # Préférences soft
    preferred_categories: list[str] = field(default_factory=list)
    avoided_categories: list[str] = field(default_factory=list)
    preferred_pace: str = "moderate"  # "relaxed", "moderate", "intense"
    morning_preference: str = "culture"  # catégorie préférée le matin

    # Contraintes logiques
    must_visit: list[str] = field(default_factory=list)       # activités obligatoires
    must_avoid: list[str] = field(default_factory=list)       # activités exclues
    incompatible_pairs: list[tuple[str, str]] = field(default_factory=list)
    prerequisites: dict[str, str] = field(default_factory=dict)  # {B: A} = A avant B

    # Cardinalité
    max_activities_per_day: int = 4
    min_activities_per_day: int = 1
    max_per_category: dict[str, int] = field(default_factory=dict)
    min_per_category: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Base de données d'activités (Rome)
# ─────────────────────────────────────────────

ROME_ACTIVITIES = [
    Activity("colosseum", "Colisée", "culture", 2.5, 18, 8, 19, "centro", 10),
    Activity("vatican", "Musées du Vatican + Chapelle Sixtine", "culture", 3.5, 17, 8, 18, "vatican", 10),
    Activity("pantheon", "Panthéon", "culture", 1.0, 0, 8, 19, "centro", 8),
    Activity("forum", "Forum Romain", "culture", 2.0, 16, 8, 19, "centro", 9),
    Activity("trastevere", "Balade dans le Trastevere", "nature", 2.0, 0, 9, 23, "trastevere", 7),
    Activity("borghese", "Galerie Borghèse", "culture", 2.0, 15, 9, 19, "borghese", 9),
    Activity("trevi", "Fontaine de Trevi + Piazza Navona", "culture", 1.5, 0, 7, 23, "centro", 8),
    Activity("catacombs", "Catacombes de San Callisto", "culture", 1.5, 10, 9, 17, "appia", 6),
    Activity("cooking_class", "Cours de cuisine italienne", "gastro", 3.0, 65, 10, 14, "trastevere", 7),
    Activity("food_tour", "Food tour au Testaccio", "gastro", 3.0, 45, 11, 15, "testaccio", 8),
    Activity("wine_tasting", "Dégustation de vins", "gastro", 2.0, 35, 16, 21, "centro", 6),
    Activity("villa_borghese", "Parc Villa Borghèse", "nature", 2.0, 0, 7, 20, "borghese", 6),
    Activity("appian_way", "Via Appia Antica à vélo", "nature", 3.0, 15, 8, 18, "appia", 7),
    Activity("ostia", "Excursion Ostia Antica", "culture", 4.0, 12, 8, 17, "ostia", 6),
    Activity("shopping_condotti", "Shopping Via dei Condotti", "shopping", 2.0, 50, 10, 20, "centro", 4),
    Activity("trastevere_night", "Soirée Trastevere (bars/restos)", "nightlife", 3.0, 40, 19, 24, "trastevere", 7),
    Activity("piazza_sunset", "Coucher de soleil Pincio", "nature", 1.5, 0, 17, 21, "borghese", 8),
    Activity("jewish_quarter", "Quartier juif + Ghetto", "culture", 1.5, 0, 9, 18, "centro", 5),
    Activity("castel_angelo", "Château Saint-Ange", "culture", 2.0, 15, 9, 19, "vatican", 7),
    Activity("gelato_tour", "Tour des meilleurs gelati", "gastro", 1.5, 15, 12, 22, "centro", 7),
]

# Temps de trajet entre zones (en minutes)
TRAVEL_TIMES = {
    ("centro", "centro"): 10,
    ("centro", "vatican"): 25,
    ("centro", "trastevere"): 15,
    ("centro", "borghese"): 20,
    ("centro", "appia"): 30,
    ("centro", "testaccio"): 15,
    ("centro", "ostia"): 60,
    ("vatican", "vatican"): 10,
    ("vatican", "trastevere"): 20,
    ("vatican", "borghese"): 25,
    ("vatican", "appia"): 40,
    ("vatican", "testaccio"): 25,
    ("vatican", "ostia"): 70,
    ("trastevere", "trastevere"): 10,
    ("trastevere", "borghese"): 30,
    ("trastevere", "appia"): 25,
    ("trastevere", "testaccio"): 10,
    ("trastevere", "ostia"): 55,
    ("borghese", "borghese"): 10,
    ("borghese", "appia"): 35,
    ("borghese", "testaccio"): 25,
    ("borghese", "ostia"): 65,
    ("appia", "appia"): 10,
    ("appia", "testaccio"): 20,
    ("appia", "ostia"): 40,
    ("testaccio", "testaccio"): 10,
    ("testaccio", "ostia"): 50,
    ("ostia", "ostia"): 10,
}


def get_travel_time(zone_a: str, zone_b: str) -> int:
    """Retourne le temps de trajet en minutes entre deux zones."""
    if (zone_a, zone_b) in TRAVEL_TIMES:
        return TRAVEL_TIMES[(zone_a, zone_b)]
    if (zone_b, zone_a) in TRAVEL_TIMES:
        return TRAVEL_TIMES[(zone_b, zone_a)]
    return 30  # défaut


# ─────────────────────────────────────────────
# Solveur CP-SAT
# ─────────────────────────────────────────────

class TravelPlannerSolver:
    """
    Solveur de planification de voyage basé sur CP-SAT.
    
    Variables de décision :
      - assign[a, d] : booléen, l'activité a est assignée au jour d
      - slot[a, d]   : entier, créneau horaire de début (en demi-heures depuis 7h)
      - intervals     : IntervalVar pour le scheduling
    
    L'objectif maximise la satisfaction (score de priorité pondéré par les préférences)
    tout en respectant toutes les contraintes hard.
    """

    # Granularité : 30 minutes
    SLOT_DURATION = 30  # minutes
    DAY_START = 7       # 7h du matin
    DAY_END = 24        # minuit
    SLOTS_PER_DAY = (DAY_END - DAY_START) * 2  # 34 créneaux de 30min

    def __init__(
        self,
        activities: list[Activity],
        constraints: TravelConstraints,
        travel_matrix: Optional[list[list[int]]] = None,
    ):
        self.activities = {a.id: a for a in activities}
        self.constraints = constraints
        self.model = cp_model.CpModel()

        # Index activity_id -> position dans travel_matrix
        self._act_order = [a.id for a in activities]
        self._act_index = {a_id: i for i, a_id in enumerate(self._act_order)}
        self.travel_matrix = travel_matrix

        # Variables
        self.assign = {}    # (activity_id, day) -> BoolVar
        self.start = {}     # (activity_id, day) -> IntVar
        self.intervals = {} # (activity_id, day) -> IntervalVar
        self.selected = {}  # activity_id -> BoolVar (sélectionnée au moins un jour)

        # Soft constraint penalties
        self.soft_penalties = []
        self.soft_bonuses = []

        self._build_model()

    def _build_model(self):
        self._create_variables()
        self._add_budget_constraints()          # Type 1
        self._add_temporal_constraints()         # Type 2
        self._add_logical_constraints()          # Type 3
        self._add_capacity_constraints()         # Type 4
        self._add_soft_preferences()             # Type 5
        self._add_cardinality_constraints()      # Type 6
        self._set_objective()

    def _create_variables(self):
        C = self.constraints
        for a_id, act in self.activities.items():
            # Variable globale : cette activité est-elle sélectionnée ?
            self.selected[a_id] = self.model.new_bool_var(f"sel_{a_id}")

            for d in range(C.num_days):
                if d % 7 not in act.available_days:
                    continue

                # Assigner l'activité a au jour d
                self.assign[a_id, d] = self.model.new_bool_var(f"assign_{a_id}_d{d}")

                # Créneau de début (en slots de 30min depuis 7h)
                dur_slots = int(act.duration_hours * 2)
                max_start = self.SLOTS_PER_DAY - dur_slots

                self.start[a_id, d] = self.model.new_int_var(
                    0, max(0, max_start), f"start_{a_id}_d{d}"
                )

                # IntervalVar optionnel (actif seulement si assigné)
                self.intervals[a_id, d] = self.model.new_optional_fixed_size_interval_var(
                    self.start[a_id, d],
                    dur_slots,
                    self.assign[a_id, d],
                    f"interval_{a_id}_d{d}"
                )

            # Lien : selected <=> assignée au moins un jour
            day_vars = [
                self.assign[a_id, d]
                for d in range(C.num_days)
                if (a_id, d) in self.assign
            ]
            if day_vars:
                self.model.add_max_equality(self.selected[a_id], day_vars)
            else:
                self.model.add(self.selected[a_id] == 0)

            # Chaque activité assignée au plus 1 jour
            if day_vars:
                self.model.add(sum(day_vars) <= 1)

    # ── TYPE 1 : Contraintes de budget ──────────────

    def _add_budget_constraints(self):
        C = self.constraints

        # Budget total pour les activités
        hotel_total = C.hotel_per_night * C.num_days * C.num_travelers
        food_total = C.daily_food_budget * C.num_days * C.num_travelers
        activity_budget = C.total_budget - hotel_total - food_total

        activity_cost = sum(
            self.selected[a_id] * act.cost_euros * C.num_travelers
            for a_id, act in self.activities.items()
        )
        self.model.add(activity_cost <= max(0, activity_budget))

        # Budget quotidien food : les activités gastro comptent dans le budget food
        for d in range(C.num_days):
            daily_gastro = sum(
                self.assign[a_id, d] * act.cost_euros * C.num_travelers
                for a_id, act in self.activities.items()
                if act.category == "gastro" and (a_id, d) in self.assign
            )
            self.model.add(daily_gastro <= C.daily_food_budget * C.num_travelers)

    # ── TYPE 2 : Contraintes temporelles ────────────

    def _add_temporal_constraints(self):
        for a_id, act in self.activities.items():
            for d in range(self.constraints.num_days):
                if (a_id, d) not in self.assign:
                    continue

                # Respect des horaires d'ouverture
                open_slot = (act.opening_hour - self.DAY_START) * 2
                close_slot = (min(act.closing_hour, self.DAY_END) - self.DAY_START) * 2
                dur_slots = int(act.duration_hours * 2)

                # Si assignée, le début doit respecter les horaires
                self.model.add(
                    self.start[a_id, d] >= open_slot
                ).only_enforce_if(self.assign[a_id, d])

                self.model.add(
                    self.start[a_id, d] + dur_slots <= close_slot
                ).only_enforce_if(self.assign[a_id, d])

    # ── TYPE 3 : Contraintes logiques ───────────────

    def _add_logical_constraints(self):
        C = self.constraints

        # Must visit : forcer la sélection
        for a_id in C.must_visit:
            if a_id in self.selected:
                self.model.add(self.selected[a_id] == 1)

        # Must avoid : interdire la sélection
        for a_id in C.must_avoid:
            if a_id in self.selected:
                self.model.add(self.selected[a_id] == 0)

        # Incompatibilités : pas le même jour
        for a1, a2 in C.incompatible_pairs:
            for d in range(C.num_days):
                if (a1, d) in self.assign and (a2, d) in self.assign:
                    self.model.add(
                        self.assign[a1, d] + self.assign[a2, d] <= 1
                    )

        # Prérequis : A doit être fait avant B (jour strict)
        for b_id, a_id in C.prerequisites.items():
            if a_id not in self.selected or b_id not in self.selected:
                continue
            # Si B est sélectionné, A doit l'être aussi
            self.model.add(
                self.selected[a_id] >= self.selected[b_id]
            )
            # Et A doit être un jour avant B
            for d_b in range(C.num_days):
                if (b_id, d_b) not in self.assign:
                    continue
                for d_a in range(d_b, C.num_days):
                    if (a_id, d_a) not in self.assign:
                        continue
                    # Si B est au jour d_b et A au jour d_a >= d_b, c'est interdit
                    self.model.add(
                        self.assign[b_id, d_b] + self.assign[a_id, d_a] <= 1
                    )

    def _travel_minutes(self, a1_id: str, a2_id: str) -> int:
        """Temps de trajet entre deux activités. Utilise la matrice OSRM si dispo,
        sinon fallback sur la table de zones hardcodée."""
        if self.travel_matrix is not None:
            i = self._act_index.get(a1_id)
            j = self._act_index.get(a2_id)
            if i is not None and j is not None:
                try:
                    return int(self.travel_matrix[i][j])
                except (IndexError, TypeError):
                    pass
        return get_travel_time(self.activities[a1_id].zone, self.activities[a2_id].zone)

    # ── TYPE 4 : Capacité et ressources ─────────────

    def _add_capacity_constraints(self):
        C = self.constraints

        # Non-chevauchement des activités chaque jour (NoOverlap)
        for d in range(C.num_days):
            day_intervals = [
                self.intervals[a_id, d]
                for a_id in self.activities
                if (a_id, d) in self.intervals
            ]
            if day_intervals:
                self.model.add_no_overlap(day_intervals)

        # Temps de trajet entre activités consécutives
        # Implémenté via un espacement minimum entre les intervalles
        act_list = list(self.activities.keys())
        for d in range(C.num_days):
            for i, a1 in enumerate(act_list):
                for a2 in act_list[i + 1:]:
                    if (a1, d) not in self.assign or (a2, d) not in self.assign:
                        continue

                    travel = self._travel_minutes(a1, a2)
                    travel_slots = max(1, travel // self.SLOT_DURATION)
                    dur1 = int(self.activities[a1].duration_hours * 2)
                    dur2 = int(self.activities[a2].duration_hours * 2)

                    # Soit a1 finit + trajet avant a2, soit l'inverse
                    # (seulement si les deux sont assignées ce jour)
                    both_assigned = self.model.new_bool_var(f"both_{a1}_{a2}_d{d}")
                    self.model.add_min_equality(
                        both_assigned,
                        [self.assign[a1, d], self.assign[a2, d]]
                    )

                    # a1 avant a2
                    order_var = self.model.new_bool_var(f"order_{a1}_{a2}_d{d}")

                    self.model.add(
                        self.start[a1, d] + dur1 + travel_slots <= self.start[a2, d]
                    ).only_enforce_if([both_assigned, order_var])

                    self.model.add(
                        self.start[a2, d] + dur2 + travel_slots <= self.start[a1, d]
                    ).only_enforce_if([both_assigned, order_var.negated()])

    # ── TYPE 5 : Préférences soft ───────────────────

    def _add_soft_preferences(self):
        C = self.constraints

        for a_id, act in self.activities.items():
            score = act.priority_score

            # Bonus si catégorie préférée
            if act.category in C.preferred_categories:
                score += 3

            # Malus si catégorie évitée (mais pas interdit)
            if act.category in C.avoided_categories:
                score -= 5

            self.soft_bonuses.append(self.selected[a_id] * score)

        # Préférence de rythme
        pace_map = {"relaxed": 2, "moderate": 3, "intense": 4}
        target = pace_map.get(C.preferred_pace, 3)

        for d in range(C.num_days):
            day_count = sum(
                self.assign[a_id, d]
                for a_id in self.activities
                if (a_id, d) in self.assign
            )
            # Pénalité pour s'éloigner du rythme cible
            deviation = self.model.new_int_var(0, 10, f"pace_dev_d{d}")
            diff = self.model.new_int_var(-10, 10, f"pace_diff_d{d}")
            self.model.add(diff == day_count - target)
            self.model.add_abs_equality(deviation, diff)
            self.soft_penalties.append(deviation * 2)

        # Préférence matin : culture le matin
        if C.morning_preference:
            for a_id, act in self.activities.items():
                if act.category == C.morning_preference:
                    for d in range(C.num_days):
                        if (a_id, d) not in self.assign:
                            continue
                        # Bonus si commence avant midi (slot 10 = 12h)
                        is_morning = self.model.new_bool_var(f"morn_{a_id}_d{d}")
                        self.model.add(self.start[a_id, d] <= 10).only_enforce_if(is_morning)
                        self.model.add(self.start[a_id, d] > 10).only_enforce_if(is_morning.negated())

                        # Bonus seulement si assignée ET le matin
                        morning_bonus = self.model.new_bool_var(f"morn_bonus_{a_id}_d{d}")
                        self.model.add_min_equality(
                            morning_bonus,
                            [self.assign[a_id, d], is_morning]
                        )
                        self.soft_bonuses.append(morning_bonus * 2)

    # ── TYPE 6 : Contraintes de cardinalité ─────────

    def _add_cardinality_constraints(self):
        C = self.constraints

        for d in range(C.num_days):
            day_count = sum(
                self.assign[a_id, d]
                for a_id in self.activities
                if (a_id, d) in self.assign
            )
            self.model.add(day_count <= C.max_activities_per_day)
            self.model.add(day_count >= C.min_activities_per_day)

        # Min/max par catégorie sur tout le voyage
        categories = set(a.category for a in self.activities.values())
        for cat in categories:
            cat_count = sum(
                self.selected[a_id]
                for a_id, act in self.activities.items()
                if act.category == cat
            )
            if cat in C.max_per_category:
                self.model.add(cat_count <= C.max_per_category[cat])
            if cat in C.min_per_category:
                self.model.add(cat_count >= C.min_per_category[cat])

    # ── Objectif ────────────────────────────────────

    def _set_objective(self):
        total_bonus = sum(self.soft_bonuses) if self.soft_bonuses else 0
        total_penalty = sum(self.soft_penalties) if self.soft_penalties else 0
        self.model.maximize(total_bonus - total_penalty)

    # ── Résolution ──────────────────────────────────

    def solve(self, time_limit_seconds: int = 10) -> Optional[dict]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = False
        solver.parameters.num_workers = 4

        status = solver.solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution(solver, status)
        else:
            return {
                "status": "INFEASIBLE",
                "message": "Aucun plan ne satisfait toutes les contraintes. "
                           "Essayez d'assouplir le budget ou le nombre de jours.",
                "stats": {
                    "status_name": solver.status_name(status),
                }
            }

    def _extract_solution(self, solver: cp_model.CpSolver, status) -> dict:
        C = self.constraints
        plan = {"days": [], "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"}

        total_cost = C.hotel_per_night * C.num_days * C.num_travelers
        total_cost += C.daily_food_budget * C.num_days * C.num_travelers

        for d in range(C.num_days):
            day_activities = []
            for a_id, act in self.activities.items():
                if (a_id, d) not in self.assign:
                    continue
                if solver.value(self.assign[a_id, d]):
                    start_slot = solver.value(self.start[a_id, d])
                    start_hour = self.DAY_START + start_slot * 0.5
                    end_hour = start_hour + act.duration_hours

                    day_activities.append({
                        "id": a_id,
                        "name": act.name,
                        "category": act.category,
                        "zone": act.zone,
                        "start_time": f"{int(start_hour):02d}:{int((start_hour % 1) * 60):02d}",
                        "end_time": f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}",
                        "duration_hours": act.duration_hours,
                        "cost": act.cost_euros * C.num_travelers,
                    })
                    total_cost += act.cost_euros * C.num_travelers

            # Trier par heure de début
            day_activities.sort(key=lambda x: x["start_time"])

            plan["days"].append({
                "day": d + 1,
                "activities": day_activities,
                "activity_count": len(day_activities),
            })

        plan["summary"] = {
            "total_cost": total_cost,
            "budget": C.total_budget,
            "remaining_budget": C.total_budget - total_cost,
            "total_activities": sum(len(day["activities"]) for day in plan["days"]),
            "hotel_cost": C.hotel_per_night * C.num_days * C.num_travelers,
            "food_cost": C.daily_food_budget * C.num_days * C.num_travelers,
            "activity_cost": total_cost - (C.hotel_per_night * C.num_days * C.num_travelers)
                             - (C.daily_food_budget * C.num_days * C.num_travelers),
            "objective_value": solver.objective_value,
        }

        plan["stats"] = {
            "status_name": solver.status_name(cp_model.OPTIMAL if plan["status"] == "OPTIMAL" else cp_model.FEASIBLE),
            "solve_time_ms": round(solver.wall_time * 1000, 1),
            "branches": solver.num_branches,
            "conflicts": solver.num_conflicts,
        }

        return plan


def solve_travel_plan(constraints_dict: dict) -> dict:
    """
    Point d'entrée legacy : utilise les ROME_ACTIVITIES hardcodées.
    Conservé pour les tests, préférer solve_with_city_data() en production.
    """
    constraints = TravelConstraints(**{
        k: v for k, v in constraints_dict.items()
        if k in TravelConstraints.__dataclass_fields__
    })

    solver = TravelPlannerSolver(ROME_ACTIVITIES, constraints)
    return solver.solve()


def solve_with_city_data(
    constraints_dict: dict,
    city_data: dict,
    time_limit_seconds: int = 10,
) -> dict:
    """
    Entrypoint principal. Consomme la sortie de data_provider.build_city_data().

    Args:
        constraints_dict: contraintes extraites par le LLM (dict)
        city_data: {"city": {...}, "activities": [dict, ...], "travel_matrix": [[...]], ...}
        time_limit_seconds: timeout CP-SAT

    Returns:
        plan dict (même format que solve_travel_plan) enrichi de city_info.
    """
    constraints = TravelConstraints(**{
        k: v for k, v in constraints_dict.items()
        if k in TravelConstraints.__dataclass_fields__
    })

    activities = [dict_to_activity(a) for a in city_data.get("activities", [])]
    if not activities:
        return {
            "status": "INFEASIBLE",
            "message": f"Aucune activité disponible pour {constraints.destination}.",
            "days": [],
            "summary": {},
            "stats": {},
        }

    travel_matrix = city_data.get("travel_matrix")

    solver = TravelPlannerSolver(activities, constraints, travel_matrix=travel_matrix)
    result = solver.solve(time_limit_seconds=time_limit_seconds)

    # Enrichir avec infos ville + lat/lon dans chaque activité du plan
    result["city"] = city_data.get("city", {})
    result["data_source"] = city_data.get("data_source", "unknown")
    if "days" in result:
        act_by_id = {a.id: a for a in activities}
        for day in result["days"]:
            for act in day.get("activities", []):
                src = act_by_id.get(act["id"])
                if src:
                    act["latitude"] = src.latitude
                    act["longitude"] = src.longitude
    return result


# ─────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_constraints = {
        "destination": "Rome",
        "num_days": 5,
        "total_budget": 2000,
        "num_travelers": 2,
        "hotel_per_night": 90,
        "daily_food_budget": 50,
        "preferred_categories": ["culture", "gastro"],
        "avoided_categories": ["shopping"],
        "preferred_pace": "moderate",
        "must_visit": ["colosseum", "vatican"],
        "incompatible_pairs": [("vatican", "colosseum")],
        "prerequisites": {"forum": "colosseum"},
        "max_activities_per_day": 3,
        "min_activities_per_day": 2,
        "min_per_category": {"culture": 3},
        "max_per_category": {"nightlife": 2},
    }

    result = solve_travel_plan(test_constraints)
    print(json.dumps(result, indent=2, ensure_ascii=False))
