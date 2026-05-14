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
import math


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
    must_visit: list[str] = field(default_factory=list)       # activités obligatoires (nom ou ID)
    must_avoid: list[str] = field(default_factory=list)       # activités exclues (nom ou ID)
    must_visit_on_day: dict[str, int] = field(default_factory=dict)  # {nom_ou_id: jour_1indexed}
    incompatible_pairs: list[tuple[str, str]] = field(default_factory=list)
    prerequisites: dict[str, str] = field(default_factory=dict)  # {B: A} = A avant B

    # Cardinalité
    max_activities_per_day: int = 6
    min_activities_per_day: int = 1
    max_per_category: dict[str, int] = field(default_factory=dict)
    min_per_category: dict[str, int] = field(default_factory=dict)

    # Fenêtre horaire journalière (None = pas de contrainte)
    day_start_hour: Optional[int] = None   # heure souhaitée de début (ex: 9)
    day_end_hour: Optional[int] = None     # heure souhaitée de fin   (ex: 18)


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


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS (formule haversine)."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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

    Modes :
      - "flexible" (défaut) : préférences = soft constraints (bonus/pénalités)
      - "strict"            : préférences catégories = hard constraints (filtre dur)

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
        mode: str = "flexible",
        transport_mode: str = "foot",
        previous_plan: Optional[dict[int, list]] = None,
        touched_days: Optional[set[int]] = None,
    ):
        self.activities = {a.id: a for a in activities}
        self.constraints = constraints
        self.mode = mode  # "flexible" | "strict"
        self._transport_mode = transport_mode  # "foot", "bike", "car"
        # Plan précédent. Format accepté :
        #   {day_idx_0based: [activity_id, ...]}              (legacy, bonus seul)
        #   {day_idx_0based: [(activity_id, start_slot), ...]} (pin strict + bonus)
        self._previous_plan = previous_plan or {}
        # Jours explicitement modifiés ce tour-ci. None = pas de pinning
        # (premier tour, ou changement structurel). set() vide = tous les jours pinned.
        self._touched_days = touched_days
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
        # Variables temporaires pour la pénalité de trajet (remplies par _add_capacity_constraints)
        self._pair_both_assigned: dict = {}
        self._create_variables()
        self._add_budget_constraints()          # Type 1
        self._add_temporal_constraints()         # Type 2
        self._add_logical_constraints()          # Type 3
        self._add_capacity_constraints()         # Type 4
        self._add_soft_preferences()             # Type 5
        self._add_cardinality_constraints()      # Type 6
        self._add_travel_penalty()               # Optimisation : minimiser temps de trajet
        self._add_pin_constraints()              # Multi-tours : pin hard des jours non touchés
        self._add_stability_bonus()              # Multi-tours : bonus soft pour le reste
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

            # Lien : selected <=> assignée exactement 1 jour (jamais plus)
            day_vars = [
                self.assign[a_id, d]
                for d in range(C.num_days)
                if (a_id, d) in self.assign
            ]
            if day_vars:
                # Contrainte HARD : chaque activité sur au plus 1 jour
                # add_at_most_one est le propagateur dédié CP-SAT (plus robuste que sum <= 1)
                self.model.add_at_most_one(day_vars)

                # selected = 1 ssi au moins un jour assigné
                # → si selected=1 : au moins un day_var=1
                self.model.add_bool_or(day_vars + [self.selected[a_id].negated()])
                # → si un jour assigné : selected=1
                for dv in day_vars:
                    self.model.add_implication(dv, self.selected[a_id])
            else:
                self.model.add(self.selected[a_id] == 0)

    # ── TYPE 1 : Contraintes de budget ──────────────

    def _add_budget_constraints(self):
        C = self.constraints

        # Budget total pour les activités
        hotel_total = C.hotel_per_night * C.num_days * C.num_travelers
        food_total = C.daily_food_budget * C.num_days * C.num_travelers
        activity_budget = max(0, C.total_budget - hotel_total - food_total)
        self._activity_budget = activity_budget

        activity_cost = sum(
            self.selected[a_id] * act.cost_euros * C.num_travelers
            for a_id, act in self.activities.items()
        )
        self.model.add(activity_cost <= activity_budget)

        # Soft : pénaliser une trop grosse sous-utilisation du budget activités.
        # Cible : dépenser au moins 70 % du budget activités s'il dépasse 100 €.
        # On évite de pénaliser quand le budget est très serré (≤ 100 €).
        # On n'applique PAS cette pénalité en pace=relaxed : l'utilisateur veut
        # peu d'activités, le forcer à utiliser le budget contredirait le pace.
        if activity_budget > 100 and C.preferred_pace != "relaxed":
            target_spend = (activity_budget * 7) // 10
            underspend = self.model.new_int_var(
                0, activity_budget, "activity_underspend"
            )
            self.model.add(underspend >= target_spend - activity_cost)
            self.model.add(underspend >= 0)
            # Pénalité = underspend // 20 (1 point par 20€ de sous-dépense).
            # Doit rester ~comparable aux bonus d'activité (~5-13 par act).
            penalty_var = self.model.new_int_var(
                0, activity_budget // 20 + 1, "activity_underspend_pen"
            )
            self.model.add_division_equality(penalty_var, underspend, 20)
            self.soft_penalties.append(penalty_var)

        # Budget quotidien food : les activités gastro comptent dans le budget food
        for d in range(C.num_days):
            daily_gastro = sum(
                self.assign[a_id, d] * act.cost_euros * C.num_travelers
                for a_id, act in self.activities.items()
                if act.category == "gastro" and (a_id, d) in self.assign
            )
            self.model.add(daily_gastro <= C.daily_food_budget * C.num_travelers)

    # ── TYPE 2 : Contraintes temporelles ────────────

    def _day_window_slots(self) -> tuple[int, int]:
        """
        Calcule la fenêtre horaire journalière en slots (1 slot = 30 min depuis DAY_START).
        La fenêtre fin est stricte (pas de marge) pour respecter l'heure de fin demandée.
        La fenêtre début admet 30 min de tolérance avant l'heure souhaitée.

        Si aucune préférence : fenêtre = toute la journée [DAY_START, DAY_END].
        """
        C = self.constraints
        START_MARGIN = 1  # tolérance de 30 min avant l'heure de début demandée

        if C.day_start_hour is not None:
            raw_start = max(self.DAY_START, C.day_start_hour)
            win_start = max(0, (raw_start - self.DAY_START) * 2 - START_MARGIN)
        else:
            win_start = 0

        if C.day_end_hour is not None:
            raw_end = min(self.DAY_END, C.day_end_hour)
            # Fin stricte : l'activité doit être terminée à l'heure indiquée
            win_end = min(self.SLOTS_PER_DAY, (raw_end - self.DAY_START) * 2)
        else:
            win_end = self.SLOTS_PER_DAY

        return win_start, win_end

    def _add_temporal_constraints(self):
        win_start, win_end = self._day_window_slots()

        for a_id, act in self.activities.items():
            for d in range(self.constraints.num_days):
                if (a_id, d) not in self.assign:
                    continue

                dur_slots = int(act.duration_hours * 2)

                # Respect des horaires d'ouverture de l'activité
                open_slot = max(0, (act.opening_hour - self.DAY_START) * 2)
                close_slot = min(self.SLOTS_PER_DAY,
                                 (min(act.closing_hour, self.DAY_END) - self.DAY_START) * 2)

                # Fenêtre effective = intersection(horaires activité, fenêtre utilisateur)
                eff_start = max(open_slot, win_start)
                eff_end = min(close_slot, win_end)

                if eff_end - eff_start < dur_slots:
                    # L'activité ne peut pas tenir dans la fenêtre → l'exclure
                    self.model.add(self.assign[a_id, d] == 0)
                    continue

                self.model.add(
                    self.start[a_id, d] >= eff_start
                ).only_enforce_if(self.assign[a_id, d])

                self.model.add(
                    self.start[a_id, d] + dur_slots <= eff_end
                ).only_enforce_if(self.assign[a_id, d])

    # ── Résolution fuzzy d'un nom/ID vers un ID interne ─────────────────

    def _resolve_activity(self, name_or_id: str) -> Optional[str]:
        """
        Résout un nom ou ID d'activité vers l'ID interne du solveur.

        Ordre de priorité :
          1. Correspondance exacte sur l'ID
          2. Correspondance exacte sur le nom (insensible à la casse)
          3. Sous-chaîne : le terme est contenu dans le nom ou vice-versa
             (min 4 caractères pour éviter les faux positifs)
        """
        # 1. Exact ID match
        if name_or_id in self.activities:
            return name_or_id

        term = name_or_id.lower().strip()

        # 2. Exact name match (case-insensitive)
        for a_id, act in self.activities.items():
            if act.name.lower() == term:
                return a_id

        # 3. Substring match (term in name, or name in term)
        if len(term) >= 4:
            for a_id, act in self.activities.items():
                name_l = act.name.lower()
                if term in name_l or name_l in term:
                    return a_id

        return None

    # ── TYPE 3 : Contraintes logiques ───────────────

    def _add_logical_constraints(self):
        C = self.constraints

        # Must visit : forcer la sélection (résolution fuzzy nom→ID)
        for name_or_id in C.must_visit:
            a_id = self._resolve_activity(name_or_id)
            if a_id:
                self.model.add(self.selected[a_id] == 1)

        # Must visit on day : forcer sur un jour précis (jours 1-indexés → 0-indexés)
        for name_or_id, day_1 in C.must_visit_on_day.items():
            a_id = self._resolve_activity(name_or_id)
            if not a_id:
                continue
            day = day_1 - 1  # conversion en 0-indexé
            if 0 <= day < C.num_days and (a_id, day) in self.assign:
                # Contrainte hard : cette activité DOIT être placée ce jour-là
                self.model.add(self.assign[a_id, day] == 1)
                self.model.add(self.selected[a_id] == 1)
                # Interdire les autres jours pour cette activité
                for d in range(C.num_days):
                    if d != day and (a_id, d) in self.assign:
                        self.model.add(self.assign[a_id, d] == 0)
            elif a_id in self.selected:
                # Jour hors plage : au moins forcer la sélection globale
                self.model.add(self.selected[a_id] == 1)

        # Must avoid : interdire la sélection (résolution fuzzy)
        for name_or_id in C.must_avoid:
            a_id = self._resolve_activity(name_or_id)
            if a_id:
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

    def _choose_segment_mode(self, foot_minutes: int, dist_m):
        """
        Choisit le mode de transport pour un segment donné.

        Règle :
          - Si le mode global est "car" ou "bike", on garde ce mode (la matrice
            travel_time est déjà calculée pour ce mode).
          - Si le mode global est "foot" (défaut) :
              * marche ≤ 25 min → on garde "foot"
              * sinon → on bascule sur transports en commun (métro/bus), estimé
                à ~3× plus rapide que la marche (vitesse moyenne urbaine
                ~12-15 km/h vs 4 km/h pour la marche), avec un minimum de 8 min
                (temps d'attente / accès).
        """
        base = getattr(self, "_transport_mode", "foot")
        if base in ("car", "bike"):
            return (base, foot_minutes)
        # Mode "foot" : trop long à pied ?
        long_walk = foot_minutes > 25 and (dist_m is None or dist_m > 1500)
        if long_walk:
            transit_min = max(8, foot_minutes // 3 + 5)
            return ("transit", transit_min)
        return ("foot", foot_minutes)

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
                    # Exposé pour _add_travel_penalty
                    self._pair_both_assigned[(a1, a2, d)] = (both_assigned, travel)

                    # a1 avant a2
                    order_var = self.model.new_bool_var(f"order_{a1}_{a2}_d{d}")

                    self.model.add(
                        self.start[a1, d] + dur1 + travel_slots <= self.start[a2, d]
                    ).only_enforce_if([both_assigned, order_var])

                    self.model.add(
                        self.start[a2, d] + dur2 + travel_slots <= self.start[a1, d]
                    ).only_enforce_if([both_assigned, order_var.negated()])

    # ── TYPE 5 : Préférences soft (ou hard en mode strict) ─────────────────

    def _add_soft_preferences(self):
        C = self.constraints

        # Mode strict : les catégories préférées deviennent des contraintes hard.
        # Seules les activités appartenant aux catégories préférées (ou must_visit)
        # peuvent être sélectionnées.
        if self.mode == "strict" and C.preferred_categories:
            for a_id, act in self.activities.items():
                if (
                    act.category not in C.preferred_categories
                    and a_id not in C.must_visit
                ):
                    self.model.add(self.selected[a_id] == 0)

        for a_id, act in self.activities.items():
            score = act.priority_score

            # Bonus si catégorie préférée
            if act.category in C.preferred_categories:
                # En mode strict le filtre hard suffit ; en flexible on booste davantage
                score += 5 if self.mode == "strict" else 3

            # Malus si catégorie évitée (soft même en strict, car must_visit peut primer)
            if act.category in C.avoided_categories:
                score -= 5

            self.soft_bonuses.append(self.selected[a_id] * score)

        # Préférence de rythme : pénalités asymétriques distinctes par rythme.
        # C'est ce qui différencie réellement relaxed / moderate / intense :
        #   - relaxed : pénalité forte sur l'overflow (n'en mets pas trop)
        #   - intense : pénalité forte sur le shortfall (remplis bien la journée)
        # Sans cette asymétrie les trois rythmes convergent vers le maximum
        # naturel autorisé par les durées + le budget.
        PACE_CONFIG = {
            "relaxed":  {"target": 3, "short_w": 4,  "over_w": 10},
            "moderate": {"target": 4, "short_w": 8,  "over_w": 4},
            "intense":  {"target": 5, "short_w": 12, "over_w": 1},
        }
        pace = C.preferred_pace if C.preferred_pace in PACE_CONFIG else "moderate"
        cfg = PACE_CONFIG[pace]
        target = cfg["target"]

        for d in range(C.num_days):
            day_count = sum(
                self.assign[a_id, d]
                for a_id in self.activities
                if (a_id, d) in self.assign
            )
            # shortfall = max(0, target - day_count)
            shortfall = self.model.new_int_var(0, target, f"shortfall_d{d}")
            self.model.add(shortfall >= target - day_count)
            self.model.add(shortfall >= 0)
            self.soft_penalties.append(shortfall * cfg["short_w"])

            # overflow = max(0, day_count - target)
            max_over = max(0, C.max_activities_per_day - target)
            if cfg["over_w"] > 0 and max_over > 0:
                overflow = self.model.new_int_var(0, max_over, f"overflow_d{d}")
                self.model.add(overflow >= day_count - target)
                self.model.add(overflow >= 0)
                self.soft_penalties.append(overflow * cfg["over_w"])

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

        # Calcul du max dynamique basé sur la durée disponible
        win_start, win_end = self._day_window_slots()
        available_hours = (win_end - win_start) / 2  # en heures

        # Durée minimale d'une activité parmi les candidats (plancher à 0.5h)
        min_act_duration = min(
            (act.duration_hours for act in self.activities.values()),
            default=1.0
        )
        min_act_duration = max(0.5, min_act_duration)

        # Max théorique = heures dispo / durée min, plafonné à C.max_activities_per_day
        dynamic_max = min(C.max_activities_per_day, int(available_hours / min_act_duration))
        dynamic_max = max(1, dynamic_max)  # au moins 1

        for d in range(C.num_days):
            day_count = sum(
                self.assign[a_id, d]
                for a_id in self.activities
                if (a_id, d) in self.assign
            )
            self.model.add(day_count <= dynamic_max)
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

    # ── Pénalité de trajet (optimisation des distances) ────────────────

    def _add_travel_penalty(self):
        """
        Nudge soft : encourage le regroupement géographique sans bloquer
        la sélection de plusieurs activités par jour.

        Échelle : un trajet ≤ 15 min ne coûte rien (intra-quartier),
        un trajet de 60 min vaut ~4 points (vs ~10 points de bonus par
        activité), un trajet de 100 min en vaut ~8.
        """
        for (a1, a2, d), (both_assigned, travel_min) in self._pair_both_assigned.items():
            weight = max(0, (int(travel_min) - 15) // 10)
            if weight > 0:
                self.soft_penalties.append(both_assigned * weight)

    # ── Bonus de stabilité + pinning multi-tours ──────────────

    @staticmethod
    def _iter_prev_plan(entries):
        """Itère sur le plan précédent (format legacy ou tuple)."""
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                yield entry[0], entry[1]   # (act_id, start_slot)
            else:
                yield entry, None          # legacy : act_id seul

    def _add_stability_bonus(self):
        """
        Bonus SOFT pour conservation du plan précédent. Utile pour les jours
        explicitement touchés par l'utilisateur : on préfère bouger le moins
        possible parmi les options compatibles avec sa demande.
        """
        if not self._previous_plan:
            return
        STABILITY_WEIGHT = 4
        for day_idx, entries in self._previous_plan.items():
            for act_id, _slot in self._iter_prev_plan(entries):
                if (act_id, day_idx) in self.assign:
                    self.soft_bonuses.append(
                        self.assign[act_id, day_idx] * STABILITY_WEIGHT
                    )

    def _add_pin_constraints(self):
        """
        Pinning HARD des jours non touchés par l'utilisateur ce tour-ci.
        Garantit que les jours non mentionnés sont identiques au tour précédent
        (mêmes activités sur les mêmes créneaux). Évite le reshuffle pour des
        gains marginaux d'objectif.

        - self._touched_days = None  → pas de pinning (premier tour ou
          changement structurel comme budget/durée/rythme/catégories).
        - self._touched_days = set() → tous les jours pinned (aucune modif
          mentionnée, utilisateur a juste discuté).
        - self._touched_days = {2, 4} → jours 2 et 4 libres, les autres pinned.
        """
        if self._touched_days is None or not self._previous_plan:
            return

        for day_idx, entries in self._previous_plan.items():
            if day_idx in self._touched_days:
                continue  # jour explicitement modifié → laisser libre
            pinned_ids: set[str] = set()
            for act_id, slot in self._iter_prev_plan(entries):
                if (act_id, day_idx) in self.assign:
                    # Activité forcée sur ce jour
                    self.model.add(self.assign[act_id, day_idx] == 1)
                    # Créneau de début forcé si on l'a
                    if slot is not None and (act_id, day_idx) in self.start:
                        self.model.add(self.start[act_id, day_idx] == int(slot))
                    pinned_ids.add(act_id)
            # Interdire toute autre activité sur ce jour pinned
            for a_id in self.activities:
                if a_id in pinned_ids:
                    continue
                if (a_id, day_idx) in self.assign:
                    self.model.add(self.assign[a_id, day_idx] == 0)

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
                        "start_slot": int(start_slot),  # pour pinning multi-tours
                        "duration_hours": act.duration_hours,
                        "cost": act.cost_euros * C.num_travelers,
                    })
                    total_cost += act.cost_euros * C.num_travelers

            # Trier par heure de début
            day_activities.sort(key=lambda x: x["start_time"])

            # Calculer les transitions (trajets) entre activités consécutives.
            # Choix du mode par segment : on n'oblige pas l'utilisateur à marcher 1h.
            transitions = []
            for i in range(len(day_activities) - 1):
                a1 = day_activities[i]
                a2 = day_activities[i + 1]
                travel_min = self._travel_minutes(a1["id"], a2["id"])
                src = self.activities.get(a1["id"])
                dst = self.activities.get(a2["id"])
                dist_m = None
                if src and dst and src.latitude and dst.latitude:
                    dist_m = _haversine_meters(
                        src.latitude, src.longitude,
                        dst.latitude, dst.longitude,
                    )
                mode, minutes = self._choose_segment_mode(int(travel_min), dist_m)
                transitions.append({
                    "from_id": a1["id"],
                    "to_id": a2["id"],
                    "from_name": a1["name"],
                    "to_name": a2["name"],
                    "minutes": minutes,
                    "distance_m": int(round(dist_m)) if dist_m is not None else None,
                    "mode": mode,
                })

            plan["days"].append({
                "day": d + 1,
                "activities": day_activities,
                "activity_count": len(day_activities),
                "transitions": transitions,
                "total_travel_minutes": sum(t["minutes"] for t in transitions),
            })

        hotel_cost = C.hotel_per_night * C.num_days * C.num_travelers
        food_cost = C.daily_food_budget * C.num_days * C.num_travelers
        activity_cost = total_cost - hotel_cost - food_cost
        remaining = C.total_budget - total_cost

        plan["summary"] = {
            "total_cost": total_cost,
            "budget": C.total_budget,
            "remaining_budget": remaining,
            "total_activities": sum(len(day["activities"]) for day in plan["days"]),
            "hotel_cost": hotel_cost,
            "food_cost": food_cost,
            "activity_cost": activity_cost,
            "objective_value": solver.objective_value,
        }

        plan["stats"] = {
            "status_name": solver.status_name(
                cp_model.OPTIMAL if plan["status"] == "OPTIMAL" else cp_model.FEASIBLE
            ),
            "solve_time_ms": round(solver.wall_time * 1000, 1),
            "branches": solver.num_branches,
            "conflicts": solver.num_conflicts,
        }

        plan["mode"] = self.mode

        # ── Contraintes respectées et violées ──────────────────────────────
        respected, violated = self._audit_constraints(plan)
        plan["respected_constraints"] = respected
        plan["violated_soft_constraints"] = violated

        return plan

    def _audit_constraints(self, plan: dict) -> tuple[list[str], list[str]]:
        """
        Analyse la solution et retourne les listes de contraintes respectées/violées.
        """
        C = self.constraints
        respected: list[str] = []
        violated: list[str] = []

        summary = plan.get("summary", {})
        days = plan.get("days", [])

        selected_ids = {
            act["id"]
            for day in days
            for act in day.get("activities", [])
        }
        selected_categories = [
            act["category"]
            for day in days
            for act in day.get("activities", [])
        ]

        # Budget
        if summary.get("remaining_budget", 0) >= 0:
            respected.append(f"Budget respecté (reste {summary['remaining_budget']}€)")
        else:
            violated.append(f"Budget dépassé de {abs(summary['remaining_budget'])}€")

        # Durée
        respected.append(f"Durée exacte : {C.num_days} jour(s)")

        # Activités par jour
        for day in days:
            count = day["activity_count"]
            if count > C.max_activities_per_day:
                violated.append(
                    f"Jour {day['day']} : {count} activités > max ({C.max_activities_per_day})"
                )
            elif count < C.min_activities_per_day:
                violated.append(
                    f"Jour {day['day']} : {count} activités < min ({C.min_activities_per_day})"
                )
            else:
                respected.append(
                    f"Jour {day['day']} : {count} activité(s) dans les limites"
                )

        # Must-visit (résolution fuzzy)
        for name_or_id in C.must_visit:
            a_id = self._resolve_activity(name_or_id)
            act_name = (self.activities[a_id].name if a_id and a_id in self.activities
                        else name_or_id)
            if a_id and a_id in selected_ids:
                respected.append(f"Activité obligatoire présente : {act_name}")
            else:
                violated.append(f"Activité obligatoire absente : {act_name}")

        # Must-visit-on-day (résolution fuzzy)
        for name_or_id, day_1 in C.must_visit_on_day.items():
            a_id = self._resolve_activity(name_or_id)
            act_name = (self.activities[a_id].name if a_id and a_id in self.activities
                        else name_or_id)
            if a_id:
                day_acts = days[day_1 - 1].get("activities", []) if 0 < day_1 <= len(days) else []
                on_day = any(act["id"] == a_id for act in day_acts)
                if on_day:
                    respected.append(f"{act_name} planifiée au jour {day_1} ✓")
                else:
                    violated.append(f"{act_name} demandée au jour {day_1} mais absente")

        # Must-avoid (résolution fuzzy)
        for name_or_id in C.must_avoid:
            a_id = self._resolve_activity(name_or_id)
            act_name = (self.activities[a_id].name if a_id and a_id in self.activities
                        else name_or_id)
            if a_id and a_id not in selected_ids:
                respected.append(f"Activité exclue bien absente : {act_name}")
            elif a_id:
                violated.append(f"Activité exclue présente : {act_name}")

        # Préférences catégories (soft)
        for cat in C.preferred_categories:
            count = selected_categories.count(cat)
            if count > 0:
                respected.append(f"Catégorie préférée '{cat}' : {count} activité(s)")
            else:
                violated.append(f"Catégorie préférée '{cat}' : aucune activité planifiée")

        # Catégories évitées (soft)
        for cat in C.avoided_categories:
            count = selected_categories.count(cat)
            if count == 0:
                respected.append(f"Catégorie évitée '{cat}' : bien absente")
            else:
                violated.append(f"Catégorie évitée '{cat}' : {count} activité(s) présente(s)")

        return respected, violated


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
    mode: str = "flexible",
    previous_plan: Optional[dict[int, list]] = None,
    touched_days: Optional[set[int]] = None,
) -> dict:
    """
    Entrypoint principal. Consomme la sortie de data_provider.build_city_data().

    Args:
        constraints_dict: contraintes extraites par le LLM (dict)
        city_data: {"city": {...}, "activities": [dict, ...], "travel_matrix": [[...]], ...}
        time_limit_seconds: timeout CP-SAT
        mode: "flexible" (défaut) ou "strict" (préférences = hard constraints)

    Returns:
        plan dict enrichi de city_info, respected_constraints, violated_soft_constraints.
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
            "respected_constraints": [],
            "violated_soft_constraints": [],
        }

    # Vérif amont : le budget couvre-t-il au moins hôtel + repas ?
    fixed_cost = (constraints.hotel_per_night + constraints.daily_food_budget) \
        * constraints.num_days * constraints.num_travelers
    if fixed_cost > constraints.total_budget:
        deficit = fixed_cost - constraints.total_budget
        return {
            "status": "INFEASIBLE",
            "message": (
                f"Budget {constraints.total_budget}€ trop faible : "
                f"l'hébergement et les repas coûtent déjà {fixed_cost}€ "
                f"(manque {deficit}€). Augmente le budget, réduis hotel_per_night, "
                f"ou diminue daily_food_budget."
            ),
            "days": [],
            "summary": {
                "total_cost": fixed_cost,
                "budget": constraints.total_budget,
                "remaining_budget": -deficit,
                "total_activities": 0,
                "hotel_cost": constraints.hotel_per_night * constraints.num_days * constraints.num_travelers,
                "food_cost": constraints.daily_food_budget * constraints.num_days * constraints.num_travelers,
                "activity_cost": 0,
            },
            "stats": {"status_name": "INFEASIBLE_BUDGET"},
            "respected_constraints": [],
            "violated_soft_constraints": [
                f"Budget dépassé de {deficit}€ rien que pour hôtel + repas"
            ],
        }

    travel_matrix = city_data.get("travel_matrix")
    transport_mode = city_data.get("transport_mode", "foot")

    planner = TravelPlannerSolver(
        activities, constraints,
        travel_matrix=travel_matrix, mode=mode, transport_mode=transport_mode,
        previous_plan=previous_plan, touched_days=touched_days,
    )
    result = planner.solve(time_limit_seconds=time_limit_seconds)

    # Si le pinning rend le problème infaisable, on retente sans pin
    # (fallback : on relâche le pinning pour permettre la modif demandée).
    if result.get("status") == "INFEASIBLE" and touched_days is not None:
        planner_unpinned = TravelPlannerSolver(
            activities, constraints,
            travel_matrix=travel_matrix, mode=mode, transport_mode=transport_mode,
            previous_plan=previous_plan, touched_days=None,
        )
        result = planner_unpinned.solve(time_limit_seconds=time_limit_seconds)
        result["pin_fallback"] = True

    # En mode strict, si INFEASIBLE, retenter en mode flexible
    if result.get("status") == "INFEASIBLE" and mode == "strict":
        result["message"] = (
            "Mode strict infaisable avec les préférences données. "
            "Passage en mode flexible (compromis acceptés)."
        )
        planner_flex = TravelPlannerSolver(
            activities, constraints,
            travel_matrix=travel_matrix, mode="flexible", transport_mode=transport_mode,
            previous_plan=previous_plan,
        )
        result = planner_flex.solve(time_limit_seconds=time_limit_seconds)
        result["mode_fallback"] = "strict→flexible"

    # Enrichir avec infos ville + lat/lon dans chaque activité du plan
    result["city"] = city_data.get("city", {})
    result["data_source"] = city_data.get("data_source", "unknown")
    result["transport_mode"] = transport_mode
    # Sélectionner un hôtel (si disponible dans city_data) dans le budget de l'utilisateur
    hotel_options = city_data.get("hotels") or []
    if hotel_options:
        budget_per_night = constraints.hotel_per_night
        # Préférer l'hôtel le plus cher ≤ budget, sinon le moins cher
        below = [h for h in hotel_options if (h.get("price_per_night") or 0) <= budget_per_night]
        if below:
            chosen = max(below, key=lambda h: h.get("price_per_night") or 0)
        else:
            chosen = min(hotel_options, key=lambda h: h.get("price_per_night") or 0)
        result["hotel"] = chosen
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
# explain_solution : explication des compromis
# ─────────────────────────────────────────────

def explain_solution(solution: dict, constraints: dict) -> str:
    """
    Génère une explication naturelle des compromis de la solution CP-SAT.

    Explique :
    - Pourquoi certaines préférences ne sont pas respectées
    - L'utilisation du budget
    - Le rythme moyen vs le rythme souhaité
    - Les alternatives possibles pour les préférences non satisfaites

    Args:
        solution: plan retourné par solve_with_city_data()
        constraints: dict plat des contraintes (état de session)

    Returns:
        Texte explicatif en français.
    """
    if not solution:
        return "Aucune solution disponible."

    if solution.get("status") == "INFEASIBLE":
        msg = solution.get("message", "")
        lines = ["Aucun plan ne satisfait toutes les contraintes."]
        if msg:
            lines.append(msg)
        budget = constraints.get("total_budget", 0)
        num_days = constraints.get("num_days", 0)
        lines.append(
            f"Suggestions : augmentez le budget (actuel : {budget}€), "
            f"allongez le séjour (actuel : {num_days} jour(s)), "
            "ou assouplissez les préférences de catégories."
        )
        return "\n".join(lines)

    lines: list[str] = []
    summary = solution.get("summary", {})
    days = solution.get("days", [])
    mode = solution.get("mode", "flexible")

    if mode == "strict":
        lines.append("Mode strict : seules les activités des catégories préférées ont été sélectionnées.")
    if solution.get("mode_fallback"):
        lines.append(
            "Note : le mode strict était infaisable, le plan a été généré en mode flexible."
        )

    # Budget
    remaining = summary.get("remaining_budget", 0)
    total_cost = summary.get("total_cost", 0)
    budget = summary.get("budget", 0)
    if remaining < 0:
        lines.append(f"⚠ Budget dépassé de {abs(remaining)}€ (coût total : {total_cost}€).")
    elif remaining < 50:
        lines.append(
            f"Budget quasi épuisé : {total_cost}€ dépensés sur {budget}€ ({remaining}€ restants)."
        )
    else:
        lines.append(f"Budget respecté : {total_cost}€ dépensés sur {budget}€ ({remaining}€ de marge).")

    # Rythme
    num_days = len(days)
    total_acts = sum(len(d.get("activities", [])) for d in days)
    avg = total_acts / num_days if num_days > 0 else 0
    pace_map = {"relaxed": 2, "moderate": 3, "intense": 4}
    target = pace_map.get(constraints.get("preferred_pace", "moderate"), 3)

    if abs(avg - target) > 0.9:
        direction = "moins" if avg < target else "plus"
        lines.append(
            f"Rythme ajusté : {avg:.1f} activité(s)/jour en moyenne "
            f"({direction} que le rythme '{constraints.get('preferred_pace', 'moderate')}' souhaité de {target}/jour). "
            f"Raison : contraintes de budget ou d'horaires d'ouverture."
        )

    # Préférences non satisfaites
    selected_cats = [
        act["category"]
        for day in days
        for act in day.get("activities", [])
    ]
    for cat in constraints.get("preferred_categories", []):
        count = selected_cats.count(cat)
        if count == 0:
            lines.append(
                f"⚠ Catégorie préférée '{cat}' absente du plan. "
                f"Causes possibles : budget insuffisant, horaires incompatibles, "
                f"ou quota journalier déjà atteint. "
                f"Alternative : augmentez le budget ou réduisez d'autres catégories."
            )
        else:
            lines.append(f"✓ {count} activité(s) '{cat}' planifiée(s).")

    # Catégories évitées présentes quand même
    for cat in constraints.get("avoided_categories", []):
        count = selected_cats.count(cat)
        if count > 0:
            lines.append(
                f"ℹ {count} activité(s) '{cat}' incluse(s) malgré la préférence d'évitement. "
                f"Ces activités étaient nécessaires pour respecter le minimum d'activités par jour."
            )

    # Activités obligatoires manquantes
    selected_ids = {
        act["id"]
        for day in days
        for act in day.get("activities", [])
    }
    for act_id in constraints.get("must_visit", []):
        if act_id not in selected_ids:
            lines.append(
                f"⚠ Activité obligatoire '{act_id}' absente. "
                f"Vérifiez que son coût et ses horaires sont compatibles avec le plan."
            )

    # Contraintes violées (depuis l'audit)
    for v in solution.get("violated_soft_constraints", []):
        if not any(v in line for line in lines):
            lines.append(f"⚠ {v}")

    return "\n".join(lines) if lines else "✓ Toutes les contraintes et préférences sont respectées."


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
