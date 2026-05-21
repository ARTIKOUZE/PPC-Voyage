"""Solveur CP-SAT (8 types de contraintes) + entrypoint solve_with_city_data. Modèles de domaine → solver_models.py"""
from __future__ import annotations

import logging
from ortools.sat.python import cp_model
from typing import Optional

from solver_models import (
    Activity, TravelConstraints, dict_to_activity, haversine_meters,
)

logger = logging.getLogger(__name__)

class TravelPlannerSolver:
    """Solveur CP-SAT pour la planification de voyage. Variables : `assign[a,d]` BoolVar (activité a assignée au jour d),"""

    SLOT_DURATION = 30
    DAY_START = 7
    DAY_END = 24
    SLOTS_PER_DAY = (DAY_END - DAY_START) * 2

    def __init__(
        self,
        activities: list[Activity],
        constraints: TravelConstraints,
        travel_matrix: Optional[list[list[int]]] = None,
        mode: str = "flexible",
        transport_mode: str = "foot",
        previous_plan: Optional[dict[int, list]] = None,
        touched_days: Optional[set[int]] = None,
        block_new_on_pinned: bool = True,
    ):
        self.activities = {a.id: a for a in activities}
        self.constraints = constraints
        self.mode = mode
        self._transport_mode = transport_mode
        self._previous_plan = previous_plan or {}
        self._touched_days = touched_days
        # Si False : on autorise une nouvelle activité sur un jour pinned
        # (cas "ajoute X" sans jour précisé → X doit pouvoir atterrir quelque part).
        self._block_new_on_pinned = block_new_on_pinned
        self.model = cp_model.CpModel()

        self._act_order = [a.id for a in activities]
        self._act_index = {a_id: i for i, a_id in enumerate(self._act_order)}
        self.travel_matrix = travel_matrix

        self.assign: dict = {}
        self.start: dict = {}
        self.intervals: dict = {}
        self.selected: dict = {}
        self.soft_penalties: list = []
        self.soft_bonuses: list = []

        self._build_model()

    def _build_model(self):
        self._pair_both_assigned: dict = {}
        self._create_variables()
        self._add_budget_constraints()
        self._add_temporal_constraints()
        self._add_logical_constraints()
        self._add_capacity_constraints()
        self._add_soft_preferences()
        self._add_cardinality_constraints()
        self._add_meal_time_constraints()
        self._add_travel_penalty()
        self._add_pin_constraints()
        self._add_stability_bonus()
        self._set_objective()

    def _create_variables(self):
        C = self.constraints
        for a_id, act in self.activities.items():
            self.selected[a_id] = self.model.new_bool_var(f"sel_{a_id}")

            for d in range(C.num_days):
                if d % 7 not in act.available_days:
                    continue
                self.assign[a_id, d] = self.model.new_bool_var(f"assign_{a_id}_d{d}")
                dur_slots = int(act.duration_hours * 2)
                max_start = self.SLOTS_PER_DAY - dur_slots
                self.start[a_id, d] = self.model.new_int_var(
                    0, max(0, max_start), f"start_{a_id}_d{d}"
                )
                self.intervals[a_id, d] = self.model.new_optional_fixed_size_interval_var(
                    self.start[a_id, d], dur_slots, self.assign[a_id, d],
                    f"interval_{a_id}_d{d}",
                )

            day_vars = [self.assign[a_id, d] for d in range(C.num_days)
                        if (a_id, d) in self.assign]
            if day_vars:
                self.model.add_at_most_one(day_vars)
                self.model.add_bool_or(day_vars + [self.selected[a_id].negated()])
                for dv in day_vars:
                    self.model.add_implication(dv, self.selected[a_id])
            else:
                self.model.add(self.selected[a_id] == 0)

    def _add_budget_constraints(self):
        C = self.constraints
        hotel_total = C.hotel_per_night * C.num_days
        food_total = C.daily_food_budget * C.num_days * C.num_travelers
        activity_budget = max(0, C.total_budget - hotel_total - food_total)
        self._activity_budget = activity_budget

        activity_cost = sum(
            self.selected[a_id] * act.cost_euros * C.num_travelers
            for a_id, act in self.activities.items()
        )
        self.model.add(activity_cost <= activity_budget)

        if activity_budget > 100 and C.preferred_pace != "relaxed":
            target_spend = (activity_budget * 7) // 10
            underspend = self.model.new_int_var(0, activity_budget, "act_underspend")
            self.model.add(underspend >= target_spend - activity_cost)
            self.model.add(underspend >= 0)
            penalty_var = self.model.new_int_var(
                0, activity_budget // 20 + 1, "act_underspend_pen"
            )
            self.model.add_division_equality(penalty_var, underspend, 20)
            self.soft_penalties.append(penalty_var)

        for d in range(C.num_days):
            daily_gastro = sum(
                self.assign[a_id, d] * act.cost_euros * C.num_travelers
                for a_id, act in self.activities.items()
                if act.category == "gastro" and (a_id, d) in self.assign
            )
            self.model.add(daily_gastro <= C.daily_food_budget * C.num_travelers)

    def _day_window_slots(self, day_idx: int = 0) -> tuple[int, int]:
        """Fenêtre journalière en slots pour un jour donné (0-indexed).
        Les overrides par jour (day_specific_*_hour, 1-indexed) prennent le pas."""
        C = self.constraints
        START_MARGIN = 1
        day_1 = day_idx + 1
        start_h = C.day_specific_start_hour.get(day_1, C.day_start_hour) \
            if isinstance(C.day_specific_start_hour, dict) else C.day_start_hour
        end_h = C.day_specific_end_hour.get(day_1, C.day_end_hour) \
            if isinstance(C.day_specific_end_hour, dict) else C.day_end_hour

        if start_h is not None:
            raw_start = max(self.DAY_START, start_h)
            win_start = max(0, (raw_start - self.DAY_START) * 2 - START_MARGIN)
        else:
            win_start = 0
        if end_h is not None:
            raw_end = min(self.DAY_END, end_h)
            win_end = min(self.SLOTS_PER_DAY, (raw_end - self.DAY_START) * 2)
        else:
            win_end = self.SLOTS_PER_DAY
        return win_start, win_end

    def _add_temporal_constraints(self):
        for a_id, act in self.activities.items():
            for d in range(self.constraints.num_days):
                if (a_id, d) not in self.assign:
                    continue
                win_start, win_end = self._day_window_slots(d)
                dur_slots = int(act.duration_hours * 2)
                open_slot = max(0, (act.opening_hour - self.DAY_START) * 2)
                close_slot = min(self.SLOTS_PER_DAY,
                                 (min(act.closing_hour, self.DAY_END) - self.DAY_START) * 2)
                eff_start = max(open_slot, win_start)
                eff_end = min(close_slot, win_end)
                if eff_end - eff_start < dur_slots:
                    self.model.add(self.assign[a_id, d] == 0)
                    continue

                self.model.add(
                    self.start[a_id, d] >= eff_start
                ).only_enforce_if(self.assign[a_id, d])

                self.model.add(
                    self.start[a_id, d] + dur_slots <= eff_end
                ).only_enforce_if(self.assign[a_id, d])

    _STOPWORDS = frozenset([
        "le", "la", "les", "l", "de", "du", "des", "d", "un", "une",
        "the", "a", "an", "of", "to", "and", "et",
    ])

    @staticmethod
    def _norm_name(s: str) -> str:
        import re as _re
        import unicodedata as _ud
        s = (s or "").lower().strip()
        s = s.replace("œ", "oe").replace("æ", "ae")
        s = "".join(c for c in _ud.normalize("NFD", s) if _ud.category(c) != "Mn")
        s = _re.sub(r"[^\w\s]", " ", s)
        return _re.sub(r"\s+", " ", s).strip()

    def _tokens(self, s: str) -> list[str]:
        return [t for t in self._norm_name(s).split() if t not in self._STOPWORDS]

    def _resolve_activity(self, name_or_id: str) -> Optional[str]:
        """Résout un nom/ID utilisateur vers l'ID interne (exact > tokens > sous-chaîne > fuzzy)."""
        from difflib import SequenceMatcher

        if name_or_id in self.activities:
            return name_or_id
        term = self._norm_name(name_or_id)
        if not term:
            return None

        # Match exact normalisé.
        for a_id, act in self.activities.items():
            if self._norm_name(act.name) == term:
                return a_id

        # Match par tokens : tous les tokens significatifs du query présents dans le nom.
        q_tokens = self._tokens(name_or_id)
        if q_tokens:
            for a_id, act in self.activities.items():
                a_tokens = set(self._tokens(act.name))
                if all(t in a_tokens for t in q_tokens):
                    return a_id

        # Sous-chaîne normalisée.
        if len(term) >= 4:
            for a_id, act in self.activities.items():
                name_l = self._norm_name(act.name)
                if term in name_l or name_l in term:
                    return a_id

        # Fuzzy SequenceMatcher en dernier recours, seuil 0.75.
        best_id, best_score = None, 0.0
        for a_id, act in self.activities.items():
            score = SequenceMatcher(None, term, self._norm_name(act.name)).ratio()
            if score > best_score:
                best_id, best_score = a_id, score
        if best_score >= 0.75:
            return best_id

        return None

    def _add_logical_constraints(self):
        C = self.constraints

        for name_or_id in C.must_visit:
            a_id = self._resolve_activity(name_or_id)
            if a_id:
                self.model.add(self.selected[a_id] == 1)
            else:
                logger.warning(
                    "[Solver] must_visit '%s' n'a matche aucune activite du pool — "
                    "elle ne sera pas ajoutee.", name_or_id,
                )

        for name_or_id, day_1 in C.must_visit_on_day.items():
            a_id = self._resolve_activity(name_or_id)
            if not a_id:
                logger.warning(
                    "[Solver] must_visit_on_day '%s' (jour %s) n'a matche aucune "
                    "activite — l'ajout au jour cible va echouer.", name_or_id, day_1,
                )
                continue
            day = day_1 - 1
            if 0 <= day < C.num_days and (a_id, day) in self.assign:
                self.model.add(self.assign[a_id, day] == 1)
                self.model.add(self.selected[a_id] == 1)
                for d in range(C.num_days):
                    if d != day and (a_id, d) in self.assign:
                        self.model.add(self.assign[a_id, d] == 0)
            elif a_id in self.selected:
                self.model.add(self.selected[a_id] == 1)

        for name_or_id in C.must_avoid:
            a_id = self._resolve_activity(name_or_id)
            if a_id:
                self.model.add(self.selected[a_id] == 0)
            else:
                logger.warning(
                    "[Solver] must_avoid '%s' n'a pas pu être résolu vers une activité "
                    "du pool — l'utilisateur va voir l'activité rester dans le plan.",
                    name_or_id,
                )

        for a1, a2 in C.incompatible_pairs:
            for d in range(C.num_days):
                if (a1, d) in self.assign and (a2, d) in self.assign:
                    self.model.add(self.assign[a1, d] + self.assign[a2, d] <= 1)

        for b_id, a_id in C.prerequisites.items():
            if a_id not in self.selected or b_id not in self.selected:
                continue
            self.model.add(self.selected[a_id] >= self.selected[b_id])
            for d_b in range(C.num_days):
                if (b_id, d_b) not in self.assign:
                    continue
                for d_a in range(d_b, C.num_days):
                    if (a_id, d_a) not in self.assign:
                        continue
                    self.model.add(self.assign[b_id, d_b] + self.assign[a_id, d_a] <= 1)

        _WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for d in range(C.num_days):
            if d >= len(C.trip_weekdays):
                continue
            weekday_name = _WEEKDAY_NAMES[C.trip_weekdays[d]]
            for a_id, act in self.activities.items():
                closed_days = getattr(act, "closed_days", None) or []
                if weekday_name in closed_days and (a_id, d) in self.assign:
                    self.model.add(self.assign[a_id, d] == 0)

    def _choose_segment_mode(self, foot_minutes: int, dist_m, src_act=None):
        """Mode de transport pour un segment. Si bike/car globaux, on garde. Sinon foot ≤ 25 min, ou bascule transit (~3× plus rapide) sinon."""
        base = getattr(self, "_transport_mode", "foot")
        if base in ("car", "bike"):
            return (base, foot_minutes)
        if foot_minutes > 25 and (dist_m is None or dist_m > 1500):
            transit_min = max(8, foot_minutes // 3 + 5)
            specific_mode = "transit"
            if src_act and src_act.transit_options:
                raw_type = (src_act.transit_options[0].get("type") or "").lower()
                if raw_type in ("metro", "bus", "rer", "tram", "train",
                                "funiculaire", "ferry", "navette"):
                    specific_mode = raw_type
            return (specific_mode, transit_min)
        return ("foot", foot_minutes)

    def _travel_minutes(self, a1_id: str, a2_id: str) -> int:
        """Temps de trajet (OSRM, pas de fallback)."""
        if self.travel_matrix is None:
            raise RuntimeError("travel_matrix obligatoire (aucun fallback).")
        return int(self.travel_matrix[self._act_index[a1_id]][self._act_index[a2_id]])

    def _add_capacity_constraints(self):
        C = self.constraints

        for d in range(C.num_days):
            day_intervals = [self.intervals[a_id, d] for a_id in self.activities
                             if (a_id, d) in self.intervals]
            if day_intervals:
                self.model.add_no_overlap(day_intervals)

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

                    both_assigned = self.model.new_bool_var(f"both_{a1}_{a2}_d{d}")
                    self.model.add_min_equality(both_assigned,
                        [self.assign[a1, d], self.assign[a2, d]])
                    self._pair_both_assigned[(a1, a2, d)] = (both_assigned, travel)

                    order_var = self.model.new_bool_var(f"order_{a1}_{a2}_d{d}")
                    self.model.add(
                        self.start[a1, d] + dur1 + travel_slots <= self.start[a2, d]
                    ).only_enforce_if([both_assigned, order_var])
                    self.model.add(
                        self.start[a2, d] + dur2 + travel_slots <= self.start[a1, d]
                    ).only_enforce_if([both_assigned, order_var.negated()])

    _PACE_CONFIG = {
        "relaxed":  {"target": 3, "short_w": 15, "over_w": 10},
        "moderate": {"target": 4, "short_w": 20, "over_w": 4},
        "intense":  {"target": 5, "short_w": 25, "over_w": 1},
    }

    def _add_soft_preferences(self):
        C = self.constraints

        if self.mode == "strict" and C.preferred_categories:
            for a_id, act in self.activities.items():
                if (act.category not in C.preferred_categories
                        and a_id not in C.must_visit):
                    self.model.add(self.selected[a_id] == 0)

        for a_id, act in self.activities.items():
            score = act.priority_score
            if act.category in C.preferred_categories:
                score += 5 if self.mode == "strict" else 3
            if act.category in C.avoided_categories:
                score -= 5
            self.soft_bonuses.append(self.selected[a_id] * score)

        pace = C.preferred_pace if C.preferred_pace in self._PACE_CONFIG else "moderate"
        cfg = self._PACE_CONFIG[pace]
        target = cfg["target"]
        for d in range(C.num_days):
            day_count = sum(self.assign[a_id, d] for a_id in self.activities
                            if (a_id, d) in self.assign)
            shortfall = self.model.new_int_var(0, target, f"shortfall_d{d}")
            self.model.add(shortfall >= target - day_count)
            self.model.add(shortfall >= 0)
            self.soft_penalties.append(shortfall * cfg["short_w"])

            max_over = max(0, C.max_activities_per_day - target)
            if cfg["over_w"] > 0 and max_over > 0:
                overflow = self.model.new_int_var(0, max_over, f"overflow_d{d}")
                self.model.add(overflow >= day_count - target)
                self.model.add(overflow >= 0)
                self.soft_penalties.append(overflow * cfg["over_w"])

        if C.morning_preference:
            for a_id, act in self.activities.items():
                if act.category == C.morning_preference:
                    for d in range(C.num_days):
                        if (a_id, d) not in self.assign:
                            continue
                        is_morning = self.model.new_bool_var(f"morn_{a_id}_d{d}")
                        self.model.add(self.start[a_id, d] <= 10).only_enforce_if(is_morning)
                        self.model.add(self.start[a_id, d] > 10).only_enforce_if(is_morning.negated())
                        morning_bonus = self.model.new_bool_var(f"morn_bonus_{a_id}_d{d}")
                        self.model.add_min_equality(morning_bonus,
                            [self.assign[a_id, d], is_morning]
                        )
                        self.soft_bonuses.append(morning_bonus * 2)

    def _add_cardinality_constraints(self):
        C = self.constraints

        min_act_duration = min(
            (act.duration_hours for act in self.activities.values()),
            default=1.0
        )
        min_act_duration = max(0.5, min_act_duration)

        HARD_FLOOR = 1
        soft_target = max(HARD_FLOOR, int(C.min_activities_per_day))

        for d in range(C.num_days):
            win_start, win_end = self._day_window_slots(d)
            available_hours = (win_end - win_start) / 2
            day_max = min(C.max_activities_per_day, int(available_hours / min_act_duration))
            day_max = max(1, day_max)

            day_count = sum(
                self.assign[a_id, d]
                for a_id in self.activities
                if (a_id, d) in self.assign
            )
            self.model.add(day_count <= day_max)
            # Floor : si la fenêtre du jour est < 1h, on relâche à 0 pour éviter INFEASIBLE.
            day_floor = HARD_FLOOR if available_hours >= 1 else 0
            self.model.add(day_count >= day_floor)

            day_soft_target = min(soft_target, day_max)
            if day_soft_target > day_floor:
                min_shortfall = self.model.new_int_var(
                    0, day_soft_target, f"min_short_d{d}"
                )
                self.model.add(min_shortfall >= day_soft_target - day_count)
                self.model.add(min_shortfall >= 0)
                self.soft_penalties.append(min_shortfall * 20)

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

    _NON_RESTAURANT_KEYWORDS = (
        "marché", "market", "halles",
        "tour", "tournée", "tasting", "dégustation",
        "class", "cours", "atelier", "cooking",
        "gelato", "glace", "ice cream",
        "café", "coffee",
        "vineyard", "winery", "cave",
        "pub ", "bar à",
    )

    def _is_restaurant(self, act: Activity) -> bool:
        """Heuristique : gastro avec durée 1-3h et nom non-exclusif."""
        if act.category != "gastro":
            return False
        name = act.name.lower()
        if any(kw in name for kw in self._NON_RESTAURANT_KEYWORDS):
            return False
        return 1.0 <= act.duration_hours <= 3.0

    def _add_meal_time_constraints(self):
        """HARD : start d'un restaurant ∈ [12h, 14h] ∨ [19h30, 21h30]. Disjunction modélisée par 2 BoolVar (lunch/dinner) + only_enforce_if."""
        LUNCH_START = (12 - self.DAY_START) * 2
        LUNCH_END = (14 - self.DAY_START) * 2
        DINNER_START = int((19.5 - self.DAY_START) * 2)
        DINNER_END = int((21.5 - self.DAY_START) * 2)

        for a_id, act in self.activities.items():
            if not self._is_restaurant(act):
                continue
            for d in range(self.constraints.num_days):
                if (a_id, d) not in self.assign:
                    continue
                is_lunch = self.model.new_bool_var(f"lunch_{a_id}_d{d}")
                is_dinner = self.model.new_bool_var(f"dinner_{a_id}_d{d}")
                self.model.add(is_lunch + is_dinner == 1).only_enforce_if(
                    self.assign[a_id, d])
                self.model.add(self.start[a_id, d] >= LUNCH_START
                    ).only_enforce_if([self.assign[a_id, d], is_lunch])
                self.model.add(self.start[a_id, d] <= LUNCH_END
                    ).only_enforce_if([self.assign[a_id, d], is_lunch])
                self.model.add(self.start[a_id, d] >= DINNER_START
                    ).only_enforce_if([self.assign[a_id, d], is_dinner])
                self.model.add(self.start[a_id, d] <= DINNER_END
                    ).only_enforce_if([self.assign[a_id, d], is_dinner])

    def _add_travel_penalty(self):
        """Nudge soft : trajet ≤ 15 min = gratuit, sinon (travel-15)//10 par paire."""
        for (a1, a2, d), (both_assigned, travel_min) in self._pair_both_assigned.items():
            weight = max(0, (int(travel_min) - 15) // 10)
            if weight > 0:
                self.soft_penalties.append(both_assigned * weight)

    @staticmethod
    def _iter_prev_plan(entries):
        """Itère le plan précédent (tuples ou IDs legacy)."""
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                yield entry[0], entry[1]
            else:
                yield entry, None

    def _add_stability_bonus(self):
        """Bonus soft pour préserver les (act, day) du tour précédent. Poids dynamique : 20 si pas de pin (modif globale), 8 si pin partiel."""
        if not self._previous_plan:
            return
        STABILITY_WEIGHT = 20 if self._touched_days is None else 8
        for day_idx, entries in self._previous_plan.items():
            for act_id, _slot in self._iter_prev_plan(entries):
                if (act_id, day_idx) in self.assign:
                    self.soft_bonuses.append(
                        self.assign[act_id, day_idx] * STABILITY_WEIGHT)

    def _add_pin_constraints(self):
        """Pin HARD des jours non touchés. On NE pin PAS les activités que
        l'utilisateur veut déplacer (must_visit_on_day) ou retirer (must_avoid),
        car ça créerait un conflit hard avec leurs propres contraintes."""
        if self._touched_days is None or not self._previous_plan:
            return

        # Activités à ne pas pin (elles sont en mouvement ou en sortie)
        skip_pin: set[str] = set()
        for name_or_id in self.constraints.must_avoid:
            a_id = self._resolve_activity(name_or_id)
            if a_id:
                skip_pin.add(a_id)
        for name_or_id in self.constraints.must_visit_on_day:
            a_id = self._resolve_activity(name_or_id)
            if a_id:
                skip_pin.add(a_id)

        for day_idx, entries in self._previous_plan.items():
            if day_idx in self._touched_days:
                continue
            pinned_ids: set[str] = set()
            for act_id, slot in self._iter_prev_plan(entries):
                if act_id in skip_pin:
                    continue
                if (act_id, day_idx) in self.assign:
                    self.model.add(self.assign[act_id, day_idx] == 1)
                    if slot is not None and (act_id, day_idx) in self.start:
                        self.model.add(self.start[act_id, day_idx] == int(slot))
                    pinned_ids.add(act_id)
            # Bloquer les nouvelles activités sauf si block_new_on_pinned=False
            if self._block_new_on_pinned:
                for a_id in self.activities:
                    if a_id in pinned_ids:
                        continue
                    if (a_id, day_idx) in self.assign:
                        self.model.add(self.assign[a_id, day_idx] == 0)

    def _set_objective(self):
        total_bonus = sum(self.soft_bonuses) if self.soft_bonuses else 0
        total_penalty = sum(self.soft_penalties) if self.soft_penalties else 0
        self.model.maximize(total_bonus - total_penalty)

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

        hotel_cost = C.hotel_per_night * C.num_days
        food_cost_actual = 0
        activity_cost_actual = 0

        for d in range(C.num_days):
            day_activities = []
            for a_id, act in self.activities.items():
                if (a_id, d) not in self.assign:
                    continue
                if solver.value(self.assign[a_id, d]):
                    start_slot = solver.value(self.start[a_id, d])
                    start_hour = self.DAY_START + start_slot * 0.5
                    end_hour = start_hour + act.duration_hours

                    line_cost = act.cost_euros * C.num_travelers
                    day_activities.append({
                        "id": a_id,
                        "name": act.name,
                        "category": act.category,
                        "zone": act.zone,
                        "start_time": f"{int(start_hour):02d}:{int((start_hour % 1) * 60):02d}",
                        "end_time": f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}",
                        "start_slot": int(start_slot),
                        "duration_hours": act.duration_hours,
                        "cost": line_cost,
                    })
                    if act.category == "gastro":
                        food_cost_actual += line_cost
                    else:
                        activity_cost_actual += line_cost

            day_activities.sort(key=lambda x: x["start_time"])

            transitions = []
            for i in range(len(day_activities) - 1):
                a1 = day_activities[i]
                a2 = day_activities[i + 1]
                travel_min = self._travel_minutes(a1["id"], a2["id"])
                src = self.activities.get(a1["id"])
                dst = self.activities.get(a2["id"])
                dist_m = None
                if src and dst and src.latitude and dst.latitude:
                    dist_m = haversine_meters(
                        src.latitude, src.longitude,
                        dst.latitude, dst.longitude,
                    )
                mode, minutes = self._choose_segment_mode(int(travel_min), dist_m, src)
                transition = {
                    "from_id": a1["id"],
                    "to_id": a2["id"],
                    "from_name": a1["name"],
                    "to_name": a2["name"],
                    "minutes": minutes,
                    "distance_m": int(round(dist_m)) if dist_m is not None else None,
                    "mode": mode,
                }
                if mode == "transit":
                    if src and src.nearest_stop:
                        transition["from_stop"] = src.nearest_stop
                        transition["from_transit"] = src.transit_options
                        if src.transit_exit:
                            transition["from_exit"] = src.transit_exit
                    if dst and dst.nearest_stop:
                        transition["to_stop"] = dst.nearest_stop
                        transition["to_transit"] = dst.transit_options
                        if dst.transit_exit:
                            transition["to_exit"] = dst.transit_exit
                transitions.append(transition)

            plan["days"].append({
                "day": d + 1,
                "activities": day_activities,
                "activity_count": len(day_activities),
                "transitions": transitions,
                "total_travel_minutes": sum(t["minutes"] for t in transitions),
            })

        total_cost = hotel_cost + food_cost_actual + activity_cost_actual
        remaining = C.total_budget - total_cost

        plan["summary"] = {
            "total_cost": total_cost,
            "budget": C.total_budget,
            "remaining_budget": remaining,
            "total_activities": sum(len(day["activities"]) for day in plan["days"]),
            "hotel_cost": hotel_cost,
            "food_cost": food_cost_actual,
            "activity_cost": activity_cost_actual,
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

        respected, violated = self._audit_constraints(plan)
        plan["respected_constraints"] = respected
        plan["violated_soft_constraints"] = violated

        return plan

    def _audit_constraints(self, plan: dict) -> tuple[list[str], list[str]]:
        """Inventaire des contraintes respectées vs violées (post-solve)."""
        C = self.constraints
        respected: list[str] = []
        violated: list[str] = []
        summary = plan.get("summary", {})
        days = plan.get("days", [])
        selected_ids = {a["id"] for day in days for a in day.get("activities", [])}
        cats = [a["category"] for day in days for a in day.get("activities", [])]

        if summary.get("remaining_budget", 0) >= 0:
            respected.append(f"Budget respecté (reste {summary['remaining_budget']}€)")
        else:
            violated.append(f"Budget dépassé de {abs(summary['remaining_budget'])}€")
        respected.append(f"Durée exacte : {C.num_days} jour(s)")

        for day in days:
            count = day["activity_count"]
            if count > C.max_activities_per_day:
                violated.append(f"Jour {day['day']} : {count} > max ({C.max_activities_per_day})")
            elif count < C.min_activities_per_day:
                violated.append(f"Jour {day['day']} : {count} < min ({C.min_activities_per_day})")
            else:
                respected.append(f"Jour {day['day']} : {count} activité(s) OK")

        def _name(a_id, fallback):
            return self.activities[a_id].name if a_id and a_id in self.activities else fallback

        for name_or_id in C.must_visit:
            a_id = self._resolve_activity(name_or_id)
            n = _name(a_id, name_or_id)
            (respected if a_id and a_id in selected_ids else violated).append(
                f"Obligatoire {'présente' if a_id in selected_ids else 'absente'} : {n}")

        for name_or_id, day_1 in C.must_visit_on_day.items():
            a_id = self._resolve_activity(name_or_id)
            n = _name(a_id, name_or_id)
            if a_id:
                day_acts = days[day_1 - 1].get("activities", []) if 0 < day_1 <= len(days) else []
                on_day = any(a["id"] == a_id for a in day_acts)
                (respected if on_day else violated).append(
                    f"{n} {'planifiée' if on_day else 'absente'} au jour {day_1}")

        for name_or_id in C.must_avoid:
            a_id = self._resolve_activity(name_or_id)
            n = _name(a_id, name_or_id)
            if a_id:
                (respected if a_id not in selected_ids else violated).append(
                    f"Exclue {'bien absente' if a_id not in selected_ids else 'présente'} : {n}")

        for cat in C.preferred_categories:
            count = cats.count(cat)
            (respected if count else violated).append(
                f"Catégorie préférée '{cat}' : {count} activité(s)")
        for cat in C.avoided_categories:
            count = cats.count(cat)
            (respected if not count else violated).append(
                f"Catégorie évitée '{cat}' : {count} présente(s)")

        return respected, violated

_ENRICHED_ACTIVITY_FIELDS = (
    "opening_hours", "flexible_hours", "closed_days",
    "price_info", "student_discount", "student_cost_euros",
    "last_entry_before_close_minutes", "address",
    "nearest_stop", "transit_options", "transit_exit",
    "data_confidence", "data_source",
)

def _infeasible_budget(constraints, hotel_cost, food_cost):
    fixed = hotel_cost + food_cost
    deficit = fixed - constraints.total_budget
    return {
        "status": "INFEASIBLE",
        "message": (
            f"Budget {constraints.total_budget}€ trop faible : hébergement + "
            f"repas = {fixed}€ (manque {deficit}€)."
        ),
        "days": [], "summary": {
            "total_cost": fixed, "budget": constraints.total_budget,
            "remaining_budget": -deficit, "total_activities": 0,
            "hotel_cost": hotel_cost, "food_cost": food_cost, "activity_cost": 0,
        },
        "stats": {"status_name": "INFEASIBLE_BUDGET"},
        "respected_constraints": [],
        "violated_soft_constraints": [f"Budget dépassé de {deficit}€"],
    }

def _derive_trip_weekdays(constraints):
    if not constraints.start_date or constraints.trip_weekdays:
        return
    try:
        from datetime import date, timedelta
        y, m, d = map(int, constraints.start_date.split("-"))
        d0 = date(y, m, d)
        constraints.trip_weekdays = [
            (d0 + timedelta(days=i)).weekday() for i in range(constraints.num_days)
        ]
    except (ValueError, TypeError):
        constraints.trip_weekdays = []

def _select_hotel(city_data, cap):
    """Hôtel le plus cher ≤ cap. Fallback : le moins cher si aucun ne rentre. Renvoie le dict ou None."""
    hotels = city_data.get("hotels") or []
    if not hotels:
        return None
    below = [h for h in hotels if (h.get("price_per_night") or 0) <= cap]
    if below:
        return max(below, key=lambda h: h.get("price_per_night") or 0)
    return min(hotels, key=lambda h: h.get("price_per_night") or 0)

def solve_with_city_data(
    constraints_dict: dict,
    city_data: dict,
    time_limit_seconds: int = 10,
    mode: str = "flexible",
    previous_plan: Optional[dict[int, list]] = None,
    touched_days: Optional[set[int]] = None,
    block_new_on_pinned: bool = True,
) -> dict:
    """Entrypoint principal : consomme un city_data, renvoie le plan complet."""
    constraints = TravelConstraints(**{
        k: v for k, v in constraints_dict.items()
        if k in TravelConstraints.__dataclass_fields__
    })

    def _coerce_int_keys(d):
        if not isinstance(d, dict):
            return {}
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = v
            except (TypeError, ValueError):
                continue
        return out
    constraints.day_specific_start_hour = _coerce_int_keys(constraints.day_specific_start_hour)
    constraints.day_specific_end_hour = _coerce_int_keys(constraints.day_specific_end_hour)

    _derive_trip_weekdays(constraints)

    activities = [dict_to_activity(a) for a in city_data.get("activities", [])]
    if not activities:
        return {
            "status": "INFEASIBLE",
            "message": f"Aucune activité disponible pour {constraints.destination}.",
            "days": [], "summary": {}, "stats": {},
            "respected_constraints": [], "violated_soft_constraints": [],
        }

    preselected_hotel = _select_hotel(city_data, constraints.hotel_per_night)
    if preselected_hotel:
        real_price = int(preselected_hotel.get("price_per_night") or 0)
        if real_price > 0:
            constraints.hotel_per_night = real_price

    hotel_cost = constraints.hotel_per_night * constraints.num_days
    food_cost = constraints.daily_food_budget * constraints.num_days * constraints.num_travelers
    if hotel_cost + food_cost > constraints.total_budget:
        return _infeasible_budget(constraints, hotel_cost, food_cost)

    travel_matrix = city_data.get("travel_matrix")
    transport_mode = city_data.get("transport_mode", "foot")
    planner = TravelPlannerSolver(
        activities, constraints,
        travel_matrix=travel_matrix, mode=mode, transport_mode=transport_mode,
        previous_plan=previous_plan, touched_days=touched_days,
        block_new_on_pinned=block_new_on_pinned,
    )
    result = planner.solve(time_limit_seconds=time_limit_seconds)

    result["city"] = city_data.get("city", {})
    result["data_source"] = city_data.get("data_source", "unknown")
    result["transport_mode"] = transport_mode
    result["start_date"] = constraints.start_date
    result["end_date"] = constraints.end_date
    result["trip_weekdays"] = constraints.trip_weekdays
    if preselected_hotel:
        result["hotel"] = preselected_hotel

    if "days" in result:
        act_by_id = {a.id: a for a in activities}
        src_dict_by_id = {a["id"]: a for a in city_data.get("activities", [])}
        for day in result["days"]:
            for act in day.get("activities", []):
                src = act_by_id.get(act["id"])
                if src:
                    act["latitude"] = src.latitude
                    act["longitude"] = src.longitude
                src_dict = src_dict_by_id.get(act["id"], {})
                for f in _ENRICHED_ACTIVITY_FIELDS:
                    if f in src_dict:
                        act[f] = src_dict[f]

    if result.get("days"):
        src_addr = {a["id"]: a.get("address", "") for a in city_data.get("activities", [])}
        for day in result["days"]:
            for trans in day.get("transitions", []):
                trans.setdefault("from_address", src_addr.get(trans.get("from_id", ""), ""))
                trans.setdefault("to_address", src_addr.get(trans.get("to_id", ""), ""))
        try:
            from transit_router import enrich_transitions_with_routing
            city_name = city_data.get("city", {}).get("name", constraints.destination)
            result = enrich_transitions_with_routing(result, city_name)
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("[Solver] Transit routing skipped: %s", e)

    pool_size = len(activities)
    selected = result.get("summary", {}).get("total_activities", 0)
    result["pool_stats"] = {
        "pool_size": pool_size, "selected": selected,
        "saturation_ratio": round(selected / pool_size, 2) if pool_size else 0,
        "pool_exhausted": selected >= pool_size,
    }
    return result

from solver_explain import explain_solution  # noqa: E402, F401
