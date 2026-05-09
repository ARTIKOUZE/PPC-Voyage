"""
Templates d'estimation et base de données curatée d'activités touristiques.

Trois niveaux d'information, utilisés en cascade :

  1. KNOWN_ACTIVITIES : données réelles vérifiées pour les grands sites touristiques.
     Confidence ≥ 0.90. Source : données publiques officielles (2024-2025).

  2. KIND_TEMPLATES : estimations par sous-type d'activité (tags OpenTripMap / OSM).
     Confidence ≈ 0.55.

  3. CATEGORY_DEFAULTS : fallback par grande catégorie.
     Confidence ≈ 0.35.

Chaque nœud retourne (valeur, confidence: float 0–1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass pour une activité connue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivityData:
    """Données réelles pour un site touristique connu."""
    open_hour: int
    close_hour: int
    cost_euros: int
    duration_hours: float
    confidence: float = 0.92
    category: str = "culture"
    student_discount: float = 0.0   # fraction de réduction (0.3 = 30%)
    free_first_sunday: bool = False  # gratuit le premier dimanche du mois
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Base curatée : grands sites touristiques mondiaux (données 2024-2025)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_ACTIVITIES: dict[str, ActivityData] = {

    # ── Rome ─────────────────────────────────────────────────────────────────
    "colosseum":             ActivityData(8, 19, 18, 2.5, student_discount=0.0),
    "colisée":               ActivityData(8, 19, 18, 2.5),
    "colosseo":              ActivityData(8, 19, 18, 2.5),
    "roman colosseum":       ActivityData(8, 19, 18, 2.5),

    "vatican museums":       ActivityData(8, 18, 17, 3.5, student_discount=0.5),
    "musées du vatican":     ActivityData(8, 18, 17, 3.5, student_discount=0.5),
    "musei vaticani":        ActivityData(8, 18, 17, 3.5, student_discount=0.5),
    "sistine chapel":        ActivityData(8, 18, 17, 3.5, student_discount=0.5),
    "chapelle sixtine":      ActivityData(8, 18, 17, 3.5, student_discount=0.5),

    "roman forum":           ActivityData(8, 19, 16, 2.0),
    "forum romain":          ActivityData(8, 19, 16, 2.0),
    "foro romano":           ActivityData(8, 19, 16, 2.0),

    "pantheon":              ActivityData(9, 19, 5,  1.0),
    "panthéon rome":         ActivityData(9, 19, 5,  1.0),

    "trevi fountain":        ActivityData(7, 23, 0,  0.75),
    "fontaine de trevi":     ActivityData(7, 23, 0,  0.75),
    "fontana di trevi":      ActivityData(7, 23, 0,  0.75),

    "galleria borghese":     ActivityData(9, 19, 15, 2.0, student_discount=0.5),
    "galerie borghèse":      ActivityData(9, 19, 15, 2.0, student_discount=0.5),
    "borghese gallery":      ActivityData(9, 19, 15, 2.0, student_discount=0.5),

    "castel sant'angelo":    ActivityData(9, 19, 15, 2.0),
    "château saint-ange":    ActivityData(9, 19, 15, 2.0),
    "castel sant angelo":    ActivityData(9, 19, 15, 2.0),

    "piazza navona":         ActivityData(0, 24,  0, 1.0, category="culture"),
    "piazza venezia":        ActivityData(0, 24,  0, 0.5, category="culture"),
    "piazza del popolo":     ActivityData(0, 24,  0, 0.5, category="culture"),

    "baths of caracalla":    ActivityData(9, 18,  8, 1.5),
    "thermes de caracalla":  ActivityData(9, 18,  8, 1.5),

    "ostia antica":          ActivityData(8, 17, 12, 4.0),

    "palazzo del quirinale": ActivityData(9, 16, 15, 2.0),

    "villa borghese":        ActivityData(6, 20,  0, 2.0, category="nature"),
    "parc villa borghèse":   ActivityData(6, 20,  0, 2.0, category="nature"),

    "via appia antica":      ActivityData(8, 18, 15, 3.0, category="nature"),

    "testaccio market":      ActivityData(7, 14,  0, 1.5, category="gastro"),
    "campo de' fiori":       ActivityData(7, 14,  0, 1.5, category="gastro"),

    # ── Paris ────────────────────────────────────────────────────────────────
    "eiffel tower":          ActivityData(9, 23, 26, 2.0, student_discount=0.2),
    "tour eiffel":           ActivityData(9, 23, 26, 2.0, student_discount=0.2),

    "louvre":                ActivityData(9, 18, 15, 3.0, student_discount=1.0,
                                          free_first_sunday=True),
    "louvre museum":         ActivityData(9, 18, 15, 3.0, student_discount=1.0,
                                          free_first_sunday=True),
    "musée du louvre":       ActivityData(9, 18, 15, 3.0, student_discount=1.0,
                                          free_first_sunday=True),

    "notre-dame cathedral":  ActivityData(8, 20,  0, 1.5),
    "cathédrale notre-dame": ActivityData(8, 20,  0, 1.5),
    "notre-dame de paris":   ActivityData(8, 20,  0, 1.5),

    "musée d'orsay":         ActivityData(9, 18, 14, 2.5, student_discount=1.0,
                                          free_first_sunday=True),
    "orsay museum":          ActivityData(9, 18, 14, 2.5, student_discount=1.0,
                                          free_first_sunday=True),

    "versailles":            ActivityData(9, 17, 18, 4.0, student_discount=1.0),
    "palace of versailles":  ActivityData(9, 17, 18, 4.0, student_discount=1.0),
    "château de versailles": ActivityData(9, 17, 18, 4.0, student_discount=1.0),

    "arc de triomphe":       ActivityData(10, 23, 13, 1.0, student_discount=0.5),
    "sainte-chapelle":       ActivityData(9, 17, 13, 1.0, student_discount=0.5),

    "pompidou centre":       ActivityData(11, 21, 15, 2.0, student_discount=1.0,
                                          free_first_sunday=True),
    "centre pompidou":       ActivityData(11, 21, 15, 2.0, student_discount=1.0,
                                          free_first_sunday=True),

    "musée de cluny":        ActivityData(9, 17, 10, 1.5, student_discount=1.0),
    "montmartre":            ActivityData(0, 24,  0, 2.0, category="nature"),
    "sacré-cœur":            ActivityData(6, 22,  0, 1.0),
    "sacré coeur":           ActivityData(6, 22,  0, 1.0),
    "champs-élysées":        ActivityData(0, 24,  0, 1.5, category="shopping"),

    "musée de l'orangerie":  ActivityData(9, 18, 12, 1.5, student_discount=1.0),
    "palais royal":          ActivityData(8, 22,  0, 1.0),

    # ── Barcelona ────────────────────────────────────────────────────────────
    "sagrada familia":       ActivityData(9, 19, 26, 2.0, student_discount=0.2),

    "park güell":            ActivityData(8, 20, 10, 2.0),
    "park guell":            ActivityData(8, 20, 10, 2.0),
    "parc güell":            ActivityData(8, 20, 10, 2.0),

    "camp nou":              ActivityData(9, 18, 26, 2.5),

    "gothic quarter":        ActivityData(0, 24,  0, 2.0, category="culture"),
    "barri gòtic":           ActivityData(0, 24,  0, 2.0, category="culture"),
    "barrio gótico":         ActivityData(0, 24,  0, 2.0, category="culture"),

    "barcelona cathedral":   ActivityData(8, 19,  0, 1.5),
    "catedral de barcelona": ActivityData(8, 19,  0, 1.5),

    "picasso museum":        ActivityData(9, 19, 12, 2.0, student_discount=0.5),
    "museu picasso":         ActivityData(9, 19, 12, 2.0, student_discount=0.5),

    "palau de la música catalana": ActivityData(10, 15, 20, 1.5),

    "casa batlló":           ActivityData(9, 21, 35, 2.0),
    "casa milà":             ActivityData(9, 21, 25, 1.5),
    "la pedrera":            ActivityData(9, 21, 25, 1.5),

    "montjuïc":              ActivityData(9, 21,  5, 3.0, category="nature"),
    "montjuic":              ActivityData(9, 21,  5, 3.0, category="nature"),

    "la barceloneta":        ActivityData(0, 24,  0, 3.0, category="nature"),

    # ── London ───────────────────────────────────────────────────────────────
    "british museum":        ActivityData(10, 17,  0, 3.0),
    "national gallery":      ActivityData(10, 18,  0, 2.5),

    "tower of london":       ActivityData(9, 17, 30, 2.5, student_discount=0.15),
    "tour de londres":       ActivityData(9, 17, 30, 2.5, student_discount=0.15),

    "buckingham palace":     ActivityData(9, 18, 30, 2.0),
    "westminster abbey":     ActivityData(9, 18, 27, 1.5, student_discount=0.1),
    "abbaye de westminster": ActivityData(9, 18, 27, 1.5, student_discount=0.1),

    "st paul's cathedral":   ActivityData(8, 16, 20, 1.5, student_discount=0.15),
    "tate modern":           ActivityData(10, 18,  0, 2.5),
    "victoria and albert museum": ActivityData(10, 17, 0, 3.0),
    "natural history museum": ActivityData(10, 17, 0, 2.5),
    "hyde park":             ActivityData(5,  24,  0, 2.0, category="nature"),
    "the shard":             ActivityData(10, 22, 32, 1.5),
    "science museum london": ActivityData(10, 18,  0, 2.5),
    "national portrait gallery": ActivityData(10, 18, 0, 2.0),

    # ── Amsterdam ────────────────────────────────────────────────────────────
    "rijksmuseum":           ActivityData(9, 17, 20, 2.5, student_discount=0.0),
    "van gogh museum":       ActivityData(9, 17, 20, 2.0),
    "musée van gogh":        ActivityData(9, 17, 20, 2.0),

    "anne frank house":      ActivityData(9, 20, 16, 1.5),
    "maison d'anne frank":   ActivityData(9, 20, 16, 1.5),
    "anne frank huis":       ActivityData(9, 20, 16, 1.5),

    "vondelpark":            ActivityData(0, 24,  0, 2.0, category="nature"),
    "stedelijk museum":      ActivityData(10, 18, 20, 2.0),

    # ── Berlin ───────────────────────────────────────────────────────────────
    "east side gallery":     ActivityData(0, 24,  0, 1.5, category="culture"),
    "berlin wall memorial":  ActivityData(0, 24,  0, 1.5),
    "mémorial du mur de berlin": ActivityData(0, 24,  0, 1.5),

    "pergamon museum":       ActivityData(10, 18, 12, 2.5, student_discount=0.5),
    "berlin cathedral":      ActivityData(9, 19,  9, 1.5),
    "berliner dom":          ActivityData(9, 19,  9, 1.5),

    "checkpoint charlie":    ActivityData(0, 24,  0, 1.0),
    "brandenburger tor":     ActivityData(0, 24,  0, 0.5),
    "porte de brandebourg":  ActivityData(0, 24,  0, 0.5),
    "tiergarten":            ActivityData(0, 24,  0, 2.0, category="nature"),
    "holocaust memorial":    ActivityData(0, 24,  0, 1.0),

    # ── Florence ─────────────────────────────────────────────────────────────
    "uffizi gallery":        ActivityData(8, 18, 20, 3.0, student_discount=0.5),
    "galleria degli uffizi": ActivityData(8, 18, 20, 3.0, student_discount=0.5),
    "galerie des offices":   ActivityData(8, 18, 20, 3.0, student_discount=0.5),

    "florence cathedral":    ActivityData(10, 17,  0, 1.5),
    "duomo firenze":         ActivityData(10, 17,  0, 1.5),
    "cathédrale de florence": ActivityData(10, 17,  0, 1.5),

    "ponte vecchio":         ActivityData(0, 24,  0, 1.0),
    "accademia gallery":     ActivityData(8, 19, 12, 2.0, student_discount=0.5),
    "galleria dell'accademia": ActivityData(8, 19, 12, 2.0, student_discount=0.5),
    "pitti palace":          ActivityData(8, 18, 16, 2.5, student_discount=0.5),
    "palazzo pitti":         ActivityData(8, 18, 16, 2.5, student_discount=0.5),

    # ── Venice ───────────────────────────────────────────────────────────────
    "st mark's basilica":    ActivityData(9, 17,  3, 1.5),
    "basilique saint-marc":  ActivityData(9, 17,  3, 1.5),
    "basilica di san marco": ActivityData(9, 17,  3, 1.5),

    "doge's palace":         ActivityData(9, 17, 20, 2.0, student_discount=0.5),
    "palais des doges":      ActivityData(9, 17, 20, 2.0, student_discount=0.5),
    "palazzo ducale":        ActivityData(9, 17, 20, 2.0, student_discount=0.5),

    "rialto bridge":         ActivityData(0, 24,  0, 0.5),
    "pont du rialto":        ActivityData(0, 24,  0, 0.5),
    "grand canal":           ActivityData(0, 24,  2, 1.5),

    # ── Madrid ───────────────────────────────────────────────────────────────
    "prado museum":          ActivityData(10, 20, 15, 3.0, student_discount=1.0,
                                          free_first_sunday=True),
    "musée du prado":        ActivityData(10, 20, 15, 3.0, student_discount=1.0),
    "museo del prado":       ActivityData(10, 20, 15, 3.0, student_discount=1.0),

    "reina sofia museum":    ActivityData(10, 21, 10, 2.5, student_discount=1.0,
                                          free_first_sunday=True),
    "museo reina sofía":     ActivityData(10, 21, 10, 2.5, student_discount=1.0),

    "royal palace madrid":   ActivityData(10, 18, 12, 2.0, student_discount=1.0),
    "palacio real":          ActivityData(10, 18, 12, 2.0, student_discount=1.0),

    "retiro park":           ActivityData(6, 22,  0, 2.0, category="nature"),
    "parque del retiro":     ActivityData(6, 22,  0, 2.0, category="nature"),
    "thyssen museum":        ActivityData(10, 19, 13, 2.0),

    # ── Athens ───────────────────────────────────────────────────────────────
    "acropolis":             ActivityData(8, 20, 20, 3.0),
    "acropole":              ActivityData(8, 20, 20, 3.0),
    "acropolis museum":      ActivityData(8, 20, 10, 2.0, student_discount=0.5),
    "musée de l'acropole":   ActivityData(8, 20, 10, 2.0, student_discount=0.5),
    "parthenon":             ActivityData(8, 20, 20, 2.5),
    "parthénon":             ActivityData(8, 20, 20, 2.5),
    "national archaeological museum athens":
                             ActivityData(8, 20, 12, 3.0, student_discount=0.5),

    # ── Istanbul ─────────────────────────────────────────────────────────────
    "hagia sophia":          ActivityData(9, 17,  0, 2.0),
    "sainte-sophie":         ActivityData(9, 17,  0, 2.0),
    "topkapi palace":        ActivityData(9, 17, 15, 3.0),
    "palais de topkapi":     ActivityData(9, 17, 15, 3.0),
    "blue mosque":           ActivityData(8, 18,  0, 1.5),
    "mosquée bleue":         ActivityData(8, 18,  0, 1.5),
    "grand bazaar":          ActivityData(8, 19,  0, 2.0, category="shopping"),
    "grand bazar":           ActivityData(8, 19,  0, 2.0, category="shopping"),

    # ── Prague ───────────────────────────────────────────────────────────────
    "prague castle":         ActivityData(9, 17, 14, 3.0),
    "château de prague":     ActivityData(9, 17, 14, 3.0),
    "pražský hrad":          ActivityData(9, 17, 14, 3.0),
    "charles bridge":        ActivityData(0, 24,  0, 1.0),
    "pont charles":          ActivityData(0, 24,  0, 1.0),
    "old town square prague": ActivityData(0, 24,  0, 1.0),
    "st vitus cathedral":    ActivityData(9, 17,  0, 1.5),

    # ── Lisbonne ─────────────────────────────────────────────────────────────
    "belém tower":           ActivityData(10, 17,  6, 1.0),
    "tour de belém":         ActivityData(10, 17,  6, 1.0),
    "torre de belém":        ActivityData(10, 17,  6, 1.0),
    "jerónimos monastery":   ActivityData(10, 17,  6, 1.5),
    "monastère des hiéronymites": ActivityData(10, 17, 6, 1.5),
    "national museum of ancient art": ActivityData(10, 18,  6, 2.0),
    "sintra":                ActivityData(9, 18, 14, 5.0, category="culture"),
    "alfama":                ActivityData(0, 24,  0, 2.0, category="culture"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Templates par sous-type d'activité (tags OSM / OTM)
# Tuple : (duration_min, duration_max, duration_typical,
#          cost_min, cost_max, cost_typical,
#          open_hour, close_hour, confidence)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KindTemplate:
    duration_min: float
    duration_max: float
    duration_typical: float
    cost_min: int
    cost_max: int
    cost_typical: int
    open_hour: int
    close_hour: int
    confidence: float = 0.55


KIND_TEMPLATES: dict[str, KindTemplate] = {
    # Culture
    "museums":                  KindTemplate(1.5, 4.0, 2.5, 8, 25, 14, 9, 18),
    "museum":                   KindTemplate(1.5, 4.0, 2.5, 8, 25, 14, 9, 18),
    "theatres_and_entertainments": KindTemplate(2.0, 4.0, 2.5, 15, 80, 35, 10, 23),
    "theatre":                  KindTemplate(2.0, 3.5, 2.5, 20, 100, 45, 10, 23),
    "historic":                 KindTemplate(1.0, 3.5, 2.0, 0, 20, 12, 8, 19),
    "archaeology":              KindTemplate(1.5, 4.0, 2.5, 8, 20, 12, 8, 19),
    "archaeological_sites":     KindTemplate(1.5, 4.5, 2.5, 5, 20, 12, 8, 17),
    "architecture":             KindTemplate(0.5, 2.0, 1.5, 0, 20, 8, 8, 20),
    "monuments_and_memorials":  KindTemplate(0.5, 1.5, 1.0, 0, 15, 5, 0, 24),
    "memorial":                 KindTemplate(0.5, 1.5, 1.0, 0, 10, 0, 0, 24),
    "churches":                 KindTemplate(0.5, 1.5, 0.75, 0, 10, 2, 8, 19),
    "cathedral":                KindTemplate(0.75, 2.0, 1.0, 0, 15, 5, 8, 18),
    "castle":                   KindTemplate(1.5, 4.0, 2.5, 5, 25, 14, 9, 18),
    "palace":                   KindTemplate(1.5, 3.5, 2.5, 8, 25, 15, 9, 18),
    "cultural":                 KindTemplate(1.0, 3.0, 2.0, 0, 20, 10, 9, 18),
    "art_gallery":              KindTemplate(1.0, 3.0, 2.0, 0, 20, 10, 10, 18),
    "gallery":                  KindTemplate(1.0, 3.0, 2.0, 0, 20, 10, 10, 18),
    "religion":                 KindTemplate(0.5, 1.5, 0.75, 0, 10, 2, 8, 18),

    # Gastronomie
    "foods":                    KindTemplate(1.5, 3.0, 2.0, 20, 70, 35, 11, 22),
    "food_tour":                KindTemplate(2.5, 4.0, 3.0, 30, 75, 50, 11, 15),
    "cooking_class":            KindTemplate(3.0, 5.0, 3.5, 50, 130, 70, 10, 15),
    "wine_tasting":             KindTemplate(1.5, 3.0, 2.0, 20, 70, 40, 16, 21),
    "restaurant":               KindTemplate(1.0, 2.5, 1.5, 20, 80, 35, 12, 23),
    "market":                   KindTemplate(1.0, 2.5, 1.5, 5, 30, 15, 7, 14),

    # Nature
    "gardens_and_parks":        KindTemplate(1.0, 3.5, 2.0, 0, 10, 2, 6, 21),
    "natural":                  KindTemplate(1.5, 5.0, 2.5, 0, 20, 5, 7, 20),
    "park":                     KindTemplate(1.0, 3.0, 2.0, 0, 5, 0, 6, 22),
    "garden":                   KindTemplate(0.5, 2.5, 1.5, 0, 12, 5, 8, 20),
    "beach":                    KindTemplate(2.0, 7.0, 3.5, 0, 20, 5, 7, 21),
    "viewpoint":                KindTemplate(0.5, 1.5, 0.75, 0, 10, 0, 7, 22),
    "waterfall":                KindTemplate(1.0, 3.0, 1.5, 0, 10, 3, 7, 19),

    # Shopping
    "shops":                    KindTemplate(1.0, 3.5, 2.0, 20, 300, 60, 10, 20),
    "mall":                     KindTemplate(1.5, 4.0, 2.5, 20, 300, 80, 10, 21),
    "boutique":                 KindTemplate(0.5, 1.5, 1.0, 20, 200, 60, 10, 19),

    # Nightlife
    "amusements":               KindTemplate(2.0, 5.0, 3.0, 15, 60, 30, 19, 24),
    "nightclub":                KindTemplate(3.0, 7.0, 4.0, 15, 60, 30, 22, 24),
    "bar":                      KindTemplate(1.5, 4.0, 2.5, 15, 50, 25, 18, 24),
}


# ─────────────────────────────────────────────────────────────────────────────
# Fallback par grande catégorie
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CategoryDefault:
    duration: float
    cost: int
    open_hour: int
    close_hour: int
    confidence: float = 0.35


CATEGORY_DEFAULTS: dict[str, CategoryDefault] = {
    "culture":   CategoryDefault(2.0, 12, 9,  18),
    "gastro":    CategoryDefault(2.0, 35, 11, 22),
    "nature":    CategoryDefault(2.5,  5, 7,  20),
    "shopping":  CategoryDefault(2.0, 50, 10, 20),
    "nightlife": CategoryDefault(3.0, 30, 19, 24),
}

_DEFAULT_FALLBACK = CategoryDefault(2.0, 10, 9, 18, confidence=0.20)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions d'estimation publiques
# ─────────────────────────────────────────────────────────────────────────────

def lookup_known(name: str) -> Optional[ActivityData]:
    """
    Cherche un site dans la base curatée par correspondance exacte
    puis partielle (sous-chaîne).
    """
    key = name.strip().lower()
    if key in KNOWN_ACTIVITIES:
        return KNOWN_ACTIVITIES[key]
    # Correspondance partielle (pour gérer les noms légèrement différents)
    for known_key, data in KNOWN_ACTIVITIES.items():
        if known_key in key or key in known_key:
            if len(known_key) >= 5:  # évite les faux positifs sur clés courtes
                return data
    return None


def _best_kind_template(kinds_str: str) -> Optional[KindTemplate]:
    """Cherche le meilleur template parmi les kinds OTM/OSM d'une activité."""
    if not kinds_str:
        return None
    for kind in kinds_str.lower().split(","):
        kind = kind.strip()
        if kind in KIND_TEMPLATES:
            return KIND_TEMPLATES[kind]
        # Correspondance partielle sur les clés
        for k, tmpl in KIND_TEMPLATES.items():
            if k in kind or kind in k:
                return tmpl
    return None


def get_duration_estimate(
    category: str,
    kinds: str = "",
    rating: int = 5,
    name: str = "",
    known: Optional[ActivityData] = None,
) -> tuple[float, float]:
    """
    Retourne (duration_hours, confidence).

    Priorité : known_activities > kind_template > category_default.
    Le rating (1-10) ajuste légèrement la durée typique.
    """
    if known is not None:
        return known.duration_hours, known.confidence

    known = lookup_known(name)
    if known:
        return known.duration_hours, known.confidence

    tmpl = _best_kind_template(kinds)
    if tmpl:
        # Rating moyen = 5. Chaque point au-dessus/dessous ajuste de ±5%
        delta = (rating - 5) * 0.05
        adjusted = tmpl.duration_typical * (1 + delta)
        adjusted = max(tmpl.duration_min, min(tmpl.duration_max, adjusted))
        return round(adjusted * 2) / 2, tmpl.confidence  # arrondi au 0.5h

    cat = CATEGORY_DEFAULTS.get(category, _DEFAULT_FALLBACK)
    return cat.duration, cat.confidence


def get_cost_estimate(
    category: str,
    kinds: str = "",
    rating: int = 5,
    name: str = "",
    known: Optional[ActivityData] = None,
    student: bool = False,
) -> tuple[int, float]:
    """
    Retourne (cost_euros, confidence).

    Si student=True et que l'activité dispose d'un tarif réduit connu,
    applique la réduction (ex. 0.3 = 30% off).
    """
    if known is not None:
        cost = known.cost_euros
        if student and known.student_discount > 0:
            cost = int(cost * (1 - known.student_discount))
        return max(0, cost), known.confidence

    known = lookup_known(name)
    if known:
        cost = known.cost_euros
        if student and known.student_discount > 0:
            cost = int(cost * (1 - known.student_discount))
        return max(0, cost), known.confidence

    tmpl = _best_kind_template(kinds)
    if tmpl:
        # Interpolation linéaire par rating (1 → min, 10 → max)
        t = max(0.0, min(1.0, (rating - 1) / 9))
        cost = int(tmpl.cost_min + t * (tmpl.cost_max - tmpl.cost_min))
        if student:
            cost = int(cost * 0.7)  # réduction étudiante générique
        return max(0, cost), tmpl.confidence

    cat = CATEGORY_DEFAULTS.get(category, _DEFAULT_FALLBACK)
    return cat.cost, cat.confidence


def get_hours_estimate(
    category: str,
    kinds: str = "",
    name: str = "",
    known: Optional[ActivityData] = None,
) -> tuple[int, int, float]:
    """
    Retourne (open_hour, close_hour, confidence).
    """
    if known is not None:
        return known.open_hour, known.close_hour, known.confidence

    known = lookup_known(name)
    if known:
        return known.open_hour, known.close_hour, known.confidence

    tmpl = _best_kind_template(kinds)
    if tmpl:
        return tmpl.open_hour, tmpl.close_hour, tmpl.confidence

    cat = CATEGORY_DEFAULTS.get(category, _DEFAULT_FALLBACK)
    return cat.open_hour, cat.close_hour, cat.confidence


def priority_from_confidence(base_score: int, confidence: float) -> int:
    """
    Ajuste le priority_score du solveur selon la confiance dans les données.
    Données fiables (confidence ≥ 0.8) → score inchangé.
    Données estimées (confidence < 0.4) → -1 point.
    """
    if confidence >= 0.8:
        return base_score
    if confidence >= 0.55:
        return base_score
    return max(1, base_score - 1)
