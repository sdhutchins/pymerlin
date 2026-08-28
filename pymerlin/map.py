"""Genetic map utilities."""

from __future__ import annotations

import math
from decimal import Decimal


def map_distance_cm(left_position_cm: float, right_position_cm: float) -> float:
    """Return the intended decimal distance between two map coordinates."""

    left_position = Decimal(str(left_position_cm))
    right_position = Decimal(str(right_position_cm))
    return float(abs(right_position - left_position))


def haldane_recombination_fraction(distance_cm: float) -> float:
    """Convert cM distance to recombination fraction using Merlin's Haldane map."""

    distance_morgans = abs(distance_cm) / 100.0
    return -0.5 * math.expm1(-2.0 * distance_morgans)
