"""American odds ↔ implied probability helpers."""

from __future__ import annotations


def american_to_implied(price: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if price == 0:
        raise ValueError("American odds cannot be 0")
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def implied_to_american(prob: float) -> float:
    """Convert a fair probability to American odds."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be in (0,1), got {prob}")
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def devig_two_way(price_a: float, price_b: float) -> tuple[float, float]:
    """Remove vig from a two-way market; returns (fair_prob_a, fair_prob_b)."""
    raw_a = american_to_implied(price_a)
    raw_b = american_to_implied(price_b)
    total = raw_a + raw_b
    if total <= 0:
        raise ValueError("invalid implied probabilities")
    return raw_a / total, raw_b / total
