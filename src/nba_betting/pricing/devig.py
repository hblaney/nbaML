"""Shin and Power devigging methods for two-way and multi-way sportsbook markets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from nba_betting.pricing.kelly import american_to_decimal


def devig_shin(decimal_odds: Sequence[float]) -> np.ndarray:
    """De-vig decimal odds using Shin's (1993) / Štrumbelj (2014) insider-trading model."""
    dec = np.asarray(decimal_odds, dtype=float)
    if dec.ndim != 1 or dec.size < 2 or np.any(dec <= 1.0):
        raise ValueError(f"decimal_odds must be 1D with >=2 entries > 1.0, got {decimal_odds}")
    q = 1.0 / dec
    q_sum = float(q.sum())
    if q_sum <= 1.0 + 1e-9:
        return q / q_sum
    lo, hi = 0.0, 0.999
    p = q / q_sum
    for _ in range(45):
        z = 0.5 * (lo + hi)
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * (q * q) / q_sum) - z) / (2.0 * (1.0 - z))
        if float(p.sum()) > 1.0:
            lo = z
        else:
            hi = z
    return p / float(p.sum())


def devig_shin_two_way(price_a: float, price_b: float) -> tuple[float, float]:
    """Two-way Shin de-vigging from American odds; returns ``(p_a, p_b)``."""
    da = american_to_decimal(float(price_a))
    db = american_to_decimal(float(price_b))
    probs = devig_shin([da, db])
    return float(probs[0]), float(probs[1])


def devig_power(decimal_odds: Sequence[float]) -> np.ndarray:
    """Power method de-vigging solving ``sum(q_i ** k) == 1``."""
    dec = np.asarray(decimal_odds, dtype=float)
    if dec.ndim != 1 or dec.size < 2 or np.any(dec <= 1.0):
        raise ValueError(f"decimal_odds must be 1D with >=2 entries > 1.0, got {decimal_odds}")
    q = 1.0 / dec
    if float(q.sum()) <= 1.0 + 1e-9:
        return q / float(q.sum())
    k = float(brentq(lambda exp_k: float(np.sum(q ** exp_k)) - 1.0, 1.0, 60.0))
    p = q ** k
    return p / float(p.sum())
