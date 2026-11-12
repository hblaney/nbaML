"""Shin and Power two-way / multi-way devigging (Strumbelj 2014)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from nba_betting.pricing.kelly import american_to_decimal


def devig_shin(decimal_odds: Sequence[float]) -> np.ndarray:
    """Remove bookmaker margin using Shin's insider-trading model."""
    d = np.asarray(decimal_odds, dtype=float)
    if d.ndim != 1 or d.size < 2 or np.any(d <= 1.0):
        raise ValueError("decimal_odds must be a 1D sequence with values > 1.0")
    q = 1.0 / d
    booksum = float(q.sum())
    if booksum <= 1.0 + 1e-9:
        return q / booksum
    lo, hi = 0.0, 0.999
    for _ in range(40):
        z = 0.5 * (lo + hi)
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * (q * q) / booksum) - z) / (2.0 * (1.0 - z))
        if float(p.sum()) > 1.0:
            lo = z
        else:
            hi = z
    return p / float(p.sum())


def devig_shin_american(price_a: float, price_b: float) -> tuple[float, float]:
    """Two-way Shin devigging directly from American odds."""
    probs = devig_shin([american_to_decimal(price_a), american_to_decimal(price_b)])
    return float(probs[0]), float(probs[1])


def devig_power(decimal_odds: Sequence[float]) -> np.ndarray:
    """Remove bookmaker margin via the power method ``sum(q_i ** k) = 1``."""
    d = np.asarray(decimal_odds, dtype=float)
    if d.ndim != 1 or d.size < 2 or np.any(d <= 1.0):
        raise ValueError("decimal_odds must be a 1D sequence with values > 1.0")
    q = 1.0 / d
    if float(q.sum()) <= 1.0 + 1e-9:
        return q / float(q.sum())
    k = brentq(lambda exp_k: float(np.sum(q ** exp_k) - 1.0), 1.0, 50.0)
    p = q ** float(k)
    return p / float(p.sum())
