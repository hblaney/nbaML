"""Slate-level Quadratic Fractional Kelly & Sharpe-Optimal portfolio sizing.

Implements Hubacek et al. (2019, 2021) and Busseti, Ryu & Boyd (2016)
variance-penalized daily slate allocation across straight and parlay tickets.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize


def solve_slate_quadratic_kelly(
    probs: Sequence[float],
    decimals: Sequence[float],
    *,
    corr_matrix: np.ndarray | None = None,
    omega: float = 0.25,
    max_bet_frac: float = 0.03,
    max_slate_frac: float = 0.20,
) -> np.ndarray:
    """Maximize ``e^T f - (1 / (2 * omega)) * f^T Sigma f`` subject to box & slate caps.

    For distinct-game bets (``corr_matrix`` is None / identity), ``Sigma`` is
    diagonal with ``Sigma_kk = p_k * (1 - p_k) * d_k^2``.
    """
    p = np.asarray(probs, dtype=float)
    d = np.asarray(decimals, dtype=float)
    k = p.size
    if k == 0:
        return np.zeros(0, dtype=float)
    edges = p * d - 1.0
    active = edges > 0.0
    if not np.any(active):
        return np.zeros(k, dtype=float)

    std_ret = np.sqrt(np.maximum(1e-9, p * (1.0 - p))) * d
    if corr_matrix is None:
        # Exact closed-form diagonal solution + simplex cap projection
        f_uncapped = np.where(active, float(omega) * edges / np.maximum(1e-9, std_ret ** 2), 0.0)
        f_box = np.clip(f_uncapped, 0.0, float(max_bet_frac))
        tot = float(f_box.sum())
        if tot <= float(max_slate_frac):
            return f_box
        # Solve exact KKT water-filling lambda: f_i = clip(omega*(e_i - lam)/sigma_i^2, 0, max_bet_frac)
        lo, hi = 0.0, float(np.max(edges))
        for _ in range(40):
            lam = 0.5 * (lo + hi)
            cand = np.clip(float(omega) * (edges - lam) / np.maximum(1e-9, std_ret ** 2), 0.0, float(max_bet_frac))
            cand = np.where(active, cand, 0.0)
            if float(cand.sum()) > float(max_slate_frac):
                lo = lam
            else:
                hi = lam
        return cand

    cov = np.outer(std_ret, std_ret) * np.asarray(corr_matrix, dtype=float)
    inv_om = 1.0 / max(1e-6, float(omega))

    def obj(f: np.ndarray) -> tuple[float, np.ndarray]:
        val = -float(edges @ f) + 0.5 * inv_om * float(f @ cov @ f)
        grad = -edges + inv_om * (cov @ f)
        return val, grad

    bounds = [(0.0, float(max_bet_frac)) if active[i] else (0.0, 0.0) for i in range(k)]
    cons = [{"type": "ineq", "fun": lambda f: float(max_slate_frac) - float(np.sum(f)), "jac": lambda f: -np.ones(k)}]
    f0 = np.minimum(float(max_bet_frac), np.maximum(0.0, float(omega) * edges / np.maximum(1e-9, std_ret ** 2)))
    if float(f0.sum()) > float(max_slate_frac):
        f0 *= float(max_slate_frac) / float(f0.sum())
    res = minimize(obj, f0, method="SLSQP", jac=True, bounds=bounds, constraints=cons)
    return np.clip(res.x, 0.0, float(max_bet_frac))
