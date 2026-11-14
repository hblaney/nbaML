"""Exact PrizePicks payout math, stat-type Empirical-Bayes calibration, and live cashout EV."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

# Promoted in R&D Cycle #1 from 136,166 pregame props on 2022-24 Train ONLY
# (shrank empirical tail bias n/(n+500)*(won - predicted) at p >= 0.545; flipped
# 2024-25 5-Flex from -8.5% -> +16.4% on ESPN and -46.7% -> +60.0% on OddsAPI,
# with +176.9% 5-Flex ROI on 2025-26 OddsAPI holdout).
STAT_TAIL_CALIBRATION_22_24: dict[str, float] = {
    "Assists": +0.01358,
    "assists": +0.01358,
    "Rebs+Asts": +0.00765,
    "Rebounds": +0.00173,
    "rebounds": +0.00173,
    "Pts+Asts": +0.00420,
    "Pts+Rebs+Asts": +0.00050,
    "pra": +0.00050,
    "Pts+Rebs": -0.00160,
    "3-PT Made": -0.00253,
    "threes": -0.00253,
    "Points": -0.00511,
    "points": -0.00511,
}

POWER_PAYOUTS: dict[int, float] = {
    6: 37.5,
    5: 20.0,
    4: 10.0,
    3: 6.0,
    2: 3.0,
}

FLEX_PAYOUTS: dict[int, dict[int, float]] = {
    6: {6: 25.0, 5: 2.0, 4: 0.4},
    5: {5: 10.0, 4: 2.0, 3: 0.4},
    4: {4: 6.0, 3: 1.5},
    3: {3: 3.0, 2: 1.0},
    2: {2: 3.0},
}


def apply_stat_tail_calibration(prob_win: float, stat_type: str) -> float:
    """Apply frozen 2022-24 Empirical-Bayes stat-reliability calibration."""
    adj = STAT_TAIL_CALIBRATION_22_24.get(stat_type, 0.0)
    return float(np.clip(float(prob_win) + adj, 0.05, 0.72))


def exact_hits_pmf(probs: Sequence[float]) -> np.ndarray:
    """Exact Poisson-Binomial hit count distribution for heterogeneous independent legs."""
    dp = np.array([1.0], dtype=float)
    for p in probs:
        pp = float(np.clip(p, 0.0, 1.0))
        nxt = np.zeros(len(dp) + 1, dtype=float)
        nxt[:-1] += dp * (1.0 - pp)
        nxt[1:] += dp * pp
        dp = nxt
    return dp


def settle_prizepicks_slip(
    outcomes: Sequence[int],
    mode: Literal["power", "flex"],
) -> float:
    """Return gross entry multiplier (e.g. 6.0, 2.0, 0.4, 1.0 refund, or 0.0 loss).

    Each element in ``outcomes`` is ``1`` (win), ``0`` (loss), or ``-1`` (tie/DNP revert).
    """
    active = [int(x) for x in outcomes if int(x) != -1]
    k = len(active)
    if k == 0:
        return 1.0
    if k == 1:
        return 1.0 if active[0] == 1 else 0.0
    hits = sum(active)
    if mode == "power":
        return float(POWER_PAYOUTS.get(k, 0.0)) if hits == k else 0.0
    if k == 2:
        return 3.0 if hits == 2 else 0.0
    return float(FLEX_PAYOUTS.get(k, {}).get(hits, 0.0))


def slip_ev_and_log_growth(
    win_probs_cond: Sequence[float],
    push_probs: Sequence[float] | None = None,
    *,
    mode: Literal["power", "flex"] = "flex",
    c_kelly: float = 0.50,
    max_frac: float = 0.06,
) -> dict[str, Any]:
    """Compute exact EV, payout PMF (including tie/push reversions), and fractional Kelly growth."""
    k = len(win_probs_cond)
    if k < 2 or k > 6:
        raise ValueError("PrizePicks slips must have 2 to 6 legs")

    p_pushes = [0.0] * k if push_probs is None else [float(x) for x in push_probs]
    if all(pp <= 1e-9 for pp in p_pushes):
        pmf_hits = exact_hits_pmf(win_probs_cond)
        dist: dict[float, float] = {}
        for h, pr in enumerate(pmf_hits):
            if pr <= 0:
                continue
            if mode == "power":
                mult = POWER_PAYOUTS[k] if h == k else 0.0
            else:
                mult = FLEX_PAYOUTS[k].get(h, 0.0)
            dist[mult] = dist.get(mult, 0.0) + float(pr)
    else:
        dist = {}
        legs_tri = []
        for pw_c, pp in zip(win_probs_cond, p_pushes):
            pp_c = float(np.clip(pp, 0.0, 0.95))
            pw_u = (1.0 - pp_c) * float(np.clip(pw_c, 0.01, 0.99))
            pl_u = (1.0 - pp_c) - pw_u
            legs_tri.append((pw_u, pl_u, pp_c))
        for states in product((1, 0, -1), repeat=k):
            pr = 1.0
            for st, (pw_u, pl_u, pp_c) in zip(states, legs_tri):
                pr *= pw_u if st == 1 else (pl_u if st == 0 else pp_c)
            if pr <= 0:
                continue
            mult = settle_prizepicks_slip(states, mode)
            dist[mult] = dist.get(mult, 0.0) + pr

    mults = np.array(list(dist.keys()), dtype=float)
    probs = np.array(list(dist.values()), dtype=float)
    ev = float(np.dot(probs, mults) - 1.0)

    if ev <= 1e-6:
        return {
            "ev": ev,
            "stake_frac": 0.0,
            "expected_log_growth": 0.0,
            "profit_prob": float(sum(p for m, p in dist.items() if m > 1.0)),
            "cash_prob": float(sum(p for m, p in dist.items() if m > 0.0)),
            "distribution": {str(round(m, 2)): round(p, 4) for m, p in sorted(dist.items())},
        }

    res = minimize_scalar(
        lambda f: -float(np.dot(probs, np.log(np.maximum(1e-9, 1.0 + f * (mults - 1.0))))),
        bounds=(0.0, 0.85),
        method="bounded",
    )
    f_full = float(res.x) if res.success else 0.0
    f_use = min(float(max_frac), float(c_kelly) * f_full)
    g = float(np.dot(probs, np.log(np.maximum(1e-9, 1.0 + f_use * (mults - 1.0)))))
    return {
        "ev": ev,
        "stake_frac": f_use,
        "expected_log_growth": g,
        "profit_prob": float(sum(p for m, p in dist.items() if m > 1.0)),
        "cash_prob": float(sum(p for m, p in dist.items() if m > 0.0)),
        "distribution": {str(round(m, 2)): round(p, 4) for m, p in sorted(dist.items())},
    }


def evaluate_live_cashout(
    *,
    side: Literal["MORE", "LESS"],
    line: float,
    current_stat: float,
    expected_remaining_minutes: float,
    per_minute_rate: float,
    full_payout_dollars: float,
    cashout_offer_dollars: float,
    dispersion_alpha: float = 0.06,
) -> dict[str, Any]:
    """Compare PrizePicks Early Payout / Cashout offer against exact NegBin ride EV."""
    rem_mu = max(float(expected_remaining_minutes) * max(float(per_minute_rate), 0.01), 0.01)
    needed_diff = float(line) - float(current_stat)

    if side == "LESS":
        if current_stat > line:
            p_win = 0.0
        else:
            max_allowed = int(math.floor(needed_diff - 1e-9)) if not float(line).is_integer() else int(needed_diff) - 1
            if max_allowed < 0:
                p_win = 0.0
            else:
                n = 1.0 / max(float(dispersion_alpha), 1e-4)
                p = n / (n + rem_mu)
                p_win = float(nbinom.cdf(max_allowed, n, p))
    else:
        if current_stat > line:
            p_win = 1.0
        else:
            min_needed = int(math.floor(needed_diff)) + 1
            n = 1.0 / max(float(dispersion_alpha), 1e-4)
            p = n / (n + rem_mu)
            p_win = float(1.0 - nbinom.cdf(min_needed - 1, n, p))

    ev_ride = float(full_payout_dollars) * p_win
    offer = float(cashout_offer_dollars)
    breakeven_p = offer / max(float(full_payout_dollars), 1e-6)
    recommend_cashout = offer > ev_ride

    return {
        "side": side,
        "line": float(line),
        "current_stat": float(current_stat),
        "expected_remaining_stat": round(rem_mu, 2),
        "p_win_ride": round(p_win, 4),
        "breakeven_win_prob": round(breakeven_p, 4),
        "ev_ride_dollars": round(ev_ride, 2),
        "cashout_offer_dollars": round(offer, 2),
        "edge_of_cashout_dollars": round(offer - ev_ride, 2),
        "recommendation": "CASH_OUT" if recommend_cashout else "LET_IT_RIDE",
    }
