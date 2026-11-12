"""Rigorous vig-removal (devigging) methods for two-way and multi-way markets.

Implements five classic and modern sportsbook margin-removal models:

1. ``multiplicative`` (proportional / basic): ``p_i = pi_i / sum(pi)``.
   Assumes identical proportional hold on favorites and longshots; systematically
   overestimates longshot win probabilities when books shade underdogs.
2. ``additive`` (equal-margin / simplex projection): ``p_i = pi_i - (S - 1) / N``
   projected onto the probability simplex so extreme longshots stay strictly positive.
3. ``power`` (logarithmic / odds-ratio exponent): ``p_i = pi_i ** k`` where
   ``k > 0`` solves ``sum(pi_i ** k) = 1``. For ``S > 1``, ``k > 1``, which
   concentrates margin removal on longshots and corrects favorite-longshot bias.
4. ``shin`` (Shin 1991/1992/1993 insider-trading equilibrium): solves for latent
   insider volume share ``z`` such that
   ``sum((sqrt(z^2 + 4*(1 - z)*pi_i^2 / S) - z) / (2*(1 - z))) = 1``.
5. ``probit`` (normal-quantile / Gaussian latent shift): ``p_i = Phi(Phi^-1(pi_i) - c)``
   where ``c`` solves ``sum(Phi(Phi^-1(pi_i) - c)) = 1``. Log-concavity of ``Phi``
   guarantees longshots are penalized relative to multiplicative devig when ``S > 1``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from nba_betting.pricing.american import american_to_implied

DevigMethod = Literal["multiplicative", "additive", "power", "shin", "probit"]

SUPPORTED_DEVIG_METHODS: tuple[DevigMethod, ...] = (
    "multiplicative",
    "additive",
    "power",
    "shin",
    "probit",
)

_METHOD_ALIASES: dict[str, DevigMethod] = {
    "multiplicative": "multiplicative",
    "mult": "multiplicative",
    "proportional": "multiplicative",
    "basic": "multiplicative",
    "additive": "additive",
    "add": "additive",
    "equal": "additive",
    "power": "power",
    "pow": "power",
    "logarithmic": "power",
    "shin": "shin",
    "probit": "probit",
    "normal": "probit",
    "gaussian": "probit",
}

_PROB_FLOOR = 1e-9


def normalize_devig_method(method: str) -> DevigMethod:
    key = str(method).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unsupported devig method {method!r}; expected one of {SUPPORTED_DEVIG_METHODS}"
        )
    return _METHOD_ALIASES[key]


def _validate_implied(implied_probs: Sequence[float]) -> np.ndarray:
    arr = np.asarray(implied_probs, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("implied_probs must be a 1-D sequence with at least 2 outcomes")
    if not np.all(np.isfinite(arr)):
        raise ValueError("implied_probs must all be finite")
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError(f"each implied probability must be in (0, 1), got {arr.tolist()}")
    return arr


def _project_simplex_bounded(v: np.ndarray, *, floor: float = _PROB_FLOOR) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``{p : sum(p) = 1, p_i >= floor}``.

    Matches exact ``v_i - (sum(v) - 1) / N`` whenever all shifted entries exceed
    ``floor``; otherwise water-fills active coordinates cleanly.
    """
    n = v.size
    if n * floor >= 1.0:
        raise ValueError("floor too large for simplex dimension")
    # Shift by floor so we project onto standard non-negative simplex of mass (1 - n*floor).
    target = 1.0 - n * floor
    y = v - floor
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - target
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    p = np.maximum(y - theta, 0.0) + floor
    return p / p.sum()


def power_exponent(implied_probs: Sequence[float]) -> float:
    """Solve ``sum(pi_i ** k) = 1`` for ``k > 0``."""
    pi = _validate_implied(implied_probs)
    s = float(pi.sum())
    if abs(s - 1.0) <= 1e-12:
        return 1.0
    log_pi = np.log(pi)

    def obj(k: float) -> float:
        # logsumexp(k * log(pi)) == 0  <=>  sum(pi**k) == 1
        return float(logsumexp(k * log_pi))

    lo, hi = (1.0, 4.0) if s > 1.0 else (0.25, 1.0)
    while obj(lo) < 0.0:
        lo *= 0.5
    while obj(hi) > 0.0:
        hi *= 2.0
    return float(brentq(obj, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200))


def _shin_probs_for_z(pi: np.ndarray, z: float, s: float) -> np.ndarray:
    disc = np.maximum(z * z + 4.0 * (1.0 - z) * (pi * pi) / s, 0.0)
    return (np.sqrt(disc) - z) / (2.0 * (1.0 - z))


def shin_insider_fraction(implied_probs: Sequence[float]) -> float:
    """Solve Shin (1993) equilibrium insider trading share ``z``.

    For standard overround books (``sum(pi) > 1``), ``z`` lies strictly in ``(0, 1)``.
    For zero-vig books (``sum(pi) == 1``), ``z = 0``. For synthetic negative-hold
    arb books (``sum(pi) < 1``), returns the smooth negative root ``z < 0``.
    """
    pi = _validate_implied(implied_probs)
    s = float(pi.sum())
    if abs(s - 1.0) <= 1e-12:
        return 0.0

    def obj(z: float) -> float:
        return float(_shin_probs_for_z(pi, z, s).sum() - 1.0)

    if s > 1.0:
        return float(brentq(obj, 0.0, 1.0 - 1e-12, xtol=1e-13, rtol=1e-13, maxiter=200))

    lo = -0.5
    while obj(lo) < 0.0:
        lo *= 2.0
    return float(brentq(obj, lo, 0.0, xtol=1e-13, rtol=1e-13, maxiter=200))


def probit_shift(implied_probs: Sequence[float]) -> float:
    """Solve ``sum(Phi(Phi^-1(pi_i) - c)) = 1`` for latent Gaussian margin shift ``c``."""
    pi = _validate_implied(implied_probs)
    s = float(pi.sum())
    if abs(s - 1.0) <= 1e-12:
        return 0.0
    q = norm.ppf(pi)

    def obj(c: float) -> float:
        return float(norm.cdf(q - c).sum() - 1.0)

    lo, hi = (0.0, 1.0) if s > 1.0 else (-1.0, 0.0)
    while obj(lo) < 0.0:
        lo -= 1.0
    while obj(hi) > 0.0:
        hi += 1.0
    return float(brentq(obj, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200))


def devig_probabilities(
    implied_probs: Sequence[float],
    *,
    method: str = "shin",
) -> tuple[float, ...]:
    """Remove bookmaker margin from raw implied probabilities ``(pi_1, ..., pi_N)``."""
    pi = _validate_implied(implied_probs)
    canon = normalize_devig_method(method)
    s = float(pi.sum())
    if abs(s - 1.0) <= 1e-12:
        return tuple(float(x) for x in (pi / s))

    if canon == "multiplicative":
        out = pi / s
    elif canon == "additive":
        out = _project_simplex_bounded(pi - (s - 1.0) / float(pi.size))
    elif canon == "power":
        k = power_exponent(pi)
        log_p = k * np.log(pi)
        log_p -= logsumexp(log_p)
        out = np.exp(log_p)
    elif canon == "shin":
        z = shin_insider_fraction(pi)
        out = _shin_probs_for_z(pi, z, s)
        out = np.clip(out, _PROB_FLOOR, 1.0)
        out = out / out.sum()
    elif canon == "probit":
        c = probit_shift(pi)
        out = norm.cdf(norm.ppf(pi) - c)
        out = np.clip(out, _PROB_FLOOR, 1.0)
        out = out / out.sum()
    else:  # pragma: no cover - guarded by normalize_devig_method
        raise ValueError(f"Unhandled method {canon!r}")

    return tuple(float(x) for x in out)


def devig_decimal(
    decimal_odds: Sequence[float],
    *,
    method: str = "shin",
) -> tuple[float, ...]:
    """Remove vig from decimal odds ``(d_1, ..., d_N)`` where each ``d_i > 1``."""
    dec = np.asarray(decimal_odds, dtype=float)
    if dec.ndim != 1 or dec.size < 2 or not np.all(np.isfinite(dec)) or np.any(dec <= 1.0):
        raise ValueError(f"decimal_odds must be finite and > 1.0, got {decimal_odds}")
    return devig_probabilities(1.0 / dec, method=method)


def devig_american(
    american_prices: Sequence[float],
    *,
    method: str = "shin",
) -> tuple[float, ...]:
    """Remove vig from American odds ``(a_1, ..., a_N)``."""
    implied = [american_to_implied(float(p)) for p in american_prices]
    return devig_probabilities(implied, method=method)


def compare_devig_methods(
    american_prices: Sequence[float],
) -> dict[DevigMethod, tuple[float, ...]]:
    """Evaluate all five devigging methods on the same American odds vector."""
    implied = [american_to_implied(float(p)) for p in american_prices]
    return {m: devig_probabilities(implied, method=m) for m in SUPPORTED_DEVIG_METHODS}
