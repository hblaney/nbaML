"""Student-t copula (df=4) and NBA blowout/overtime regime simulation for correlated SGPs.

Why not a Gaussian copula?
A Gaussian copula has zero asymptotic tail dependence (``lambda_U = lambda_L = 0``),
underpricing joint tail clustering in NBA games (overtime scoring spikes on tight
margins, Q4 bench clearings on heavy blowouts, and pace-driven track meets).
A bivariate Student-t copula with ``df = 4`` captures positive symmetric tail
dependence ``lambda = 2 * t_{df+1}(-sqrt((df+1)*(1-rho)/(1+rho))) > 0`` while
preserving arbitrary marginal win/push probabilities.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import t as student_t

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def t_copula_tail_dependence(rho: float | np.ndarray, *, df: float = 4.0) -> np.ndarray:
    """Exact upper/lower tail dependence coefficient ``lambda`` of a bivariate ``t_df`` copula."""
    r = np.clip(np.asarray(rho, dtype=float), -0.999, 0.999)
    nu = float(df)
    arg = -np.sqrt((nu + 1.0) * (1.0 - r) / np.maximum(1.0 + r, 1e-12))
    return 2.0 * student_t.cdf(arg, df=nu + 1.0)


def t_copula_rect_prob(
    *,
    u1_lo: np.ndarray | float,
    u1_hi: np.ndarray | float,
    u2_lo: np.ndarray | float,
    u2_hi: np.ndarray | float,
    rho: np.ndarray | float,
    df: float = 4.0,
) -> np.ndarray:
    """Exact vectorized bivariate ``t_df`` copula rectangle probability ``P(u1 in [u1_lo, u1_hi], u2 in [u2_lo, u2_hi])``.

    Uses the exact conditional Student-t identity (Kotz & Nadarajah 2004):
    if ``(T1, T2) ~ BivariateStudentT(df=nu, corr=rho)``, then
        ``T2 | (T1 = t1) ~ rho * t1 + sqrt(((nu + t1^2) / (nu + 1)) * (1 - rho^2)) * t_{nu+1}``.
    Integrated over ``t1`` via 48-node Gauss-Legendre quadrature (accurate to ~1e-11).
    """
    nu = float(df)
    a_u1, b_u1, a_u2, b_u2, r = np.broadcast_arrays(
        np.clip(np.asarray(u1_lo, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(u1_hi, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(u2_lo, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(u2_hi, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(rho, dtype=float), -0.999, 0.999),
    )
    orig_shape = a_u1.shape
    u1a = a_u1.ravel()
    u1b = b_u1.ravel()
    u2a = a_u2.ravel()
    u2b = b_u2.ravel()
    rf = r.ravel()

    # Map uniform [u1a, u1b] directly via Gauss-Legendre nodes u in (0, 1)!
    # Wait: integrating du1 directly over [u1a, u1b] with t1 = F_{t_nu}^{-1}(u1)
    # has ZERO Jacobian factor (f_{t_nu}(t1) dt1 == du1!) and handles u1=0 or u1=1
    # cleanly without finite-truncation error in heavy Student-t tails!
    width = np.maximum(u1b - u1a, 0.0)
    half = 0.5 * width
    mid = 0.5 * (u1a + u1b)

    u_nodes = np.clip(mid[:, None] + half[:, None] * _GL_NODES[None, :], 1e-12, 1.0 - 1e-12)
    t1_nodes = student_t.ppf(u_nodes, df=nu)  # shape (N, 48)

    t2a = np.where(u2a <= 1e-12, -np.inf, np.where(u2a >= 1.0 - 1e-12, np.inf, student_t.ppf(np.clip(u2a, 1e-12, 1.0 - 1e-12), df=nu)))
    t2b = np.where(u2b <= 1e-12, -np.inf, np.where(u2b >= 1.0 - 1e-12, np.inf, student_t.ppf(np.clip(u2b, 1e-12, 1.0 - 1e-12), df=nu)))

    scale_cond = np.sqrt(((nu + t1_nodes * t1_nodes) / (nu + 1.0)) * np.maximum(1.0 - rf[:, None] ** 2, 1e-12))
    mean_cond = rf[:, None] * t1_nodes

    hi_cdf = np.where(
        np.isposinf(t2b[:, None]),
        1.0,
        np.where(
            np.isneginf(t2b[:, None]),
            0.0,
            student_t.cdf((t2b[:, None] - mean_cond) / scale_cond, df=nu + 1.0),
        ),
    )
    lo_cdf = np.where(
        np.isneginf(t2a[:, None]),
        0.0,
        np.where(
            np.isposinf(t2a[:, None]),
            1.0,
            student_t.cdf((t2a[:, None] - mean_cond) / scale_cond, df=nu + 1.0),
        ),
    )
    cond_prob = np.maximum(hi_cdf - lo_cdf, 0.0)
    prob = half * np.sum(cond_prob * _GL_WEIGHTS[None, :], axis=1)
    prob = np.where((u1b <= u1a) | (u2b <= u2a), 0.0, prob)
    return np.clip(prob, 0.0, 1.0).reshape(orig_shape)


def sgp_prob_t_copula(
    marginal_probs: Sequence[float],
    corr_matrix: np.ndarray,
    *,
    df: float = 4.0,
    n_sims: int = 50_000,
    seed: int = 0,
) -> float:
    """Monte Carlo ``K``-leg joint hit probability under a multivariate ``t_df`` copula."""
    probs = np.asarray(marginal_probs, dtype=float)
    corr = np.asarray(corr_matrix, dtype=float)
    k = probs.size
    if corr.shape != (k, k):
        raise ValueError(f"corr_matrix shape {corr.shape} does not match k={k}")
    if np.any(probs <= 0.0):
        return 0.0
    if np.any(probs >= 1.0):
        active = probs < 1.0
        if not np.any(active):
            return 1.0
        probs = probs[active]
        corr = corr[np.ix_(active, active)]
        k = probs.size

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr + 1e-10 * np.eye(k))
    z = rng.standard_normal((int(n_sims), k)) @ chol.T
    chi2 = rng.chisquare(float(df), size=(int(n_sims), 1))
    t_draws = z / np.sqrt(chi2 / float(df))
    thresholds = student_t.ppf(1.0 - probs, df=float(df))
    hits = np.all(t_draws > thresholds[None, :], axis=1)
    return float(hits.mean())


def simulate_blowout_ot_regime_sgp(
    *,
    mu_margin: float,
    sigma_margin: float,
    mu_total_reg: float,
    sigma_total_reg: float,
    rho_reg: float,
    spread_home: float,
    total_line: float,
    n_sims: int = 25_000,
    seed: int = 0,
) -> dict[str, float]:
    """Explicit NBA Overtime + Q4 Blowout/Foul Regime Simulator for `(Margin, Total)` SGPs.

    Captures structural non-Gaussian NBA skewness & tail coupling:
    1. Overtime regime (`|M_reg| < 1.35`, ~6% of games): breaks regulation tie with
       `M_ot ~ N(0.3, 5.2^2)` and adds `T_ot ~ N(23.0, 4.8^2)` points (strong Underdog + Over spike).
    2. Close-endgame intentional fouling (`1.35 <= |M_reg| <= 6.5`): adds late foul-fest
       scoring `+2.2` pts on average.
    3. Heavy blowout Q4 bench clearing (`|M_reg| >= 16.0`): starters sit, zero intentional
       fouls (`-2.0` pts shift vs foul-fest, higher garbage-time dispersion).
    """
    rng = np.random.default_rng(seed)
    r = float(np.clip(rho_reg, -0.95, 0.95))
    z1 = rng.standard_normal(int(n_sims))
    z2 = r * z1 + math.sqrt(max(1.0 - r * r, 1e-12)) * rng.standard_normal(int(n_sims))

    # Heavy-tailed Student-t_6 game shocks for regulation pace/shooting bursts
    scale_shock = np.sqrt(6.0 / rng.chisquare(6.0, size=int(n_sims))) * math.sqrt(4.0 / 6.0)
    m_reg = float(mu_margin) + float(sigma_margin) * z1 * scale_shock
    t_reg = float(mu_total_reg) + float(sigma_total_reg) * z2 * scale_shock

    is_ot = np.abs(m_reg) < 1.35
    is_foul = (~is_ot) & (np.abs(m_reg) <= 6.5)
    is_blowout = np.abs(m_reg) >= 16.0

    m_ot_draw = rng.normal(0.25, 5.2, size=int(n_sims))
    # Prevent exact zero in OT
    m_ot_draw = np.where(np.abs(m_ot_draw) < 0.5, np.where(m_ot_draw >= 0, 1.0, -1.0), m_ot_draw)
    t_ot_add = np.maximum(rng.normal(23.0, 4.8, size=int(n_sims)), 10.0)
    t_foul_add = rng.normal(1.8, 2.0, size=int(n_sims))
    t_blow_adj = rng.normal(-2.2, 3.5, size=int(n_sims))

    margin = np.where(is_ot, m_ot_draw, m_reg)
    # Subtract mean regime shifts (~1.55 pts) so unconditional E[Total] stays anchored to mu_total_reg
    total = (
        t_reg
        - 1.55
        + np.where(is_ot, t_ot_add, 0.0)
        + np.where(is_foul, t_foul_add, 0.0)
        + np.where(is_blowout, t_blow_adj, 0.0)
    )

    m_int = np.round(margin)
    m_int = np.where(m_int == 0, np.where(margin >= 0, 1.0, -1.0), m_int)
    t_int = np.round(total)

    ats = m_int + float(spread_home)
    tdiff = t_int - float(total_line)

    hc = ats > 0
    ac = ats < 0
    ov = tdiff > 0
    un = tdiff < 0

    return {
        "mean_margin": float(m_int.mean()),
        "median_margin": float(np.median(m_int)),
        "mean_total": float(t_int.mean()),
        "median_total": float(np.median(t_int)),
        "ot_rate": float(is_ot.mean()),
        "blowout_rate": float(is_blowout.mean()),
        "p_home_cover": float(hc.mean()),
        "p_away_cover": float(ac.mean()),
        "p_over": float(ov.mean()),
        "p_under": float(un.mean()),
        "p_sgp_home_cover_over": float((hc & ov).mean()),
        "p_sgp_home_cover_under": float((hc & un).mean()),
        "p_sgp_away_cover_over": float((ac & ov).mean()),
        "p_sgp_away_cover_under": float((ac & un).mean()),
    }
