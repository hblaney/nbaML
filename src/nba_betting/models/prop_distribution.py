"""Bimodal blowout minute truncation & Negative Binomial / Poisson-Gamma stat mixture.

NBA starters lose 5–8 fourth-quarter minutes when games enter garbage time, while
bench players gain garbage-time minutes and softer defensive matchups. Conversely,
close games carry a ~6–7% overtime tail (+5 starter minutes) that vanishes in large
spreads. Linear unimodal mean models misprice both tails.

For each player and stat ``s in {points, rebounds, assists, threes, pra}``:
1. Condition regime weights ``(w_comp, w_blowout, w_ot)`` on pregame ``team_spread``
   (signed: negative = favorite, positive = underdog).
2. Map pregame adjusted minutes ``m_adj`` into regime-specific minute modes
   ``(m_comp, m_blowout, m_ot)`` anchored at typical NBA spread ``|spread| ≈ 5.5``
   so unconditional expectations remain unbiased while big spreads shift starter
   mass downward and bench mass upward.
3. Model conditional counts within each regime via a Poisson-Gamma mixture
   (exact Negative Binomial ``NB(r = 1/alpha_s, p = r / (r + mu_regime))``),
   yielding an exact finite Negative Binomial mixture CDF for Over/Under pricing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import nbinom, poisson

from nba_betting.features.prop_redistribution import RedistributedRates

# Empirical Poisson-Gamma overdispersion parameters (Var = mu + alpha * mu^2)
# calibrated on 2022-24 NBA boxscores (accounting for minute + rate variance).
DEFAULT_NB_ALPHA: dict[str, float] = {
    "points": 0.14,
    "rebounds": 0.10,
    "assists": 0.08,
    "threes": 0.10,
    "pra": 0.10,
}

# Reference regime probabilities at a typical NBA spread (|spread| ≈ 5.5)
REF_P_BLOWOUT = 0.13
REF_P_OT = 0.045
REF_P_COMP = 1.0 - REF_P_BLOWOUT - REF_P_OT


@dataclass(frozen=True)
class RegimeMixtureParams:
    """Three-regime (Competitive / Blowout / Overtime) mixture parameters."""

    stat: str
    nb_alpha: float
    w_comp: float
    w_blowout: float
    w_ot: float
    min_comp: float
    min_blowout: float
    min_ot: float
    mu_comp: float
    mu_blowout: float
    mu_ot: float

    @property
    def expected_minutes(self) -> float:
        return float(
            self.w_comp * self.min_comp
            + self.w_blowout * self.min_blowout
            + self.w_ot * self.min_ot
        )

    @property
    def expected_stat(self) -> float:
        return float(
            self.w_comp * self.mu_comp + self.w_blowout * self.mu_blowout + self.w_ot * self.mu_ot
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_minutes"] = self.expected_minutes
        d["expected_stat"] = self.expected_stat
        return d


def regime_weights_from_spread(team_spread: float) -> tuple[float, float, float]:
    """Return ``(w_comp, w_blowout, w_ot)`` from signed pregame ``team_spread``.

    ``team_spread < 0`` indicates the player's team is favored; ``> 0`` underdog.
    Heavy favorites sit starters slightly more reliably in Q4 leads than heavy
    underdogs (where coaches occasionally leave starters in during comeback runs).
    """
    sp = float(team_spread)
    abs_sp = abs(sp)
    p_blow = 0.10 + 0.012 * max(0.0, abs_sp - 3.0) + 0.016 * max(0.0, abs_sp - 8.5)
    p_blow = float(np.clip(p_blow, 0.08, 0.52))
    if abs_sp >= 7.0:
        fav_mult = 1.12 if sp < 0.0 else 0.92
        p_blow = float(np.clip(p_blow * fav_mult, 0.06, 0.55))
    p_ot = float(np.clip(0.068 * math.exp(-0.075 * abs_sp), 0.015, 0.075))
    p_comp = max(0.0, 1.0 - p_blow - p_ot)
    total = p_comp + p_blow + p_ot
    return float(p_comp / total), float(p_blow / total), float(p_ot / total)


def build_regime_mixture(
    adj_minutes: float,
    rate_per_min: float,
    stat: str,
    *,
    team_spread: float = 0.0,
    nb_alpha: float | None = None,
) -> RegimeMixtureParams:
    """Build bimodal blowout/OT Negative Binomial mixture for one player-stat."""
    s_key = {
        "points": "points",
        "player_points": "points",
        "rebounds": "rebounds",
        "player_rebounds": "rebounds",
        "assists": "assists",
        "player_assists": "assists",
        "threes": "threes",
        "player_threes": "threes",
        "pra": "pra",
        "player_points_rebounds_assists": "pra",
    }.get(stat)
    if s_key is None:
        raise ValueError(f"unknown prop stat/market: {stat!r}")

    alpha = float(DEFAULT_NB_ALPHA[s_key] if nb_alpha is None else nb_alpha)
    base_m = max(float(adj_minutes), 0.5)
    rpm = max(float(rate_per_min), 1e-4)
    w_comp, w_blow, w_ot = regime_weights_from_spread(team_spread)

    if base_m >= 24.0:
        starter_frac = float(np.clip((base_m - 20.0) / 14.0, 0.0, 1.0))
        m_blow = max(8.0, base_m - (2.5 + 5.0 * starter_frac))
        m_ot = min(45.0, base_m + 2.5 + 3.0 * starter_frac)
        m_comp = max(4.0, (base_m - REF_P_BLOWOUT * m_blow - REF_P_OT * m_ot) / REF_P_COMP)
        rate_blow = rpm * 0.98
    else:
        bench_frac = float(np.clip((24.0 - base_m) / 14.0, 0.0, 1.0))
        m_blow = min(32.0, base_m + 2.2 * bench_frac)
        m_ot = min(36.0, base_m + 1.2)
        m_comp = max(2.0, (base_m - REF_P_BLOWOUT * m_blow - REF_P_OT * m_ot) / REF_P_COMP)
        rate_blow = rpm * (1.0 + 0.04 * bench_frac)

    mu_comp = max(0.03, m_comp * rpm)
    mu_blow = max(0.03, m_blow * rate_blow)
    mu_ot = max(0.03, m_ot * rpm)

    return RegimeMixtureParams(
        stat=s_key,
        nb_alpha=alpha,
        w_comp=w_comp,
        w_blowout=w_blow,
        w_ot=w_ot,
        min_comp=float(m_comp),
        min_blowout=float(m_blow),
        min_ot=float(m_ot),
        mu_comp=float(mu_comp),
        mu_blowout=float(mu_blow),
        mu_ot=float(mu_ot),
    )


def p_over_nbinom(expected: float, line: float, *, nb_alpha: float = 0.10) -> float:
    """``P(X > line)`` under Poisson-Gamma / Negative Binomial ``Var = mu + alpha*mu^2``.

    Integer lines split push mass 50/50 to match two-way probability scoring.
    """
    mu = max(float(expected), 1e-5)
    alpha = float(nb_alpha)
    is_int = float(line).is_integer()
    k = int(line) if is_int else int(math.floor(float(line)))

    if alpha <= 1e-6:
        p_le = float(poisson.cdf(k, mu))
        p_eq = float(poisson.pmf(k, mu)) if is_int else 0.0
        return float(np.clip(1.0 - p_le + 0.5 * p_eq, 1e-6, 1.0 - 1e-6))

    r = 1.0 / alpha
    p = r / (r + mu)
    p_le = float(nbinom.cdf(k, r, p))
    p_eq = float(nbinom.pmf(k, r, p)) if is_int else 0.0
    return float(np.clip(1.0 - p_le + 0.5 * p_eq, 1e-6, 1.0 - 1e-6))


def p_over_mixture(mix: RegimeMixtureParams, line: float) -> float:
    """Exact ``P(X > line)`` under the 3-regime Negative Binomial mixture."""
    p = (
        mix.w_comp * p_over_nbinom(mix.mu_comp, line, nb_alpha=mix.nb_alpha)
        + mix.w_blowout * p_over_nbinom(mix.mu_blowout, line, nb_alpha=mix.nb_alpha)
        + mix.w_ot * p_over_nbinom(mix.mu_ot, line, nb_alpha=mix.nb_alpha)
    )
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def p_under_mixture(mix: RegimeMixtureParams, line: float) -> float:
    return float(1.0 - p_over_mixture(mix, line))


def unimodal_p_over_from_rates(
    adj_minutes: float,
    rate_per_min: float,
    stat: str,
    line: float,
    *,
    nb_alpha: float | None = None,
) -> float:
    """Single-regime NB probability without blowout/OT minute bimodality."""
    s_key = build_regime_mixture(adj_minutes, rate_per_min, stat).stat
    alpha = float(DEFAULT_NB_ALPHA[s_key] if nb_alpha is None else nb_alpha)
    return p_over_nbinom(
        max(float(adj_minutes), 0.5) * max(float(rate_per_min), 1e-4), line, nb_alpha=alpha
    )


@dataclass(frozen=True)
class DistributionalPropProjection:
    """Full player prop projection carrying bimodal mixtures across all 5 markets."""

    player_id: str
    player_name: str
    team_id: int
    team_abbreviation: str
    game_id: str
    game_date: str
    team_spread: float
    total_line: float
    base_minutes: float
    adj_minutes: float
    exp_minutes: float
    points: float
    rebounds: float
    assists: float
    threes: float
    pra: float
    mixtures: dict[str, RegimeMixtureParams]
    delta_missing_min: float = 0.0
    delta_missing_usg: float = 0.0
    spot_starter_flag: float = 0.0
    bench_demotion_flag: float = 0.0
    is_b2b: float = 0.0

    def expected(self, market_or_stat: str) -> float:
        key = {
            "points": "points",
            "player_points": "points",
            "rebounds": "rebounds",
            "player_rebounds": "rebounds",
            "assists": "assists",
            "player_assists": "assists",
            "threes": "threes",
            "player_threes": "threes",
            "pra": "pra",
            "player_points_rebounds_assists": "pra",
        }.get(market_or_stat)
        if key is None:
            raise ValueError(f"unknown prop market/stat: {market_or_stat!r}")
        return float(getattr(self, key))

    def p_over(self, market_or_stat: str, line: float) -> float:
        key = {
            "points": "points",
            "player_points": "points",
            "rebounds": "rebounds",
            "player_rebounds": "rebounds",
            "assists": "assists",
            "player_assists": "assists",
            "threes": "threes",
            "player_threes": "threes",
            "pra": "pra",
            "player_points_rebounds_assists": "pra",
        }[market_or_stat]
        return p_over_mixture(self.mixtures[key], line)

    def p_under(self, market_or_stat: str, line: float) -> float:
        return float(1.0 - self.p_over(market_or_stat, line))

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team_id": self.team_id,
            "team_abbreviation": self.team_abbreviation,
            "game_id": self.game_id,
            "game_date": self.game_date,
            "team_spread": self.team_spread,
            "total_line": self.total_line,
            "base_minutes": self.base_minutes,
            "adj_minutes": self.adj_minutes,
            "exp_minutes": self.exp_minutes,
            "points": self.points,
            "rebounds": self.rebounds,
            "assists": self.assists,
            "threes": self.threes,
            "pra": self.pra,
            "delta_missing_min": self.delta_missing_min,
            "delta_missing_usg": self.delta_missing_usg,
            "spot_starter_flag": self.spot_starter_flag,
            "bench_demotion_flag": self.bench_demotion_flag,
            "is_b2b": self.is_b2b,
        }


def project_distributional_props(
    rates: RedistributedRates,
    *,
    game_id: str,
    game_date: str,
    team_spread: float = 0.0,
    total_line: float = 226.5,
) -> DistributionalPropProjection:
    """Combine pregame redistributed rates with spread-conditioned blowout mixture."""
    mix_pts = build_regime_mixture(
        rates.adj_minutes, rates.pts_per_min, "points", team_spread=team_spread
    )
    mix_reb = build_regime_mixture(
        rates.adj_minutes, rates.reb_per_min, "rebounds", team_spread=team_spread
    )
    mix_ast = build_regime_mixture(
        rates.adj_minutes, rates.ast_per_min, "assists", team_spread=team_spread
    )
    mix_fg3 = build_regime_mixture(
        rates.adj_minutes, rates.fg3m_per_min, "threes", team_spread=team_spread
    )
    mix_pra = build_regime_mixture(
        rates.adj_minutes, rates.pra_per_min, "pra", team_spread=team_spread
    )

    return DistributionalPropProjection(
        player_id=rates.player_id,
        player_name=rates.player_name,
        team_id=rates.team_id,
        team_abbreviation=rates.team_abbreviation,
        game_id=str(game_id),
        game_date=str(game_date),
        team_spread=float(team_spread),
        total_line=float(total_line),
        base_minutes=rates.base_minutes,
        adj_minutes=rates.adj_minutes,
        exp_minutes=mix_pts.expected_minutes,
        points=mix_pts.expected_stat,
        rebounds=mix_reb.expected_stat,
        assists=mix_ast.expected_stat,
        threes=mix_fg3.expected_stat,
        pra=mix_pra.expected_stat,
        mixtures={
            "points": mix_pts,
            "rebounds": mix_reb,
            "assists": mix_ast,
            "threes": mix_fg3,
            "pra": mix_pra,
        },
        delta_missing_min=rates.delta_missing_min,
        delta_missing_usg=rates.delta_missing_usg,
        spot_starter_flag=rates.spot_starter_flag,
        bench_demotion_flag=rates.bench_demotion_flag,
        is_b2b=rates.is_b2b,
    )
