"""Player prop projections conditioned on expected minutes priors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.stats import poisson

from nba_betting.features.player_priors import PlayerPrior

PROP_STATS = ("points", "rebounds", "assists", "threes", "pra")


@dataclass(frozen=True)
class PropProjection:
    player_id: int
    player_name: str
    team_abbreviation: str
    game_id: str
    game_date: str
    exp_minutes: float
    points: float
    rebounds: float
    assists: float
    threes: float
    pra: float

    def to_dict(self) -> dict:
        return asdict(self)

    def expected(self, market: str) -> float:
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
        }.get(market)
        if key is None:
            raise ValueError(f"unknown prop market: {market}")
        return float(getattr(self, key))


def project_from_prior(
    prior: PlayerPrior,
    *,
    game_id: str,
    game_date: str,
    minutes: float | None = None,
) -> PropProjection:
    """Scale per-minute rates by expected (or override) minutes."""
    mins = float(prior.exp_minutes if minutes is None else minutes)
    mins = max(mins, 0.0)
    points = mins * prior.pts_per_min
    rebounds = mins * prior.reb_per_min
    assists = mins * prior.ast_per_min
    threes = mins * prior.fg3m_per_min
    return PropProjection(
        player_id=prior.player_id,
        player_name=prior.player_name,
        team_abbreviation=prior.team_abbreviation,
        game_id=game_id,
        game_date=game_date,
        exp_minutes=mins,
        points=float(points),
        rebounds=float(rebounds),
        assists=float(assists),
        threes=float(threes),
        pra=float(points + rebounds + assists),
    )


def project_players(
    priors: Iterable[PlayerPrior],
    *,
    game_id: str,
    game_date: str,
) -> list[PropProjection]:
    return [project_from_prior(p, game_id=game_id, game_date=game_date) for p in priors]


def p_over_poisson(expected: float, line: float) -> float:
    """P(stat > line) under Poisson(expected). Half-lines use strict inequality."""
    lam = max(float(expected), 1e-6)
    # For X ~ Poisson, P(X > L) = 1 - F(floor(L)) when L is .5 style,
    # and 1 - F(L) + 0.5*P(X=L) push split when L is integer.
    if float(line).is_integer():
        k = int(line)
        p_eq = float(poisson.pmf(k, lam))
        p_le = float(poisson.cdf(k, lam))
        return float(1.0 - p_le + 0.5 * p_eq)
    # line like 24.5 → need X >= 25
    threshold = int(np.floor(line))
    return float(1.0 - poisson.cdf(threshold, lam))


def p_under_poisson(expected: float, line: float) -> float:
    return float(1.0 - p_over_poisson(expected, line))
