"""Game score engine + player prop projections."""

from nba_betting.models.game import GamePrediction, expected_scores, predict_game
from nba_betting.models.prop_distribution import (
    DistributionalPropProjection,
    RegimeMixtureParams,
    build_regime_mixture,
    p_over_mixture,
    p_over_nbinom,
    p_under_mixture,
    project_distributional_props,
    regime_weights_from_spread,
)
from nba_betting.models.props import PropProjection, project_from_prior

__all__ = [
    "DistributionalPropProjection",
    "GamePrediction",
    "PropProjection",
    "RegimeMixtureParams",
    "build_regime_mixture",
    "expected_scores",
    "p_over_mixture",
    "p_over_nbinom",
    "p_under_mixture",
    "predict_game",
    "project_distributional_props",
    "project_from_prior",
    "regime_weights_from_spread",
]
