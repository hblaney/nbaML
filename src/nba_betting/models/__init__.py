"""Game score engine, joint (margin, total) distribution, and player prop projections."""

from nba_betting.models.game import GamePrediction, expected_scores, predict_game
from nba_betting.models.joint_distribution import (
    DMOCHOWSKI_QUANTILES,
    attach_bivariate_market_probs,
    attach_dmochowski_quantiles,
    build_pregame_joint_feature_frame,
    bvn_rect_prob,
    fit_expanding_joint_distribution,
)
from nba_betting.models.props import PropProjection, project_from_prior

__all__ = [
    "DMOCHOWSKI_QUANTILES",
    "GamePrediction",
    "PropProjection",
    "attach_bivariate_market_probs",
    "attach_dmochowski_quantiles",
    "build_pregame_joint_feature_frame",
    "bvn_rect_prob",
    "expected_scores",
    "fit_expanding_joint_distribution",
    "predict_game",
    "project_from_prior",
]
