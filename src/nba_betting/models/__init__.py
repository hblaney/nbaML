"""Game score engine, player prop projections, and market-residual alpha engine."""

from nba_betting.models.game import GamePrediction, expected_scores, predict_game
from nba_betting.models.losses import (
    CustomNewtonGBDT,
    classwise_ece,
    decorrelated_brier_grad_hess,
    fit_hubacek_linear,
    fit_logit_offset_residual,
)
from nba_betting.models.market_residual import (
    DEFAULT_ORTH_FEATURES,
    simulate_seasonal_kelly_portfolio,
    walkforward_anchored_hubacek_probs,
)
from nba_betting.models.props import PropProjection, project_from_prior

__all__ = [
    "CustomNewtonGBDT",
    "DEFAULT_ORTH_FEATURES",
    "GamePrediction",
    "PropProjection",
    "classwise_ece",
    "decorrelated_brier_grad_hess",
    "expected_scores",
    "fit_hubacek_linear",
    "fit_logit_offset_residual",
    "predict_game",
    "project_from_prior",
    "simulate_seasonal_kelly_portfolio",
    "walkforward_anchored_hubacek_probs",
]
