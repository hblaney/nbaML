"""Expanding Platt calibration must not use future labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.backtest.calibration import apply_expanding_platt, swap_calibrated_ml_probs
from nba_betting.backtest.metrics import brier_score


def test_expanding_platt_no_future_leak() -> None:
    rng = np.random.default_rng(0)
    n = 300
    # Synthetic: true logit scales with a latent; model p is noisy.
    latent = rng.normal(0, 1, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(float)
    p = 1 / (1 + np.exp(-(0.6 * latent + rng.normal(0, 0.5, size=n))))
    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "game_id": [f"g{i:04d}" for i in range(n)],
            "p_home_win": p,
            "home_win": y,
        }
    )
    cal = apply_expanding_platt(df, min_train=50, shrink=1.0, C=10.0, refit_every=25)
    # First min_train rows stay near raw (shrink=1).
    assert np.allclose(
        cal.loc[:49, "p_home_win_cal"].to_numpy(),
        cal.loc[:49, "p_home_win"].to_numpy(),
        atol=1e-9,
    )
    # Calibrated Brier on the tail should not explode vs raw.
    tail = cal.iloc[50:]
    raw_b = brier_score(tail["p_home_win"].to_numpy(), tail["home_win"].to_numpy())
    cal_b = brier_score(tail["p_home_win_cal"].to_numpy(), tail["home_win"].to_numpy())
    assert cal_b <= raw_b + 0.05


def test_swap_calibrated_refreshes_picks() -> None:
    df = pd.DataFrame(
        {
            "game_date": ["2024-01-01", "2024-01-02"],
            "game_id": ["a", "b"],
            "p_home_win": [0.4, 0.6],
            "p_away_win": [0.6, 0.4],
            "home_win": [1.0, 0.0],
            "p_home_win_cal": [0.55, 0.45],
        }
    )
    out = swap_calibrated_ml_probs(df)
    assert out.loc[0, "ml_pick_home"] == True
    assert out.loc[1, "ml_pick_home"] == False
    assert out.loc[0, "ml_correct"] == True
    assert out.loc[1, "ml_correct"] == True
