"""Tests for ESPN closing ML parse helpers and market blend weight choice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.backtest.market_blend import blend_probs, choose_blend_weight, ml_prob_metrics
from nba_betting.data.espn_closing import _extract_provider_mls, _pick_best_provider
from nba_betting.pricing.american import american_to_implied, devig_two_way


def test_devig_two_way_sums_to_one() -> None:
    a, b = devig_two_way(-150, 130)
    assert abs(a + b - 1.0) < 1e-9
    assert 0.0 < a < 1.0


def test_extract_close_ml_from_espn_shape() -> None:
    odds_obj = {
        "provider": {"id": "58", "name": "ESPN BET"},
        "close": {"total": {"american": "220.5"}},
        "homeTeamOdds": {
            "moneyLine": 110,
            "close": {"moneyLine": {"american": "+110"}},
        },
        "awayTeamOdds": {
            "moneyLine": -130,
            "close": {"moneyLine": {"american": "-130"}},
        },
    }
    parsed = _extract_provider_mls(odds_obj)
    assert parsed["home_ml"] == 110.0
    assert parsed["away_ml"] == -130.0
    best = _pick_best_provider([parsed, {"provider_id": "59", "provider_name": "ESPN Bet - Live Odds", "home_ml": -3000, "away_ml": 1000}])
    assert best is not None
    assert best["provider_id"] == "58"


def test_choose_blend_prefers_better_confirm() -> None:
    rng = np.random.default_rng(0)
    n = 400
    # Market is well calibrated; model is noisier.
    latent = rng.normal(0, 1, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(float)
    p_mkt = 1 / (1 + np.exp(-0.9 * latent))
    p_model = 1 / (1 + np.exp(-0.3 * latent + rng.normal(0, 0.8, size=n)))
    seasons = np.array(["2022-23"] * 150 + ["2023-24"] * 150 + ["2024-25"] * 100)
    df = pd.DataFrame(
        {
            "season": seasons,
            "p_model": p_model,
            "p_home_novig": p_mkt,
            "home_win": y,
        }
    )
    chosen = choose_blend_weight(
        df,
        train_seasons=["2022-23", "2023-24"],
        confirm_season="2024-25",
    )
    assert 0.0 <= chosen["w_model"] <= 1.0
    # Market-heavy blend should win over pure model on this synthetic.
    assert chosen["w_model"] < 0.75
    mkt = ml_prob_metrics(df.loc[df.season == "2024-25", "p_home_novig"], df.loc[df.season == "2024-25", "home_win"])
    blend = blend_probs(
        df.loc[df.season == "2024-25", "p_model"],
        df.loc[df.season == "2024-25", "p_home_novig"],
        w_model=chosen["w_model"],
    )
    blended = ml_prob_metrics(blend, df.loc[df.season == "2024-25", "home_win"])
    assert blended["ml_brier"] <= mkt["ml_brier"] + 0.02
