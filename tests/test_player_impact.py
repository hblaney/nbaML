"""Player-impact availability and expanding ridge smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.backtest.player_impact_model import expanding_ridge_normal_probs
from nba_betting.features.player_impact import (
    PlayerState,
    learn_play_rate,
    prepare_player_box,
    team_player_impact_asof_states,
    update_player_states_after_game,
)
from nba_betting.features.opponent_adjusted import verify_harder_schedule_raises_adj_offense


def test_miss_streak_reduces_expected_impact() -> None:
    states = {
        "star": PlayerState(
            games=10,
            ema_minutes=34.0,
            ema_pm_per_min=0.2,
            ema_pts_per_min=0.6,
            miss_streak=0,
            last_team_id=1,
        ),
        "role": PlayerState(
            games=10,
            ema_minutes=22.0,
            ema_pm_per_min=0.05,
            ema_pts_per_min=0.3,
            miss_streak=0,
            last_team_id=1,
        ),
    }
    rates = {0: 0.95, 1: 0.25, 2: 0.1}
    full = team_player_impact_asof_states(states, 1, play_rate_by_streak=rates)
    states["star"].miss_streak = 2
    missing = team_player_impact_asof_states(states, 1, play_rate_by_streak=rates)
    assert missing["impact"] < full["impact"]
    assert missing["missing_impact"] > 0


def test_update_player_states_asof_only() -> None:
    states: dict[str, PlayerState] = {}
    box = prepare_player_box(
        pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "team_id": 1,
                    "player_id": "a",
                    "minutes": 30.0,
                    "plus_minus": 6.0,
                    "pts": 18.0,
                    "dnp": False,
                }
            ]
        )
    )
    update_player_states_after_game(states, box)
    assert states["a"].games == 1
    assert states["a"].miss_streak == 0
    # DNP next game
    box2 = prepare_player_box(
        pd.DataFrame(
            [
                {
                    "game_id": "g2",
                    "game_date": "2024-01-03",
                    "team_id": 1,
                    "player_id": "a",
                    "minutes": 0.0,
                    "plus_minus": 0.0,
                    "pts": 0.0,
                    "dnp": True,
                }
            ]
        )
    )
    update_player_states_after_game(states, box2)
    assert states["a"].miss_streak == 1


def test_learn_play_rate_monotonic_in_streak() -> None:
    hist = [(0, 1)] * 50 + [(0, 0)] * 5 + [(3, 1)] * 5 + [(3, 0)] * 40
    rates = learn_play_rate(hist)
    assert rates[0] > rates[3]


def test_expanding_ridge_shapes() -> None:
    rng = np.random.default_rng(0)
    n = 400
    impact = rng.normal(0, 5, size=n)
    margin = 0.4 * impact + 2.5 + rng.normal(0, 12, size=n)
    y = (margin > 0).astype(float)
    df = pd.DataFrame(
        {
            "season": ["2024-25"] * n,
            "game_id": [str(i) for i in range(n)],
            "game_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "home_win": y,
            "actual_margin": margin,
            "impact_diff": impact,
            "missing_impact_diff": 0.0,
            "exp_min_diff": 0.0,
            "adj_net_diff": impact * 0.2,
            "net_diff": impact * 0.2,
            "elo_diff": impact * 10,
            "rest_diff": 0.0,
            "b2b_diff": 0.0,
            "three_diff": 0.0,
            "games7_diff": 0.0,
        }
    )
    out = expanding_ridge_normal_probs(df, min_train=100)
    assert len(out) == n
    assert out["p_home_win"].between(0, 1).all()


def test_opp_adj_still_verified() -> None:
    assert verify_harder_schedule_raises_adj_offense()
