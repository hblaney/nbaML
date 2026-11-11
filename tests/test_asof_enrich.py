"""Tests for opponent-adjusted identity, availability as-of, and enrich leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.backtest.asof_enrich import (
    enrich_graded_predictions,
    expanding_logit_enrich,
    margin_shift_from_features,
)
from nba_betting.features.opponent_adjusted import (
    adj_offense_observation,
    verify_harder_schedule_raises_adj_offense,
)
from nba_betting.features.player_availability import availability_asof, prepare_player_minutes


def test_opponent_adjusted_harder_schedule_property() -> None:
    assert adj_offense_observation(110.0, 100.0) > adj_offense_observation(110.0, 120.0)
    assert verify_harder_schedule_raises_adj_offense()


def test_availability_uses_only_prior_games() -> None:
    rows = []
    # Two prior games + same-day game that must be ignored.
    for gid, day, mins in [
        ("g1", "2024-01-01", 30.0),
        ("g2", "2024-01-03", 0.0),  # star out last game
        ("g3", "2024-01-05", 28.0),  # same asof day — must not leak
    ]:
        rows.append(
            {
                "game_id": gid,
                "game_date": day,
                "team_id": 1,
                "player_id": "star",
                "minutes": mins,
                "dnp": mins == 0.0,
            }
        )
        rows.append(
            {
                "game_id": gid,
                "game_date": day,
                "team_id": 1,
                "player_id": "role",
                "minutes": 20.0,
                "dnp": False,
            }
        )
    pm = prepare_player_minutes(pd.DataFrame(rows))
    snap = availability_asof(pm, 1, "2024-01-05", window=10, top_n=2, min_avg_minutes=15.0)
    # Last prior game is g2 where star had 0 — missing minutes > 0
    assert snap.missing_minutes > 0
    assert snap.asof_date == "2024-01-05"


def test_expanding_logit_no_future_fit() -> None:
    rng = np.random.default_rng(0)
    n = 400
    # Construct a signal that only appears late; early probs should stay near prior.
    p_raw = np.full(n, 0.55)
    y = (rng.random(n) < 0.55).astype(float)
    shift = np.zeros(n)
    # Strong shift correlated with y only in second half — model must not use it early.
    shift[200:] = (y[200:] - 0.5) * 4.0
    out = expanding_logit_enrich(p_raw, y, shift, min_train=100, refit_every=50, C=1.0)
    assert out.shape == (n,)
    # Before min_train, unchanged.
    assert np.allclose(out[:100], p_raw[:100])


def test_enrich_join_and_shift_sign() -> None:
    graded = pd.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "game_id": ["1", "2", "3"],
            "game_date": ["2024-11-01", "2024-11-02", "2024-11-03"],
            "p_home_win": [0.55, 0.60, 0.45],
            "home_win": [1.0, 0.0, 1.0],
        }
    )
    feats = pd.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "game_id": ["1", "2", "3"],
            "game_date": ["2024-11-01", "2024-11-02", "2024-11-03"],
            "miss_diff": [10.0, -10.0, 0.0],
            "three_diff": [0.0, 0.0, 0.0],
            "games7_diff": [0.0, 0.0, 0.0],
            "adj_residual": [0.0, 0.0, 0.0],
        }
    )
    shift = margin_shift_from_features(feats, avail_k=0.05, sched_k=0.0, adj_k=0.0)
    # Home missing more minutes ⇒ negative shift.
    assert shift[0] < 0
    assert shift[1] > 0
    out = enrich_graded_predictions(graded, feats, C=1.0)
    assert len(out) == 3
    assert "p_away_win" in out.columns
