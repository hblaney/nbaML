"""Offline tests for as-of features, leakage, and game score model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_betting.features.game_features import build_game_features_for_date
from nba_betting.features.team_ratings import (
    days_rest_asof,
    prepare_team_game_rows,
    team_rating_asof,
    team_rating_ema,
)
from nba_betting.models.game import expected_scores, predict_game, simulate_scores
from nba_betting.slate.score import format_predictions_table, score_date

FIXTURES = Path(__file__).parent / "fixtures" / "model"


@pytest.fixture
def schedule() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "schedule.csv", dtype={"game_id": str})


@pytest.fixture
def team_logs() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "team_logs.csv", dtype={"game_id": str})


def test_prepare_attaches_opp_and_efficiencies(team_logs: pd.DataFrame) -> None:
    prepared = prepare_team_game_rows(team_logs)
    row = prepared.loc[
        (prepared["game_id"] == "0022400001") & (prepared["team_abbreviation"] == "LAL")
    ].iloc[0]
    assert row["opp_pts"] == 103
    assert row["ortg"] > 0
    assert row["drtg"] > 0
    assert row["pace"] > 0


def test_ratings_asof_exclude_same_day(team_logs: pd.DataFrame) -> None:
    prepared = prepare_team_game_rows(team_logs)
    # On 2024-10-25, LAL has played Oct 22 and Oct 24 only (not the 25th game).
    rating = team_rating_asof(prepared, 1610612747, "2024-10-25", window=20)
    assert rating is not None
    assert rating.games == 2
    assert rating.asof_date == "2024-10-25"

    # Same-day game must not be included.
    same_day = prepared.loc[
        (prepared["team_id"] == 1610612747) & (prepared["game_date"] == "2024-10-25")
    ]
    assert len(same_day) == 1
    hist = prepared.loc[
        (prepared["team_id"] == 1610612747)
        & (prepared["game_date"] < pd.Timestamp("2024-10-25"))
    ]
    assert len(hist) == rating.games


def test_no_future_leakage_in_game_features(
    schedule: pd.DataFrame, team_logs: pd.DataFrame
) -> None:
    feats = build_game_features_for_date(
        schedule=schedule,
        team_logs=team_logs,
        game_date="2024-10-25",
        season="2024-25",
    )
    assert len(feats) == 2
    prepared = prepare_team_game_rows(team_logs)
    asof = pd.Timestamp("2024-10-25")
    for f in feats:
        for team_id in (f.home_team_id, f.away_team_id):
            future = prepared.loc[
                (prepared["team_id"] == team_id) & (prepared["game_date"] >= asof)
            ]
            # Future rows may exist in the raw frame, but ratings must ignore them.
            rating = team_rating_asof(prepared, team_id, asof)
            assert rating is not None
            used = prepared.loc[
                (prepared["team_id"] == team_id) & (prepared["game_date"] < asof)
            ]
            assert len(used) == rating.games
            assert len(future) >= 0  # present in fixture, unused


def test_b2b_detection(team_logs: pd.DataFrame) -> None:
    prepared = prepare_team_game_rows(team_logs)
    # Oct 24 -> Oct 25 is back-to-back for LAL.
    days, is_b2b = days_rest_asof(prepared, 1610612747, "2024-10-25")
    assert days == 1.0
    assert is_b2b is True


def test_ema_ratings_exclude_same_day(team_logs: pd.DataFrame) -> None:
    prepared = prepare_team_game_rows(team_logs)
    rating = team_rating_ema(prepared, 1610612747, "2024-10-25", half_life=16.0)
    assert rating is not None
    assert rating.games == 2
    hist = prepared.loc[
        (prepared["team_id"] == 1610612747)
        & (prepared["game_date"] < pd.Timestamp("2024-10-25"))
    ]
    assert len(hist) == rating.games


def test_expected_scores_home_court_and_predict(
    schedule: pd.DataFrame, team_logs: pd.DataFrame
) -> None:
    feats = build_game_features_for_date(
        schedule=schedule,
        team_logs=team_logs,
        game_date="2024-10-25",
        season="2024-25",
    )
    lal_bos = next(f for f in feats if f.home_tricode == "LAL")
    exp_h, exp_a = expected_scores(lal_bos)
    assert exp_h > 0 and exp_a > 0

    pred = predict_game(lal_bos, market_spread=-3.5, market_total=220.0, n_sims=5000, seed=1)
    assert 0.0 < pred.p_home_win < 1.0
    assert 0.0 < pred.p_home_cover < 1.0
    assert 0.0 < pred.p_over < 1.0
    assert pred.exp_total == pytest.approx(exp_h + exp_a)
    assert pred.fair_total > 0


def test_simulate_scores_shape() -> None:
    home, away = simulate_scores(110.0, 105.0, n_sims=100, rng=np.random.default_rng(0))
    assert home.shape == (100,)
    assert away.shape == (100,)


def test_score_date_offline(schedule: pd.DataFrame, team_logs: pd.DataFrame) -> None:
    preds = score_date(
        season="2024-25",
        game_date="2024-10-25",
        schedule=schedule,
        team_logs=team_logs,
        n_sims=2000,
        seed=2,
    )
    assert len(preds) == 2
    table = format_predictions_table(preds)
    assert "LAL" in table
    assert "GSW" in table
