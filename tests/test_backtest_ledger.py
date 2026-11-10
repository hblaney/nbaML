"""Fixture tests for walk-forward backtest, metrics, Kelly, and blind ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_betting.backtest.metrics import brier_score, log_loss
from nba_betting.backtest.walkforward import grade_prediction_row, run_walkforward
from nba_betting.pricing.kelly import kelly_fraction, stake_amount
from nba_betting.slate.ledger import grade_ledger, lock_date, read_ledger

FIXTURES = Path(__file__).parent / "fixtures" / "model"


@pytest.fixture
def schedule() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "schedule.csv", dtype={"game_id": str})


@pytest.fixture
def team_logs() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "team_logs.csv", dtype={"game_id": str})


def test_kelly_caps_and_zeros() -> None:
    assert kelly_fraction(0.55, -110) > 0
    assert kelly_fraction(0.40, -110) == 0.0
    stake = stake_amount(100.0, 0.90, -110, max_bet_frac=0.05)
    assert stake <= 5.0 + 1e-9


def test_brier_log_loss_sane() -> None:
    import numpy as np

    probs = np.array([0.7, 0.2])
    y = np.array([1.0, 0.0])
    assert 0.0 < brier_score(probs, y) < 1.0
    assert log_loss(probs, y) > 0


def test_walkforward_no_future_features(
    schedule: pd.DataFrame, team_logs: pd.DataFrame
) -> None:
    graded, metrics = run_walkforward(
        season="2024-25",
        schedule=schedule,
        team_logs=team_logs,
        start_date="2024-10-24",
        n_sims=1500,
    )
    assert metrics["n_games"] >= 2
    assert "ml_hit_rate" in metrics
    assert "ml_brier" in metrics
    assert "kelly_roi" in metrics
    # Predictions only for dates with enough prior history.
    assert set(graded["game_date"]) <= {"2024-10-24", "2024-10-25"}
    # Leakage: as-of ratings for a row cannot include that game's box.
    from nba_betting.features.team_ratings import prepare_team_game_rows, team_rating_asof

    prepared = prepare_team_game_rows(team_logs)
    for _, row in graded.iterrows():
        rating = team_rating_asof(prepared, 1610612747, row["game_date"])
        if rating is None:
            continue
        used = prepared.loc[
            (prepared["team_id"] == 1610612747)
            & (prepared["game_date"] < pd.Timestamp(row["game_date"]))
        ]
        assert len(used) == rating.games


def test_grade_prediction_row_logic() -> None:
    row = pd.Series(
        {
            "p_home_win": 0.6,
            "p_away_win": 0.4,
            "p_over": 0.55,
            "fair_spread": -3.5,
            "fair_total": 220.0,
            "home_score": 110,
            "away_score": 100,
            "home_win": 1.0,
            "margin": 10.0,
            "total_pts": 210.0,
        }
    )
    graded = grade_prediction_row(row)
    assert graded["ml_correct"] is True
    assert graded["spread_correct"] is True  # 10 + (-3.5) > 0
    assert graded["total_correct"] is False  # picked over, went under


def test_blind_ledger_lock_no_overwrite(
    tmp_path: Path, schedule: pd.DataFrame, team_logs: pd.DataFrame
) -> None:
    path = tmp_path / "blind_predictions.jsonl"
    added1, _ = lock_date(
        season="2024-25",
        game_date="2024-10-25",
        path=path,
        schedule=schedule,
        team_logs=team_logs,
        n_sims=800,
    )
    assert added1 == 2
    rows = read_ledger(path)
    locked_at = rows[0]["locked_at"]
    p_home = rows[0]["p_home_win"]

    added2, _ = lock_date(
        season="2024-25",
        game_date="2024-10-25",
        path=path,
        schedule=schedule,
        team_logs=team_logs,
        n_sims=800,
    )
    assert added2 == 0
    rows2 = read_ledger(path)
    assert len(rows2) == 2
    assert rows2[0]["locked_at"] == locked_at
    assert rows2[0]["p_home_win"] == p_home


def test_blind_ledger_grade(
    tmp_path: Path, schedule: pd.DataFrame, team_logs: pd.DataFrame
) -> None:
    path = tmp_path / "blind_predictions.jsonl"
    lock_date(
        season="2024-25",
        game_date="2024-10-25",
        path=path,
        schedule=schedule,
        team_logs=team_logs,
        n_sims=800,
    )
    n, _ = grade_ledger(season="2024-25", path=path, schedule=schedule)
    assert n == 2
    rows = read_ledger(path)
    assert all(r["status"] == "graded" for r in rows)
    assert all(r["home_score"] is not None for r in rows)
    # Re-grade should not change locked prediction fields / double-count.
    original = json.dumps(rows[0]["p_home_win"])
    n2, _ = grade_ledger(season="2024-25", path=path, schedule=schedule)
    assert n2 == 0
    rows2 = read_ledger(path)
    assert json.dumps(rows2[0]["p_home_win"]) == original
