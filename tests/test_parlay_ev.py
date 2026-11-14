"""Tests for same-day parlay EV selector."""

from __future__ import annotations

import pandas as pd

from nba_betting.pricing.kelly import american_to_decimal
from nba_betting.pricing.parlay_ev import (
    build_eligible_legs,
    clears_parlay_price,
    enumerate_same_day_parlays,
    parlay_decimal,
    parlay_ev_per_unit,
    run_parlay_selector,
)


def test_parlay_decimal_product() -> None:
    d = parlay_decimal([-110.0, -110.0])
    assert abs(d - american_to_decimal(-110) ** 2) < 1e-12


def test_combined_must_beat_parlay_price() -> None:
    # Fair 50/50 legs at -110: joint does not clear vigged parlay
    assert not clears_parlay_price([0.5, 0.5], [-110.0, -110.0], min_price_edge=0.0)
    assert clears_parlay_price([0.58, 0.58], [-110.0, -110.0], min_price_edge=0.0)
    assert not clears_parlay_price([0.58, 0.58], [-110.0, -110.0], min_price_edge=0.5)
    ev_fair = parlay_ev_per_unit([0.5, 0.5], [-110.0, -110.0])
    assert ev_fair < 0


def test_eligible_legs_and_same_day_only() -> None:
    games = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "game_id": ["1", "2", "3", "4"],
            "game_date": pd.to_datetime(
                ["2024-11-01", "2024-11-01", "2024-11-01", "2024-11-02"]
            ),
            "p_model_home": [0.70, 0.65, 0.40, 0.80],
            "p_home_novig": [0.55, 0.55, 0.50, 0.55],
            "home_ml": [-150.0, -140.0, 100.0, -200.0],
            "away_ml": [130.0, 120.0, -120.0, 170.0],
            "home_win": [1.0, 0.0, 1.0, 1.0],
        }
    )
    legs = build_eligible_legs(games, min_fair_edge=0.05, require_vs_vigged=True)
    assert set(legs.game_id) >= {"1", "2", "3", "4"}
    cands = enumerate_same_day_parlays(legs, sizes=(2,), min_price_edge=0.0)
    assert not cands.empty
    for _, row in cands.iterrows():
        gids = set(row.game_ids.split("|"))
        assert not ({"4"} <= gids and gids & {"1", "2", "3"})
        assert row.p_joint_model >= row.book_implied - 1e-12


def test_run_top_edge_fewer_tickets() -> None:
    games = pd.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "game_id": ["a", "b", "c"],
            "game_date": pd.to_datetime(["2024-12-01"] * 3),
            "p_model_home": [0.75, 0.72, 0.70],
            "p_home_novig": [0.55, 0.55, 0.55],
            "home_ml": [-120.0, -120.0, -120.0],
            "away_ml": [100.0, 100.0, 100.0],
            "home_win": [1.0, 1.0, 0.0],
        }
    )
    legs, graded, summary = run_parlay_selector(
        games,
        min_fair_edge=0.05,
        min_price_edge=0.1,
        sizes=(2,),
        max_per_day=1,
        walkforward_quantile=None,
        stake=1.0,
        starting_bankroll=100.0,
    )
    assert len(legs) >= 3
    assert summary["n_tickets"] <= 1
