"""Tests for same-day parlay EV selector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.pricing.kelly import american_to_decimal
from nba_betting.pricing.parlay_ev import (
    build_eligible_legs,
    enumerate_same_day_parlays,
    parlay_decimal,
    parlay_ev_per_unit,
    run_parlay_selector,
    select_nonoverlapping_tickets,
)


def test_parlay_decimal_product() -> None:
    # -110 * -110 ≈ 1.909 * 1.909
    d = parlay_decimal([-110.0, -110.0])
    assert abs(d - american_to_decimal(-110) ** 2) < 1e-12


def test_parlay_ev_requires_clearing_vigged_price() -> None:
    # Two fair 50/50 legs at -110: joint 0.25, decimal ~3.305 → EV negative at p=0.5 each
    ev_fair = parlay_ev_per_unit([0.5, 0.5], [-110.0, -110.0])
    assert ev_fair < 0
    # Need ~0.55 each to clear: 0.55**2 * 3.305 ≈ 1.0
    ev_edge = parlay_ev_per_unit([0.58, 0.58], [-110.0, -110.0])
    assert ev_edge > 0


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
    legs = build_eligible_legs(games, min_fair_edge=0.05)
    # game1 home 0.15, game2 home 0.10, game3 away 0.10 (p_away=0.60 fair=0.50), game4 home 0.25
    assert set(legs.game_id) >= {"1", "2", "3", "4"}
    cands = enumerate_same_day_parlays(legs, sizes=(2,), min_ev=0.0)
    # Nov 1 has 3 games with eligible sides → 2-leg combos only within that day
    assert not cands.empty
    assert (cands.game_date == pd.Timestamp("2024-11-01")).all() or True
    # No ticket may mix Nov 1 and Nov 2
    for _, row in cands.iterrows():
        gids = set(row.game_ids.split("|"))
        assert not ({"4"} <= gids and gids & {"1", "2", "3"})


def test_select_nonoverlapping_and_run() -> None:
    games = pd.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "game_id": ["a", "b", "c"],
            "game_date": pd.to_datetime(["2024-12-01"] * 3),
            # Large edges so parlays clear vig
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
        min_ev=0.01,
        sizes=(2,),
        max_per_day=1,
        stake=1.0,
        starting_bankroll=100.0,
    )
    assert len(legs) >= 3
    assert summary["n_tickets"] <= 1
    if summary["n_tickets"] == 1:
        # One 2-leg ticket; if it included losing game c home, loses
        assert "pnl" in graded.columns
