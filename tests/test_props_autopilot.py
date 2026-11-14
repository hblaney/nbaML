"""Unit tests for PrizePicks payout math, push reversions, stat calibration, and live cashout EV."""

from __future__ import annotations

from nba_betting.props_autopilot.prizepicks_math import (
    apply_stat_tail_calibration,
    evaluate_live_cashout,
    exact_hits_pmf,
    settle_prizepicks_slip,
    slip_ev_and_log_growth,
)


def test_prizepicks_breakeven_rates_and_push_reversion() -> None:
    # 5-Pick Flex at 54.25% per leg is ~breakeven (EV >= 0); at 53.0% is negative EV
    m_pos = slip_ev_and_log_growth([0.55] * 5, mode="flex")
    m_neg = slip_ev_and_log_growth([0.53] * 5, mode="flex")
    assert m_pos["ev"] > 0.0
    assert m_neg["ev"] < 0.0

    # Push reversion: 4-Flex with 1 push (-1) and 3 wins reverts to 3-Flex 3/3 = 3.0x
    assert settle_prizepicks_slip([1, 1, 1, -1], "flex") == 3.0
    # 3-Power with 1 push (-1) and 2 wins reverts to 2-Power = 3.0x
    assert settle_prizepicks_slip([1, 1, -1], "power") == 3.0
    assert abs(float(exact_hits_pmf([0.6, 0.5]).sum()) - 1.0) < 1e-12


def test_stat_tail_calibration_boosts_assists_over_points() -> None:
    p_ast = apply_stat_tail_calibration(0.555, "Assists")
    p_pts = apply_stat_tail_calibration(0.555, "Points")
    assert p_ast > 0.555 > p_pts


def test_live_cashout_calculator_matches_jalen_green_scenario() -> None:
    # Jalen Green LESS 20.5 PRA, sitting at 16 PRA late Q3 in 2-pt game (~10 rem min * 0.68 PRA/min = 6.8),
    # $24 full payout vs $10 cashout offer -> must recommend CASH_OUT!
    res = evaluate_live_cashout(
        side="LESS",
        line=20.5,
        current_stat=16.0,
        expected_remaining_minutes=10.0,
        per_minute_rate=0.68,
        full_payout_dollars=24.0,
        cashout_offer_dollars=10.0,
    )
    assert res["recommendation"] == "CASH_OUT"
    assert res["ev_ride_dollars"] < 10.0
    assert res["edge_of_cashout_dollars"] > 2.0
