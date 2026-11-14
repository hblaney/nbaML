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


def test_postmortem_classifies_clv_win_and_3pt_variance_loss(tmp_path) -> None:
    from nba_betting.props_autopilot.notifier import send_advancement_alert
    from nba_betting.props_autopilot.postmortem import LegSpec, analyze_leg_from_box

    # 1. Ayo Dosunmu MORE 17.5 PTS (DK 18.5) landing on exact 18 -> CLV_SAVED_WIN
    ayo = analyze_leg_from_box(
        LegSpec(
            game_id="401909978",
            player_name="Ayo Dosunmu",
            team="MIN",
            stat_type="Points",
            side="MORE",
            pp_line=17.5,
            dk_line_at_lock=18.5,
            exp_minutes=33.0,
            pregame_spread_abs=5.5,
        ),
        matchup="DAL@MIN",
        final_score="DAL 94, MIN 133",
        final_margin_abs=39,
        overtime_periods=0,
        player_stats={"MIN": "27", "PTS": "18", "REB": "2", "AST": "1", "FG": "8-15", "3PT": "2-4", "FT": "0-0", "PF": "1"},
    )
    assert ayo.outcome == "WIN"
    assert ayo.verdict_tag == "CLV_SAVED_WIN"
    assert ayo.clv_pts == 1.0

    # 2. Desmond Bane LESS 20.5 PTS (DK 19.5) going 6-10 from 3PT for 21 PTS -> GOOD_PROCESS_3PT_VARIANCE_LOSS
    bane = analyze_leg_from_box(
        LegSpec(
            game_id="401909297",
            player_name="Desmond Bane",
            team="ORL",
            stat_type="Points",
            side="LESS",
            pp_line=20.5,
            dk_line_at_lock=19.5,
            exp_minutes=35.0,
            pregame_spread_abs=8.5,
        ),
        matchup="MIL@ORL",
        final_score="MIL 99, ORL 107",
        final_margin_abs=8,
        overtime_periods=0,
        player_stats={"MIN": "35", "PTS": "21", "REB": "5", "AST": "3", "FG": "7-13", "3PT": "6-10", "FT": "1-1", "PF": "6"},
    )
    assert bane.outcome == "LOSS"
    assert bane.verdict_tag == "GOOD_PROCESS_3PT_VARIANCE_LOSS"
    assert bane.three_pt_points_over_expected >= 6.0

    # 3. Alert outbox logs cleanly when no SMS env vars are set
    outbox = tmp_path / "alerts.jsonl"
    rec = send_advancement_alert("RD-001 Promoted", "5-Flex OOS +60% ROI", log_path=outbox)
    assert rec["status"] == "queued_awaiting_provider"
    assert outbox.exists()
