"""Unit tests for PrizePicks payout math, push reversions, stat calibration, and live cashout EV."""

from __future__ import annotations

from nba_betting.props_autopilot.prizepicks_math import (
    apply_stat_tail_calibration,
    evaluate_live_cashout,
    exact_hits_pmf,
    passes_soft_stale_line,
    select_slip_legs,
    settle_prizepicks_slip,
    signed_stale_soft,
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


def test_select_slip_legs_enforces_rd091_prop_type_diversity() -> None:
    """v2.5: 5-Flex must span ≥2 distinct prop_type categories (RD-091)."""
    # Top-5 by p would be all Points across unique games — diversity must swap in Pts+Asts.
    cands = [
        {"game_id": "g1", "stat": "Points", "side": "MORE", "player": "A", "p_calibrated": 0.60},
        {"game_id": "g2", "stat": "Points", "side": "LESS", "player": "B", "p_calibrated": 0.59},
        {"game_id": "g3", "stat": "Points", "side": "LESS", "player": "C", "p_calibrated": 0.58},
        {"game_id": "g4", "stat": "Points", "side": "LESS", "player": "D", "p_calibrated": 0.57},
        {"game_id": "g5", "stat": "Points", "side": "LESS", "player": "E", "p_calibrated": 0.56},
        {"game_id": "g6", "stat": "Pts+Asts", "side": "MORE", "player": "F", "p_calibrated": 0.555},
    ]
    legs = select_slip_legs(cands, n_legs=5, min_distinct_prop_types=2)
    assert len(legs) == 5
    assert {x["stat"] for x in legs} == {"Points", "Pts+Asts"}
    assert legs[-1]["player"] == "F"

    # Same-game duplicate skipped; still forms diverse 5.
    with_dupe = [
        {"game_id": "g1", "stat": "Points", "side": "MORE", "player": "A", "p_calibrated": 0.61},
        {"game_id": "g1", "stat": "Assists", "side": "MORE", "player": "A2", "p_calibrated": 0.605},
        *cands[1:],
    ]
    legs2 = select_slip_legs(with_dupe, n_legs=5, min_distinct_prop_types=2)
    assert len(legs2) == 5
    assert len({x["game_id"] for x in legs2}) == 5
    assert len({x["stat"] for x in legs2}) >= 2

    # Impossible diversity → empty (all Points, only 5 games).
    mono = cands[:5]
    assert select_slip_legs(mono, n_legs=5, min_distinct_prop_types=2) == []

    # Without the floor, mono 5-Flex is allowed (legacy behavior for non-5 or opt-out).
    assert len(select_slip_legs(mono, n_legs=5, min_distinct_prop_types=None)) == 5


def test_soft1_signed_stale_rejects_flat_and_wrong_way() -> None:
    """v2.5 soft1.0: |PP-DK|<1 and wrong-way gaps cannot qualify."""
    # Sabonis-class flat LESS (Mon 12 AM CT preview bug).
    assert signed_stale_soft("LESS", 4.5, 4.5) == 0.0
    assert not passes_soft_stale_line("LESS", 4.5, 4.5, min_abs=1.0)
    # Shannon-class flat MORE.
    assert not passes_soft_stale_line("MORE", 5.5, 5.5, min_abs=1.0)
    # Wrong-way: MORE with PP above DK, LESS with PP below DK.
    assert signed_stale_soft("MORE", 12.5, 11.5) == -1.0
    assert not passes_soft_stale_line("MORE", 12.5, 11.5, min_abs=1.0)
    assert signed_stale_soft("LESS", 10.5, 11.5) == -1.0
    assert not passes_soft_stale_line("LESS", 10.5, 11.5, min_abs=1.0)
    # Aligned soft ≥1.0 passes (Fears MORE / Camara LESS).
    assert passes_soft_stale_line("MORE", 22.5, 24.5, min_abs=1.0)
    assert passes_soft_stale_line("LESS", 11.5, 10.5, min_abs=1.0)
    # Sub-1 absolute gap fails even if directionally aligned.
    assert not passes_soft_stale_line("MORE", 20.0, 20.5, min_abs=1.0)
    assert not passes_soft_stale_line("LESS", 15.5, 15.0, min_abs=1.0)


def test_recommended_slip_candidates_exclude_soft_violations() -> None:
    """Live recommended pool must drop soft1.0 failures before select_slip_legs."""
    cands = [
        # Flat LESS — Sabonis class (must not enter).
        {
            "game_id": "g_sab",
            "cname": "domantas sabonis",
            "player": "Domantas Sabonis",
            "stat": "Assists",
            "side": "LESS",
            "pp_line": 4.5,
            "dk_line": 4.5,
            "p_calibrated": 0.99,
        },
        # Wrong-way MORE.
        {
            "game_id": "g_ww",
            "cname": "wrong way",
            "player": "Wrong Way",
            "stat": "Points",
            "side": "MORE",
            "pp_line": 18.5,
            "dk_line": 17.5,
            "p_calibrated": 0.98,
        },
        # Valid soft MORE / LESS board that can fill 5-Flex under RD-091/103.
        {
            "game_id": "g1",
            "cname": "a",
            "player": "A",
            "stat": "Pts+Asts",
            "side": "MORE",
            "pp_line": 22.5,
            "dk_line": 24.5,
            "p_calibrated": 0.60,
        },
        {
            "game_id": "g2",
            "cname": "b",
            "player": "B",
            "stat": "Points",
            "side": "LESS",
            "pp_line": 11.5,
            "dk_line": 10.5,
            "p_calibrated": 0.59,
        },
        {
            "game_id": "g3",
            "cname": "c",
            "player": "C",
            "stat": "Points",
            "side": "LESS",
            "pp_line": 13.5,
            "dk_line": 12.5,
            "p_calibrated": 0.58,
        },
        {
            "game_id": "g4",
            "cname": "d",
            "player": "D",
            "stat": "Rebs+Asts",
            "side": "MORE",
            "pp_line": 5.5,
            "dk_line": 6.5,
            "p_calibrated": 0.57,
        },
        {
            "game_id": "g5",
            "cname": "e",
            "player": "E",
            "stat": "Assists",
            "side": "LESS",
            "pp_line": 6.5,
            "dk_line": 5.5,
            "p_calibrated": 0.56,
        },
    ]
    soft_ok = [
        x
        for x in cands
        if passes_soft_stale_line(x["side"], x["pp_line"], x["dk_line"], min_abs=1.0)
    ]
    assert all(x["cname"] not in {"domantas sabonis", "wrong way"} for x in soft_ok)
    legs = select_slip_legs(
        soft_ok, n_legs=5, min_distinct_prop_types=2, max_points_more=1
    )
    assert len(legs) == 5
    names = {x["cname"] for x in legs}
    assert "domantas sabonis" not in names
    assert "wrong way" not in names
    for leg in legs:
        assert passes_soft_stale_line(leg["side"], leg["pp_line"], leg["dk_line"], min_abs=1.0)


def test_select_slip_legs_enforces_unique_player() -> None:
    cands = [
        {"game_id": "g1", "cname": "same", "player": "Same", "stat": "Points", "side": "MORE", "p_calibrated": 0.62},
        {"game_id": "g2", "cname": "same", "player": "Same", "stat": "Assists", "side": "MORE", "p_calibrated": 0.61},
        {"game_id": "g3", "cname": "b", "player": "B", "stat": "Pts+Asts", "side": "MORE", "p_calibrated": 0.60},
        {"game_id": "g4", "cname": "c", "player": "C", "stat": "Rebounds", "side": "LESS", "p_calibrated": 0.59},
        {"game_id": "g5", "cname": "d", "player": "D", "stat": "Points", "side": "LESS", "p_calibrated": 0.58},
        {"game_id": "g6", "cname": "e", "player": "E", "stat": "Assists", "side": "LESS", "p_calibrated": 0.57},
    ]
    legs = select_slip_legs(cands, n_legs=5, min_distinct_prop_types=2, max_points_more=1)
    assert len(legs) == 5
    assert len({x["cname"] for x in legs}) == 5


def test_select_slip_legs_enforces_rd103_max_points_more() -> None:
    """v2.5-soft1.0-div2-ptsmore1: at most one Points MORE leg on 5-Flex."""
    cands = [
        {"game_id": "g1", "stat": "Points", "side": "MORE", "player": "A", "p_calibrated": 0.62},
        {"game_id": "g2", "stat": "Points", "side": "MORE", "player": "B", "p_calibrated": 0.61},
        {"game_id": "g3", "stat": "Pts+Asts", "side": "MORE", "player": "C", "p_calibrated": 0.60},
        {"game_id": "g4", "stat": "Pts+Rebs", "side": "LESS", "player": "D", "p_calibrated": 0.59},
        {"game_id": "g5", "stat": "Assists", "side": "MORE", "player": "E", "p_calibrated": 0.58},
        {"game_id": "g6", "stat": "Points", "side": "LESS", "player": "F", "p_calibrated": 0.57},
    ]
    legs = select_slip_legs(
        cands, n_legs=5, min_distinct_prop_types=2, max_points_more=1
    )
    assert len(legs) == 5
    points_more = [x for x in legs if x["stat"] == "Points" and x["side"] == "MORE"]
    assert len(points_more) == 1
    assert points_more[0]["player"] == "A"
    # Second Points MORE (B) skipped; Points LESS (F) still allowed.
    assert "B" not in {x["player"] for x in legs}
    assert "F" in {x["player"] for x in legs}

    # Combined with diversity: all Points MORE board cannot fill 5 under both gates.
    mono_more = [
        {"game_id": f"g{i}", "stat": "Points", "side": "MORE", "player": f"P{i}", "p_calibrated": 0.60 - 0.01 * i}
        for i in range(6)
    ]
    assert select_slip_legs(mono_more, n_legs=5, min_distinct_prop_types=2, max_points_more=1) == []


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
