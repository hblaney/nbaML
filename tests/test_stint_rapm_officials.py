"""Offline unit & anti-leakage tests for Stint RAPM, Garbage-Time Filter, and Officials."""

from __future__ import annotations

import pandas as pd

from nba_betting.backtest.stint_officials_eval import (
    CLEAN_RAPM_MARGIN_COLS,
    expanding_ridge_margin_walkforward,
    select_daily_parlay_tickets,
    simulate_seasonal_bankroll,
)
from nba_betting.data.espn_summary import (
    competitive_leverage_weight,
    is_garbage_time_state,
    normalize_official_name,
    parse_espn_summary,
)
from nba_betting.features.leverage_stints import (
    ExpandingTwoWayRapm,
    build_leverage_and_rapm_features,
)
from nba_betting.features.officials import build_officials_and_interaction_features


def _synthetic_summary(
    *,
    home_id: str = "1",
    away_id: str = "2",
    home_q4: int = 15,
    away_q4: int = 35,
    officials: list[str] | None = None,
) -> dict:
    offs = officials or ["J.T. Orr", "Pat Fraher", "Zach Zarba", "Alternate Ref"]
    return {
        "header": {
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "id": home_id,
                            "score": str(90 + home_q4),
                            "team": {"abbreviation": "HOM"},
                            "linescores": [
                                {"displayValue": "30"},
                                {"displayValue": "30"},
                                {"displayValue": "30"},
                                {"displayValue": str(home_q4)},
                            ],
                        },
                        {
                            "homeAway": "away",
                            "id": away_id,
                            "score": str(65 + away_q4),
                            "team": {"abbreviation": "AWY"},
                            "linescores": [
                                {"displayValue": "20"},
                                {"displayValue": "22"},
                                {"displayValue": "23"},
                                {"displayValue": str(away_q4)},
                            ],
                        },
                    ]
                }
            ]
        },
        "gameInfo": {
            "attendance": 18500,
            "officials": [
                {"displayName": name, "order": idx + 1} for idx, name in enumerate(offs)
            ],
        },
        "boxscore": {
            "teams": [
                {
                    "team": {"id": home_id},
                    "statistics": [
                        {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "40-85"},
                        {
                            "name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                            "displayValue": "12-35",
                        },
                        {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "13-16"},
                        {"name": "offensiveRebounds", "displayValue": "8"},
                        {"name": "defensiveRebounds", "displayValue": "34"},
                        {"name": "totalTurnovers", "displayValue": "12"},
                        {"name": "fouls", "displayValue": "18"},
                    ],
                },
                {
                    "team": {"id": away_id},
                    "statistics": [
                        {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "36-88"},
                        {
                            "name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                            "displayValue": "10-32",
                        },
                        {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "18-24"},
                        {"name": "offensiveRebounds", "displayValue": "10"},
                        {"name": "defensiveRebounds", "displayValue": "30"},
                        {"name": "totalTurnovers", "displayValue": "14"},
                        {"name": "fouls", "displayValue": "22"},
                    ],
                },
            ],
            "players": [
                {
                    "team": {"id": home_id},
                    "statistics": [
                        {
                            "keys": [
                                "minutes",
                                "points",
                                "fieldGoalsMade-fieldGoalsAttempted",
                                "freeThrowsMade-freeThrowsAttempted",
                                "fouls",
                                "plusMinus",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"id": f"h{k}", "displayName": f"Home {k}"},
                                    "starter": k <= 5,
                                    "didNotPlay": False,
                                    "stats": ["32", "18", "7-14", "2-3", "3", "+5"],
                                }
                                for k in range(1, 7)
                            ],
                        }
                    ],
                },
                {
                    "team": {"id": away_id},
                    "statistics": [
                        {
                            "keys": [
                                "minutes",
                                "points",
                                "fieldGoalsMade-fieldGoalsAttempted",
                                "freeThrowsMade-freeThrowsAttempted",
                                "fouls",
                                "plusMinus",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"id": f"a{k}", "displayName": f"Away {k}"},
                                    "starter": k <= 5,
                                    "didNotPlay": False,
                                    "stats": ["32", "16", "6-15", "3-4", "4", "-5"],
                                }
                                for k in range(1, 7)
                            ],
                        }
                    ],
                },
            ],
        },
        "plays": [
            # Q1 competitive basket (+30 home run across regulation simulated concisely)
            {
                "period": {"number": 1},
                "clock": {"displayValue": "10:00"},
                "type": {"id": "92", "text": "Jump Shot"},
                "team": {"id": home_id},
                "scoringPlay": True,
                "shootingPlay": True,
                "scoreValue": 28,
                "participants": [{"athlete": {"id": "h1"}}],
            },
            # Q4 blowout state (home up 28 with 4:00 left -> garbage time!)
            {
                "period": {"number": 4},
                "clock": {"displayValue": "4:00"},
                "type": {"id": "584", "text": "Substitution"},
                "team": {"id": home_id},
                "scoringPlay": False,
                "shootingPlay": False,
                "scoreValue": 0,
                "participants": [{"athlete": {"id": "h6"}}, {"athlete": {"id": "h1"}}],
            },
            # Away bench scores 20 meaningless garbage-time points in final 3 minutes
            {
                "period": {"number": 4},
                "clock": {"displayValue": "2:00"},
                "type": {"id": "92", "text": "Jump Shot"},
                "team": {"id": away_id},
                "scoringPlay": True,
                "shootingPlay": True,
                "scoreValue": 20,
                "participants": [{"athlete": {"id": "a5"}}],
            },
        ],
    }


def test_official_name_normalization_and_alternate_exclusion() -> None:
    assert normalize_official_name("J.T. Orr") == "jt orr"
    assert normalize_official_name("Joe DeRosa") == normalize_official_name("J.B. DeRosa")
    parsed = parse_espn_summary(_synthetic_summary(), "9001")
    g = parsed["game"]
    assert g["n_officials"] == 3
    assert (g["ref_1"], g["ref_2"], g["ref_3"]) == ("jt orr", "pat fraher", "zach zarba")


def test_garbage_time_filters_late_blowout_points() -> None:
    assert is_garbage_time_state(4, 240.0, 22.0) is True
    assert is_garbage_time_state(4, 240.0, 6.0) is False
    assert is_garbage_time_state(5, 60.0, 18.0) is False  # OT never garbage time
    assert competitive_leverage_weight(4, 120.0, 24.0) < competitive_leverage_weight(4, 120.0, 3.0)

    parsed = parse_espn_summary(_synthetic_summary(), "9002")
    g = parsed["game"]
    # The +28 Q1 points are clean; the +20 Q4 2:00 blowout points are filtered as garbage time!
    assert g["home_clean_pts"] == 28.0
    assert g["away_clean_pts"] == 0.0
    assert g["garb_sec"] > 0.0


def test_two_way_stint_rapm_identifies_dominant_lineup() -> None:
    rapm = ExpandingTwoWayRapm(lambda_off=200.0, lambda_def=200.0)
    stints = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "home_p1": "s1",
                "home_p2": "s2",
                "home_p3": "s3",
                "home_p4": "s4",
                "home_p5": "s5",
                "away_p1": "b1",
                "away_p2": "b2",
                "away_p3": "b3",
                "away_p4": "b4",
                "away_p5": "b5",
                "clean_poss_home": 60.0,
                "clean_poss_away": 60.0,
                "lev_poss_home": 60.0,
                "lev_poss_away": 60.0,
                "clean_pts_home": 84.0,  # 140 ORTG
                "clean_pts_away": 54.0,  # 90 ORTG
                "lev_pts_home": 84.0,
                "lev_pts_away": 54.0,
            }
        ]
    )
    rapm.add_game_stints(stints, league_ortg=114.5)
    orapm, drapm = rapm.current_ratings()
    assert orapm["s1"] > orapm["b1"]
    assert drapm["s1"] > drapm["b1"]


def test_strict_asof_zero_future_leakage_across_leverage_and_officials() -> None:
    """Mutating outcomes/fouls on Day 2 MUST NOT change any feature emitted on Day 1 or Day 2."""
    p1 = parse_espn_summary(_synthetic_summary(home_q4=25, away_q4=25), "1001")
    p2_a = parse_espn_summary(_synthetic_summary(home_q4=10, away_q4=45), "1002")
    p2_b = parse_espn_summary(_synthetic_summary(home_q4=55, away_q4=10), "1002")
    # Also mutate fouls & FTAs drastically in p2_b
    p2_b["game"]["home_pf"] = 38.0
    p2_b["game"]["away_pf"] = 39.0
    p2_b["game"]["home_fta"] = 58.0
    p2_b["game"]["away_fta"] = 55.0

    def _run(second_parsed: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        mkt = pd.DataFrame(
            [
                {
                    "season": "2024-25",
                    "season_type": "Regular Season",
                    "game_id": "1001",
                    "game_date": "2024-11-01",
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_tricode": "HOM",
                    "away_tricode": "AWY",
                    "home_score": 115.0,
                    "away_score": 90.0,
                    "spread_home": -5.5,
                    "total_line": 222.5,
                    "is_neutral": False,
                },
                {
                    "season": "2024-25",
                    "season_type": "Regular Season",
                    "game_id": "1002",
                    "game_date": "2024-11-03",
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_tricode": "HOM",
                    "away_tricode": "AWY",
                    "home_score": float(second_parsed["game"]["reg_home_pts"]),
                    "away_score": float(second_parsed["game"]["reg_away_pts"]),
                    "spread_home": -5.5,
                    "total_line": 222.5,
                    "is_neutral": False,
                },
            ]
        )
        sg = pd.DataFrame([p1["game"], second_parsed["game"]])
        sp = pd.DataFrame(p1["players"] + second_parsed["players"])
        ss = pd.DataFrame(p1["stints"] + second_parsed["stints"])
        lev = build_leverage_and_rapm_features(mkt, sg, sp, ss)
        off = build_officials_and_interaction_features(mkt, sg, lev)
        return lev, off

    lev_a, off_a = _run(p2_a)
    lev_b, off_b = _run(p2_b)
    pd.testing.assert_frame_equal(lev_a, lev_b)
    pd.testing.assert_frame_equal(off_a, off_b)


def test_parlay_push_reduction_and_50_bankroll_compounding() -> None:
    legs = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "game_date": "2024-11-10",
                "game_id": "g1",
                "market": "ats",
                "side": "home_ats",
                "p_model": 0.58,
                "p_vigged": 0.5238,
                "vig_edge": 0.0562,
                "american": -110.0,
                "decimal": 1.9090909,
                "outcome": "win",
            },
            {
                "season": "2024-25",
                "game_date": "2024-11-10",
                "game_id": "g2",
                "market": "total",
                "side": "over",
                "p_model": 0.57,
                "p_vigged": 0.5238,
                "vig_edge": 0.0462,
                "american": -110.0,
                "decimal": 1.9090909,
                "outcome": "push",
            },
        ]
    )
    tix = select_daily_parlay_tickets(legs, n_legs=2, min_price_edge=0.05)
    assert len(tix) == 1
    # Win + Push reduces 2-leg parlay to single winning leg decimal (~1.9091x)
    assert abs(float(tix.iloc[0]["settled_mult"]) - 1.9090909) < 1e-5
    sim = simulate_seasonal_bankroll(tix, starting_bankroll=50.0, stake_pct=0.04, seasons=("2024-25",))
    # 4% of $50 = $2.00 stake; payout = round(2.00 * 1.9090909, 2) = $3.82 -> +$1.82 -> $51.82
    assert sim["2024-25"]["end_bankroll"] == 51.82
    assert sim["2024-25"]["beat_start"] is True


def test_expanding_ridge_margin_walkforward_smoke() -> None:
    n = 320
    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2023-10-24", periods=n, freq="D").astype(str),
            "game_id": [f"g{i}" for i in range(n)],
            "is_neutral": [False] * n,
            "home_margin": [4.0 if i % 2 == 0 else -2.0 for i in range(n)],
            **{c: [0.5 if i % 2 == 0 else -0.5 for i in range(n)] for c in CLEAN_RAPM_MARGIN_COLS},
        }
    )
    _m, _s, p = expanding_ridge_margin_walkforward(df, CLEAN_RAPM_MARGIN_COLS, min_train_games=80)
    assert len(p) == n
    assert (p > 0.0).all() and (p < 1.0).all()
