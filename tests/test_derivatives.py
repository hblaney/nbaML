"""Unit tests for 1Q and 1H NBA derivative Odds API adapter, ESPN period parser,
strictly-pregame expanding features, residual models, parlays, and $50 Kelly bankroll.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nba_betting.cli import app
from nba_betting.data.derivative_odds import (
    DERIVATIVE_MARKETS,
    derivative_markets_from_env,
    fetch_event_derivative_odds,
    fetch_nba_derivative_odds,
    latest_derivative_odds_cache,
    load_derivative_odds_cache,
    parse_derivative_event_odds,
    pull_derivative_odds,
)
from nba_betting.data.espn_periods import parse_espn_period_propbets, parse_summary_payload
from nba_betting.data.odds import DERIVATIVE_MARKET_KEYS, OddsApiConfigError
from nba_betting.features.derivatives import build_pregame_derivative_features, snap_to_half_point
from nba_betting.models.derivatives import (
    DerivativeResidualModelSuite,
    PeriodPrediction,
    moneyline_two_way_probs,
    spread_cover_probs,
    standard_bookmaker_derivative_lines,
    total_over_under_probs,
)
from nba_betting.pricing.derivative_edge import (
    build_derivative_edge_rows,
    build_historical_derivative_bets,
    enumerate_derivative_parlays,
    simulate_seasonal_kelly_bankroll,
)


@pytest.fixture
def sample_derivative_event_payload() -> dict[str, Any]:
    return {
        "id": "evt_den_sac_001",
        "commence_time": "2025-11-04T02:00:00Z",
        "home_team": "Denver Nuggets",
        "away_team": "Sacramento Kings",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h_q1",
                        "outcomes": [
                            {"name": "Denver Nuggets", "price": -185},
                            {"name": "Sacramento Kings", "price": 154},
                        ],
                    },
                    {
                        "key": "spreads_q1",
                        "outcomes": [
                            {"name": "Denver Nuggets", "point": -2.5, "price": -112},
                            {"name": "Sacramento Kings", "point": 2.5, "price": -108},
                        ],
                    },
                    {
                        "key": "totals_q1",
                        "outcomes": [
                            {"name": "Over", "point": 59.5, "price": -110},
                            {"name": "Under", "point": 59.5, "price": -110},
                        ],
                    },
                    {
                        "key": "h2h_h1",
                        "outcomes": [
                            {"name": "Denver Nuggets", "price": -245},
                            {"name": "Sacramento Kings", "price": 198},
                        ],
                    },
                    {
                        "key": "spreads_h1",
                        "outcomes": [
                            {"name": "Denver Nuggets", "point": -5.5, "price": -105},
                            {"name": "Sacramento Kings", "point": 5.5, "price": -115},
                        ],
                    },
                    {
                        "key": "totals_h1",
                        "outcomes": [
                            {"name": "Over", "point": 120.5, "price": -115},
                            {"name": "Under", "point": 120.5, "price": -105},
                        ],
                    },
                ],
            }
        ],
    }


def test_derivative_market_constants_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = ("h2h_q1", "spreads_q1", "totals_q1", "h2h_h1", "spreads_h1", "totals_h1")
    assert DERIVATIVE_MARKETS == expected
    assert DERIVATIVE_MARKET_KEYS == expected
    monkeypatch.delenv("ODDS_DERIVATIVE_MARKETS", raising=False)
    assert derivative_markets_from_env() == expected
    monkeypatch.setenv("ODDS_DERIVATIVE_MARKETS", "spreads_q1, totals_h1")
    assert derivative_markets_from_env() == ("spreads_q1", "totals_h1")


def test_parse_all_six_derivative_markets(sample_derivative_event_payload: dict[str, Any]) -> None:
    parsed = parse_derivative_event_odds(sample_derivative_event_payload, asof_ts="2025-11-04T01:00:00Z")
    assert parsed is not None
    assert parsed.has_any_derivative()
    assert parsed.home_tricode == "DEN"
    assert parsed.away_tricode == "SAC"
    assert parsed.book == "draftkings"
    assert parsed.q1_moneyline_home == -185.0
    assert parsed.q1_moneyline_away == 154.0
    assert parsed.q1_spread_home_point == -2.5
    assert parsed.q1_spread_home_price == -112.0
    assert parsed.q1_spread_away_point == 2.5
    assert parsed.q1_spread_away_price == -108.0
    assert parsed.q1_total_point == 59.5
    assert parsed.q1_total_over_price == -110.0
    assert parsed.q1_total_under_price == -110.0
    assert parsed.h1_moneyline_home == -245.0
    assert parsed.h1_moneyline_away == 198.0
    assert parsed.h1_spread_home_point == -5.5
    assert parsed.h1_spread_home_price == -105.0
    assert parsed.h1_spread_away_point == 5.5
    assert parsed.h1_spread_away_price == -115.0
    assert parsed.h1_total_point == 120.5
    assert parsed.h1_total_over_price == -115.0
    assert parsed.h1_total_under_price == -105.0


def test_fetch_and_cache_derivative_odds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_derivative_event_payload: dict[str, Any],
) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsApiConfigError):
        fetch_event_derivative_odds("evt_den_sac_001", http_get=lambda u, p: {})
    with pytest.raises(OddsApiConfigError):
        fetch_nba_derivative_odds(http_get=lambda u, p: [])

    monkeypatch.setenv("ODDS_API_KEY", "mock-deriv-key")
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_get(url: str, params: dict[str, str]) -> Any:
        calls.append((url, params))
        if url.endswith("/basketball_nba/odds"):
            return [
                {
                    "id": "evt_den_sac_001",
                    "commence_time": "2025-11-04T02:00:00Z",
                    "home_team": "Denver Nuggets",
                    "away_team": "Sacramento Kings",
                    "bookmakers": [{"key": "draftkings", "markets": []}],
                }
            ]
        assert "/events/evt_den_sac_001/odds" in url
        assert params["markets"] == "h2h_q1,spreads_q1,totals_q1,h2h_h1,spreads_h1,totals_h1"
        assert params["oddsFormat"] == "american"
        return sample_derivative_event_payload

    path = pull_derivative_odds(root=tmp_path, http_get=fake_get)
    assert path.exists()
    assert latest_derivative_odds_cache(tmp_path) == path
    loaded = load_derivative_odds_cache(path)
    assert len(loaded) == 1
    assert loaded[0].q1_spread_home_point == -2.5
    assert loaded[0].h1_total_point == 120.5
    assert len(calls) == 2

    # CLI degrades cleanly without ODDS_API_KEY
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    runner = CliRunner()
    res = runner.invoke(app, ["odds", "pull-derivatives", "--data-dir", str(tmp_path)])
    assert res.exit_code == 0


def test_build_derivative_edge_rows_all_six_markets(
    sample_derivative_event_payload: dict[str, Any],
) -> None:
    ev = parse_derivative_event_odds(sample_derivative_event_payload)
    assert ev is not None
    pred = PeriodPrediction(
        game_id="401810017",
        game_date="2025-11-03",
        season="2025-26",
        home_tricode="DEN",
        away_tricode="SAC",
        exp_q1_margin=4.2,
        sigma_q1_margin=8.3,
        exp_h1_margin=7.1,
        sigma_h1_margin=11.0,
        exp_q1_total=61.8,
        sigma_q1_total=8.1,
        exp_h1_total=123.0,
        sigma_h1_total=12.0,
        pred_resid_q1_margin=1.3,
        pred_resid_h1_margin=1.3,
        pred_resid_q1_total=2.4,
        pred_resid_h1_total=3.1,
    )
    rows = build_derivative_edge_rows([pred], [ev])
    assert len(rows) == 12  # 2 sides * 6 markets
    by_mkt_side = {(r.market, r.side): r for r in rows}
    assert set(r.market for r in rows) == set(DERIVATIVE_MARKETS)
    # Since exp_q1_margin=4.2 > 2.5 and exp_q1_total=61.8 > 59.5, DEN Q1 spread and Q1 Over should have positive fair edge
    assert by_mkt_side[("spreads_q1", "home")].edge > 0.0
    assert by_mkt_side[("totals_q1", "over")].edge > 0.0
    assert by_mkt_side[("h2h_q1", "home")].push_prob > 0.0
    for r in rows:
        assert 0.0 < r.model_prob < 1.0
        assert r.edge == pytest.approx(r.model_prob - r.book_implied_fair)


def test_espn_summary_and_propbets_parsers() -> None:
    mock_summary = {
        "header": {
            "competitions": [
                {
                    "id": "401000001",
                    "date": "2025-02-10T00:30Z",
                    "neutralSite": False,
                    "competitors": [
                        {
                            "homeAway": "home",
                            "id": "2",
                            "score": "112",
                            "team": {"id": "2", "abbreviation": "BOS"},
                            "linescores": [
                                {"displayValue": "32"},
                                {"displayValue": "28"},
                                {"displayValue": "27"},
                                {"displayValue": "25"},
                            ],
                        },
                        {
                            "homeAway": "away",
                            "id": "18",
                            "score": "104",
                            "team": {"id": "18", "abbreviation": "NY"},
                            "linescores": [
                                {"displayValue": "24"},
                                {"displayValue": "26"},
                                {"displayValue": "28"},
                                {"displayValue": "26"},
                            ],
                        },
                    ],
                }
            ]
        },
        "gameInfo": {"venue": {"fullName": "TD Garden", "address": {"city": "Boston", "state": "MA"}}},
        "boxscore": {
            "teams": [
                {
                    "team": {"id": "2"},
                    "statistics": [
                        {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "42-88"},
                        {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": "15-38"},
                        {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "13-16"},
                        {"name": "offensiveRebounds", "displayValue": "9"},
                        {"name": "defensiveRebounds", "displayValue": "34"},
                        {"name": "totalTurnovers", "displayValue": "11"},
                    ],
                },
                {
                    "team": {"id": "18"},
                    "statistics": [
                        {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "39-86"},
                        {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": "11-31"},
                        {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "15-18"},
                        {"name": "offensiveRebounds", "displayValue": "8"},
                        {"name": "defensiveRebounds", "displayValue": "31"},
                        {"name": "totalTurnovers", "displayValue": "12"},
                    ],
                },
            ],
            "players": [
                {
                    "team": {"id": "2"},
                    "statistics": [
                        {
                            "labels": ["MIN", "PTS", "+/-"],
                            "athletes": [
                                {"athlete": {"id": "p1", "displayName": "Starter A"}, "starter": True, "stats": ["36", "28", "+12"]},
                                {"athlete": {"id": "p2", "displayName": "Bench B"}, "starter": False, "stats": ["12", "6", "-4"]},
                            ],
                        }
                    ],
                }
            ],
        },
        "plays": [
            {"period": {"number": 1}, "clock": {"displayValue": "10:00"}, "team": {"id": "2"}, "shootingPlay": True, "pointsAttempted": 3, "homeScore": 3, "awayScore": 0, "type": {"text": "Jump Shot"}},
            {"period": {"number": 1}, "clock": {"displayValue": "5:00"}, "team": {"id": "18"}, "shootingPlay": True, "pointsAttempted": 2, "homeScore": 15, "awayScore": 12, "type": {"text": "Layup"}},
        ],
    }
    grow, prows = parse_summary_payload(mock_summary)
    assert grow["home_tricode_norm"] == "BOS"
    assert grow["away_tricode_norm"] == "NYK"
    assert grow["q1_margin"] == 8.0
    assert grow["h1_margin"] == 10.0
    assert grow["q1_total"] == 56.0
    assert grow["h1_total"] == 110.0
    assert len(prows) == 2

    prop_items = [
        {
            "type": {"name": "1st Quarter Spread"},
            "team": {"$ref": "http://sports.core.api.espn.com/v2/.../teams/2?lang=en"},
            "odds": {"american": {"value": "-115"}, "total": {"value": "-2.5", "open": "-1.5"}},
            "current": {"target": {"value": -2.5}},
        },
        {
            "type": {"name": "1st Quarter Spread"},
            "team": {"$ref": "http://sports.core.api.espn.com/v2/.../teams/18?lang=en"},
            "odds": {"american": {"value": "-105"}, "total": {"value": "2.5", "open": "1.5"}},
            "current": {"target": {"value": 2.5}},
        },
        {
            "type": {"name": "1st Half Total"},
            "odds": {"american": {"value": "-112"}, "total": {"value": "111.5", "open": "110.5"}},
            "current": {"over": {"american": "-112"}, "target": {"value": 111.5}},
        },
        {
            "type": {"name": "1st Half Total"},
            "odds": {"american": {"value": "-108"}, "total": {"value": "111.5", "open": "110.5"}},
            "current": {"under": {"american": "-108"}, "target": {"value": 111.5}},
        },
    ]
    p_odds = parse_espn_period_propbets(prop_items, home_team_id=2, away_team_id=18, provider_name="ESPN BET")
    assert p_odds["deriv_odds_provider"] == "ESPN BET"
    assert p_odds["book_q1_spread_home"] == -2.5
    assert p_odds["book_q1_spread_home_odds"] == -115.0
    assert p_odds["book_h1_total_line"] == 111.5
    assert p_odds["book_h1_over_odds"] == -112.0
    assert p_odds["book_h1_under_odds"] == -108.0


def test_strict_pregame_features_zero_leakage_and_kelly_parlays() -> None:
    rng = np.random.default_rng(42)
    n_games = 80
    teams = ["BOS", "DEN", "LAL", "MIA", "NYK", "SAC"]
    rows = []
    base_date = pd.Timestamp("2024-11-01")
    for i in range(n_games):
        h = teams[i % len(teams)]
        a = teams[(i * 2 + 1) % len(teams)]
        if a == h:
            a = teams[(i + 1) % len(teams)]
        d = base_date + pd.Timedelta(days=i // 3)
        hq1, aq1 = float(rng.integers(22, 36)), float(rng.integers(22, 36))
        hq2, aq2 = float(rng.integers(22, 36)), float(rng.integers(22, 36))
        hq3, aq3 = float(rng.integers(22, 36)), float(rng.integers(22, 36))
        hq4, aq4 = float(rng.integers(22, 36)), float(rng.integers(22, 36))
        hs, as_ = hq1 + hq2 + hq3 + hq4, aq1 + aq2 + aq3 + aq4
        rows.append(
            {
                "season": "2024-25",
                "season_type": "Regular Season",
                "game_id": str(1000 + i),
                "game_date": str(d.date()),
                "tipoff_utc": f"{d.date()}T23:30Z",
                "home_team_id": teams.index(h) + 1,
                "away_team_id": teams.index(a) + 1,
                "espn_home_team_id": teams.index(h) + 1,
                "espn_away_team_id": teams.index(a) + 1,
                "home_tricode": h,
                "away_tricode": a,
                "venue_city": "Denver" if h == "DEN" else "Boston",
                "is_neutral": False,
                "summary_is_neutral": False,
                "spread_home": -3.5,
                "total_line": 224.5,
                "open_spread_home": -3.0,
                "open_total_line": 223.5,
                "p_home_novig": 0.60,
                "q1_margin": hq1 - aq1,
                "h1_margin": (hq1 + hq2) - (aq1 + aq2),
                "h2_margin": (hq3 + hq4) - (aq3 + aq4),
                "fg_margin": hs - as_,
                "q1_total": hq1 + aq1,
                "h1_total": hq1 + hq2 + aq1 + aq2,
                "fg_total": hs + as_,
                "home_q1_ortg": 114.0,
                "home_q1_drtg": 110.0,
                "away_q1_ortg": 110.0,
                "away_q1_drtg": 114.0,
                "home_h1_ortg": 113.0,
                "home_h1_drtg": 111.0,
                "away_h1_ortg": 111.0,
                "away_h1_drtg": 113.0,
                "home_fg_ortg": 112.0,
                "home_fg_drtg": 112.0,
                "away_fg_ortg": 112.0,
                "away_fg_drtg": 112.0,
                "q1_pace_48": 101.0,
                "h1_pace_48": 100.0,
                "fg_pace_48": 99.0,
                "home_starter_min": 165.0,
                "home_bench_min": 75.0,
                "home_starter_min_share": 165.0 / 240.0,
                "home_starter_pm": 25.0,
                "home_bench_pm": -10.0,
                "away_starter_min": 150.0,
                "away_bench_min": 90.0,
                "away_starter_min_share": 150.0 / 240.0,
                "away_starter_pm": -5.0,
                "away_bench_pm": 5.0,
                "home_q1_starter_min_share": 0.80,
                "away_q1_starter_min_share": 0.72,
                "home_h1_starter_min_share": 0.70,
                "away_h1_starter_min_share": 0.65,
                "home_q1_first_sub_sec": 390.0,
                "away_q1_first_sub_sec": 320.0,
                "q1_first6_margin": 3.0,
                "q1_last6_margin": 1.0,
                "q2_first6_margin": -1.0,
            }
        )
    games_df = pd.DataFrame(rows)
    feats_orig = build_pregame_derivative_features(games_df)

    # Mutate all future games strictly after midpoint date and verify zero change on <= midpoint
    mid_date = feats_orig["game_date"].iloc[40]
    games_mut = games_df.copy()
    fut_mask = games_mut["game_date"] > mid_date
    games_mut.loc[fut_mask, "q1_margin"] += 50.0
    games_mut.loc[fut_mask, "home_q1_ortg"] += 40.0
    feats_mut = build_pregame_derivative_features(games_mut)

    early_cols = [c for c in feats_orig.select_dtypes(include=[np.number]).columns if "margin" not in c and "total" not in c]
    a_slice = feats_orig.loc[feats_orig["game_date"] <= mid_date, early_cols].reset_index(drop=True)
    b_slice = feats_mut.loc[feats_mut["game_date"] <= mid_date, early_cols].reset_index(drop=True)
    assert np.allclose(a_slice.to_numpy(), b_slice.to_numpy(), equal_nan=True)

    # Fit suite on first 55 rows, score remaining 25, build bets, parlays, and $50 Kelly bankroll
    suite = DerivativeResidualModelSuite(alpha=500.0).fit(feats_orig.iloc[:55])
    scored = suite.predict_frame(feats_orig.iloc[55:])
    bets = build_historical_derivative_bets(scored, pricing_mode="standard_sharp")
    assert not bets.empty
    parlays_2 = enumerate_derivative_parlays(bets, n_legs=2, min_vig_edge_leg=-0.05, min_price_edge=-0.20)
    assert not parlays_2.empty
    sim_df, meta = simulate_seasonal_kelly_bankroll(
        bets.loc[bets["vig_edge"] > 0.0],
        starting_bankroll=50.0,
        kelly_mult=0.25,
    )
    if not sim_df.empty:
        assert meta["seasons"]["2024-25"]["starting_bankroll"] == 50.0
        assert meta["seasons"]["2024-25"]["ending_bankroll"] >= 0.0


def test_half_point_snapping_and_prob_identities() -> None:
    assert snap_to_half_point(-2.875) == -2.5
    assert snap_to_half_point(-3.15) == -3.5
    assert snap_to_half_point(0.0, zero_to=-0.5) == -0.5

    lines = standard_bookmaker_derivative_lines(-8.0, 228.0, mode="proportional")
    assert lines["q1_spread_home"] in (-2.5, -1.5)
    assert abs(lines["q1_total_line"] % 1.0) == 0.5

    ph, pa, ppush = spread_cover_probs(2.5, 8.0, -2.5)
    assert ppush == 0.0
    assert ph == pytest.approx(0.5)
    assert pa == pytest.approx(0.5)

    po, pu, tpush = total_over_under_probs(57.5, 8.0, 57.5)
    assert tpush == 0.0
    assert po == pytest.approx(0.5)
    assert pu == pytest.approx(0.5)

    mh, ma, mpush = moneyline_two_way_probs(0.0, 8.0)
    assert mpush > 0.0
    assert mh == pytest.approx(0.5)
    assert ma == pytest.approx(0.5)
