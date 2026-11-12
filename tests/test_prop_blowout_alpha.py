"""Unit tests for non-linear teammate redistribution, bimodal blowout mixtures,
live Odds API multi-book prop markets, and cross-game prop parlay + Kelly sizing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_betting.backtest.prop_walkforward import (
    build_walkforward_prop_frame,
    fit_expanding_prop_calibrators,
    summarize_prop_calibration,
)
from nba_betting.data.prop_odds import (
    DEFAULT_PROP_MARKETS,
    fetch_event_prop_odds,
    parse_prop_markets,
)
from nba_betting.features.prop_redistribution import (
    RedistributionTracker,
    build_tracker_asof,
)
from nba_betting.models.prop_distribution import (
    build_regime_mixture,
    p_over_mixture,
    p_over_nbinom,
    p_under_mixture,
    project_distributional_props,
    regime_weights_from_spread,
)
from nba_betting.pricing.prop_edge import build_prop_edge_rows
from nba_betting.pricing.prop_parlay import (
    build_eligible_prop_legs,
    enumerate_prop_parlays,
    parlay_kelly_fraction,
    select_nonoverlapping_prop_parlays,
    simulate_prop_parlay_kelly,
)


def _synthetic_team_logs() -> pd.DataFrame:
    """Create 8 prior games for a team with a star, secondary creator, 6th man, and deep bench."""
    rows = []
    dates = pd.date_range("2024-11-01", periods=9, freq="2D")
    for i, gd in enumerate(dates):
        gid = f"G{i + 1:02d}"
        # Game 8 (index 7): Star sits (0 minutes / DNP), 6th man starts!
        star_min = 0.0 if i == 7 else 35.0
        sixth_start = True if i == 7 else False
        sixth_min = 31.0 if i == 7 else 21.0
        sec_min = 36.0 if i == 7 else 32.0
        deep_min = 12.0 if i == 7 else 6.0

        players = [
            (
                "P_STAR",
                "Alpha Star",
                True,
                star_min,
                30.0 if star_min > 0 else 0.0,
                6.0,
                8.0,
                3.0,
                22.0,
                7.0,
                3.5,
            ),
            (
                "P_SEC",
                "Beta Guard",
                True,
                sec_min,
                22.0 if i == 7 else 18.0,
                4.0,
                7.0 if i == 7 else 5.5,
                2.5,
                15.0,
                4.0,
                2.5,
            ),
            (
                "P_SIX",
                "Sixth Wing",
                sixth_start,
                sixth_min,
                16.0 if i == 7 else 10.0,
                6.0,
                2.5,
                2.0,
                10.0,
                2.0,
                1.0,
            ),
            ("P_BIG", "Rim Center", True, 29.0, 12.0, 11.0, 2.0, 0.0, 8.0, 3.0, 1.5),
            ("P_DEEP", "Deep Bench", False, deep_min, 3.0, 1.5, 0.5, 0.5, 3.0, 0.5, 0.5),
        ]
        for pid, pname, starter, mins, pts, reb, ast, fg3m, fga, fta, tov in players:
            rows.append(
                {
                    "season": "2024-25",
                    "game_id": gid,
                    "game_date": str(gd.date()),
                    "team_id": 10,
                    "opp_team_id": 20,
                    "team_tricode": "BOS",
                    "is_home": True,
                    "player_id": pid,
                    "player_name": pname,
                    "starter": starter,
                    "dnp": mins == 0.0,
                    "minutes": mins,
                    "pts": pts,
                    "reb": reb,
                    "ast": ast,
                    "fg3m": fg3m,
                    "fga": fga,
                    "fta": fta,
                    "tov": tov,
                    "pra": pts + reb + ast,
                }
            )
    return pd.DataFrame(rows)


def test_nonlinear_redistribution_strictly_pregame_no_leakage() -> None:
    logs = _synthetic_team_logs()
    # Replay before Game 8 (star healthy across G01..G07) vs before Game 9 (star missed G08)
    asof_healthy = pd.Timestamp("2024-11-15")  # G08 date -> uses G01..G07 only
    asof_star_out = pd.Timestamp("2024-11-17")  # G09 date -> uses G01..G08 (knows star missed G08)

    tr_healthy = build_tracker_asof(logs, asof_healthy)
    tr_out = build_tracker_asof(logs, asof_star_out)

    # Verify G09 row was NOT touched in tr_out
    assert tr_out.states["P_SEC"].games == 8

    proj_h = tr_healthy.project_team_pregame(
        10, 20, asof_healthy, team_spread=-4.0, total_line=226.5
    )
    proj_o = tr_out.project_team_pregame(10, 20, asof_star_out, team_spread=-4.0, total_line=226.5)

    # Secondary high-usage creator gets a larger usage/assist rate boost than low-usage center
    sec_pts_gain = proj_o["P_SEC"].pts_per_min / proj_h["P_SEC"].pts_per_min - 1.0
    big_pts_gain = proj_o["P_BIG"].pts_per_min / proj_h["P_BIG"].pts_per_min - 1.0
    assert sec_pts_gain > big_pts_gain
    assert proj_o["P_SEC"].ast_per_min > proj_h["P_SEC"].ast_per_min

    # Promoted 6th man (headroom + spot-starter flag) gains more minutes than already-32m Beta Guard
    assert proj_o["P_SIX"].spot_starter_flag == 1.0
    six_min_bump = proj_o["P_SIX"].adj_minutes - proj_o["P_SIX"].base_minutes
    sec_min_bump = proj_o["P_SEC"].adj_minutes - proj_o["P_SEC"].base_minutes
    assert six_min_bump > sec_min_bump > 0.0


def test_bimodal_blowout_truncates_starters_and_boosts_bench() -> None:
    w_close = regime_weights_from_spread(-2.0)
    w_blow = regime_weights_from_spread(-14.5)
    assert w_blow[1] > w_close[1] * 2.0  # blowout weight more than doubles
    assert w_blow[2] < w_close[2]  # OT weight shrinks in big spreads

    # 35-minute star scoring 0.80 pts/min (line 27.5):
    star_close = build_regime_mixture(35.0, 0.80, "points", team_spread=-2.0)
    star_big_fav = build_regime_mixture(35.0, 0.80, "points", team_spread=-15.0)
    assert star_big_fav.expected_minutes < star_close.expected_minutes - 1.2
    assert p_over_mixture(star_big_fav, 27.5) < p_over_mixture(star_close, 27.5)
    assert p_over_mixture(star_big_fav, 27.5) + p_under_mixture(
        star_big_fav, 27.5
    ) == pytest.approx(1.0)

    # 16-minute bench rotation player (line 8.5): gains garbage-time upside in big spreads!
    bench_close = build_regime_mixture(16.0, 0.52, "points", team_spread=-2.0)
    bench_big_fav = build_regime_mixture(16.0, 0.52, "points", team_spread=-15.0)
    assert bench_big_fav.expected_minutes > bench_close.expected_minutes
    assert p_over_mixture(bench_big_fav, 8.5) > p_over_mixture(bench_close, 8.5)


def test_nbinom_overdispersion_fattens_tails_vs_poisson() -> None:
    # At expected = 20, P(X > 28.5) is higher under overdispersed NB than thin-tailed Poisson
    p_pois = p_over_nbinom(20.0, 28.5, nb_alpha=0.0)
    p_nb = p_over_nbinom(20.0, 28.5, nb_alpha=0.14)
    assert p_nb > p_pois
    # Integer line splits push 50/50
    p_int = p_over_nbinom(10.0, 10.0, nb_alpha=0.10)
    assert 0.35 < p_int < 0.65


def test_all_five_live_odds_api_prop_markets_and_consensus() -> None:
    assert set(DEFAULT_PROP_MARKETS) == {
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_points_rebounds_assists",
    }
    payload = {
        "id": "evt_1",
        "commence_time": "2026-11-12T00:30:00Z",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": mkt,
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Jayson Tatum Jr.",
                                "point": 26.5,
                                "price": -115,
                            },
                            {
                                "name": "Under",
                                "description": "Jayson Tatum Jr.",
                                "point": 26.5,
                                "price": -105,
                            },
                        ],
                    }
                    for mkt in DEFAULT_PROP_MARKETS
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": mkt,
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Jayson Tatum",
                                "point": 26.5,
                                "price": -108,
                            },
                            {
                                "name": "Under",
                                "description": "Jayson Tatum",
                                "point": 26.5,
                                "price": -112,
                            },
                        ],
                    }
                    for mkt in DEFAULT_PROP_MARKETS
                ],
            },
        ],
    }

    # Consensus line-shopping picks best Over (-108 FD) and best Under (-105 DK)
    rows = fetch_event_prop_odds(
        "evt_1",
        api_key="dummy",
        consensus=True,
        http_get=lambda url, params: payload,
    )
    assert len(rows) == 5
    assert {r.stat for r in rows} == {"points", "rebounds", "assists", "threes", "pra"}
    pts = next(r for r in rows if r.stat == "points")
    assert pts.over_price == -108.0
    assert pts.under_price == -105.0

    # Distributional projection joins cleanly across generational suffix mismatch
    tr = RedistributionTracker()
    logs = _synthetic_team_logs().replace({"Alpha Star": "Jayson Tatum"})
    tr = build_tracker_asof(logs, "2024-11-20")
    rates = tr.project_team_pregame(10, 20, "2024-11-20", team_spread=-12.0, total_line=224.0)[
        "P_STAR"
    ]
    dproj = project_distributional_props(
        rates, game_id="evt_1", game_date="2024-11-20", team_spread=-12.0
    )
    single_rows = parse_prop_markets(payload, book_key="draftkings")
    edges = build_prop_edge_rows([dproj], single_rows)
    assert len(edges) == 5
    assert all(0.0 < e.model_p_over < 1.0 for e in edges)


def test_prop_parlay_cross_game_gate_and_kelly_from_50() -> None:
    props_df = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "game_date": "2025-01-10",
                "game_id": "G1",
                "player_id": "P1",
                "player_name": "Player One",
                "team_abbreviation": "BOS",
                "market": "player_points",
                "stat": "points",
                "line": 24.5,
                "model_p_over": 0.39,  # Under model p = 0.61 > vigged(-110)=0.5238
                "book_implied_over_fair": 0.50,
                "book_over": -110.0,
                "book_under": -110.0,
                "y_over": 0.0,
            },
            {
                "season": "2024-25",
                "game_date": "2025-01-10",
                "game_id": "G1",  # Same game as P1 -> must not pair when allow_same_game=False
                "player_id": "P1B",
                "player_name": "Teammate OneB",
                "team_abbreviation": "BOS",
                "market": "player_rebounds",
                "stat": "rebounds",
                "line": 7.5,
                "model_p_over": 0.60,
                "book_implied_over_fair": 0.50,
                "book_over": -110.0,
                "book_under": -110.0,
                "y_over": 1.0,
            },
            {
                "season": "2024-25",
                "game_date": "2025-01-10",
                "game_id": "G2",
                "player_id": "P2",
                "player_name": "Player Two",
                "team_abbreviation": "NYK",
                "market": "player_assists",
                "stat": "assists",
                "line": 6.5,
                "model_p_over": 0.60,
                "book_implied_over_fair": 0.50,
                "book_over": -110.0,
                "book_under": -110.0,
                "y_over": 1.0,
            },
            {
                "season": "2024-25",
                "game_date": "2025-01-10",
                "game_id": "G3",
                "player_id": "P3",
                "player_name": "Player Three",
                "team_abbreviation": "LAL",
                "market": "player_threes",
                "stat": "threes",
                "line": 2.5,
                "model_p_over": 0.40,  # Under model p = 0.60
                "book_implied_over_fair": 0.50,
                "book_over": -110.0,
                "book_under": -110.0,
                "y_over": 0.0,
            },
        ]
    )
    legs = build_eligible_prop_legs(props_df, require_vs_vigged=True)
    assert len(legs) == 4

    cands_2 = enumerate_prop_parlays(legs, sizes=(2,), min_price_edge=0.10, allow_same_game=False)
    assert not cands_2.empty
    for _, row in cands_2.iterrows():
        assert len(set(row["game_ids"].split("|"))) == 2

    cands_3 = enumerate_prop_parlays(legs, sizes=(3,), min_price_edge=0.10, allow_same_game=False)
    assert len(cands_3) == 1
    picked = select_nonoverlapping_prop_parlays(cands_2, max_per_day=1)
    assert len(picked) == 1

    fk = parlay_kelly_fraction(0.366, 3.6446, fraction=0.15, max_bet_frac=0.03)
    assert 0.0 < fk <= 0.03

    sim_df, summary = simulate_prop_parlay_kelly(picked, starting_bankroll=50.0)
    assert summary["starting_bankroll"] == 50.0
    assert summary["n_tickets"] == 1
    assert summary["ending_bankroll"] > 50.0
    assert sim_df["kelly_stake"].iloc[0] <= 1.50


def test_walkforward_prop_frame_and_expanding_calibrator_smoke() -> None:
    logs = _synthetic_team_logs()
    mkt = pd.DataFrame(
        [{"game_id": f"G{i + 1:02d}", "spread_home": -8.5, "total_line": 228.5} for i in range(9)]
    )
    hist_props = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "game_id": "G09",
                "game_date": "2024-11-17",
                "player_id": "P_SEC",
                "market": "player_points",
                "stat": "points",
                "open_line": 17.5,
                "open_over_price": -115.0,
                "open_under_price": -115.0,
                "book": "DraftKings",
            }
        ]
    )
    wf = build_walkforward_prop_frame(
        player_boxscores=logs,
        game_markets=mkt,
        historical_props=hist_props,
        min_prior_games=5,
    )
    assert len(wf) == 1
    cal = fit_expanding_prop_calibrators(wf, min_train_days=1)
    summary = summarize_prop_calibration(cal["model_p_over"].to_numpy(), cal["y_over"].to_numpy())
    assert summary.n == 1
    assert 0.0 < summary.brier < 1.0
