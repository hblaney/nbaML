"""Unit tests for tail probability shrinkage and exact daily joint Kelly optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.pricing.kelly_portfolio import (
    build_prop_candidate_legs,
    build_shrunk_legs,
    compute_shin_clv_bps,
    debiased_hubacek_probs,
    enumerate_kelly_candidates,
    fit_oos_shrinkage,
    run_kelly_compounding_selector,
    run_unified_kelly_portfolio,
    select_daily_kelly_portfolio,
    shrink_probs,
    simulate_seasonal_kelly,
    single_ticket_expected_log_growth,
    single_ticket_kelly_fraction,
    solve_daily_joint_kelly,
    winsorize_prob_by_ev_cap,
)


def test_shrink_probs_modes_damping() -> None:
    # Underdog tail: market=0.12, noisy model=0.28
    p_lin = float(shrink_probs(0.28, 0.12, w=0.25, mode="linear"))
    p_log = float(shrink_probs(0.28, 0.12, w=0.25, mode="logit"))
    p_td = float(shrink_probs(0.28, 0.12, w=0.25, mode="tail_damped"))
    assert abs(p_lin - (0.25 * 0.28 + 0.75 * 0.12)) < 1e-12
    # Both logit and tail_damped shrink tails harder toward market than linear
    assert 0.12 < p_log < p_lin
    assert 0.12 < p_td < p_lin


def test_fit_oos_shrinkage_recovers_market_when_model_is_noise() -> None:
    rng = np.random.default_rng(42)
    n = 1200
    p_true = rng.uniform(0.2, 0.8, size=n)
    y = (rng.random(n) < p_true).astype(float)
    p_noisy = np.clip(0.5 + 0.3 * (p_true - 0.5) + rng.normal(0, 0.15, size=n), 0.05, 0.95)
    folds = np.array(["2022-23"] * 600 + ["2023-24"] * 600)
    fit = fit_oos_shrinkage(p_noisy, p_true, y, fold_ids=folds, mode="linear", metric="log_loss")
    assert 0.0 <= fit["w_opt"] < 0.15
    assert fit["oof_log_loss"] < fit["model_raw_log_loss"]


def test_log_growth_ranking_penalizes_longshot_variance() -> None:
    # Same +10% EV: moderate 2-leg (decimal 3.0, p=0.3667) vs longshot (decimal 12.0, p=0.0917)
    g_mod, f_mod = single_ticket_expected_log_growth(1.10 / 3.0, 3.0, c_kelly=1.0, max_frac=0.5)
    g_long, f_long = single_ticket_expected_log_growth(1.10 / 12.0, 12.0, c_kelly=1.0, max_frac=0.5)
    assert g_mod > 5.0 * g_long
    assert f_mod > f_long


def test_solve_daily_joint_kelly_matches_grid_optimum() -> None:
    probs = [0.40, 0.35]
    decs = [3.0, 3.4]
    fracs, g_opt = solve_daily_joint_kelly(probs, decs, c_kelly=1.0, daily_cap=0.50)
    assert fracs.shape == (2,)
    assert np.all(fracs > 0.0)
    assert float(fracs.sum()) <= 0.50 + 1e-9

    # Verify against fine 2D grid over [0, 0.25]^2
    grid = np.linspace(0.0, 0.20, 201)
    best_grid_g = -1e9
    p1, p2 = probs
    o1, o2 = decs
    for f1 in grid:
        for f2 in grid:
            g = (
                (1 - p1) * (1 - p2) * np.log(1 - f1 - f2)
                + p1 * (1 - p2) * np.log(1 - f1 - f2 + f1 * o1)
                + (1 - p1) * p2 * np.log(1 - f1 - f2 + f2 * o2)
                + p1 * p2 * np.log(1 - f1 - f2 + f1 * o1 + f2 * o2)
            )
            if g > best_grid_g:
                best_grid_g = float(g)
    assert g_opt >= best_grid_g - 1e-6
    # Simultaneous Kelly allocates slightly less per bet than isolated single-bet Kelly
    assert fracs[0] < single_ticket_kelly_fraction(p1, o1)
    assert fracs[1] < single_ticket_kelly_fraction(p2, o2)


def test_end_to_end_daily_portfolio_and_compounding() -> None:
    games = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_date": pd.to_datetime(["2024-11-01", "2024-11-01", "2024-11-02", "2024-11-02"]),
            "p_model_home": [0.72, 0.70, 0.68, 0.71],
            "p_home_novig": [0.55, 0.55, 0.55, 0.55],
            "home_ml": [-120.0, -120.0, -120.0, -120.0],
            "away_ml": [100.0, 100.0, 100.0, 100.0],
            "home_win": [1.0, 1.0, 1.0, 0.0],
        }
    )
    legs = build_shrunk_legs(games, shrink_w=0.50, shrink_mode="linear")
    cands = enumerate_kelly_candidates(legs, sizes=(1, 2), c_kelly=0.5, daily_cap=0.15, max_decimal=4.5)
    assert not cands.empty
    port = select_daily_kelly_portfolio(cands, max_tickets_per_day=2, c_kelly=0.5, daily_cap=0.15)
    assert not port.empty
    graded, summary = simulate_seasonal_kelly(port, initial_bankroll=50.0)
    assert summary["initial_bankroll"] == 50.0
    assert summary["n_staked"] >= 1
    assert not summary["broke"]
    assert "end_day_bankroll" in graded.columns


def test_run_kelly_compounding_selector_multimarket() -> None:
    games = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_date": pd.to_datetime(["2024-11-01", "2024-11-01", "2024-11-02", "2024-11-02"]),
            "p_model_home": [0.75, 0.55, 0.55, 0.80],
            "p_home_novig": [0.50, 0.50, 0.50, 0.50],
            "home_ml": [-110.0, -110.0, -110.0, -110.0],
            "away_ml": [-110.0, -110.0, -110.0, -110.0],
            "home_win": [1.0, 0.0, 1.0, 1.0],
            "p_model_cover_home": [0.55, 0.82, 0.50, 0.50],
            "p_spread_home_novig": [0.50, 0.50, 0.50, 0.50],
            "spread_home_odds": [-110.0, -110.0, -110.0, -110.0],
            "spread_away_odds": [-110.0, -110.0, -110.0, -110.0],
            "spread_overround": [1.0476] * 4,
            "spread_push": [False] * 4,
            "home_cover": [1.0, 1.0, 0.0, 1.0],
            "p_model_over": [0.50, 0.50, 0.85, 0.50],
            "p_over_novig": [0.50, 0.50, 0.50, 0.50],
            "over_odds": [-110.0, -110.0, -110.0, -110.0],
            "under_odds": [-110.0, -110.0, -110.0, -110.0],
            "total_overround": [1.0476] * 4,
            "total_push": [False] * 4,
            "over_hit": [0.0, 0.0, 1.0, 1.0],
        }
    )
    legs, graded, summary = run_kelly_compounding_selector(games, initial_bankroll=50.0)
    assert not legs.empty
    assert summary["ending_bankroll"] > 50.0
    assert summary["n_wins"] == summary["n_staked"]


def test_hubacek_debiasing_and_ev_winsorization() -> None:
    # Stationary Hubáček formula: p_hub = pi + (p_true - pi) / (1 - c)
    pi = 0.55
    p_true = 0.61
    c = 0.65
    p_hub = pi + (p_true - pi) / (1.0 - c)
    recovered = float(debiased_hubacek_probs(p_hub, pi, c_decorr=c))
    assert abs(recovered - p_true) < 1e-12

    # EV cap at +8% on 2.00 decimal caps effective p at 0.54
    p_capped = float(winsorize_prob_by_ev_cap(0.70, 2.00, ev_cap=0.08))
    assert abs(p_capped - 0.54) < 1e-12
    assert abs(float(compute_shin_clv_bps(0.51, 0.545)) - 350.0) < 1e-9


def test_unified_prop_and_game_portfolio_no_future_leakage() -> None:
    props = pd.DataFrame(
        {
            "season": ["2024-25"] * 6,
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g4"],
            "game_date": pd.to_datetime(
                ["2024-11-01", "2024-11-01", "2024-11-01", "2024-11-01", "2024-11-02", "2024-11-02"]
            ),
            "prop_type": ["points", "rebounds", "assists", "points", "points", "pra"],
            "p_model_over": [0.38, 0.42, 0.36, 0.48, 0.37, 0.35],
            "p_mkt_over": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            "over_decimal": [1.87] * 6,
            "under_decimal": [1.91] * 6,
            "best_over_decimal_same_line": [1.91] * 6,
            "best_under_decimal_same_line": [1.95] * 6,
            "over_win": [0, 1, 0, 1, 0, 0],
        }
    )
    prop_legs = build_prop_candidate_legs(props, shrink_w=0.82, min_leg_ev=0.015, ev_cap=0.08, use_best_same_line=True)
    # Dedupes to at most one prop per game_id
    assert len(prop_legs) == 4
    assert prop_legs["game_id"].nunique() == 4

    graded_a, summary_a = run_unified_kelly_portfolio(
        prop_legs,
        sizes=(1, 2),
        max_decimal=4.2,
        c_kelly=1.0,
        daily_cap=0.20,
        per_bet_cap=0.08,
        max_tickets_per_day=2,
        initial_bankroll=50.0,
    )
    assert summary_a["ending_bankroll"] > 50.0

    # Mutate outcomes on day 2 (2024-11-02) -> day 1 stakes must remain bitwise identical
    props_mut = props.copy()
    props_mut.loc[props_mut["game_date"] == pd.Timestamp("2024-11-02"), "over_win"] = 1
    prop_legs_mut = build_prop_candidate_legs(
        props_mut, shrink_w=0.82, min_leg_ev=0.015, ev_cap=0.08, use_best_same_line=True
    )
    graded_b, _ = run_unified_kelly_portfolio(
        prop_legs_mut,
        sizes=(1, 2),
        max_decimal=4.2,
        c_kelly=1.0,
        daily_cap=0.20,
        per_bet_cap=0.08,
        max_tickets_per_day=2,
        initial_bankroll=50.0,
    )
    d1_a = graded_a[graded_a["game_date"] == pd.Timestamp("2024-11-01")]["stake"].to_numpy()
    d1_b = graded_b[graded_b["game_date"] == pd.Timestamp("2024-11-01")]["stake"].to_numpy()
    np.testing.assert_array_equal(d1_a, d1_b)
