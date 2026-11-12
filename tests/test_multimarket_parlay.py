"""Unit tests for joint (margin, total) distribution, push parlay settlement, and SGP pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from nba_betting.models.joint_distribution import (
    attach_bivariate_market_probs,
    attach_dmochowski_quantiles,
    build_pregame_joint_feature_frame,
    bvn_rect_prob,
    fit_expanding_joint_distribution,
)
from nba_betting.pricing.copula_sgp import (
    sgp_prob_t_copula,
    simulate_blowout_ot_regime_sgp,
    t_copula_rect_prob,
    t_copula_tail_dependence,
)
from nba_betting.pricing.kelly import american_to_decimal
from nba_betting.pricing.multimarket_parlay import (
    build_multimarket_legs,
    enumerate_multimarket_parlays,
    fast_seasonal_kelly_summary,
    settle_parlay_legs,
    simulate_seasonal_kelly_bankroll,
)


def test_bvn_rect_prob_matches_scipy_mvn_and_sums_to_one() -> None:
    mu_m, sig_m, mu_t, sig_t, rho = 4.0, 12.5, 226.0, 17.5, 0.25
    mvn = multivariate_normal(
        mean=[mu_m, mu_t],
        cov=[[sig_m**2, rho * sig_m * sig_t], [rho * sig_m * sig_t, sig_t**2]],
    )
    p_scipy = 1.0 - mvn.cdf([-3.5, np.inf]) - mvn.cdf([np.inf, 222.5]) + mvn.cdf([-3.5, 222.5])
    p_quad = float(
        bvn_rect_prob(
            mu_m=mu_m,
            sigma_m=sig_m,
            mu_t=mu_t,
            sigma_t=sig_t,
            rho=rho,
            m_lo=-3.5,
            m_hi=np.inf,
            t_lo=222.5,
            t_hi=np.inf,
        )
    )
    assert abs(p_scipy - p_quad) < 1e-11

    # Partition of R^2 across whole-number spread (-6.0 -> thr +6.0) & whole-number total (225.0)
    df = pd.DataFrame(
        {
            "mu_margin": [mu_m],
            "sigma_margin": [sig_m],
            "mu_total": [mu_t],
            "sigma_total": [sig_t],
            "rho_margin_total": [rho],
            "spread_home": [-6.0],
            "total_line": [225.0],
        }
    )
    priced = attach_bivariate_market_probs(df).iloc[0]
    assert abs(
        priced["p_model_home_cover"] + priced["p_model_away_cover"] + priced["p_model_spread_push"] - 1.0
    ) < 1e-12
    assert abs(
        priced["p_model_over"] + priced["p_model_under"] + priced["p_model_total_push"] - 1.0
    ) < 1e-12

    nine_cells = (
        priced["p_sgp_home_cover_over"]
        + priced["p_sgp_home_cover_under"]
        + priced["p_sgp_home_cover_push_over"]
        + priced["p_sgp_away_cover_over"]
        + priced["p_sgp_away_cover_under"]
        + priced["p_sgp_away_cover_push_over"]
        + priced["p_sgp_push_home_cover_over"]
        + priced["p_sgp_push_home_cover_under"]
        + priced["p_sgp_both_push"]
    )
    assert abs(float(nine_cells) - 1.0) < 1e-11


def test_parlay_push_drops_to_n_minus_1_legs() -> None:
    d1 = american_to_decimal(-110.0)
    d2 = american_to_decimal(120.0)
    d3 = american_to_decimal(-115.0)

    # 3-leg with 1 push and 2 wins -> pays 2-leg decimal d1 * d3!
    status, eff_dec, n_wins = settle_parlay_legs(["win", "push", "win"], [d1, d2, d3])
    assert status == "win"
    assert n_wins == 2
    assert abs(eff_dec - d1 * d3) < 1e-12

    # 2-leg with 1 push and 1 loss -> still loses!
    status_l, eff_l, _ = settle_parlay_legs(["push", "loss"], [d1, d2])
    assert status_l == "loss"
    assert eff_l == 0.0

    # 2-leg with 2 pushes -> full stake refund (1.0x decimal, 0 profit)
    status_p, eff_p, nw_p = settle_parlay_legs(["push", "push"], [d1, d2])
    assert status_p == "push"
    assert eff_p == 1.0
    assert nw_p == 0


def test_correlated_sgp_reflects_rho_and_blocks_ml_plus_spread() -> None:
    base = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "game_id": ["g1", "g2"],
            "game_date": ["2024-11-10", "2024-11-10"],
            "home_tricode": ["BOS", "DEN"],
            "away_tricode": ["NYK", "LAL"],
            "mu_margin": [8.5, 4.0],
            "sigma_margin": [12.0, 12.0],
            "mu_total": [231.0, 224.0],
            "sigma_total": [17.0, 17.0],
            "rho_margin_total": [0.30, 0.0],
            "home_win": [1.0, 1.0],
            "home_ml": [-150.0, -140.0],
            "away_ml": [130.0, 120.0],
            "p_home_novig": [0.58, 0.56],
            "p_away_novig": [0.42, 0.44],
            "spread_home": [-4.5, -2.5],
            "spread_home_odds": [-110.0, -110.0],
            "spread_away_odds": [-110.0, -110.0],
            "p_spread_home_novig": [0.50, 0.50],
            "p_spread_away_novig": [0.50, 0.50],
            "ats_result": ["home", "push"],
            "total_line": [224.5, 224.0],
            "over_odds": [-110.0, -110.0],
            "under_odds": [-110.0, -110.0],
            "p_over_novig": [0.50, 0.50],
            "p_under_novig": [0.50, 0.50],
            "total_result": ["over", "over"],
        }
    )
    priced_pos = attach_bivariate_market_probs(base)
    base_neg = base.copy()
    base_neg["rho_margin_total"] = [-0.30, 0.0]
    priced_neg = attach_bivariate_market_probs(base_neg)

    # Positive rho increases P(Home Cover & Over) on g1 vs negative rho
    assert priced_pos.loc[0, "p_sgp_home_cover_over"] > priced_neg.loc[0, "p_sgp_home_cover_over"]

    legs = build_multimarket_legs(priced_pos, markets=("ml", "spread", "total"))
    cands = enumerate_multimarket_parlays(
        legs,
        priced_pos,
        sizes=(2,),
        min_price_edge=0.01,
        allow_cross_game=False,
        allow_same_game=True,
    )
    assert not cands.empty
    # Every same-game ticket must combine spread|total or ml|total — NEVER ml|spread!
    for mkts in cands["markets"]:
        assert set(mkts.split("|")) in ({"spread", "total"}, {"ml", "total"})


def test_strict_no_future_leakage_in_joint_distribution() -> None:
    """Mutating late-season scores must change ZERO earlier features or predictions."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-10-22", periods=35, freq="D")
    sched_rows = []
    log_rows = []
    pbox_rows = []
    gid_num = 1000
    for d_idx, dt in enumerate(dates):
        for pair in ((1, 2), (3, 4), (5, 6)):
            gid_num += 1
            gid = str(gid_num)
            hs = float(108 + rng.integers(-12, 15))
            as_ = float(106 + rng.integers(-12, 15))
            if hs == as_:
                hs += 1.0
            sched_rows.append(
                {
                    "season": "2024-25",
                    "season_type": "Regular Season",
                    "game_id": gid,
                    "game_date_est": str(dt.date()),
                    "home_team_id": pair[0],
                    "home_tricode": f"T{pair[0]}",
                    "away_team_id": pair[1],
                    "away_tricode": f"T{pair[1]}",
                    "home_score": hs,
                    "away_score": as_,
                    "is_neutral": False,
                }
            )
            for tid, pts in ((pair[0], hs), (pair[1], as_)):
                log_rows.append(
                    {
                        "season": "2024-25",
                        "game_id": gid,
                        "game_date": str(dt.date()),
                        "team_id": tid,
                        "fga": 88.0,
                        "fgm": 42.0,
                        "fg3a": 36.0,
                        "fg3m": 13.0,
                        "fta": 22.0,
                        "oreb": 10.0,
                        "tov": 13.0,
                        "pf": 19.0,
                        "pts": pts,
                    }
                )
                pbox_rows.append(
                    {
                        "season": "2024-25",
                        "game_id": gid,
                        "game_date": str(dt.date()),
                        "team_id": tid,
                        "player_id": f"p_{tid}_1",
                        "minutes": 34.0,
                        "pts": pts * 0.25,
                        "plus_minus": (hs - as_) if tid == pair[0] else (as_ - hs),
                        "dnp": False,
                    }
                )

    sched1 = pd.DataFrame(sched_rows)
    logs1 = pd.DataFrame(log_rows)
    pbox1 = pd.DataFrame(pbox_rows)

    # Corrupt all games after day 25 with extreme scores (+90 points!)
    cutoff = str(dates[25].date())
    sched2 = sched1.copy()
    logs2 = logs1.copy()
    late_s = sched2["game_date_est"] > cutoff
    sched2.loc[late_s, "home_score"] += 90.0
    late_l = logs2["game_date"] > cutoff
    logs2.loc[late_l, "pts"] += 90.0

    f1 = build_pregame_joint_feature_frame(seasons=[("2024-25", sched1, logs1)], player_box=pbox1)
    f2 = build_pregame_joint_feature_frame(seasons=[("2024-25", sched2, logs2)], player_box=pbox1)

    d1 = fit_expanding_joint_distribution(f1, min_train=30, refit_every_days=1)
    d2 = fit_expanding_joint_distribution(f2, min_train=30, refit_every_days=1)

    early1 = d1[d1["game_date"] <= cutoff].reset_index(drop=True)
    early2 = d2[d2["game_date"] <= cutoff].reset_index(drop=True)
    for col in ("mu_margin", "sigma_margin", "mu_total", "sigma_total", "rho_margin_total"):
        np.testing.assert_allclose(early1[col], early2[col], rtol=0.0, atol=1e-12)


def test_seasonal_kelly_bankroll_no_intraday_compounding() -> None:
    tickets = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "game_date": pd.to_datetime(["2024-11-10", "2024-11-10"]),
            "ev_per_unit": [0.20, 0.15],
            "parlay_decimal": [3.50, 3.50],
            "ticket_status": ["win", "loss"],
            "effective_decimal": [3.50, 0.0],
            "n_pushed_legs": [0, 0],
            "same_game": [True, False],
        }
    )
    graded, meta = simulate_seasonal_kelly_bankroll(
        tickets, starting_bankroll=50.0, fixed_pct=0.05, max_daily_frac=0.20
    )
    # Both tickets on same day must stake 5% of $50.00 = $2.50 (not 5% of post-ticket-1 balance!)
    assert list(graded["stake"]) == [2.50, 2.50]
    # Ticket 1 pnl = 2.50 * 2.50 = +6.25; Ticket 2 pnl = -2.50 -> end $53.75
    assert meta["by_season"]["2024-25"]["end"] == 53.75
    fast_meta = fast_seasonal_kelly_summary(
        tickets, starting_bankroll=50.0, fixed_pct=0.05, max_daily_frac=0.20
    )
    assert fast_meta["2024-25"]["end"] == 53.75


def test_t4_copula_tail_dependence_and_mc_agreement() -> None:
    # Positive symmetric tail dependence even at rho=0, and strictly higher than Gaussian in tails
    lam0 = float(t_copula_tail_dependence(0.0, df=4.0))
    lam_pos = float(t_copula_tail_dependence(0.25, df=4.0))
    assert 0.05 < lam0 < lam_pos

    # Check exact quadrature vs 150,000-draw Monte Carlo for 2-leg tail event (p1=0.18, p2=0.20, rho=0.20)
    p1, p2, rho = 0.18, 0.20, 0.20
    p_quad = float(
        t_copula_rect_prob(u1_lo=1.0 - p1, u1_hi=1.0, u2_lo=1.0 - p2, u2_hi=1.0, rho=rho, df=4.0)
    )
    p_mc = sgp_prob_t_copula(
        [p1, p2], np.array([[1.0, rho], [rho, 1.0]]), df=4.0, n_sims=150_000, seed=7
    )
    assert abs(p_quad - p_mc) < 0.003

    # Compare t_4 vs Gaussian copula on joint upper tail: t_4 must assign higher joint tail mass!
    from scipy.stats import norm

    p_gauss = float(
        bvn_rect_prob(
            mu_m=0.0,
            sigma_m=1.0,
            mu_t=0.0,
            sigma_t=1.0,
            rho=rho,
            m_lo=float(norm.ppf(1.0 - p1)),
            m_hi=np.inf,
            t_lo=float(norm.ppf(1.0 - p2)),
            t_hi=np.inf,
        )
    )
    assert p_quad > p_gauss + 0.005

    # Regime simulator captures NBA OT right-skew (mean_total > median_total)
    sim = simulate_blowout_ot_regime_sgp(
        mu_margin=1.5,
        sigma_margin=12.5,
        mu_total_reg=225.0,
        sigma_total_reg=16.5,
        rho_reg=0.05,
        spread_home=-1.5,
        total_line=225.5,
        n_sims=30_000,
        seed=11,
    )
    assert sim["ot_rate"] > 0.04
    assert sim["mean_total"] > sim["median_total"]


def test_dmochowski_quantile_gate_filters_inside_band() -> None:
    df = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "game_id": ["g_inside", "g_outside"],
            "game_date": ["2024-11-12", "2024-11-12"],
            "home_tricode": ["BOS", "OKC"],
            "away_tricode": ["MIA", "UTA"],
            # g_inside: mean=230.0 looks > 228.5 because of OT right-skew, YET median=227.5
            # so 228.5 sits INSIDE [q_4762=226.5, q_5238=228.8] -> Dmochowski BLOCKS fake Over!
            "mu_margin": [5.0, 9.0],
            "sigma_margin": [12.0, 12.0],
            "mu_total": [230.0, 233.0],
            "sigma_total": [17.0, 17.0],
            "rho_margin_total": [0.05, 0.05],
            "q_margin_4762": [4.2, 8.2],
            "q_margin_5000": [5.0, 9.0],
            "q_margin_5238": [5.8, 9.8],
            "q_total_4762": [226.5, 230.5],
            "q_total_5000": [227.5, 231.5],
            "q_total_5238": [228.8, 232.8],
            "dmochowski_home_cover_margin": [4.2 - 4.5, 8.2 - 5.5],  # -0.3 (blocked) vs +2.7 (ok)
            "dmochowski_away_cover_margin": [4.5 - 5.8, 5.5 - 9.8],
            "dmochowski_over_margin": [226.5 - 228.5, 230.5 - 228.5],  # -2.0 (blocked) vs +2.0 (ok)
            "dmochowski_under_margin": [228.5 - 228.8, 228.5 - 232.8],
            "home_win": [1.0, 1.0],
            "home_ml": [-190.0, -250.0],
            "away_ml": [160.0, 205.0],
            "p_home_novig": [0.63, 0.69],
            "p_away_novig": [0.37, 0.31],
            "spread_home": [-4.5, -5.5],
            "spread_home_odds": [-110.0, -110.0],
            "spread_away_odds": [-110.0, -110.0],
            "p_spread_home_novig": [0.50, 0.50],
            "p_spread_away_novig": [0.50, 0.50],
            "ats_result": ["home", "home"],
            "total_line": [228.5, 228.5],
            "over_odds": [-110.0, -110.0],
            "under_odds": [-110.0, -110.0],
            "p_over_novig": [0.50, 0.50],
            "p_under_novig": [0.50, 0.50],
            "total_result": ["under", "over"],
        }
    )
    priced = attach_bivariate_market_probs(df, copula="t4")
    legs_ungated = build_multimarket_legs(priced, markets=("total",), require_dmochowski_gate=False)
    legs_gated = build_multimarket_legs(
        priced, markets=("total",), require_dmochowski_gate=True, dmochowski_buffer=0.5
    )
    assert "g_inside" in set(legs_ungated["game_id"])
    assert set(legs_gated["game_id"]) == {"g_outside"}
