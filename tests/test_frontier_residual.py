"""Tests for Shin devigging, Hubacek Decorrelated Brier, Logit-Offset Residual, and Slate Quadratic Kelly."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from nba_betting.models.losses import (
    CustomNewtonGBDT,
    classwise_ece,
    decorrelated_brier_grad_hess,
    fit_hubacek_linear,
    fit_logit_offset_residual,
    sigmoid,
)
from nba_betting.models.market_residual import (
    simulate_seasonal_kelly_portfolio,
    walkforward_anchored_hubacek_probs,
)
from nba_betting.pricing.american import devig_two_way
from nba_betting.pricing.devig import devig_power, devig_shin, devig_shin_american
from nba_betting.pricing.portfolio import solve_slate_quadratic_kelly


def test_shin_and_power_devig_favorite_longshot_asymmetry() -> None:
    # On -300 / +240, Shin assigns lower true prob to the +240 longshot than multiplicative devig
    mult_fav, mult_dog = devig_two_way(-300.0, 240.0)
    shin_fav, shin_dog = devig_shin_american(-300.0, 240.0)
    pow_probs = devig_power([1.0 + 100.0 / 300.0, 3.40])

    assert abs(shin_fav + shin_dog - 1.0) < 1e-12
    assert abs(float(pow_probs.sum()) - 1.0) < 1e-12
    assert shin_dog < mult_dog
    assert float(pow_probs[1]) < mult_dog


def test_hubacek_decorrelated_brier_reduces_market_correlation() -> None:
    rng = np.random.default_rng(7)
    n = 600
    z_mkt = rng.normal(0.0, 0.9, size=n)
    pi = sigmoid(z_mkt)
    # Orthogonal signal unseen by market
    x_orth = rng.normal(0.0, 1.0, size=n)
    p_true = sigmoid(z_mkt + 0.35 * x_orth)
    y = (rng.random(n) < p_true).astype(float)
    X = x_orth.reshape(-1, 1)

    w_c0 = fit_hubacek_linear(X, y, pi, c=0.0, l2_lambda=0.02, z_offset=z_mkt, mu_tether=0.01)
    w_c55 = fit_hubacek_linear(X, y, pi, c=0.55, l2_lambda=0.02, z_offset=z_mkt, mu_tether=0.01)

    X1 = np.column_stack([X, np.ones(n)])
    p_c0 = sigmoid(z_mkt + X1 @ w_c0)
    p_c55 = sigmoid(z_mkt + X1 @ w_c55)

    r_c0, _ = pearsonr(p_c0, pi)
    r_c55, _ = pearsonr(p_c55, pi)
    # Higher c amplifies orthogonal signal and lowers correlation with market pi
    assert r_c55 < r_c0
    assert float(np.std(p_c55 - pi)) > float(np.std(p_c0 - pi))


def test_grad_finite_difference_and_newton_gbdt() -> None:
    rng = np.random.default_rng(11)
    u = rng.normal(0.0, 0.5, size=50)
    pi = sigmoid(rng.normal(0.0, 0.5, size=50))
    y = (rng.random(50) < pi).astype(float)
    anchor = np.zeros(50)

    val0, grad, _ = decorrelated_brier_grad_hess(u, y, pi, c=0.5, u_anchor=anchor, mu_tether=0.1)
    eps = 1e-6
    u_pert = u.copy()
    u_pert[3] += eps
    val1, _, _ = decorrelated_brier_grad_hess(u_pert, y, pi, c=0.5, u_anchor=anchor, mu_tether=0.1)
    assert abs((val1 - val0) / eps - grad[3]) < 1e-4

    X = rng.normal(0.0, 1.0, size=(200, 3))
    gb = CustomNewtonGBDT(mode="hubacek", c=0.45, mu_tether=0.2, n_estimators=10, min_samples_leaf=25)
    gb.fit(X, (rng.random(200) < 0.5).astype(float), np.full(200, 0.5), init_score=np.zeros(200))
    preds = gb.predict_proba(X, init_score=np.zeros(200))
    assert preds.shape == (200,)
    assert np.all((preds > 0.0) & (preds < 1.0))
    assert 0.0 <= classwise_ece(preds, (rng.random(200) < 0.5).astype(float)) <= 1.0


def test_quadratic_kelly_caps_and_no_future_leakage() -> None:
    # Negative EV -> 0 stake
    f_neg = solve_slate_quadratic_kelly([0.50, 0.45], [1.91, 2.00], omega=0.25)
    np.testing.assert_allclose(f_neg, [0.0, 0.0])

    # Positive EV respects per-bet and slate caps
    f_pos = solve_slate_quadratic_kelly(
        [0.60, 0.60, 0.60, 0.60],
        [2.00, 2.00, 2.00, 2.00],
        omega=0.50,
        max_bet_frac=0.05,
        max_slate_frac=0.12,
    )
    assert np.all(f_pos <= 0.05 + 1e-9)
    assert abs(float(f_pos.sum()) - 0.12) < 1e-6

    # Walk-forward bitwise zero-leakage gate: mutating outcomes at t >= split leaves t < split identical
    rng = np.random.default_rng(23)
    X = rng.normal(0.0, 1.0, size=(300, 4))
    z_mkt = rng.normal(0.0, 0.6, size=300)
    y_orig = (rng.random(300) < sigmoid(z_mkt + 0.2 * X[:, 0])).astype(float)
    y_mutated = y_orig.copy()
    y_mutated[200:] = 1.0 - y_mutated[200:]

    def wf_predict_at_200(y_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = fit_logit_offset_residual(X[:200], y_vec[:200], z_mkt[:200], l2_lambda=0.05, mu_delta=0.4)
        X1_day = np.column_stack([X[200:210], np.ones(10)])
        p_day = sigmoid(z_mkt[200:210] + X1_day @ w)
        stakes = solve_slate_quadratic_kelly(p_day, np.full(10, 1.91), omega=0.25, max_bet_frac=0.03)
        return p_day, stakes

    p1, s1 = wf_predict_at_200(y_orig)
    p2, s2 = wf_predict_at_200(y_mutated)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(s1, s2)


def test_walkforward_anchored_hubacek_and_portfolio_no_leakage() -> None:
    import pandas as pd

    rng = np.random.default_rng(99)
    n = 120
    dates = pd.date_range("2024-10-22", periods=60, freq="D").repeat(2)
    z = rng.normal(0.0, 0.7, size=n)
    pi = sigmoid(z)
    x1 = rng.normal(0.0, 1.0, size=n)
    y = (rng.random(n) < sigmoid(z + 0.30 * x1)).astype(float)
    df = pd.DataFrame(
        {
            "game_id": [str(i) for i in range(n)],
            "game_date": dates,
            "z_ml_shin": z,
            "p_ml_shin": pi,
            "feat_a": x1,
            "home_win": y,
            "home_ml": np.where(pi >= 0.5, -150.0, 130.0),
            "away_ml": np.where(pi >= 0.5, 130.0, -150.0),
        }
    )
    p_orig = walkforward_anchored_hubacek_probs(
        df, feature_cols=["feat_a"], min_train=40, refit_days=5
    )
    df_mut = df.copy()
    df_mut.loc[80:, "home_win"] = 1.0 - df_mut.loc[80:, "home_win"]
    p_mut = walkforward_anchored_hubacek_probs(
        df_mut, feature_cols=["feat_a"], min_train=40, refit_days=5
    )
    np.testing.assert_array_equal(p_orig[:80], p_mut[:80])

    df["p_hub_c65"] = p_orig
    res = simulate_seasonal_kelly_portfolio(df.iloc[40:], starting_bankroll=50.0)
    assert res["starting_bankroll"] == 50.0
    assert res["ending_bankroll"] >= 0.0
