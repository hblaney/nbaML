"""Anchored Hubacek Decorrelated Brier residual engine + variance-capped Kelly backtester."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nba_betting.models.losses import fit_hubacek_linear, sigmoid
from nba_betting.pricing.kelly import american_to_decimal
from nba_betting.pricing.portfolio import solve_slate_quadratic_kelly

DEFAULT_ORTH_FEATURES: tuple[str, ...] = (
    "star_out_g1_diff",
    "star_ewing_g2_5_diff",
    "b2b_collapse_diff",
    "cluster_pen_diff",
    "starter_skew_diff",
    "kalman_diff",
    "has_open",
    "dp_ml_steam_shin",
    "sp_steam_home",
    "tot_steam_over",
    "ml_vs_spread_gap",
    "res_logit_ema",
    "res_logit_pi",
    "form_hot_diff",
    "fg3_luck_diff",
    "rest_diff",
    "away_road_b2b",
    "home_b2b",
    "three_in_4_diff",
    "travel_diff",
    "east_b2b_away",
    "away_alt_shock",
    "streak_hr_diff",
    "pace_clash",
    "fg3a_mismatch",
    "ref_fta_bias",
    "ref_home_ats",
)


def walkforward_anchored_hubacek_probs(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = DEFAULT_ORTH_FEATURES,
    y_col: str = "home_win",
    pi_col: str = "p_ml_shin",
    z_col: str = "z_ml_shin",
    date_col: str = "game_date",
    c: float = 0.65,
    l2_lambda: float = 0.004,
    mu_tether: float = 0.012,
    clip_logit_delta: float = 0.35,
    min_train: int = 350,
    refit_days: int = 21,
) -> np.ndarray:
    """Walk-forward out-of-fold probabilities anchored at Shin market log-odds.

    Orthogonalizes ``feature_cols`` against ``z_col`` strictly on each training
    fold (``date < current_date``) and optimizes Hubacek's Decorrelated Brier
    Loss with light tether ``mu_tether``.
    """
    dates = pd.to_datetime(df[date_col]).dt.normalize().to_numpy()
    unique_dates = np.unique(dates)
    X_raw = df[list(feature_cols)].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    pi = df[pi_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)

    out = pi.copy()
    last_fit_idx = -999
    state: tuple[float, np.ndarray, StandardScaler, np.ndarray] | None = None

    for d_idx, gd in enumerate(unique_dates):
        day_mask = dates == gd
        tr_mask = dates < gd
        if int(tr_mask.sum()) < int(min_train):
            continue
        if (d_idx - last_fit_idx) >= int(refit_days) or state is None:
            last_fit_idx = d_idx
            z_mean = float(np.mean(z[tr_mask]))
            zc = z[tr_mask] - z_mean
            betas = (zc @ (X_raw[tr_mask] - np.mean(X_raw[tr_mask], axis=0))) / (
                float(np.dot(zc, zc)) + 1e-8
            )
            X_orth_tr = X_raw[tr_mask] - np.outer(z[tr_mask] - z_mean, betas)
            scaler = StandardScaler().fit(X_orth_tr)
            w = fit_hubacek_linear(
                scaler.transform(X_orth_tr),
                y[tr_mask],
                pi[tr_mask],
                c=float(c),
                l2_lambda=float(l2_lambda),
                z_offset=z[tr_mask],
                mu_tether=float(mu_tether),
            )
            state = (z_mean, betas, scaler, w)

        z_mean, betas, scaler, w = state
        X_day = scaler.transform(X_raw[day_mask] - np.outer(z[day_mask] - z_mean, betas))
        X_day1 = np.column_stack([X_day, np.ones(X_day.shape[0], dtype=float)])
        delta = np.clip(X_day1 @ w, -float(clip_logit_delta), float(clip_logit_delta))
        out[day_mask] = sigmoid(z[day_mask] + delta)
    return out


def _floor_cent(x: float) -> float:
    return math.floor(max(0.0, float(x)) * 100.0 + 1e-12) / 100.0


def simulate_seasonal_kelly_portfolio(
    season_df: pd.DataFrame,
    *,
    prob_col: str = "p_hub_c65",
    home_ml_col: str = "home_ml",
    away_ml_col: str = "away_ml",
    y_col: str = "home_win",
    date_col: str = "game_date",
    game_id_col: str = "game_id",
    mode: str = "straight",
    min_ev: float = 0.05,
    ev_cap: float = 0.06,
    min_decimal: float = 1.20,
    max_decimal: float = 3.50,
    omega: float = 0.25,
    max_bet_frac: float = 0.02,
    max_slate_frac: float = 0.10,
    starting_bankroll: float = 50.0,
) -> dict[str, Any]:
    """Replay a seasonal bankroll using variance-capped Quadratic Fractional Kelly."""
    df = season_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    br = float(starting_bankroll)
    peak = br
    max_dd = 0.0
    n_bets = 0
    n_wins = 0
    unit_pnl = 0.0
    staked_total = 0.0
    pnl_total = 0.0

    for _, day_df in df.groupby(date_col, sort=True):
        legs: list[tuple[str, float, float, float, bool]] = []
        for _, row in day_df.iterrows():
            gid = str(row[game_id_col])
            p_h = float(row[prob_col])
            h_dec = american_to_decimal(float(row[home_ml_col]))
            a_dec = american_to_decimal(float(row[away_ml_col]))
            h_won = float(row[y_col]) == 1.0
            best: tuple[str, float, float, float, bool] | None = None
            for p_side, dec_side, won_side in ((p_h, h_dec, h_won), (1.0 - p_h, a_dec, not h_won)):
                if float(min_decimal) <= dec_side <= float(max_decimal):
                    ev = p_side * dec_side - 1.0
                    if ev >= float(min_ev):
                        ev_eff = min(ev, float(ev_cap))
                        p_eff = (1.0 + ev_eff) / dec_side
                        if best is None or ev > best[2]:
                            best = (gid, p_eff, ev, dec_side, won_side)
            if best is not None:
                legs.append(best)

        if not legs:
            continue
        legs.sort(key=lambda item: -item[2])
        cands: list[tuple[set[str], float, float, list[tuple[str, float, float, float, bool]]]] = []
        if mode in {"straight", "mixed_1_2"}:
            for leg in legs[:6]:
                cands.append(({leg[0]}, leg[1], leg[3], [leg]))
        if mode in {"parlay2", "mixed_1_2"} and len(legs) >= 2:
            for a, b in combinations(legs[:6], 2):
                dec_j = a[3] * b[3]
                if 2.20 <= dec_j <= 4.50:
                    p_j = a[1] * b[1]
                    if p_j * dec_j - 1.0 >= float(min_ev):
                        cands.append(({a[0], b[0]}, p_j, dec_j, [a, b]))
        if not cands:
            continue

        cands.sort(
            key=lambda t: (t[1] * t[2] - 1.0) / (math.sqrt(max(1e-9, t[1] * (1.0 - t[1]))) * t[2]),
            reverse=True,
        )
        picked = []
        used: set[str] = set()
        limit = 6 if mode == "straight" else 3
        for cand in cands:
            if not (cand[0] & used):
                picked.append(cand)
                used |= cand[0]
                if len(picked) >= limit:
                    break

        fracs = solve_slate_quadratic_kelly(
            [t[1] for t in picked],
            [t[2] for t in picked],
            omega=omega,
            max_bet_frac=max_bet_frac,
            max_slate_frac=max_slate_frac,
        )
        day_pnl = 0.0
        for cand, fk in zip(picked, fracs):
            mult = 1.0
            lost = False
            for leg in cand[3]:
                if leg[4]:
                    mult *= leg[3]
                else:
                    lost = True
                    break
            net_mult = -1.0 if lost else (mult - 1.0)
            if fk > 0.0 and br >= 0.50:
                stake = min(br, _floor_cent(br * float(fk)))
                if stake >= 0.05:
                    pnl = round(stake * net_mult, 2)
                    day_pnl += pnl
                    staked_total += stake
                    pnl_total += pnl
                    unit_pnl += net_mult
                    n_bets += 1
                    if net_mult > 0.0:
                        n_wins += 1

        br = round(max(0.0, br + day_pnl), 2)
        peak = max(peak, br)
        max_dd = min(max_dd, br - peak)

    return {
        "starting_bankroll": float(starting_bankroll),
        "ending_bankroll": br,
        "max_drawdown": round(max_dd, 2),
        "n_tickets": n_bets,
        "n_wins": n_wins,
        "hit_rate": round(n_wins / max(1, n_bets), 4) if n_bets else 0.0,
        "unit_roi": round(unit_pnl / max(1, n_bets), 4) if n_bets else 0.0,
        "kelly_roi": round(pnl_total / max(1e-6, staked_total), 4) if staked_total > 0 else 0.0,
        "total_staked": round(staked_total, 2),
        "total_pnl": round(pnl_total, 2),
    }
