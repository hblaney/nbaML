"""Strict walk-forward evaluation for Stint RAPM, Clean Leverage, and Officials.

Protocol:
- Train:   2022-23, 2023-24
- Confirm: 2024-25
- Holdout: 2025-26

Zero-leakage rules:
1. All model parameters (expanding Ridge / Logistic + standardization + residual
   orthogonalization) are fit strictly on earlier dates (``prior_date < game_date``).
2. Any model/hyperparameter/betting-rule freeze uses ONLY Train (2022-23..2023-24)
   (or Train + Confirm for final holdout check, clearly labeled), and never touches
   2025-26 for selection.
3. Parlays enforce distinct ``game_id``s per ticket so sportsbook independent
   decimal multiplication ``∏ d_i`` and cross-game independence hold identically,
   with exact push leg reduction (pushed leg -> 1.0x multiplier).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge

from nba_betting.pricing.kelly import american_to_decimal

TRAIN_SEASONS = ("2022-23", "2023-24")
CONFIRM_SEASON = "2024-25"
HOLDOUT_SEASON = "2025-26"
ALL_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")

# Feature groups for ablation & market-residual modeling
RAW_MARGIN_COLS = [
    "raw_net_diff",
    "rest_diff",
    "b2b_diff",
    "three_in_four_diff",
    "games_last7_diff",
]

CLEAN_LEVERAGE_MARGIN_COLS = [
    "clean_net_diff",
    "lev_net_diff",
    "garbage_distortion_diff",
    "q4_giveback_diff",
    "rest_diff",
    "b2b_diff",
    "three_in_four_diff",
    "games_last7_diff",
]

CLEAN_RAPM_MARGIN_COLS = [
    "clean_net_diff",
    "lev_net_diff",
    "q4_giveback_diff",
    "rapm_rot_diff",
    "rapm_missing_diff",
    "clean_pm_rot_diff",
    "clean_missing_min_diff",
    "rest_diff",
    "b2b_diff",
    "three_in_four_diff",
    "games_last7_diff",
]

FULL_MARGIN_COLS = CLEAN_RAPM_MARGIN_COLS + [
    "crew_home_foul_bias",
    "crew_home_fta_bias",
    "ix_crew_fta_x_net_ftr_diff",
    "ix_crew_foul_x_pf_diff",
    "ix_crew_home_bias_x_away_pf",
]

RAW_TOTAL_COLS = [
    "raw_expected_total",
    "b2b_sum",
]

FULL_TOTAL_COLS = [
    "clean_expected_total",
    "raw_expected_total",
    "clean_vs_raw_total_gap",
    "recent_ot_inflation",
    "recent_endgame_fta_sum",
    "orapm_sum",
    "drapm_sum",
    "rapm_scoring_tilt_sum",
    "b2b_sum",
    "crew_adj_fta_lift",
    "crew_adj_foul_lift",
    "crew_pace_lift",
    "crew_reg_total_lift",
    "ix_crew_fta_x_ftr_sum",
    "ix_crew_foul_x_pf_sum",
    "ix_crew_pace_x_team_pace",
]

ATS_RESIDUAL_COLS = [
    "clean_rapm_vs_spread",
    "garbage_orth_diff",
    "rapm_orth_diff",
    "q4_giveback_diff",
    "rapm_missing_diff",
    "crew_home_foul_bias",
    "crew_home_fta_bias",
    "crew_home_ats_bias",
    "ix_crew_fta_x_net_ftr_diff",
    "ix_crew_foul_x_pf_diff",
    "ix_crew_home_bias_x_away_pf",
]

OFFICIALS_ONLY_TOTAL_RESIDUAL_COLS = [
    "crew_adj_fta_lift",
    "crew_adj_foul_lift",
    "crew_pace_lift",
    "crew_total_vs_close_lift",
    "ix_crew_fta_x_ftr_sum",
    "ix_crew_foul_x_pf_sum",
    "ix_crew_pace_x_team_pace",
]

FULL_TOTAL_RESIDUAL_COLS = [
    "clean_tot_vs_line",
    "clean_vs_raw_total_gap",
    "recent_ot_inflation",
    "recent_endgame_fta_sum",
    "orapm_sum",
    "drapm_sum",
    "rapm_scoring_tilt_sum",
] + OFFICIALS_ONLY_TOTAL_RESIDUAL_COLS


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def expanding_ridge_margin_walkforward(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "home_margin",
    ridge_alpha: float = 150.0,
    min_train_games: int = 250,
    refit_every_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-grouped expanding standardized Ridge -> (pred_margin, sigma_hat, p_home_win)."""
    ordered = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    X_all = np.nan_to_num(ordered[feature_cols].to_numpy(dtype=float), nan=0.0)
    is_neutral = ordered["is_neutral"].astype(float).to_numpy()
    hca_col = 1.0 - is_neutral
    y_all = ordered[target_col].to_numpy(dtype=float)

    n = len(ordered)
    pred_m = np.empty(n, dtype=float)
    sigmas = np.empty(n, dtype=float)
    probs = np.empty(n, dtype=float)

    model: Ridge | None = None
    mu = np.zeros(X_all.shape[1], dtype=float)
    sd = np.ones(X_all.shape[1], dtype=float)
    sigma_cur = 12.2
    days_since_fit = 999

    date_groups = ordered.groupby("game_date", sort=True).indices
    for idxs in date_groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        first_idx = int(idx_arr[0])
        if first_idx < min_train_games:
            pred_m[idx_arr] = 2.2 * hca_col[idx_arr]
            sigmas[idx_arr] = sigma_cur
            probs[idx_arr] = np.clip(norm.cdf(pred_m[idx_arr] / sigma_cur), 1e-6, 1.0 - 1e-6)
            continue

        if model is None or days_since_fit >= refit_every_days:
            X_tr_raw = X_all[:first_idx]
            mu, sd = _standardize_fit(X_tr_raw)
            X_tr = np.column_stack([hca_col[:first_idx], (X_tr_raw - mu) / sd])
            y_tr = y_all[:first_idx]
            rg = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
            rg.fit(X_tr, y_tr)
            resid = y_tr - rg.predict(X_tr)
            sigma_cur = float(np.clip(np.std(resid, ddof=1), 10.5, 14.5))
            model = rg
            days_since_fit = 0

        X_day = np.column_stack([hca_col[idx_arr], (X_all[idx_arr] - mu) / sd])
        m_hat = model.predict(X_day)
        pred_m[idx_arr] = m_hat
        sigmas[idx_arr] = sigma_cur
        probs[idx_arr] = np.clip(norm.cdf(m_hat / sigma_cur), 1e-6, 1.0 - 1e-6)
        days_since_fit += 1

    return pred_m, sigmas, probs


def expanding_ridge_total_walkforward(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "actual_total",
    ridge_alpha: float = 200.0,
    min_train_games: int = 250,
    refit_every_days: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Date-grouped expanding standardized Ridge for game totals -> (pred_total, sigma_tot)."""
    ordered = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    X_all = np.nan_to_num(ordered[feature_cols].to_numpy(dtype=float), nan=0.0)
    y_all = ordered[target_col].to_numpy(dtype=float)

    n = len(ordered)
    pred_t = np.empty(n, dtype=float)
    sigmas = np.empty(n, dtype=float)

    model: Ridge | None = None
    mu = np.zeros(X_all.shape[1], dtype=float)
    sd = np.ones(X_all.shape[1], dtype=float)
    sigma_cur = 18.2
    days_since_fit = 999

    date_groups = ordered.groupby("game_date", sort=True).indices
    for idxs in date_groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        first_idx = int(idx_arr[0])
        if first_idx < min_train_games:
            pred_t[idx_arr] = 228.0 if first_idx == 0 else float(np.mean(y_all[:first_idx]))
            sigmas[idx_arr] = sigma_cur
            continue

        if model is None or days_since_fit >= refit_every_days:
            X_tr_raw = X_all[:first_idx]
            mu, sd = _standardize_fit(X_tr_raw)
            X_tr = (X_tr_raw - mu) / sd
            y_tr = y_all[:first_idx]
            rg = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
            rg.fit(X_tr, y_tr)
            resid = y_tr - rg.predict(X_tr)
            sigma_cur = float(np.clip(np.std(resid, ddof=1), 15.0, 22.0))
            model = rg
            days_since_fit = 0

        X_day = (X_all[idx_arr] - mu) / sd
        pred_t[idx_arr] = model.predict(X_day)
        sigmas[idx_arr] = sigma_cur
        days_since_fit += 1

    return pred_t, sigmas


def add_orthogonal_market_residual_columns(
    df: pd.DataFrame,
    *,
    clean_rapm_margin_pred: np.ndarray,
    clean_rapm_total_pred: np.ndarray,
    min_train_games: int = 200,
) -> pd.DataFrame:
    """Attach strictly expanding orthogonalized residuals relative to raw ratings & market."""
    out = df.sort_values(["game_date", "game_id"]).reset_index(drop=True).copy()
    out["pred_margin_no_odds"] = clean_rapm_margin_pred
    out["pred_total_no_odds"] = clean_rapm_total_pred

    spread_arr = out["spread_home"].to_numpy(dtype=float)
    mkt_margin = np.where(np.isfinite(spread_arr), -spread_arr, clean_rapm_margin_pred)
    out["clean_rapm_vs_spread"] = clean_rapm_margin_pred - mkt_margin
    out["clean_tot_vs_line"] = clean_rapm_total_pred - out["total_line"].to_numpy(dtype=float)

    raw_n = out["raw_net_diff"].to_numpy(dtype=float)
    cln_n = out["clean_net_diff"].to_numpy(dtype=float)
    rapm_n = out["rapm_rot_diff"].to_numpy(dtype=float)

    garb_orth = np.zeros(len(out), dtype=float)
    rapm_orth = np.zeros(len(out), dtype=float)

    date_groups = out.groupby("game_date", sort=True).indices
    slope_c = 1.0
    int_c = 0.0
    slope_r = 0.5
    int_r = 0.0
    for idxs in date_groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        first_idx = int(idx_arr[0])
        if first_idx >= min_train_games:
            rx = raw_n[:first_idx]
            vx = float(np.var(rx))
            if vx > 1e-6:
                mx = float(np.mean(rx))
                mc = float(np.mean(cln_n[:first_idx]))
                mr = float(np.mean(rapm_n[:first_idx]))
                slope_c = float(np.mean((rx - mx) * (cln_n[:first_idx] - mc)) / vx)
                int_c = mc - slope_c * mx
                slope_r = float(np.mean((rx - mx) * (rapm_n[:first_idx] - mr)) / vx)
                int_r = mr - slope_r * mx
        garb_orth[idx_arr] = cln_n[idx_arr] - (int_c + slope_c * raw_n[idx_arr])
        rapm_orth[idx_arr] = rapm_n[idx_arr] - (int_r + slope_r * raw_n[idx_arr])

    out["garbage_orth_diff"] = garb_orth
    out["rapm_orth_diff"] = rapm_orth
    return out


def expanding_market_residual_probs(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    residual_col: str,
    novig_prob_col: str,
    valid_mask_col: str | None = None,
    ridge_alpha: float = 1200.0,
    shrink_scale: float = 1.0,
    min_train_games: int = 300,
    refit_every_days: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict residual vs closing line via strong Ridge; map onto Probit(novig_prob)."""
    ordered = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    X_all = np.nan_to_num(ordered[feature_cols].to_numpy(dtype=float), nan=0.0)
    y_all = np.nan_to_num(ordered[residual_col].to_numpy(dtype=float), nan=0.0)
    p0 = np.clip(ordered[novig_prob_col].fillna(0.5).to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
    if valid_mask_col is not None:
        valid = ordered[valid_mask_col].astype(bool).to_numpy()
    else:
        valid = np.ones(len(ordered), dtype=bool)

    n = len(ordered)
    r_hat = np.zeros(n, dtype=float)
    p_out = p0.copy()

    model: Ridge | None = None
    mu = np.zeros(X_all.shape[1], dtype=float)
    sd = np.ones(X_all.shape[1], dtype=float)
    sigma_res = 13.5 if "ats" in residual_col else 18.0
    days_since_fit = 999

    date_groups = ordered.groupby("game_date", sort=True).indices
    for idxs in date_groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        first_idx = int(idx_arr[0])
        tr_mask = valid[:first_idx]
        if int(tr_mask.sum()) < min_train_games:
            continue

        if model is None or days_since_fit >= refit_every_days:
            X_tr_raw = X_all[:first_idx][tr_mask]
            y_tr = y_all[:first_idx][tr_mask]
            mu, sd = _standardize_fit(X_tr_raw)
            X_tr = (X_tr_raw - mu) / sd
            rg = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
            rg.fit(X_tr, y_tr)
            resid = y_tr - rg.predict(X_tr)
            sigma_res = float(max(np.std(resid, ddof=1), 8.0))
            model = rg
            days_since_fit = 0

        X_day = (X_all[idx_arr] - mu) / sd
        pred_r = float(shrink_scale) * model.predict(X_day)
        r_hat[idx_arr] = pred_r
        z0 = norm.ppf(p0[idx_arr])
        p_out[idx_arr] = np.clip(norm.cdf(z0 + pred_r / sigma_res), 1e-4, 1.0 - 1e-4)
        days_since_fit += 1

    return r_hat, p_out


def expanding_ml_market_anchor_probs(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    C: float = 0.08,
    min_train_games: int = 300,
    refit_every_days: int = 7,
) -> np.ndarray:
    """Expanding L2 Logistic with fixed 1.0 offset on logit(p_home_novig) + orth features."""
    ordered = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    p_mkt = np.clip(ordered["p_home_novig"].to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
    logit_mkt = np.log(p_mkt / (1.0 - p_mkt))
    X_orth = np.nan_to_num(ordered[feature_cols].to_numpy(dtype=float), nan=0.0)
    y_all = ordered["home_win"].to_numpy(dtype=float)

    p_out = p_mkt.copy()
    mu = np.zeros(X_orth.shape[1], dtype=float)
    sd = np.ones(X_orth.shape[1], dtype=float)
    model: LogisticRegression | None = None
    days_since_fit = 999

    # Use sample-weighted / offset-approximated logistic by including logit_mkt unpenalized
    # via large scaling factor (100x) so L2 penalty applies almost purely to orthogonal features!
    date_groups = ordered.groupby("game_date", sort=True).indices
    for idxs in date_groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        first_idx = int(idx_arr[0])
        if first_idx < min_train_games:
            continue
        if model is None or days_since_fit >= refit_every_days:
            X_tr_raw = X_orth[:first_idx]
            mu, sd = _standardize_fit(X_tr_raw)
            X_tr = np.column_stack([25.0 * logit_mkt[:first_idx], (X_tr_raw - mu) / sd])
            lr = LogisticRegression(C=float(C), solver="lbfgs", max_iter=500)
            lr.fit(X_tr, y_all[:first_idx])
            model = lr
            days_since_fit = 0
        X_day = np.column_stack([25.0 * logit_mkt[idx_arr], (X_orth[idx_arr] - mu) / sd])
        p_out[idx_arr] = np.clip(model.predict_proba(X_day)[:, 1], 1e-4, 1.0 - 1e-4)
        days_since_fit += 1

    return p_out


def grade_straight_market(
    df: pd.DataFrame,
    *,
    market: str,
    p_col: str,
    min_vig_edge: float = 0.01,
) -> pd.DataFrame:
    """Grade flat 1-unit vigged straight bets for ML, ATS, or Totals."""
    rows: list[dict[str, Any]] = []
    p_arr = df[p_col].to_numpy(dtype=float)
    for idx, r in enumerate(df.itertuples(index=False)):
        p_pos = p_arr[idx]
        if not np.isfinite(p_pos):
            continue
        p_pos = float(p_pos)
        p_neg = 1.0 - p_pos

        if market == "ml":
            odds_pos, odds_neg = float(r.home_ml), float(r.away_ml)
            res_pos = 1.0 if float(r.home_win) == 1.0 else 0.0
            res_neg = 1.0 - res_pos
            is_push = False
            sides = (("home_ml", p_pos, odds_pos, res_pos), ("away_ml", p_neg, odds_neg, res_neg))
        elif market == "ats":
            if pd.isna(r.spread_home) or pd.isna(r.spread_home_odds) or pd.isna(r.spread_away_odds):
                continue
            odds_pos, odds_neg = float(r.spread_home_odds), float(r.spread_away_odds)
            is_push = bool(r.spread_push)
            res_pos = 0.5 if is_push else float(r.home_cover)
            res_neg = 0.5 if is_push else float(r.away_cover)
            sides = (
                ("home_ats", p_pos, odds_pos, res_pos),
                ("away_ats", p_neg, odds_neg, res_neg),
            )
        elif market == "total":
            if pd.isna(r.total_line) or pd.isna(r.over_odds) or pd.isna(r.under_odds):
                continue
            odds_pos, odds_neg = float(r.over_odds), float(r.under_odds)
            is_push = bool(r.total_push)
            res_pos = 0.5 if is_push else float(r.over_hit)
            res_neg = 0.5 if is_push else float(r.under_hit)
            sides = (("over", p_pos, odds_pos, res_pos), ("under", p_neg, odds_neg, res_neg))
        else:
            raise ValueError(f"Unknown market {market!r}")

        best_side = None
        best_edge = -1e9
        for sname, pm, am, res in sides:
            if not np.isfinite(am) or am == 0.0:
                continue
            dec = american_to_decimal(am)
            pvig = 1.0 / dec
            edge = pm - pvig
            if edge >= float(min_vig_edge) and edge > best_edge:
                best_edge = edge
                if is_push:
                    pnl = 0.0
                elif res == 1.0:
                    pnl = dec - 1.0
                else:
                    pnl = -1.0
                best_side = {
                    "season": r.season,
                    "game_date": r.game_date,
                    "game_id": r.game_id,
                    "market": market,
                    "side": sname,
                    "p_model": pm,
                    "p_vigged": pvig,
                    "vig_edge": edge,
                    "american": am,
                    "decimal": dec,
                    "is_push": is_push,
                    "won": bool(res == 1.0),
                    "pnl": pnl,
                }
        if best_side is not None:
            rows.append(best_side)
    return pd.DataFrame(rows)


def summarize_straight_bets(bets: pd.DataFrame) -> dict[str, Any]:
    if bets.empty:
        return {"n_bets": 0, "n_wins": 0, "n_pushes": 0, "hit_rate": None, "pnl": 0.0, "roi": None}
    non_push = bets[~bets["is_push"]]
    n_bets = len(bets)
    pnl = float(bets["pnl"].sum())
    return {
        "n_bets": n_bets,
        "n_wins": int(bets["won"].sum()),
        "n_pushes": int(bets["is_push"].sum()),
        "hit_rate": float(non_push["won"].mean()) if len(non_push) > 0 else None,
        "pnl": pnl,
        "roi": pnl / float(n_bets),
    }


def build_multimarket_legs(
    df: pd.DataFrame,
    *,
    markets: Sequence[str] = ("ml", "ats", "total"),
    ml_col: str = "p_ml_blend",
    ats_col: str = "p_ats_resid",
    total_col: str = "p_tot_resid",
    min_leg_vig_edge: float = 0.0,
) -> pd.DataFrame:
    """Build eligible vig-beating straight legs across ML, ATS, and Totals."""
    rows: list[dict[str, Any]] = []
    ml_arr = df[ml_col].to_numpy(dtype=float) if ml_col in df.columns else np.full(len(df), np.nan)
    ats_arr = df[ats_col].to_numpy(dtype=float) if ats_col in df.columns else np.full(len(df), np.nan)
    tot_arr = df[total_col].to_numpy(dtype=float) if total_col in df.columns else np.full(len(df), np.nan)
    for idx, r in enumerate(df.itertuples(index=False)):
        specs: list[tuple[str, str, float, float, str]] = []
        if "ml" in markets and np.isfinite(ml_arr[idx]):
            pm = float(ml_arr[idx])
            hw = float(r.home_win) == 1.0
            specs.append(("ml", "home", pm, float(r.home_ml), "win" if hw else "loss"))
            specs.append(("ml", "away", 1.0 - pm, float(r.away_ml), "loss" if hw else "win"))
        if (
            "ats" in markets
            and pd.notna(r.spread_home)
            and pd.notna(r.spread_home_odds)
            and np.isfinite(ats_arr[idx])
        ):
            pa = float(ats_arr[idx])
            if bool(r.spread_push):
                hres, ares = "push", "push"
            else:
                hc = float(r.home_cover) == 1.0
                hres, ares = ("win" if hc else "loss"), ("loss" if hc else "win")
            specs.append(("ats", "home_ats", pa, float(r.spread_home_odds), hres))
            specs.append(("ats", "away_ats", 1.0 - pa, float(r.spread_away_odds), ares))
        if (
            "total" in markets
            and pd.notna(r.total_line)
            and pd.notna(r.over_odds)
            and np.isfinite(tot_arr[idx])
        ):
            pt = float(tot_arr[idx])
            if bool(r.total_push):
                ores, ures = "push", "push"
            else:
                ov = float(r.over_hit) == 1.0
                ores, ures = ("win" if ov else "loss"), ("loss" if ov else "win")
            specs.append(("total", "over", pt, float(r.over_odds), ores))
            specs.append(("total", "under", 1.0 - pt, float(r.under_odds), ures))

        for mkt, side, p_m, am, outcome in specs:
            if not np.isfinite(am) or am == 0.0 or not (0.0 < p_m < 1.0):
                continue
            dec = american_to_decimal(am)
            pvig = 1.0 / dec
            vig_edge = p_m - pvig
            if vig_edge <= float(min_leg_vig_edge):
                continue
            rows.append(
                {
                    "season": str(r.season),
                    "game_date": str(r.game_date),
                    "game_id": str(r.game_id),
                    "market": mkt,
                    "side": side,
                    "p_model": p_m,
                    "p_vigged": pvig,
                    "vig_edge": vig_edge,
                    "american": am,
                    "decimal": dec,
                    "outcome": outcome,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["game_date", "game_id", "market", "side"]).reset_index(
        drop=True
    )


def select_daily_parlay_tickets(
    legs: pd.DataFrame,
    *,
    n_legs: int = 2,
    min_price_edge: float = 0.05,
    max_parlay_decimal: float | None = 12.0,
    max_per_day: int = 1,
) -> pd.DataFrame:
    """Enumerate distinct-game same-day parlays clearing min_price_edge; pick top daily non-overlapping."""
    if legs.empty:
        return pd.DataFrame()

    picked_rows: list[dict[str, Any]] = []
    for gdate, day_df in legs.groupby("game_date", sort=True):
        # Keep only best vig_edge leg per game_id to avoid redundant combinatorial explosion
        best_per_game = (
            day_df.sort_values("vig_edge", ascending=False)
            .drop_duplicates(subset=["game_id"], keep="first")
            .reset_index(drop=True)
        )
        if len(best_per_game) < n_legs:
            continue
        recs = list(best_per_game.to_dict("records"))
        day_cands: list[dict[str, Any]] = []
        for combo in combinations(range(len(recs)), n_legs):
            chosen = [recs[i] for i in combo]
            dec = 1.0
            joint_p = 1.0
            settled_mult = 1.0
            any_loss = False
            n_pushes = 0
            for L in chosen:
                dec *= float(L["decimal"])
                joint_p *= float(L["p_model"])
                if L["outcome"] == "loss":
                    any_loss = True
                    settled_mult = 0.0
                elif L["outcome"] == "push":
                    n_pushes += 1
                else:
                    if not any_loss:
                        settled_mult *= float(L["decimal"])
            if max_parlay_decimal is not None and dec > float(max_parlay_decimal):
                continue
            book_imp = 1.0 / dec
            price_edge = joint_p / book_imp - 1.0
            if price_edge < float(min_price_edge):
                continue
            day_cands.append(
                {
                    "season": chosen[0]["season"],
                    "game_date": gdate,
                    "n_legs": n_legs,
                    "game_ids": "|".join(c["game_id"] for c in chosen),
                    "markets": "|".join(c["market"] for c in chosen),
                    "sides": "|".join(c["side"] for c in chosen),
                    "p_joint": joint_p,
                    "parlay_decimal": dec,
                    "price_edge": price_edge,
                    "ev_per_unit": joint_p * dec - 1.0,
                    "settled_mult": 0.0 if any_loss else settled_mult,
                    "ticket_won": (not any_loss) and (n_pushes < n_legs),
                    "all_push": (not any_loss) and (n_pushes == n_legs),
                }
            )
        if not day_cands:
            continue
        day_cands.sort(key=lambda x: -float(x["price_edge"]))
        used_games: set[str] = set()
        kept = 0
        for cand in day_cands:
            gset = set(cand["game_ids"].split("|"))
            if gset & used_games:
                continue
            picked_rows.append(cand)
            used_games |= gset
            kept += 1
            if kept >= max_per_day:
                break
    return pd.DataFrame(picked_rows)


def simulate_seasonal_bankroll(
    tickets: pd.DataFrame,
    *,
    starting_bankroll: float = 50.0,
    stake_pct: float = 0.02,
    seasons: Sequence[str] = ALL_SEASONS,
) -> dict[str, dict[str, Any]]:
    """Replay $50 seasonal bankroll compounding with floor-to-cent percentage stakes."""
    out: dict[str, dict[str, Any]] = {}
    for s in seasons:
        sub = (
            tickets[tickets["season"] == s].sort_values("game_date").reset_index(drop=True)
            if not tickets.empty
            else pd.DataFrame()
        )
        br = float(starting_bankroll)
        peak = br
        max_dd = 0.0
        n_staked = 0
        n_wins = 0
        total_staked = 0.0
        total_pnl = 0.0

        for r in sub.itertuples(index=False):
            stake = math.floor(br * float(stake_pct) * 100.0) / 100.0
            if stake < 0.10 or stake > br:
                continue
            n_staked += 1
            total_staked += stake
            payout = round(stake * float(r.settled_mult), 2)
            pnl = round(payout - stake, 2)
            br = round(br + pnl, 2)
            total_pnl += pnl
            if bool(r.ticket_won):
                n_wins += 1
            peak = max(peak, br)
            max_dd = min(max_dd, br - peak)

        # Also compute 1-unit flat ROI on identical tickets
        unit_pnl = float((sub["settled_mult"] - 1.0).sum()) if not sub.empty else 0.0
        unit_roi = (unit_pnl / len(sub)) if not sub.empty else None

        out[s] = {
            "n_tickets": n_staked,
            "n_wins": n_wins,
            "start_bankroll": float(starting_bankroll),
            "end_bankroll": round(br, 2),
            "net_profit": round(br - starting_bankroll, 2),
            "max_drawdown": round(max_dd, 2),
            "beat_start": bool(br > starting_bankroll),
            "unit_roi": unit_roi,
        }
    return out
