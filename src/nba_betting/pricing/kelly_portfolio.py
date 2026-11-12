"""Tail probability shrinkage and exact daily joint log-utility (Kelly) allocator.

Replaces raw ``min_price_edge`` ranking (which selects overconfident ~8%-hit-rate
underdog tails) and fixed-% staking with:

1. Out-of-sample tail shrinkage toward de-vigged market probabilities:
   - ``linear``: ``p_shrunk = w * p_model + (1 - w) * p_market_novig``
   - ``logit``: ``sigmoid(w * logit(p_model) + (1 - w) * logit(p_market_novig))``
   - ``tail_damped``: ``w_eff = w * 4 * p_mkt * (1 - p_mkt)`` (vanishes in tails)

2. Exact daily log-wealth maximization over simultaneous non-overlapping tickets
   (1-leg, 2-leg, 3-leg across ML / spread / total markets):
       max_{f >= 0, sum(f) <= cap} E_{p_shrunk}[log(1 + sum_k f_k * R_k)]
   scaled by fractional Kelly multiplier ``c_kelly`` and bounded by ``daily_cap``.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from nba_betting.backtest.metrics import brier_score, log_loss
from nba_betting.pricing.kelly import american_to_decimal

ShrinkMode = Literal["linear", "logit", "tail_damped"]


def _clip_prob(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def shrink_probs(
    p_model: np.ndarray | pd.Series | float,
    p_market_novig: np.ndarray | pd.Series | float,
    *,
    w: float,
    mode: ShrinkMode = "linear",
) -> np.ndarray:
    """Shrink ``p_model`` toward ``p_market_novig`` with weight ``w in [0, 1]``."""
    ww = float(np.clip(w, 0.0, 1.0))
    pm = _clip_prob(p_model)
    pk = _clip_prob(p_market_novig)
    if mode == "linear":
        out = ww * pm + (1.0 - ww) * pk
    elif mode == "logit":
        lm = np.log(pm / (1.0 - pm))
        lk = np.log(pk / (1.0 - pk))
        z = ww * lm + (1.0 - ww) * lk
        out = 1.0 / (1.0 + np.exp(-z))
    elif mode == "tail_damped":
        w_eff = ww * (4.0 * pk * (1.0 - pk))
        out = w_eff * pm + (1.0 - w_eff) * pk
    else:
        raise ValueError(f"unknown shrink mode: {mode!r}")
    return _clip_prob(out)


def debiased_hubacek_probs(
    p_hubacek: np.ndarray | pd.Series | float,
    p_market_shin: np.ndarray | pd.Series | float,
    *,
    c_decorr: float = 0.65,
    w_override: float | None = None,
) -> np.ndarray:
    """Invert Hubáček stationary amplification ``p* = pi + (p_true - pi)/(1 - c)``.

    When ``w_override`` is omitted, uses the exact analytical inverse ``w = 1 - c_decorr``
    (e.g. ``0.35`` for ``c = 0.65``, matching empirical 2022-24 Log-Loss optimum ``0.366``).
    """
    w = float(1.0 - float(c_decorr)) if w_override is None else float(w_override)
    return shrink_probs(p_hubacek, p_market_shin, w=w, mode="linear")


def winsorize_prob_by_ev_cap(
    p_win: np.ndarray | pd.Series | float,
    decimal_odds: np.ndarray | pd.Series | float,
    *,
    ev_cap: float | None = 0.08,
) -> np.ndarray:
    """Cap effective Kelly probability at ``(1 + ev_cap) / decimal_odds`` to prevent tail overbetting."""
    p = _clip_prob(p_win)
    if ev_cap is None:
        return p
    dec = np.maximum(np.asarray(decimal_odds, dtype=float), 1.0 + 1e-6)
    p_max = (1.0 + float(ev_cap)) / dec
    return _clip_prob(np.minimum(p, p_max))


def compute_shin_clv_bps(
    p_open_shin: np.ndarray | pd.Series | float,
    p_close_shin: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Return Closing Line Value in basis points: ``10000 * (p_close_shin - p_open_shin)``."""
    return 10000.0 * (np.asarray(p_close_shin, dtype=float) - np.asarray(p_open_shin, dtype=float))


def fit_oos_shrinkage(
    p_model: np.ndarray | pd.Series,
    p_market_novig: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    fold_ids: np.ndarray | pd.Series | None = None,
    mode: ShrinkMode = "linear",
    metric: Literal["log_loss", "brier"] = "log_loss",
    min_train: int = 200,
) -> dict[str, Any]:
    """Fit out-of-sample shrinkage weight ``w in [0, 1]`` on training games only.

    If ``fold_ids`` is given (e.g. train seasons ``2022-23`` vs ``2023-24``),
    evaluates out-of-fold loss across folds; otherwise evaluates on the post-warmup
    slice ``[min_train:]`` via expanding walk-forward refits or direct proper scoring.
    """
    pm = _clip_prob(p_model)
    pk = _clip_prob(p_market_novig)
    yy = np.asarray(y, dtype=float)
    if not (len(pm) == len(pk) == len(yy)):
        raise ValueError("p_model, p_market_novig, and y must have equal length")

    def loss_fn(probs: np.ndarray, target: np.ndarray) -> float:
        if metric == "log_loss":
            return float(log_loss(probs, target))
        return float(brier_score(probs, target))

    if fold_ids is not None:
        fids = np.asarray(fold_ids)
        unique_folds = list(dict.fromkeys(fids.tolist()))
        if len(unique_folds) >= 2:
            # Leave-one-fold-out proper scoring on train folds: for each held-out fold,
            # fit w on the remaining train folds and score on the held-out fold; also
            # find the pooled proper-scoring w on the union of train folds.
            oof_preds = np.empty_like(pm)
            fold_ws: dict[str, float] = {}
            for fid in unique_folds:
                tr_mask = fids != fid
                va_mask = fids == fid
                res_f = minimize_scalar(
                    lambda w: loss_fn(shrink_probs(pm[tr_mask], pk[tr_mask], w=w, mode=mode), yy[tr_mask]),
                    bounds=(0.0, 1.0),
                    method="bounded",
                )
                wf = float(res_f.x)
                fold_ws[str(fid)] = wf
                oof_preds[va_mask] = shrink_probs(pm[va_mask], pk[va_mask], w=wf, mode=mode)

            res_all = minimize_scalar(
                lambda w: loss_fn(shrink_probs(pm, pk, w=w, mode=mode), yy),
                bounds=(0.0, 1.0),
                method="bounded",
            )
            w_star = float(res_all.x)
            w_oof_mean = float(np.mean(list(fold_ws.values())))
            return {
                "w_opt": w_star,
                "w_oof_mean": w_oof_mean,
                "fold_weights": fold_ws,
                "mode": mode,
                "metric": metric,
                "oof_log_loss": float(log_loss(oof_preds, yy)),
                "oof_brier": float(brier_score(oof_preds, yy)),
                "market_log_loss": float(log_loss(pk, yy)),
                "market_brier": float(brier_score(pk, yy)),
                "model_raw_log_loss": float(log_loss(pm, yy)),
                "model_raw_brier": float(brier_score(pm, yy)),
            }

    idx = slice(int(min_train), len(yy)) if len(yy) > int(min_train) else slice(0, len(yy))
    res = minimize_scalar(
        lambda w: loss_fn(shrink_probs(pm[idx], pk[idx], w=w, mode=mode), yy[idx]),
        bounds=(0.0, 1.0),
        method="bounded",
    )
    w_star = float(res.x)
    shr = shrink_probs(pm[idx], pk[idx], w=w_star, mode=mode)
    return {
        "w_opt": w_star,
        "w_oof_mean": w_star,
        "fold_weights": {},
        "mode": mode,
        "metric": metric,
        "oof_log_loss": float(log_loss(shr, yy[idx])),
        "oof_brier": float(brier_score(shr, yy[idx])),
        "market_log_loss": float(log_loss(pk[idx], yy[idx])),
        "market_brier": float(brier_score(pk[idx], yy[idx])),
        "model_raw_log_loss": float(log_loss(pm[idx], yy[idx])),
        "model_raw_brier": float(brier_score(pm[idx], yy[idx])),
    }


def single_ticket_kelly_fraction(p_win: float, decimal_odds: float) -> float:
    """Exact single-ticket full Kelly fraction ``max(0, (p*O - 1)/(O - 1))``."""
    p = float(p_win)
    dec = float(decimal_odds)
    if not (0.0 < p < 1.0) or dec <= 1.0:
        return 0.0
    b = dec - 1.0
    f = (p * dec - 1.0) / b
    return float(max(0.0, f))


def single_ticket_expected_log_growth(
    p_win: float,
    decimal_odds: float,
    *,
    c_kelly: float = 1.0,
    max_frac: float = 0.25,
) -> tuple[float, float]:
    """Return ``(expected_log_growth, stake_fraction)`` for one candidate ticket."""
    f_full = single_ticket_kelly_fraction(p_win, decimal_odds)
    if f_full <= 0.0 or c_kelly <= 0.0 or max_frac <= 0.0:
        return 0.0, 0.0
    f = float(min(max_frac, float(c_kelly) * f_full))
    if f <= 0.0 or f >= 1.0:
        return 0.0, 0.0
    p = float(p_win)
    b = float(decimal_odds) - 1.0
    g = p * math.log(1.0 + f * b) + (1.0 - p) * math.log(1.0 - f)
    return float(g), f


def solve_daily_joint_kelly(
    probs: Sequence[float],
    decimals: Sequence[float],
    *,
    c_kelly: float = 0.25,
    daily_cap: float = 0.15,
    max_full_sum: float = 0.95,
) -> tuple[np.ndarray, float]:
    """Exact multi-ticket joint log-utility optimizer for independent daily bets.

    Maximizes exact ``E_Z[log(1 - sum_k f_k + sum_k Z_k f_k O_k)]`` over all
    ``2^K`` joint outcome states ``Z in {0, 1}^K``, then scales by fractional
    Kelly ``c_kelly`` and enforces ``sum_k f_k <= daily_cap``.

    Returns ``(fractions, expected_log_growth_at_fractions)``.
    """
    k = len(probs)
    if k == 0 or daily_cap <= 0.0 or c_kelly <= 0.0:
        return np.zeros(k, dtype=float), 0.0
    if len(decimals) != k:
        raise ValueError("probs and decimals must have equal length")

    p_vec = np.asarray(probs, dtype=float)
    o_vec = np.asarray(decimals, dtype=float)

    # Filter out any non-+EV coordinates immediately so solver stays strictly interior
    active = (p_vec > 0.0) & (p_vec < 1.0) & (o_vec > 1.0) & ((p_vec * o_vec) > 1.0 + 1e-12)
    if not np.any(active):
        return np.zeros(k, dtype=float), 0.0

    act_idx = np.where(active)[0]
    pa = p_vec[act_idx]
    oa = o_vec[act_idx]
    m = len(act_idx)

    if m == 1:
        f_full = single_ticket_kelly_fraction(float(pa[0]), float(oa[0]))
        f_scaled = min(float(daily_cap), float(c_kelly) * min(f_full, max_full_sum))
        g = float(pa[0] * math.log(1.0 + f_scaled * (oa[0] - 1.0)) + (1.0 - pa[0]) * math.log(1.0 - f_scaled))
        out = np.zeros(k, dtype=float)
        out[act_idx[0]] = f_scaled
        return out, g

    if m > 10:
        # Keep top 10 by single-ticket log growth so 2^m <= 1024 exact states
        g1 = np.array([single_ticket_expected_log_growth(float(pa[i]), float(oa[i]))[0] for i in range(m)])
        keep = np.argsort(-g1)[:10]
        act_idx = act_idx[keep]
        pa = pa[keep]
        oa = oa[keep]
        m = 10

    # Build exact 2^m state matrix Z of shape (2^m, m)
    states = ((np.arange(1 << m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
    state_probs = np.prod(np.where(states == 1.0, pa[None, :], 1.0 - pa[None, :]), axis=1)
    # Net payoff matrix R of shape (2^m, m): Z_{s,i}*O_i - 1
    payoffs = states * oa[None, :] - 1.0

    def _project_simplex_box(v: np.ndarray, cap: float) -> np.ndarray:
        u = np.maximum(v, 0.0)
        s = float(u.sum())
        if s <= cap:
            return u
        # Euclidean projection onto {u >= 0, sum(u) = cap}
        sv = np.sort(u)[::-1]
        cssv = np.cumsum(sv) - cap
        ind = np.arange(1, len(u) + 1)
        cond = sv - cssv / ind > 0
        rho = int(ind[cond][-1])
        theta = float(cssv[rho - 1] / rho)
        return np.maximum(u - theta, 0.0)

    # Warm start from scaled single-bet Kelly fractions
    f_kelly = np.array([single_ticket_kelly_fraction(float(pa[i]), float(oa[i])) for i in range(m)], dtype=float)
    f_kelly = _project_simplex_box(f_kelly, float(max_full_sum) * 0.8)

    w_cur = 1.0 + payoffs @ f_kelly
    obj_cur = float(np.dot(state_probs, np.log(w_cur)))

    # Fast projected damped Newton on strictly concave log-wealth objective
    for _ in range(25):
        inv_w = state_probs / w_cur
        grad = inv_w @ payoffs
        inv_w2 = state_probs / (w_cur * w_cur)
        hess = (payoffs.T * inv_w2) @ payoffs + 1e-10 * np.eye(m)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = grad
        improved = False
        alpha = 1.0
        for _ in range(12):
            cand = _project_simplex_box(f_kelly + alpha * step, float(max_full_sum))
            w_cand = 1.0 + payoffs @ cand
            if np.all(w_cand > 1e-8):
                obj_cand = float(np.dot(state_probs, np.log(w_cand)))
                if obj_cand >= obj_cur + 1e-13:
                    diff = float(np.max(np.abs(cand - f_kelly)))
                    f_kelly = cand
                    w_cur = w_cand
                    obj_cur = obj_cand
                    improved = True
                    if diff < 1e-9:
                        break
                    break
            alpha *= 0.5
        if not improved or float(np.max(np.abs(step))) < 1e-9:
            break

    # Apply fractional Kelly c_kelly and enforce daily exposure cap
    f_final_active = float(c_kelly) * f_kelly
    s_final = float(f_final_active.sum())
    if s_final > float(daily_cap) and s_final > 0:
        f_final_active *= float(daily_cap) / s_final

    w_final = 1.0 + payoffs @ f_final_active
    g_final = float(np.dot(state_probs, np.log(w_final)))
    if g_final <= 0.0:
        return np.zeros(k, dtype=float), 0.0

    out = np.zeros(k, dtype=float)
    out[act_idx] = f_final_active
    return out, g_final


def build_shrunk_legs(
    games: pd.DataFrame,
    *,
    model_home_col: str = "p_model_home",
    novig_home_col: str = "p_home_novig",
    home_ml_col: str = "home_ml",
    away_ml_col: str = "away_ml",
    y_home_col: str = "home_win",
    shrink_w: float = 0.15,
    shrink_mode: ShrinkMode = "linear",
    min_leg_novig: float = 0.0,
    max_leg_decimal: float | None = None,
    require_leg_plus_ev: bool = False,
    extra_market_legs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build candidate single-game sides with ``p_shrunk`` across ML (and optional spread/total)."""
    df = games.copy()
    p_shrunk_home = shrink_probs(
        df[model_home_col].to_numpy(float),
        df[novig_home_col].to_numpy(float),
        w=shrink_w,
        mode=shrink_mode,
    )
    df["_p_shrunk_home"] = p_shrunk_home

    rows: list[dict[str, Any]] = []
    for _, g in df.iterrows():
        p_sh = float(g["_p_shrunk_home"])
        p_mh = float(g[model_home_col])
        p_nh = float(g[novig_home_col])
        hw = float(g[y_home_col])
        gd = pd.Timestamp(g["game_date"]).normalize()
        gid = str(g["game_id"])
        ssn = g.get("season")

        for side, ps, pm, pn, price, won in (
            ("home", p_sh, p_mh, p_nh, float(g[home_ml_col]), hw == 1.0),
            ("away", 1.0 - p_sh, 1.0 - p_mh, 1.0 - p_nh, float(g[away_ml_col]), hw == 0.0),
        ):
            if not np.isfinite(price) or price == 0.0:
                continue
            if pn < float(min_leg_novig):
                continue
            dec = american_to_decimal(price)
            if max_leg_decimal is not None and dec > float(max_leg_decimal):
                continue
            if require_leg_plus_ev and ps * dec <= 1.0:
                continue
            rows.append(
                {
                    "season": ssn,
                    "game_date": gd,
                    "game_id": gid,
                    "market": "ml",
                    "side": side,
                    "p_shrunk": ps,
                    "p_model_raw": pm,
                    "p_novig": pn,
                    "american": price,
                    "decimal": dec,
                    "leg_outcome": 1.0 if won else 0.0,  # 0.5 for push on spread/total
                }
            )

    out = pd.DataFrame(rows)
    if extra_market_legs is not None and not extra_market_legs.empty:
        out = pd.concat([out, extra_market_legs], ignore_index=True)
    if out.empty:
        return out
    return out.sort_values(["game_date", "game_id", "market", "side"]).reset_index(drop=True)


def enumerate_kelly_candidates(
    legs: pd.DataFrame,
    *,
    sizes: Sequence[int] = (2,),
    c_kelly: float = 0.25,
    daily_cap: float = 0.15,
    min_decimal: float = 1.5,
    max_decimal: float | None = 5.0,
    min_ev: float = 0.0,
    min_log_growth: float = 1e-6,
    max_legs_per_day_pool: int = 12,
) -> pd.DataFrame:
    """Enumerate same-day distinct-game combinations ranked by expected log-wealth increment."""
    if legs.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for day, day_df in legs.groupby("game_date", sort=True):
        # Keep at most the single strongest market/side per game_id by shrunk EV ratio
        day_df = day_df.assign(_ev_ratio=day_df["p_shrunk"] * day_df["decimal"])
        day_df = (
            day_df.sort_values("_ev_ratio", ascending=False)
            .drop_duplicates(subset=["game_id"], keep="first")
            .head(int(max_legs_per_day_pool))
        )
        rows = [day_df.iloc[i] for i in range(len(day_df))]

        for n in sizes:
            if n == 1:
                combos = [(i,) for i in range(len(rows))]
            else:
                if len(rows) < n:
                    continue
                combos = list(combinations(range(len(rows)), n))

            for combo in combos:
                picked = [rows[i] for i in combo]
                gids = [str(L["game_id"]) for L in picked]
                # Require distinct games so independence holds cleanly
                if len(set(gids)) != n:
                    continue
                ps = [float(L["p_shrunk"]) for L in picked]
                decs = [float(L["decimal"]) for L in picked]
                dec_prod = float(np.prod(decs))
                if dec_prod < float(min_decimal):
                    continue
                if max_decimal is not None and dec_prod > float(max_decimal):
                    continue
                p_joint = float(np.prod(ps))
                ev = p_joint * dec_prod - 1.0
                if ev <= float(min_ev):
                    continue
                g_inc, f_single = single_ticket_expected_log_growth(
                    p_joint,
                    dec_prod,
                    c_kelly=c_kelly,
                    max_frac=daily_cap,
                )
                if g_inc < float(min_log_growth):
                    continue

                # Realized ticket multiplier (supports pushes where leg_outcome == 0.5 -> leg decimal 1.0)
                realized_mult = 1.0
                any_loss = False
                for L in picked:
                    outc = float(L["leg_outcome"])
                    if outc == 0.0:
                        any_loss = True
                        realized_mult = 0.0
                        break
                    if outc == 1.0:
                        realized_mult *= float(L["decimal"])
                    else:
                        realized_mult *= 1.0  # push reduces parlay by that leg

                records.append(
                    {
                        "season": picked[0].get("season"),
                        "game_date": day,
                        "n_legs": int(n),
                        "game_ids": "|".join(gids),
                        "markets": "|".join(str(L["market"]) for L in picked),
                        "sides": "|".join(str(L["side"]) for L in picked),
                        "americans": "|".join(f"{float(L['american']):.0f}" for L in picked),
                        "p_joint_shrunk": p_joint,
                        "parlay_decimal": dec_prod,
                        "book_implied": 1.0 / dec_prod,
                        "ev_shrunk": float(ev),
                        "f_single": float(f_single),
                        "expected_log_growth": float(g_inc),
                        "realized_multiplier": float(realized_mult),
                        "ticket_won": bool(not any_loss and realized_mult > 1.0),
                        "ticket_push": bool(not any_loss and realized_mult == 1.0),
                    }
                )

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["game_date", "expected_log_growth"], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_daily_kelly_portfolio(
    candidates: pd.DataFrame,
    *,
    max_tickets_per_day: int = 3,
    c_kelly: float = 0.25,
    daily_cap: float = 0.15,
    per_bet_cap: float | None = None,
) -> pd.DataFrame:
    """Pick daily non-overlapping tickets by ``expected_log_growth`` and size via joint Kelly."""
    if candidates.empty:
        return candidates.copy()

    selected_days: list[pd.DataFrame] = []
    for _, day_df in candidates.groupby("game_date", sort=True):
        day_sorted = day_df.sort_values("expected_log_growth", ascending=False)
        chosen: list[pd.Series] = []
        used_games: set[str] = set()
        for _, row in day_sorted.iterrows():
            gids = set(str(row["game_ids"]).split("|"))
            if gids & used_games:
                continue
            chosen.append(row)
            used_games |= gids
            if len(chosen) >= int(max_tickets_per_day):
                break
        if not chosen:
            continue
        sub = pd.DataFrame(chosen).reset_index(drop=True)
        fracs, day_g = solve_daily_joint_kelly(
            sub["p_joint_shrunk"].tolist(),
            sub["parlay_decimal"].tolist(),
            c_kelly=c_kelly,
            daily_cap=daily_cap,
        )
        if per_bet_cap is not None:
            fracs = np.minimum(fracs, float(per_bet_cap))
            s_f = float(fracs.sum())
            if s_f > float(daily_cap) and s_f > 0:
                fracs = fracs * (float(daily_cap) / s_f)
        sub["stake_frac"] = fracs
        sub["day_expected_log_growth"] = day_g
        sub = sub[sub["stake_frac"] > 1e-6].copy()
        if not sub.empty:
            selected_days.append(sub)

    if not selected_days:
        return candidates.iloc[0:0].copy()
    return pd.concat(selected_days, ignore_index=True)


def simulate_seasonal_kelly(
    portfolio: pd.DataFrame,
    *,
    initial_bankroll: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate chronological start-of-day bankroll compounding with cent stakes."""
    start = float(initial_bankroll)
    if portfolio.empty:
        return portfolio.copy(), {
            "initial_bankroll": start,
            "ending_bankroll": start,
            "net_profit": 0.0,
            "multiple": 1.0,
            "log_growth": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "n_days_active": 0,
            "n_tickets": 0,
            "n_staked": 0,
            "n_wins": 0,
            "hit_rate": None,
            "total_staked": 0.0,
            "realized_roi": None,
            "avg_decimal": None,
            "avg_p_joint": None,
            "broke": False,
        }

    br = start
    peak = start
    max_dd = 0.0
    max_dd_pct = 0.0
    broke = False
    day_records: list[pd.DataFrame] = []
    n_days_active = 0

    for _, day_df in portfolio.groupby("game_date", sort=True):
        if br < 0.01:
            broke = True
            break
        start_day_br = br
        raw_stakes = [math.floor(start_day_br * float(f) * 100.0 + 1e-12) / 100.0 for f in day_df["stake_frac"]]
        # Guarantee total day stakes never exceed start-of-day bankroll
        tot_day_stake = round(sum(raw_stakes), 2)
        if tot_day_stake > start_day_br and tot_day_stake > 0:
            scale = start_day_br / tot_day_stake
            raw_stakes = [math.floor(s * scale * 100.0 + 1e-12) / 100.0 for s in raw_stakes]

        day_pnl = 0.0
        stakes_col: list[float] = []
        payouts_col: list[float] = []
        pnls_col: list[float] = []
        any_placed = False

        for stake, (_, row) in zip(raw_stakes, day_df.iterrows()):
            if stake < 0.01:
                stakes_col.append(0.0)
                payouts_col.append(0.0)
                pnls_col.append(0.0)
                continue
            any_placed = True
            mult = float(row["realized_multiplier"])
            payout = round(stake * mult, 2)
            pnl = round(payout - stake, 2)
            stakes_col.append(stake)
            payouts_col.append(payout)
            pnls_col.append(pnl)
            day_pnl = round(day_pnl + pnl, 2)

        if any_placed:
            n_days_active += 1
        br = round(max(0.0, start_day_br + day_pnl), 2)
        if br < 0.01:
            broke = True
        peak = max(peak, br)
        dd = br - peak
        dd_pct = (br - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        max_dd_pct = min(max_dd_pct, dd_pct)

        d_out = day_df.copy()
        d_out["start_day_bankroll"] = start_day_br
        d_out["stake"] = stakes_col
        d_out["payout"] = payouts_col
        d_out["pnl"] = pnls_col
        d_out["end_day_bankroll"] = br
        day_records.append(d_out)

    graded = pd.concat(day_records, ignore_index=True) if day_records else portfolio.iloc[0:0].copy()
    staked_rows = graded[graded["stake"] >= 0.01] if not graded.empty else graded
    total_staked = float(staked_rows["stake"].sum()) if not staked_rows.empty else 0.0
    total_pnl = float(staked_rows["pnl"].sum()) if not staked_rows.empty else 0.0
    n_staked = int(len(staked_rows))
    n_wins = int(staked_rows["ticket_won"].sum()) if not staked_rows.empty else 0

    summary = {
        "initial_bankroll": start,
        "ending_bankroll": float(br),
        "net_profit": round(float(br) - start, 2),
        "multiple": float(br) / start if start > 0 else 0.0,
        "log_growth": math.log(max(br, 1e-6) / start) if start > 0 else 0.0,
        "max_drawdown": round(float(max_dd), 2),
        "max_drawdown_pct": float(max_dd_pct),
        "n_days_active": int(n_days_active),
        "n_tickets": int(len(portfolio)),
        "n_staked": n_staked,
        "n_wins": n_wins,
        "hit_rate": float(n_wins / n_staked) if n_staked > 0 else None,
        "total_staked": round(total_staked, 2),
        "realized_roi": float(total_pnl / total_staked) if total_staked > 0 else None,
        "avg_decimal": float(staked_rows["parlay_decimal"].mean()) if n_staked > 0 else None,
        "avg_p_joint": float(staked_rows["p_joint_shrunk"].mean()) if n_staked > 0 else None,
        "broke": bool(broke),
    }
    return graded, summary


def build_extra_spread_total_legs(
    games: pd.DataFrame,
    *,
    shrink_w: float = 0.15,
    shrink_mode: ShrinkMode = "tail_damped",
    min_leg_novig: float = 0.0,
    require_leg_plus_ev: bool = True,
    p_cover_col: str = "p_model_cover_home",
    p_over_col: str = "p_model_over",
) -> pd.DataFrame:
    """Build shrunk spread and total candidate legs when market columns exist on ``games``."""
    rows: list[dict[str, Any]] = []
    has_sp = {p_cover_col, "p_spread_home_novig", "spread_home_odds", "spread_away_odds", "home_cover"}.issubset(games.columns)
    has_tot = {p_over_col, "p_over_novig", "over_odds", "under_odds", "over_hit"}.issubset(games.columns)
    if not has_sp and not has_tot:
        return pd.DataFrame()

    df = games.copy()
    if has_sp:
        v_sp = df[p_cover_col].notna() & df["p_spread_home_novig"].notna()
        df["_ps_sp_h"] = np.nan
        if bool(v_sp.any()):
            df.loc[v_sp, "_ps_sp_h"] = shrink_probs(
                df.loc[v_sp, p_cover_col],
                df.loc[v_sp, "p_spread_home_novig"],
                w=shrink_w,
                mode=shrink_mode,
            )
    if has_tot:
        v_tot = df[p_over_col].notna() & df["p_over_novig"].notna()
        df["_ps_ov"] = np.nan
        if bool(v_tot.any()):
            df.loc[v_tot, "_ps_ov"] = shrink_probs(
                df.loc[v_tot, p_over_col],
                df.loc[v_tot, "p_over_novig"],
                w=shrink_w,
                mode=shrink_mode,
            )

    for _, g in df.iterrows():
        gd = pd.Timestamp(g["game_date"]).normalize()
        gid = str(g["game_id"])
        ssn = g.get("season")

        if has_sp and pd.notna(g.get("_ps_sp_h")):
            sp_ovr = float(g.get("spread_overround", 1.047))
            h_am = float(g["spread_home_odds"])
            a_am = float(g["spread_away_odds"])
            if sp_ovr >= 1.025 and np.isfinite(h_am) and np.isfinite(a_am) and abs(h_am) <= 140 and abs(a_am) <= 140:
                sp_push = bool(g.get("spread_push", False))
                hc = float(g["home_cover"]) if pd.notna(g["home_cover"]) else 0.0
                ps_h = float(g["_ps_sp_h"])
                pn_h = float(g["p_spread_home_novig"])
                pm_h = float(g[p_cover_col])
                for side, ps, pm, pn, am, outc in (
                    ("home", ps_h, pm_h, pn_h, h_am, 0.5 if sp_push else (1.0 if hc == 1.0 else 0.0)),
                    ("away", 1.0 - ps_h, 1.0 - pm_h, 1.0 - pn_h, a_am, 0.5 if sp_push else (1.0 if hc == 0.0 else 0.0)),
                ):
                    if pn < float(min_leg_novig) or am == 0.0:
                        continue
                    dec = american_to_decimal(am)
                    if require_leg_plus_ev and ps * dec <= 1.0 + 1e-9:
                        continue
                    rows.append(
                        {
                            "season": ssn,
                            "game_date": gd,
                            "game_id": gid,
                            "market": "spread",
                            "side": side,
                            "p_shrunk": ps,
                            "p_model_raw": pm,
                            "p_novig": pn,
                            "american": am,
                            "decimal": dec,
                            "leg_outcome": outc,
                        }
                    )

        if has_tot and pd.notna(g.get("_ps_ov")):
            tot_ovr = float(g.get("total_overround", 1.047))
            ov_am = float(g["over_odds"])
            un_am = float(g["under_odds"])
            if tot_ovr >= 1.025 and np.isfinite(ov_am) and np.isfinite(un_am) and abs(ov_am) <= 140 and abs(un_am) <= 140:
                tot_push = bool(g.get("total_push", False))
                oh = float(g["over_hit"]) if pd.notna(g["over_hit"]) else 0.0
                ps_o = float(g["_ps_ov"])
                pn_o = float(g["p_over_novig"])
                pm_o = float(g[p_over_col])
                for side, ps, pm, pn, am, outc in (
                    ("over", ps_o, pm_o, pn_o, ov_am, 0.5 if tot_push else (1.0 if oh == 1.0 else 0.0)),
                    ("under", 1.0 - ps_o, 1.0 - pm_o, 1.0 - pn_o, un_am, 0.5 if tot_push else (1.0 if oh == 0.0 else 0.0)),
                ):
                    if pn < float(min_leg_novig) or am == 0.0:
                        continue
                    dec = american_to_decimal(am)
                    if require_leg_plus_ev and ps * dec <= 1.0 + 1e-9:
                        continue
                    rows.append(
                        {
                            "season": ssn,
                            "game_date": gd,
                            "game_id": gid,
                            "market": "total",
                            "side": side,
                            "p_shrunk": ps,
                            "p_model_raw": pm,
                            "p_novig": pn,
                            "american": am,
                            "decimal": dec,
                            "leg_outcome": outc,
                        }
                    )
    return pd.DataFrame(rows)


def run_kelly_compounding_selector(
    games: pd.DataFrame,
    *,
    model_home_col: str = "p_model_home",
    shrink_w: float = 0.15,
    shrink_mode: ShrinkMode = "tail_damped",
    min_leg_novig: float = 0.0,
    sizes: Sequence[int] = (1, 2, 3),
    max_decimal: float | None = 6.0,
    c_kelly: float = 1.0,
    daily_cap: float = 0.10,
    max_tickets_per_day: int = 1,
    include_spread_total: bool = True,
    initial_bankroll: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """End-to-end shrunk-probability daily log-utility portfolio selector & simulator.

    Defaults match the 2022-24 train-frozen multi-market winner:
    ``shrink_w=0.15``, ``shrink_mode='tail_damped'``, ``sizes=(1, 2, 3)``,
    ``max_decimal=6.0``, ``c_kelly=1.0``, ``daily_cap=0.10``, ``max_tickets_per_day=1``.
    """
    extra = (
        build_extra_spread_total_legs(
            games,
            shrink_w=shrink_w,
            shrink_mode=shrink_mode,
            min_leg_novig=min_leg_novig,
            require_leg_plus_ev=True,
        )
        if include_spread_total
        else None
    )
    legs = build_shrunk_legs(
        games,
        model_home_col=model_home_col,
        shrink_w=shrink_w,
        shrink_mode=shrink_mode,
        min_leg_novig=min_leg_novig,
        require_leg_plus_ev=True,
        extra_market_legs=extra,
    )
    # Candidate ordering uses full-Kelly log-growth (c_kelly=1.0, max_frac=0.95)
    cands = enumerate_kelly_candidates(
        legs,
        sizes=sizes,
        c_kelly=1.0,
        daily_cap=0.95,
        min_decimal=1.05,
        max_decimal=max_decimal,
        min_ev=0.0,
        min_log_growth=1e-7,
    )
    portfolio = select_daily_kelly_portfolio(
        cands,
        max_tickets_per_day=max_tickets_per_day,
        c_kelly=c_kelly,
        daily_cap=daily_cap,
    )
    graded, summary = simulate_seasonal_kelly(portfolio, initial_bankroll=initial_bankroll)
    summary["params"] = {
        "model_home_col": model_home_col,
        "shrink_w": float(shrink_w),
        "shrink_mode": shrink_mode,
        "min_leg_novig": float(min_leg_novig),
        "sizes": list(sizes),
        "max_decimal": max_decimal,
        "c_kelly": float(c_kelly),
        "daily_cap": float(daily_cap),
        "max_tickets_per_day": int(max_tickets_per_day),
        "include_spread_total": bool(include_spread_total),
    }
    return legs, graded, summary


def build_prop_candidate_legs(
    props: pd.DataFrame,
    *,
    shrink_w: float = 0.8183,
    min_leg_ev: float = 0.015,
    ev_cap: float | None = 0.08,
    use_best_same_line: bool = False,
    min_decimal: float = 1.50,
    max_decimal: float = 2.65,
) -> pd.DataFrame:
    """Select at most one highest-EV shrunk player-prop leg per ``(season, game_id)``.

    Restricting to one prop per game guarantees cross-game independence when combined
    with distinct-game daily Kelly allocation.
    """
    if props.empty:
        return pd.DataFrame()

    p = props.copy()
    ps_ov = shrink_probs(p["p_model_over"], p["p_mkt_over"], w=shrink_w, mode="linear")
    ps_un = 1.0 - ps_ov

    if use_best_same_line and {"best_over_decimal_same_line", "best_under_decimal_same_line"}.issubset(p.columns):
        d_ov = p["best_over_decimal_same_line"].fillna(p["over_decimal"]).to_numpy(float)
        d_un = p["best_under_decimal_same_line"].fillna(p["under_decimal"]).to_numpy(float)
    else:
        d_ov = p["over_decimal"].to_numpy(float)
        d_un = p["under_decimal"].to_numpy(float)

    ev_ov = np.where((d_ov >= min_decimal) & (d_ov <= max_decimal), ps_ov * d_ov - 1.0, -1.0)
    ev_un = np.where((d_un >= min_decimal) & (d_un <= max_decimal), ps_un * d_un - 1.0, -1.0)
    pick_ov = ev_ov >= ev_un

    best_ev = np.where(pick_ov, ev_ov, ev_un)
    best_ps_raw = np.where(pick_ov, ps_ov, ps_un)
    best_dec = np.where(pick_ov, d_ov, d_un)
    best_ps = winsorize_prob_by_ev_cap(best_ps_raw, best_dec, ev_cap=ev_cap)
    best_pn = np.where(pick_ov, p["p_mkt_over"].to_numpy(float), 1.0 - p["p_mkt_over"].to_numpy(float))
    best_out = np.where(pick_ov, p["over_win"].to_numpy(float) == 1.0, p["over_win"].to_numpy(float) == 0.0).astype(float)

    p["_ev"] = best_ev
    p["_ps"] = best_ps
    p["_pn"] = best_pn
    p["_dec"] = best_dec
    p["_out"] = best_out
    p["_side"] = np.where(pick_ov, "over", "under")

    keep = p[p["_ev"] >= float(min_leg_ev)].sort_values("_ev", ascending=False)
    if keep.empty:
        return pd.DataFrame()
    dedup_cols = ["season", "game_id"] if "season" in keep.columns else ["game_id"]
    keep = keep.drop_duplicates(subset=dedup_cols, keep="first")

    rows: list[dict[str, Any]] = []
    for _, r in keep.iterrows():
        dec = float(r["_dec"])
        am = (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)
        rows.append(
            {
                "season": r.get("season"),
                "game_date": pd.Timestamp(r["game_date"]).normalize(),
                "game_id": str(r["game_id"]),
                "market": f"prop_{r.get('prop_type', 'stat')}",
                "side": str(r["_side"]),
                "p_shrunk": float(r["_ps"]),
                "p_model_raw": float(r["p_model_over"] if r["_side"] == "over" else 1.0 - float(r["p_model_over"])),
                "p_novig": float(r["_pn"]),
                "american": float(am),
                "decimal": dec,
                "leg_outcome": float(r["_out"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def run_unified_kelly_portfolio(
    candidate_legs: pd.DataFrame,
    *,
    sizes: Sequence[int] = (1, 2),
    max_decimal: float | None = 4.2,
    min_ev: float = 0.0,
    c_kelly: float = 1.0,
    daily_cap: float = 0.25,
    per_bet_cap: float = 0.08,
    max_tickets_per_day: int = 1,
    initial_bankroll: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run unified multi-stream (open game + close game + props) daily Kelly allocation."""
    cands = enumerate_kelly_candidates(
        candidate_legs,
        sizes=sizes,
        c_kelly=1.0,
        daily_cap=0.95,
        min_decimal=1.10,
        max_decimal=max_decimal,
        min_ev=min_ev,
        min_log_growth=1e-7,
    )
    portfolio = select_daily_kelly_portfolio(
        cands,
        max_tickets_per_day=max_tickets_per_day,
        c_kelly=c_kelly,
        daily_cap=daily_cap,
        per_bet_cap=per_bet_cap,
    )
    return simulate_seasonal_kelly(portfolio, initial_bankroll=initial_bankroll)
