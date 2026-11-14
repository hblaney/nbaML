"""Market baseline and model↔market probability blends (no future leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nba_betting.backtest.metrics import brier_score, hit_rate, log_loss


def ml_prob_metrics(probs: np.ndarray | pd.Series, outcomes: np.ndarray | pd.Series) -> dict[str, Any]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return {"n": 0, "ml_hit_rate": None, "ml_brier": None, "ml_log_loss": None}
    pick = (p >= 0.5) == (y == 1.0)
    return {
        "n": int(p.size),
        "ml_hit_rate": float(hit_rate(pick)),
        "ml_brier": float(brier_score(p, y)),
        "ml_log_loss": float(log_loss(p, y)),
    }


def blend_probs(p_model: np.ndarray | pd.Series, p_market: np.ndarray | pd.Series, *, w_model: float) -> np.ndarray:
    """``w_model * p_model + (1 - w_model) * p_market``, clipped to (0,1)."""
    w = float(w_model)
    out = w * np.asarray(p_model, dtype=float) + (1.0 - w) * np.asarray(p_market, dtype=float)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def choose_blend_weight(
    df: pd.DataFrame,
    *,
    model_col: str = "p_model",
    market_col: str = "p_home_novig",
    y_col: str = "home_win",
    season_col: str = "season",
    train_seasons: list[str],
    confirm_season: str,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pick ``w_model`` by confirm-season Brier (train seasons used only for reporting).

    Returns the weight that minimizes confirm Brier on the provided grid.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    train = df[df[season_col].isin(train_seasons)].copy()
    confirm = df[df[season_col] == confirm_season].copy()
    if confirm.empty:
        raise ValueError(f"no rows for confirm season {confirm_season}")

    market_confirm = ml_prob_metrics(confirm[market_col], confirm[y_col])
    best: dict[str, Any] | None = None
    for w in grid:
        p_tr = blend_probs(train[model_col], train[market_col], w_model=float(w))
        p_cf = blend_probs(confirm[model_col], confirm[market_col], w_model=float(w))
        m_tr = ml_prob_metrics(p_tr, train[y_col])
        m_cf = ml_prob_metrics(p_cf, confirm[y_col])
        row = {"w_model": float(w), "train": m_tr, "confirm": m_cf}
        if best is None or m_cf["ml_brier"] < best["confirm"]["ml_brier"]:
            best = row
    assert best is not None
    best["market_confirm"] = market_confirm
    best["beats_market_on_confirm"] = bool(
        best["confirm"]["ml_brier"] < float(market_confirm["ml_brier"]) - 1e-4
    )
    return best


def expanding_odds_ratings_blend(
    p_market: np.ndarray | pd.Series,
    p_model: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    min_train: int = 200,
    refit_every: int = 25,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Walk-forward blend of closing ML (market) + ratings model.

    For each game ``i``, choose ``w_model`` that minimizes Brier on games
    ``0..i-1`` only (refit every ``refit_every`` steps after ``min_train``).
    Until ``min_train``, emit the market probability. No game-outcome leakage.
    """
    if grid is None:
        grid = np.linspace(0.0, 0.4, 17)
    mkt = np.asarray(p_market, dtype=float)
    mod = np.asarray(p_model, dtype=float)
    yy = np.asarray(y, dtype=float)
    if not (len(mkt) == len(mod) == len(yy)):
        raise ValueError("p_market, p_model, and y must have equal length")
    out = np.empty(len(yy), dtype=float)
    w_cur = 0.0
    last = -10**9
    for i in range(len(yy)):
        if i < int(min_train):
            out[i] = float(mkt[i])
            continue
        if i - last >= int(refit_every):
            best_w, best_b = 0.0, 1e9
            for w in grid:
                p = float(w) * mod[:i] + (1.0 - float(w)) * mkt[:i]
                b = float(np.mean((p - yy[:i]) ** 2))
                if b < best_b:
                    best_b = b
                    best_w = float(w)
            w_cur = best_w
            last = i
        out[i] = w_cur * mod[i] + (1.0 - w_cur) * mkt[i]
    return np.clip(out, 1e-6, 1.0 - 1e-6)
