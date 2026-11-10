"""Expanding-window probability calibration (no future leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover - sklearn is a project dependency
    LogisticRegression = None  # type: ignore[misc, assignment]


DEFAULT_PLATT_C = 10.0
DEFAULT_MIN_TRAIN = 100
DEFAULT_SHRINK = 1.05
DEFAULT_REFIT_EVERY = 50


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pc = np.clip(p.astype(float), eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def apply_expanding_platt(
    df: pd.DataFrame,
    *,
    p_col: str = "p_home_win",
    y_col: str = "home_win",
    date_col: str = "game_date",
    id_col: str = "game_id",
    min_train: int = DEFAULT_MIN_TRAIN,
    shrink: float = DEFAULT_SHRINK,
    C: float = DEFAULT_PLATT_C,
    refit_every: int = DEFAULT_REFIT_EVERY,
    out_col: str = "p_home_win_cal",
) -> pd.DataFrame:
    """Calibrate ``p_col`` with Platt scaling fit only on earlier rows.

    Rows are sorted by ``date_col`` then ``id_col``. Until ``min_train`` prior
    games exist, the raw probability is kept (optionally shrunk). After that,
    a logistic regression on logit(p) is fit on all earlier games and applied
    to the current row. ``shrink`` pulls the calibrated prob toward 0.5
    (``p' = 0.5 + shrink * (p - 0.5)``); values > 1 sharpen slightly.
    """
    if LogisticRegression is None:
        raise ImportError("scikit-learn is required for Platt calibration")
    if df.empty:
        out = df.copy()
        out[out_col] = pd.Series(dtype=float)
        return out

    ordered = df.sort_values([date_col, id_col]).reset_index(drop=True)
    p_raw = ordered[p_col].to_numpy(dtype=float)
    y = ordered[y_col].to_numpy(dtype=float)
    p_cal = np.empty(len(ordered), dtype=float)

    model: Any = None
    last_fit_at = -1
    for i in range(len(ordered)):
        if i < min_train:
            p = float(p_raw[i])
        else:
            if model is None or i == min_train or (i - last_fit_at) >= refit_every:
                logit = _logit(p_raw[:i]).reshape(-1, 1)
                lr = LogisticRegression(C=float(C), solver="lbfgs", max_iter=300)
                lr.fit(logit, y[:i])
                model = lr
                last_fit_at = i
            cur = _logit(np.asarray([p_raw[i]])).reshape(1, 1)
            p = float(model.predict_proba(cur)[0, 1])
        p = 0.5 + float(shrink) * (p - 0.5)
        p_cal[i] = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    out = ordered.copy()
    out[out_col] = p_cal
    return out


def swap_calibrated_ml_probs(
    df: pd.DataFrame,
    *,
    cal_col: str = "p_home_win_cal",
    raw_col: str = "p_home_win",
) -> pd.DataFrame:
    """Replace ML probs with calibrated values and refresh pick / correctness."""
    out = df.copy()
    if cal_col not in out.columns:
        raise KeyError(cal_col)
    if "p_home_win_raw" not in out.columns:
        out["p_home_win_raw"] = out[raw_col]
    out[raw_col] = out[cal_col].astype(float)
    out["p_away_win"] = 1.0 - out[raw_col]
    if "home_win" in out.columns:
        pick_home = out[raw_col] >= 0.5
        out["ml_pick_home"] = pick_home
        out["p_pick"] = np.where(pick_home, out[raw_col], out["p_away_win"])
        out["ml_correct"] = pick_home == (out["home_win"] == 1.0)
    return out
