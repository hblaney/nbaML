"""Backtest metrics: hit rates, Brier, log loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clip_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p.astype(float), eps, 1.0 - eps)


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = _clip_prob(np.asarray(probs))
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = _clip_prob(np.asarray(probs))
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1.0 - outcomes) * np.log(1.0 - probs)))


def hit_rate(correct: np.ndarray) -> float:
    arr = np.asarray(correct, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def summarize_graded(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a graded prediction frame with standard columns."""
    out: dict[str, Any] = {"n_games": int(len(df))}
    if df.empty:
        return out

    if "ml_correct" in df.columns:
        mask = df["ml_correct"].notna()
        out["ml_hit_rate"] = hit_rate(df.loc[mask, "ml_correct"].to_numpy())
        out["ml_n"] = int(mask.sum())
        if "p_home_win" in df.columns and "home_win" in df.columns:
            out["ml_brier"] = brier_score(
                df.loc[mask, "p_home_win"].to_numpy(),
                df.loc[mask, "home_win"].to_numpy(),
            )
            out["ml_log_loss"] = log_loss(
                df.loc[mask, "p_home_win"].to_numpy(),
                df.loc[mask, "home_win"].to_numpy(),
            )

    if "spread_correct" in df.columns:
        mask = df["spread_correct"].notna()
        out["spread_ats_hit_rate"] = hit_rate(df.loc[mask, "spread_correct"].to_numpy())
        out["spread_n"] = int(mask.sum())

    if "total_correct" in df.columns:
        mask = df["total_correct"].notna()
        out["total_hit_rate"] = hit_rate(df.loc[mask, "total_correct"].to_numpy())
        out["total_n"] = int(mask.sum())

    if "bankroll_end" in df.columns and len(df):
        out["bankroll_end"] = float(df["bankroll_end"].iloc[-1])
    if "bankroll_start" in df.attrs:
        out["bankroll_start"] = float(df.attrs["bankroll_start"])
    return out


def format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics or metrics.get("n_games", 0) == 0:
        return "(no graded games)"
    lines = [f"n_games={metrics.get('n_games')}"]
    for key in (
        "ml_hit_rate",
        "ml_brier",
        "ml_log_loss",
        "spread_ats_hit_rate",
        "total_hit_rate",
        "bankroll_start",
        "bankroll_end",
        "flat_roi",
        "kelly_roi",
    ):
        if key in metrics and metrics[key] == metrics[key]:  # not NaN
            val = metrics[key]
            if isinstance(val, float):
                lines.append(f"{key}={val:.4f}")
            else:
                lines.append(f"{key}={val}")
    return "\n".join(lines)
