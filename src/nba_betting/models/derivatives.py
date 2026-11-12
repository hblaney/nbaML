"""1st Quarter (1Q) and 1st Half (1H) residual models and probability calibration.

Predicts deviations from mechanical full-game closing line proportions:
- ``resid_q1_margin_mech = actual_q1_margin - 0.25 * (-fg_spread_home)``
- ``resid_h1_margin_mech = actual_h1_margin - 0.505 * (-fg_spread_home)``
- ``resid_q1_total_mech  = actual_q1_total  - 0.25 * fg_total_line``
- ``resid_h1_total_mech  = actual_h1_total  - 0.505 * fg_total_line``

Converts expected period margin & total distributions ``N(mu, sigma^2)`` into
cover, total, and 2-way moneyline probabilities against any offered book line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_betting.features.derivatives import MECH_H1_PROP, MECH_Q1_PROP, snap_to_half_point

DEFAULT_RIDGE_ALPHA = 1500.0

Q1_MARGIN_FEATURES: tuple[str, ...] = (
    "fg_exp_margin",
    "abs_fg_spread",
    "fg_spread_move",
    "diff_delta_net_q1",
    "diff_fast_delta_q1",
    "diff_q1_margin_resid",
    "diff_starter_bench_div",
    "diff_bench_net48",
    "diff_rot_top5_minus_bench",
    "diff_q1_starter_min_share",
    "diff_q1_first6_margin",
    "diff_missing_starter_pm",
    "three_in_four_diff",
    "home_b2b",
    "away_road_b2b",
    "rest_diff",
    "altitude_adv_kft",
    "is_early_matinee",
    "away_morning_body",
)

H1_MARGIN_FEATURES: tuple[str, ...] = (
    "fg_exp_margin",
    "abs_fg_spread",
    "fg_spread_move",
    "diff_delta_net_h1",
    "diff_fast_delta_h1",
    "diff_h1_margin_resid",
    "diff_starter_bench_div",
    "diff_bench_net48",
    "diff_rot_top5_minus_bench",
    "diff_h1_starter_min_share",
    "diff_starter_min_share",
    "diff_missing_starter_pm",
    "three_in_four_diff",
    "home_b2b",
    "away_road_b2b",
    "rest_diff",
    "altitude_adv_kft",
    "is_early_matinee",
    "away_morning_body",
)

Q1_TOTAL_FEATURES: tuple[str, ...] = (
    "fg_total_line",
    "abs_fg_spread",
    "fg_total_move",
    "sum_q1_pace_ratio",
    "matchup_pace_q1_minus_fg",
    "sum_q1_total_share_excess",
    "sum_delta_ortg_q1",
    "sum_delta_drtg_q1",
    "sum_q1_starter_min_share",
    "sum_starter_min_share",
    "three_in_four_sum",
    "venue_high_alt",
    "altitude_x_away_b2b",
    "is_early_matinee",
    "away_morning_body",
    "is_playoffs",
)

H1_TOTAL_FEATURES: tuple[str, ...] = (
    "fg_total_line",
    "abs_fg_spread",
    "fg_total_move",
    "sum_h1_pace_ratio",
    "sum_q1_pace_ratio",
    "matchup_pace_h1_minus_fg",
    "sum_h1_total_share_excess",
    "sum_delta_ortg_h1",
    "sum_delta_drtg_h1",
    "sum_starter_min_share",
    "sum_q1_starter_min_share",
    "three_in_four_sum",
    "venue_high_alt",
    "altitude_x_away_b2b",
    "is_early_matinee",
    "away_morning_body",
    "is_playoffs",
)


@dataclass(frozen=True)
class PeriodPrediction:
    game_id: str
    game_date: str
    season: str
    home_tricode: str
    away_tricode: str
    exp_q1_margin: float
    sigma_q1_margin: float
    exp_h1_margin: float
    sigma_h1_margin: float
    exp_q1_total: float
    sigma_q1_total: float
    exp_h1_total: float
    sigma_h1_total: float
    pred_resid_q1_margin: float
    pred_resid_h1_margin: float
    pred_resid_q1_total: float
    pred_resid_h1_total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def standard_bookmaker_derivative_lines(
    fg_spread_home: float,
    fg_total_line: float,
    *,
    mode: str = "sharp",
) -> dict[str, float]:
    """Derive half-point sportsbook 1Q/1H spreads and totals from closing full-game lines.

    - ``mode="proportional"``: literal mechanical proportions (25% for 1Q, 50.5% for 1H)
      snapped to standard `.5` half-points.
    - ``mode="sharp"``: garbage-time-adjusted sportsbook derivative chart calibrated strictly
      on 2022-24 train (0.315x/0.570x spread, 0.252x/0.503x total) snapped to `.5` half-points.
    """
    zero_side = -0.5 if fg_spread_home <= 0 else 0.5
    if mode == "proportional":
        return {
            "q1_spread_home": snap_to_half_point(MECH_Q1_PROP * fg_spread_home, zero_to=zero_side),
            "h1_spread_home": snap_to_half_point(MECH_H1_PROP * fg_spread_home, zero_to=zero_side),
            "q1_total_line": snap_to_half_point(MECH_Q1_PROP * fg_total_line, zero_to=55.5),
            "h1_total_line": snap_to_half_point(MECH_H1_PROP * fg_total_line, zero_to=110.5),
        }
    if mode == "sharp":
        return {
            "q1_spread_home": snap_to_half_point(0.315 * fg_spread_home, zero_to=zero_side),
            "h1_spread_home": snap_to_half_point(0.570 * fg_spread_home, zero_to=zero_side),
            "q1_total_line": snap_to_half_point(0.252 * fg_total_line, zero_to=55.5),
            "h1_total_line": snap_to_half_point(0.503 * fg_total_line, zero_to=110.5),
        }
    raise ValueError(f"unknown derivative line mode: {mode!r}")


def spread_cover_probs(
    exp_margin: float,
    sigma_margin: float,
    home_spread_line: float,
) -> tuple[float, float, float]:
    """Return ``(p_home_cover_cond, p_away_cover_cond, p_push)`` for integer-valued margins."""
    sig = max(float(sigma_margin), 1.0)
    target = -float(home_spread_line)
    # Check if half-point line vs integer line
    if abs((target % 1.0) - 0.5) < 1e-6:
        p_home = float(1.0 - norm.cdf(target, loc=exp_margin, scale=sig))
        p_home = float(np.clip(p_home, 1e-4, 1.0 - 1e-4))
        return p_home, 1.0 - p_home, 0.0
    # Integer spread: push when margin == target -> continuity interval [target-0.5, target+0.5]
    p_win = float(1.0 - norm.cdf(target + 0.5, loc=exp_margin, scale=sig))
    p_loss = float(norm.cdf(target - 0.5, loc=exp_margin, scale=sig))
    p_push = float(max(0.0, 1.0 - p_win - p_loss))
    denom = max(p_win + p_loss, 1e-8)
    p_home_cond = float(np.clip(p_win / denom, 1e-4, 1.0 - 1e-4))
    return p_home_cond, 1.0 - p_home_cond, p_push


def total_over_under_probs(
    exp_total: float,
    sigma_total: float,
    total_line: float,
) -> tuple[float, float, float]:
    """Return ``(p_over_cond, p_under_cond, p_push)`` for integer-valued period totals."""
    sig = max(float(sigma_total), 1.0)
    line = float(total_line)
    if abs((line % 1.0) - 0.5) < 1e-6:
        p_over = float(1.0 - norm.cdf(line, loc=exp_total, scale=sig))
        p_over = float(np.clip(p_over, 1e-4, 1.0 - 1e-4))
        return p_over, 1.0 - p_over, 0.0
    p_over_raw = float(1.0 - norm.cdf(line + 0.5, loc=exp_total, scale=sig))
    p_under_raw = float(norm.cdf(line - 0.5, loc=exp_total, scale=sig))
    p_push = float(max(0.0, 1.0 - p_over_raw - p_under_raw))
    denom = max(p_over_raw + p_under_raw, 1e-8)
    p_over_cond = float(np.clip(p_over_raw / denom, 1e-4, 1.0 - 1e-4))
    return p_over_cond, 1.0 - p_over_cond, p_push


def moneyline_two_way_probs(
    exp_margin: float,
    sigma_margin: float,
) -> tuple[float, float, float]:
    """Return ``(p_home_cond, p_away_cond, p_tie_push)`` for 2-way period ML (tie pushes)."""
    return spread_cover_probs(exp_margin, sigma_margin, 0.0)


class DerivativeResidualModelSuite:
    """Four-head regularized residual Ridge suite for 1Q/1H margins and totals."""

    def __init__(self, *, alpha: float = DEFAULT_RIDGE_ALPHA) -> None:
        self.alpha = float(alpha)
        self.models: dict[str, Pipeline] = {}
        self.sigmas: dict[str, float] = {
            "q1_margin": 8.33,
            "h1_margin": 11.09,
            "q1_total": 8.13,
            "h1_total": 12.01,
        }
        self._specs: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
            ("q1_margin", Q1_MARGIN_FEATURES, "resid_q1_margin_mech", "mech_q1_exp_margin"),
            ("h1_margin", H1_MARGIN_FEATURES, "resid_h1_margin_mech", "mech_h1_exp_margin"),
            ("q1_total", Q1_TOTAL_FEATURES, "resid_q1_total_mech", "mech_q1_exp_total"),
            ("h1_total", H1_TOTAL_FEATURES, "resid_h1_total_mech", "mech_h1_exp_total"),
        )

    def fit(self, train_df: pd.DataFrame) -> "DerivativeResidualModelSuite":
        valid = train_df.dropna(
            subset=["fg_spread_home", "fg_total_line", "q1_margin", "h1_margin", "q1_total", "h1_total"]
        )
        if len(valid) < 50:
            raise ValueError(f"need at least 50 completed games to fit derivative suite, got {len(valid)}")

        for key, feat_cols, res_col, base_col in self._specs:
            X = valid[list(feat_cols)].fillna(0.0).to_numpy(dtype=float)
            y_res = valid[res_col].to_numpy(dtype=float)
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])
            pipe.fit(X, y_res)
            self.models[key] = pipe
            in_err = y_res - pipe.predict(X)
            dof = max(len(y_res) - 4, 1)
            self.sigmas[key] = float(np.sqrt(np.sum(in_err**2) / dof))
        return self

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for key, feat_cols, _, base_col in self._specs:
            X = out[list(feat_cols)].fillna(0.0).to_numpy(dtype=float)
            pred_res = self.models[key].predict(X)
            out[f"pred_resid_{key}"] = pred_res
            out[f"exp_{key}"] = out[base_col].to_numpy(dtype=float) + pred_res
            out[f"sigma_{key}"] = float(self.sigmas[key])
        return out

    def predict_rows(self, df: pd.DataFrame) -> list[PeriodPrediction]:
        scored = self.predict_frame(df)
        preds: list[PeriodPrediction] = []
        for _, r in scored.iterrows():
            preds.append(
                PeriodPrediction(
                    game_id=str(r["game_id"]),
                    game_date=str(r["game_date"]),
                    season=str(r["season"]),
                    home_tricode=str(r["home_tricode"]),
                    away_tricode=str(r["away_tricode"]),
                    exp_q1_margin=float(r["exp_q1_margin"]),
                    sigma_q1_margin=float(r["sigma_q1_margin"]),
                    exp_h1_margin=float(r["exp_h1_margin"]),
                    sigma_h1_margin=float(r["sigma_h1_margin"]),
                    exp_q1_total=float(r["exp_q1_total"]),
                    sigma_q1_total=float(r["sigma_q1_total"]),
                    exp_h1_total=float(r["exp_h1_total"]),
                    sigma_h1_total=float(r["sigma_h1_total"]),
                    pred_resid_q1_margin=float(r["pred_resid_q1_margin"]),
                    pred_resid_h1_margin=float(r["pred_resid_h1_margin"]),
                    pred_resid_q1_total=float(r["pred_resid_q1_total"]),
                    pred_resid_h1_total=float(r["pred_resid_h1_total"]),
                )
            )
        return preds

    def feature_coefficients(self) -> dict[str, dict[str, float]]:
        """Return standardized Ridge coefficients per target head for diagnostics."""
        coefs: dict[str, dict[str, float]] = {}
        for key, feat_cols, _, _ in self._specs:
            pipe = self.models.get(key)
            if pipe is None:
                continue
            rg: Ridge = pipe.named_steps["ridge"]
            coefs[key] = {col: float(w) for col, w in zip(feat_cols, rg.coef_)}
        return coefs


def run_strict_walkforward_predictions(
    features_df: pd.DataFrame,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    train_seasons: tuple[str, ...] = ("2022-23", "2023-24"),
    min_warmup_games: int = 350,
    refit_every_days: int = 14,
    freeze_after_train: bool = True,
) -> pd.DataFrame:
    """Produce strictly out-of-sample walk-forward predictions across all seasons.

    - Within ``train_seasons`` (after ``min_warmup_games``), models are refit on an
      expanding window of strictly prior dates ``game_date < d`` every ``refit_every_days``.
    - After ``train_seasons``:
      - If ``freeze_after_train=True``, the suite fit on all ``train_seasons`` is frozen
        and applied untouched to confirm (2024-25) and holdout (2025-26).
      - If ``freeze_after_train=False``, expanding past-only refits continue across
        confirm and holdout using ``game_date < d``.
    """
    df = features_df.dropna(subset=["fg_spread_home", "fg_total_line"]).copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df.sort_values(["game_date_dt", "tipoff_utc", "game_id"]).reset_index(drop=True)

    unique_dates = sorted(df["game_date_dt"].unique())
    scored_chunks: list[pd.DataFrame] = []

    active_suite: DerivativeResidualModelSuite | None = None
    frozen_train_suite: DerivativeResidualModelSuite | None = None
    days_since_refit = refit_every_days

    for d_val in unique_dates:
        day_rows = df.loc[df["game_date_dt"] == d_val].copy()
        day_season = str(day_rows.iloc[0]["season"])
        prior_rows = df.loc[df["game_date_dt"] < d_val]

        if day_season in train_seasons:
            if len(prior_rows) < min_warmup_games:
                continue
            if active_suite is None or days_since_refit >= refit_every_days:
                active_suite = DerivativeResidualModelSuite(alpha=alpha).fit(prior_rows)
                days_since_refit = 0
            days_since_refit += 1
            scored_chunks.append(active_suite.predict_frame(day_rows))
        else:
            if freeze_after_train:
                if frozen_train_suite is None:
                    tr_only = prior_rows.loc[prior_rows["season"].isin(train_seasons)]
                    frozen_train_suite = DerivativeResidualModelSuite(alpha=alpha).fit(tr_only)
                scored_chunks.append(frozen_train_suite.predict_frame(day_rows))
            else:
                if active_suite is None or days_since_refit >= refit_every_days:
                    active_suite = DerivativeResidualModelSuite(alpha=alpha).fit(prior_rows)
                    days_since_refit = 0
                days_since_refit += 1
                scored_chunks.append(active_suite.predict_frame(day_rows))

    if not scored_chunks:
        return pd.DataFrame()
    return pd.concat(scored_chunks, ignore_index=True)
