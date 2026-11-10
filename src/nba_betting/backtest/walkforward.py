"""Walk-forward backtest: as-of features only, then grade vs finals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nba_betting.backtest.bankroll import simulate_bankroll
from nba_betting.backtest.calibration import (
    apply_expanding_platt,
    swap_calibrated_ml_probs,
)
from nba_betting.backtest.metrics import format_metrics, summarize_graded
from nba_betting.data.boxscores import load_boxscores
from nba_betting.data.schedule import load_schedule
from nba_betting.features.game_features import build_game_features_for_date
from nba_betting.models.game import predict_game
from nba_betting.pricing.kelly import STARTING_BANKROLL


def _schedule_dates(schedule: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(schedule["game_date_est"]).dt.normalize().drop_duplicates()
    return [str(d.date()) for d in sorted(dates)]


def _actuals_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    df = schedule.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_pts"] = df["home_score"] + df["away_score"]
    return df[
        [
            "game_id",
            "home_score",
            "away_score",
            "home_win",
            "margin",
            "total_pts",
            "home_tricode",
            "away_tricode",
        ]
    ]


def grade_prediction_row(row: pd.Series) -> dict[str, Any]:
    """Grade ML / model-fair spread / model-fair total vs actual scores."""
    out: dict[str, Any] = {
        "ml_pick_home": None,
        "p_pick": None,
        "ml_correct": None,
        "spread_correct": None,
        "total_correct": None,
        "spread_line_used": None,
        "total_line_used": None,
    }
    if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return out

    p_home = float(row["p_home_win"])
    pick_home = p_home >= 0.5
    out["ml_pick_home"] = pick_home
    out["p_pick"] = p_home if pick_home else float(row["p_away_win"])
    home_win = bool(row["home_win"])
    out["ml_correct"] = pick_home == home_win

    # Phase A: use model fair spread/total as the lines to grade against.
    spread_line = float(row["fair_spread"])
    total_line = float(row["fair_total"])
    out["spread_line_used"] = spread_line
    out["total_line_used"] = total_line

    # Home covers when margin + home_spread > 0 (home_spread negative if favored).
    cover_margin = float(row["margin"]) + spread_line
    if cover_margin == 0:
        out["spread_correct"] = None  # push
    else:
        out["spread_correct"] = cover_margin > 0

    diff = float(row["total_pts"]) - total_line
    if diff == 0:
        out["total_correct"] = None
    else:
        # Model "over" pick when p_over >= 0.5 else under.
        pick_over = float(row["p_over"]) >= 0.5
        went_over = diff > 0
        out["total_correct"] = pick_over == went_over
    return out


def run_walkforward(
    *,
    season: str,
    root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    team_logs: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_games: int = 1,
    n_sims: int = 5_000,
    starting_bankroll: float = STARTING_BANKROLL,
    paper_american: float = -110.0,
    calibrate: bool = True,
    rating_mode: str = "ema",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Walk forward over schedule dates; features use only prior games.

    When ``calibrate`` is True, moneyline probabilities are Platt-calibrated with
    an expanding window (fit only on earlier games in this run).
    """
    sched = schedule if schedule is not None else load_schedule(season, root=root)
    logs = team_logs if team_logs is not None else load_boxscores(season, level="team", root=root)
    actuals = _actuals_from_schedule(sched)

    dates = _schedule_dates(sched)
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    rows: list[dict[str, Any]] = []
    for i, game_date in enumerate(dates):
        feats = build_game_features_for_date(
            schedule=sched,
            team_logs=logs,
            game_date=game_date,
            season=season,
            min_games=min_games,
            rating_mode=rating_mode,
        )
        for j, feat in enumerate(feats):
            pred = predict_game(feat, n_sims=n_sims, seed=10_000 + i * 100 + j)
            row = pred.to_dict()
            act = actuals.loc[actuals["game_id"] == pred.game_id]
            if act.empty:
                continue
            a = act.iloc[0]
            row.update(
                {
                    "home_score": float(a["home_score"]),
                    "away_score": float(a["away_score"]),
                    "home_win": float(a["home_win"]),
                    "margin": float(a["margin"]),
                    "total_pts": float(a["total_pts"]),
                }
            )
            row.update(grade_prediction_row(pd.Series(row)))
            rows.append(row)

    graded = pd.DataFrame(rows)
    if graded.empty:
        metrics = {"n_games": 0, "season": season}
        return graded, metrics

    if calibrate and "p_home_win" in graded.columns and "home_win" in graded.columns:
        graded = apply_expanding_platt(graded)
        graded = swap_calibrated_ml_probs(graded)

    graded, bank_meta = simulate_bankroll(
        graded,
        starting_bankroll=starting_bankroll,
        american=paper_american,
    )
    metrics = summarize_graded(graded)
    metrics.update(bank_meta)
    metrics["season"] = season
    metrics["calibrate"] = bool(calibrate)
    metrics["rating_mode"] = rating_mode
    return graded, metrics


def format_backtest_report(graded: pd.DataFrame, metrics: dict[str, Any]) -> str:
    head = format_metrics(metrics)
    if graded.empty:
        return head
    sample = graded[
        [
            c
            for c in [
                "game_date",
                "away_tricode",
                "home_tricode",
                "p_home_win",
                "ml_correct",
                "spread_correct",
                "total_correct",
            ]
            if c in graded.columns
        ]
    ].head(20)
    return f"{head}\n\npreview:\n{sample.to_string(index=False)}"
