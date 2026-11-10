"""Score a calendar date from cached schedule + team box logs (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_betting.data.boxscores import load_boxscores
from nba_betting.data.schedule import load_schedule
from nba_betting.features.game_features import GameFeatures, build_game_features_for_date
from nba_betting.models.game import GamePrediction, predict_game


def score_date(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    market_spread: float = 0.0,
    market_total: float | None = None,
    n_sims: int = 20_000,
    seed: int = 0,
    schedule: pd.DataFrame | None = None,
    team_logs: pd.DataFrame | None = None,
) -> list[GamePrediction]:
    """Load caches (unless frames provided) and return model predictions for the date."""
    sched = schedule if schedule is not None else load_schedule(season, root=root)
    logs = team_logs if team_logs is not None else load_boxscores(season, level="team", root=root)
    feats = build_game_features_for_date(
        schedule=sched,
        team_logs=logs,
        game_date=game_date,
        season=season,
    )
    return [
        predict_game(
            f,
            market_spread=market_spread,
            market_total=market_total,
            n_sims=n_sims,
            seed=seed + i,
        )
        for i, f in enumerate(feats)
    ]


def predictions_to_frame(preds: list[GamePrediction]) -> pd.DataFrame:
    if not preds:
        return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in preds])


def format_predictions_table(preds: list[GamePrediction]) -> str:
    if not preds:
        return "(no games / insufficient history for this date)"
    df = predictions_to_frame(preds)
    cols = [
        "game_id",
        "away_tricode",
        "home_tricode",
        "exp_away_score",
        "exp_home_score",
        "p_home_win",
        "fair_spread",
        "fair_total",
        "p_home_cover",
        "p_over",
    ]
    view = df[cols].copy()
    for c in ["exp_away_score", "exp_home_score", "fair_spread", "fair_total"]:
        view[c] = view[c].map(lambda x: f"{x:.1f}")
    for c in ["p_home_win", "p_home_cover", "p_over"]:
        view[c] = view[c].map(lambda x: f"{x:.3f}")
    return view.to_string(index=False)


def write_predictions_json(preds: list[GamePrediction], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.to_dict() for p in preds]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def features_to_frame(features: list[GameFeatures]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    return pd.DataFrame([f.to_dict() for f in features])
