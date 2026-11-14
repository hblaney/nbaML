"""Per-game feature rows for a slate date (as-of ratings + rest/home)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from nba_betting.features.team_ratings import (
    DEFAULT_EMA_HALF_LIFE,
    DEFAULT_WINDOW,
    HOME_COURT_POINTS,
    days_rest_asof,
    prepare_team_game_rows,
    team_rating_asof,
    team_rating_ema,
)


@dataclass(frozen=True)
class GameFeatures:
    game_id: str
    game_date: str
    season: str
    home_team_id: int
    away_team_id: int
    home_tricode: str
    away_tricode: str
    home: bool  # always True for the home side context; kept for clarity
    home_pace: float
    away_pace: float
    home_ortg: float
    away_ortg: float
    home_drtg: float
    away_drtg: float
    home_rest_days: float
    away_rest_days: float
    home_b2b: bool
    away_b2b: bool
    home_games: int
    away_games: int
    home_court_points: float

    def to_dict(self) -> dict:
        return asdict(self)


def _schedule_for_date(schedule: pd.DataFrame, game_date: str | pd.Timestamp) -> pd.DataFrame:
    target = pd.Timestamp(game_date).normalize()
    df = schedule.copy()
    # Prefer EST calendar date when present.
    if "game_date_est" in df.columns:
        dates = pd.to_datetime(df["game_date_est"]).dt.normalize()
    elif "game_datetime_est" in df.columns:
        dates = pd.to_datetime(df["game_datetime_est"]).dt.normalize()
    else:
        raise ValueError("schedule missing game_date_est")
    return df.loc[dates == target].copy()


def build_game_features_for_date(
    *,
    schedule: pd.DataFrame,
    team_logs: pd.DataFrame,
    game_date: str,
    season: str,
    window: int = DEFAULT_WINDOW,
    home_court_points: float = HOME_COURT_POINTS,
    min_games: int = 1,
    rating_mode: str = "ema",
    ema_half_life: float = DEFAULT_EMA_HALF_LIFE,
) -> list[GameFeatures]:
    """Build feature rows for all scheduled games on ``game_date``.

    Ratings / rest use only team games with ``game_date < slate date``.
    ``rating_mode`` is ``\"ema\"`` (default) or ``\"window\"`` (flat trailing mean).
    """
    prepared = prepare_team_game_rows(team_logs)
    day = _schedule_for_date(schedule, game_date)
    features: list[GameFeatures] = []

    for _, row in day.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        if rating_mode == "ema":
            home_r = team_rating_ema(
                prepared, home_id, game_date, half_life=ema_half_life
            )
            away_r = team_rating_ema(
                prepared, away_id, game_date, half_life=ema_half_life
            )
        elif rating_mode == "window":
            home_r = team_rating_asof(prepared, home_id, game_date, window=window)
            away_r = team_rating_asof(prepared, away_id, game_date, window=window)
        else:
            raise ValueError(f"unknown rating_mode: {rating_mode!r}")
        if home_r is None or away_r is None:
            continue
        if home_r.games < min_games or away_r.games < min_games:
            continue

        home_rest, home_b2b = days_rest_asof(prepared, home_id, game_date)
        away_rest, away_b2b = days_rest_asof(prepared, away_id, game_date)

        features.append(
            GameFeatures(
                game_id=str(row["game_id"]),
                game_date=str(pd.Timestamp(game_date).date()),
                season=season,
                home_team_id=home_id,
                away_team_id=away_id,
                home_tricode=str(row.get("home_tricode") or home_r.team_abbreviation),
                away_tricode=str(row.get("away_tricode") or away_r.team_abbreviation),
                home=True,
                home_pace=home_r.pace,
                away_pace=away_r.pace,
                home_ortg=home_r.ortg,
                away_ortg=away_r.ortg,
                home_drtg=home_r.drtg,
                away_drtg=away_r.drtg,
                home_rest_days=home_rest,
                away_rest_days=away_rest,
                home_b2b=home_b2b,
                away_b2b=away_b2b,
                home_games=home_r.games,
                away_games=away_r.games,
                home_court_points=float(home_court_points),
            )
        )
    return features
