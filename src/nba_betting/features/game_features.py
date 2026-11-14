"""Per-game feature rows for a slate date (as-of ratings + rest/home)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from nba_betting.features.opponent_adjusted import opponent_adjusted_asof
from nba_betting.features.player_availability import (
    AvailabilitySnapshot,
    availability_asof,
    prepare_player_minutes,
)
from nba_betting.features.schedule_context import schedule_density_asof
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
    # Optional enrichments (defaults keep legacy behavior).
    home_missing_minutes: float = 0.0
    away_missing_minutes: float = 0.0
    home_three_in_four: float = 0.0
    away_three_in_four: float = 0.0
    home_games_last7: float = 0.0
    away_games_last7: float = 0.0
    adj_blend: float = 0.0  # 0 = raw EMA only; 1 = full opp-adjusted

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
    player_minutes: pd.DataFrame | None = None,
    adj_blend: float = 0.0,
    prepared_logs: pd.DataFrame | None = None,
    player_minutes_prepared: pd.DataFrame | dict | None = None,
) -> list[GameFeatures]:
    """Build feature rows for all scheduled games on ``game_date``.

    Ratings / rest use only team games with ``game_date < slate date``.
    ``rating_mode`` is ``\"ema\"`` (default) or ``\"window\"`` (flat trailing mean).

    Optional ``player_minutes`` / ``adj_blend`` add prior-box availability and
    opponent-adjusted efficiency (see ``opponent_adjusted``).
    """
    prepared = (
        prepared_logs if prepared_logs is not None else prepare_team_game_rows(team_logs)
    )
    pm = player_minutes_prepared
    if pm is None and player_minutes is not None:
        pm = prepare_player_minutes(player_minutes)

    day = _schedule_for_date(schedule, game_date)
    features: list[GameFeatures] = []
    blend = float(adj_blend)

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

        home_ortg, home_drtg = float(home_r.ortg), float(home_r.drtg)
        away_ortg, away_drtg = float(away_r.ortg), float(away_r.drtg)
        if blend > 0.0:
            home_a = opponent_adjusted_asof(
                prepared, home_id, game_date, half_life=ema_half_life
            )
            away_a = opponent_adjusted_asof(
                prepared, away_id, game_date, half_life=ema_half_life
            )
            if home_a is not None:
                home_ortg = (1.0 - blend) * home_ortg + blend * float(home_a.off)
                home_drtg = (1.0 - blend) * home_drtg + blend * float(home_a.defense)
            if away_a is not None:
                away_ortg = (1.0 - blend) * away_ortg + blend * float(away_a.off)
                away_drtg = (1.0 - blend) * away_drtg + blend * float(away_a.defense)

        home_rest, home_b2b = days_rest_asof(prepared, home_id, game_date)
        away_rest, away_b2b = days_rest_asof(prepared, away_id, game_date)

        home_miss = away_miss = 0.0
        if pm is not None:
            hav = _availability_lookup(pm, home_id, game_date)
            aav = _availability_lookup(pm, away_id, game_date)
            home_miss = float(hav.missing_minutes)
            away_miss = float(aav.missing_minutes)

        home_dates = prepared.loc[
            (prepared["team_id"] == home_id)
            & (prepared["game_date"] < pd.Timestamp(game_date).normalize()),
            "game_date",
        ]
        away_dates = prepared.loc[
            (prepared["team_id"] == away_id)
            & (prepared["game_date"] < pd.Timestamp(game_date).normalize()),
            "game_date",
        ]
        h_sched = schedule_density_asof(home_dates, game_date)
        a_sched = schedule_density_asof(away_dates, game_date)

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
                home_ortg=home_ortg,
                away_ortg=away_ortg,
                home_drtg=home_drtg,
                away_drtg=away_drtg,
                home_rest_days=home_rest,
                away_rest_days=away_rest,
                home_b2b=home_b2b,
                away_b2b=away_b2b,
                home_games=home_r.games,
                away_games=away_r.games,
                home_court_points=float(home_court_points),
                home_missing_minutes=home_miss,
                away_missing_minutes=away_miss,
                home_three_in_four=float(h_sched["three_in_four"]),
                away_three_in_four=float(a_sched["three_in_four"]),
                home_games_last7=float(h_sched["games_last_n"]),
                away_games_last7=float(a_sched["games_last_n"]),
                adj_blend=blend,
            )
        )
    return features


def _availability_lookup(
    pm: pd.DataFrame | dict, team_id: int, game_date: str
) -> AvailabilitySnapshot:
    if isinstance(pm, dict):
        frame = pm.get(int(team_id))
        if frame is None or getattr(frame, "empty", True):
            return availability_asof(
                pd.DataFrame(
                    columns=["game_id", "game_date", "team_id", "player_id", "minutes"]
                ),
                int(team_id),
                game_date,
            )
        return availability_asof(frame, int(team_id), game_date)
    return availability_asof(pm, int(team_id), game_date)
