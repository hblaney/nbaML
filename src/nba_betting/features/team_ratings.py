"""As-of-date team pace / ORTG / DRTG ratings (no future leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_betting.features.possessions import attach_opponent_points, estimate_possessions

DEFAULT_WINDOW = 20  # trailing games used when available
HOME_COURT_POINTS = 2.5  # prior; used by the game model, documented here


@dataclass(frozen=True)
class TeamRating:
    team_id: int
    team_abbreviation: str
    games: int
    pace: float
    ortg: float
    drtg: float
    asof_date: str


def prepare_team_game_rows(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, attach opp points / possessions / efficiencies."""
    if team_logs.empty:
        cols = list(team_logs.columns) + [
            "opp_pts",
            "possessions",
            "ortg",
            "drtg",
            "pace",
            "game_date",
        ]
        return pd.DataFrame(columns=cols)

    df = team_logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = attach_opponent_points(df)
    df["possessions"] = estimate_possessions(df).clip(lower=1.0)
    # Team minutes are typically 240; scale pace to per-48 possessions.
    minutes = df["min"].astype(float).replace(0, np.nan).fillna(240.0)
    df["pace"] = df["possessions"] * (48.0 * 5.0) / minutes
    df["ortg"] = 100.0 * df["pts"].astype(float) / df["possessions"]
    df["drtg"] = 100.0 * df["opp_pts"].astype(float) / df["possessions"]
    return df.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)


def team_rating_asof(
    prepared: pd.DataFrame,
    team_id: int,
    asof_date: str | pd.Timestamp,
    *,
    window: int = DEFAULT_WINDOW,
) -> TeamRating | None:
    """Aggregate trailing ratings using only games with ``game_date < asof_date``."""
    asof = pd.Timestamp(asof_date).normalize()
    hist = prepared.loc[
        (prepared["team_id"] == team_id) & (prepared["game_date"] < asof)
    ]
    if hist.empty:
        return None
    hist = hist.tail(int(window))
    abbr = str(hist.iloc[-1]["team_abbreviation"])
    return TeamRating(
        team_id=int(team_id),
        team_abbreviation=abbr,
        games=int(len(hist)),
        pace=float(hist["pace"].mean()),
        ortg=float(hist["ortg"].mean()),
        drtg=float(hist["drtg"].mean()),
        asof_date=str(asof.date()),
    )


def days_rest_asof(
    prepared: pd.DataFrame,
    team_id: int,
    asof_date: str | pd.Timestamp,
) -> tuple[float, bool]:
    """Return (days_rest, is_b2b) using the last game strictly before ``asof_date``.

    ``days_rest`` is calendar days between prior tip and ``asof_date`` (1 = B2B).
    If no prior game, returns (nan, False).
    """
    asof = pd.Timestamp(asof_date).normalize()
    hist = prepared.loc[
        (prepared["team_id"] == team_id) & (prepared["game_date"] < asof),
        "game_date",
    ]
    if hist.empty:
        return float("nan"), False
    last = pd.Timestamp(hist.max()).normalize()
    days = float((asof - last).days)
    return days, days == 1.0


def assert_no_future_rows(prepared: pd.DataFrame, asof_date: str | pd.Timestamp) -> None:
    """Raise if any prepared row is on/after ``asof_date`` (leakage guard for callers)."""
    asof = pd.Timestamp(asof_date).normalize()
    if (prepared["game_date"] >= asof).any():
        # Callers should filter; this helper is for tests / strict modes.
        raise ValueError(f"prepared frame contains rows on/after {asof.date()}")
