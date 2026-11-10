"""As-of player minutes and per-minute rate priors from cached player logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

DEFAULT_PLAYER_WINDOW = 10
MIN_MINUTES_FLOOR = 1.0

STAT_COLS = ("pts", "reb", "ast", "fg3m")


@dataclass(frozen=True)
class PlayerPrior:
    player_id: int
    player_name: str
    team_id: int
    team_abbreviation: str
    games: int
    exp_minutes: float
    pts_per_min: float
    reb_per_min: float
    ast_per_min: float
    fg3m_per_min: float
    asof_date: str

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_player_logs(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and ensure required columns exist."""
    if player_logs.empty:
        cols = list(player_logs.columns) + ["game_date"]
        return pd.DataFrame(columns=cols)
    df = player_logs.copy()
    if "player_id" not in df.columns or "player_name" not in df.columns:
        raise ValueError("player logs require player_id and player_name")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


def player_prior_asof(
    prepared: pd.DataFrame,
    player_id: int,
    asof_date: str | pd.Timestamp,
    *,
    window: int = DEFAULT_PLAYER_WINDOW,
) -> PlayerPrior | None:
    """Trailing minutes / per-minute rates using only games before ``asof_date``."""
    asof = pd.Timestamp(asof_date).normalize()
    hist = prepared.loc[
        (prepared["player_id"] == player_id) & (prepared["game_date"] < asof)
    ]
    if hist.empty:
        return None
    hist = hist.tail(int(window))
    minutes = hist["min"].astype(float).clip(lower=0.0)
    total_min = float(minutes.sum())
    if total_min <= 0:
        return None
    exp_minutes = float(minutes.mean())
    return PlayerPrior(
        player_id=int(player_id),
        player_name=str(hist.iloc[-1]["player_name"]),
        team_id=int(hist.iloc[-1]["team_id"]),
        team_abbreviation=str(hist.iloc[-1]["team_abbreviation"]),
        games=int(len(hist)),
        exp_minutes=exp_minutes,
        pts_per_min=float(hist["pts"].sum()) / total_min,
        reb_per_min=float(hist["reb"].sum()) / total_min,
        ast_per_min=float(hist["ast"].sum()) / total_min,
        fg3m_per_min=float(hist["fg3m"].sum()) / total_min,
        asof_date=str(asof.date()),
    )


def priors_for_teams(
    prepared: pd.DataFrame,
    team_ids: set[int],
    asof_date: str | pd.Timestamp,
    *,
    window: int = DEFAULT_PLAYER_WINDOW,
    min_exp_minutes: float = 12.0,
) -> list[PlayerPrior]:
    """Build as-of priors for players whose latest team is in ``team_ids``."""
    asof = pd.Timestamp(asof_date).normalize()
    hist = prepared.loc[prepared["game_date"] < asof]
    if hist.empty:
        return []
    # Latest team assignment before asof.
    latest = hist.sort_values("game_date").groupby("player_id", as_index=False).tail(1)
    latest = latest.loc[latest["team_id"].isin(team_ids)]
    priors: list[PlayerPrior] = []
    for player_id in latest["player_id"].tolist():
        prior = player_prior_asof(prepared, int(player_id), asof, window=window)
        if prior is None:
            continue
        if prior.exp_minutes < min_exp_minutes:
            continue
        if prior.team_id not in team_ids:
            continue
        priors.append(prior)
    return priors
