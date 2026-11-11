"""Player availability proxies from prior box scores only (no same-day / injury feed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AvailabilitySnapshot:
    team_id: int
    asof_date: str
    rotation_n: int
    exp_minutes_top: float
    last_minutes_top: float
    missing_minutes: float  # max(0, exp - last) summed over rotation
    availability_ratio: float  # last/exp clipped


def prepare_player_minutes(player_minutes: pd.DataFrame) -> pd.DataFrame:
    """Normalize ESPN/player minute rows for as-of joins."""
    df = player_minutes.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(str)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["dnp"] = df.get("dnp", False)
    if "dnp" in df.columns:
        df.loc[df["dnp"].astype(bool), "minutes"] = 0.0
    return df.sort_values(["team_id", "player_id", "game_date", "game_id"]).reset_index(
        drop=True
    )


def availability_asof(
    prepared: pd.DataFrame,
    team_id: int,
    asof_date: str | pd.Timestamp,
    *,
    window: int = 10,
    top_n: int = 7,
    min_avg_minutes: float = 18.0,
) -> AvailabilitySnapshot:
    """Rotation availability from prior games only (``game_date < asof_date``).

    Rotation = top ``top_n`` players by average minutes over the prior ``window``
    team games (players who appeared). Compare their last-game minutes to that
    average; ``missing_minutes`` is a pregame-usable proxy for who sat last time
    (often tracks short-term unavailability without using postgame DNP for the
    *current* slate).
    """
    asof = pd.Timestamp(asof_date).normalize()
    hist = prepared.loc[
        (prepared["team_id"] == int(team_id)) & (prepared["game_date"] < asof)
    ]
    empty = AvailabilitySnapshot(
        team_id=int(team_id),
        asof_date=str(asof.date()),
        rotation_n=0,
        exp_minutes_top=0.0,
        last_minutes_top=0.0,
        missing_minutes=0.0,
        availability_ratio=1.0,
    )
    if hist.empty:
        return empty

    # Last ``window`` distinct team games.
    game_dates = hist[["game_id", "game_date"]].drop_duplicates().sort_values("game_date")
    recent_games = game_dates.tail(int(window))
    if recent_games.empty:
        return empty
    recent = hist.loc[hist["game_id"].isin(recent_games["game_id"])]
    last_gid = str(recent_games.iloc[-1]["game_id"])

    avg = (
        recent.groupby("player_id", as_index=False)["minutes"]
        .mean()
        .rename(columns={"minutes": "avg_min"})
    )
    avg = avg.loc[avg["avg_min"] >= float(min_avg_minutes)].sort_values(
        "avg_min", ascending=False
    )
    rotation = avg.head(int(top_n))
    if rotation.empty:
        return empty

    last = recent.loc[recent["game_id"] == last_gid, ["player_id", "minutes"]]
    last_map = dict(zip(last["player_id"].astype(str), last["minutes"].astype(float)))

    exp = 0.0
    got = 0.0
    for _, row in rotation.iterrows():
        pid = str(row["player_id"])
        e = float(row["avg_min"])
        exp += e
        got += float(last_map.get(pid, 0.0))
    missing = max(0.0, exp - got)
    ratio = float(got / exp) if exp > 0 else 1.0
    return AvailabilitySnapshot(
        team_id=int(team_id),
        asof_date=str(asof.date()),
        rotation_n=int(len(rotation)),
        exp_minutes_top=float(exp),
        last_minutes_top=float(got),
        missing_minutes=float(missing),
        availability_ratio=float(np.clip(ratio, 0.0, 1.5)),
    )
