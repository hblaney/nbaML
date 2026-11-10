"""Possession / pace / efficiency estimators from traditional box lines."""

from __future__ import annotations

import pandas as pd

# Standard estimate when official pace is unavailable.
# Poss ≈ FGA - OREB + TOV + 0.44 * FTA
FTA_COEFF = 0.44


def estimate_possessions(df: pd.DataFrame) -> pd.Series:
    """Estimate team possessions from traditional counting stats."""
    return (
        df["fga"].astype(float)
        - df["oreb"].astype(float)
        + df["tov"].astype(float)
        + FTA_COEFF * df["fta"].astype(float)
    )


def attach_opponent_points(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Add ``opp_pts`` by pairing the other team's row on the same ``game_id``."""
    if team_logs.empty:
        out = team_logs.copy()
        out["opp_pts"] = pd.Series(dtype=float)
        return out

    required = {"game_id", "team_id", "pts"}
    missing = required - set(team_logs.columns)
    if missing:
        raise ValueError(f"team logs missing columns: {sorted(missing)}")

    slim = team_logs[["game_id", "team_id", "pts"]].rename(
        columns={"team_id": "opp_team_id", "pts": "opp_pts"}
    )
    merged = team_logs.merge(slim, on="game_id", how="left")
    merged = merged.loc[merged["team_id"] != merged["opp_team_id"]].copy()
    return merged.drop(columns=["opp_team_id"])
