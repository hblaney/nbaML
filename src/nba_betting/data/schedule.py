"""Season schedule ingest via ``nba_api`` ScheduleLeagueV2."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from nba_betting.data.cache import read_frame, resolve_existing, write_frame
from nba_betting.data.paths import schedule_path
from nba_betting.data.season import normalize_season

ScheduleFetcher = Callable[[str], pd.DataFrame]

# Stable subset useful for downstream features / slate building.
SCHEDULE_COLUMNS = [
    "season",
    "game_id",
    "game_date_est",
    "game_datetime_est",
    "game_status",
    "game_status_text",
    "home_team_id",
    "home_tricode",
    "home_score",
    "away_team_id",
    "away_tricode",
    "away_score",
    "arena_name",
    "is_neutral",
]


def _default_fetch_schedule(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import scheduleleaguev2

    endpoint = scheduleleaguev2.ScheduleLeagueV2(league_id="00", season=season)
    frames = endpoint.get_data_frames()
    if not frames:
        return pd.DataFrame()
    return frames[0]


def normalize_schedule(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Map ScheduleLeagueV2 columns to a stable schema."""
    if raw.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    rename = {
        "seasonYear": "season",
        "gameId": "game_id",
        "gameDateEst": "game_date_est",
        "gameDateTimeEst": "game_datetime_est",
        "gameStatus": "game_status",
        "gameStatusText": "game_status_text",
        "homeTeam_teamId": "home_team_id",
        "homeTeam_teamTricode": "home_tricode",
        "homeTeam_score": "home_score",
        "awayTeam_teamId": "away_team_id",
        "awayTeam_teamTricode": "away_tricode",
        "awayTeam_score": "away_score",
        "arenaName": "arena_name",
        "isNeutral": "is_neutral",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    if "season" not in df.columns:
        df["season"] = season
    else:
        df["season"] = df["season"].fillna(season).astype(str)

    for col in SCHEDULE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[SCHEDULE_COLUMNS].copy()
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].astype(str)
    return out.reset_index(drop=True)


def pull_schedule(
    season: str,
    *,
    root: Path | None = None,
    fetcher: ScheduleFetcher | None = None,
    force: bool = False,
    also_csv: bool = True,
) -> Path:
    """Fetch season schedule and cache under ``data/raw/schedule/``."""
    season = normalize_season(season)
    out = schedule_path(season, root=root, fmt="parquet")
    if out.exists() and not force:
        return out

    fetch = fetcher or _default_fetch_schedule
    raw = fetch(season)
    normalized = normalize_schedule(raw, season)
    return write_frame(normalized, out, also_csv=also_csv)


def load_schedule(season: str, *, root: Path | None = None) -> pd.DataFrame:
    """Load cached schedule (offline)."""
    season = normalize_season(season)
    path = resolve_existing(schedule_path(season, root=root, fmt="parquet"))
    return read_frame(path)


def schedule_client_info() -> dict[str, Any]:
    """Small metadata helper for CLI / debugging."""
    return {"endpoint": "scheduleleaguev2", "provider": "nba_api"}
