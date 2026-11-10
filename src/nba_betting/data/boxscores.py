"""Team / player game-log (box-score) ingest via ``nba_api`` LeagueGameLog.

LeagueGameLog returns one row per team (or player) per game with traditional
box-score fields — the bulk feed for pace/efficiency features. Per-game
``BoxScoreTraditionalV2`` pulls are left for a later, selective refresh.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nba_betting.data.cache import read_frame, resolve_existing, write_frame
from nba_betting.data.paths import boxscore_path
from nba_betting.data.season import normalize_season

BoxscoreFetcher = Callable[[str, str, str], pd.DataFrame]

DEFAULT_SEASON_TYPES: tuple[str, ...] = ("Regular Season", "Playoffs")

TEAM_COLUMNS = [
    "season",
    "season_type",
    "game_id",
    "game_date",
    "team_id",
    "team_abbreviation",
    "team_name",
    "matchup",
    "wl",
    "min",
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "fg3_pct",
    "ftm",
    "fta",
    "ft_pct",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
    "plus_minus",
]


def _default_fetch_game_log(season: str, season_type: str, level: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    player_or_team = "T" if level == "team" else "P"
    endpoint = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation=player_or_team,
        league_id="00",
    )
    frames = endpoint.get_data_frames()
    if not frames:
        return pd.DataFrame()
    return frames[0]


def normalize_team_game_log(
    raw: pd.DataFrame, season: str, season_type: str
) -> pd.DataFrame:
    """Map LeagueGameLog team rows to a stable lowercase schema."""
    if raw.empty:
        return pd.DataFrame(columns=TEAM_COLUMNS)

    rename = {
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbreviation",
        "TEAM_NAME": "team_name",
        "MATCHUP": "matchup",
        "WL": "wl",
        "MIN": "min",
        "FGM": "fgm",
        "FGA": "fga",
        "FG_PCT": "fg_pct",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FG3_PCT": "fg3_pct",
        "FTM": "ftm",
        "FTA": "fta",
        "FT_PCT": "ft_pct",
        "OREB": "oreb",
        "DREB": "dreb",
        "REB": "reb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "PF": "pf",
        "PTS": "pts",
        "PLUS_MINUS": "plus_minus",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    df["season"] = season
    df["season_type"] = season_type
    for col in TEAM_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[TEAM_COLUMNS].copy()
    out["game_id"] = out["game_id"].astype(str)
    return out.reset_index(drop=True)


def normalize_player_game_log(
    raw: pd.DataFrame, season: str, season_type: str
) -> pd.DataFrame:
    """Normalize player LeagueGameLog rows (keeps PLAYER_ID / PLAYER_NAME if present)."""
    if raw.empty:
        base = TEAM_COLUMNS + ["player_id", "player_name"]
        return pd.DataFrame(columns=base)

    teamish = normalize_team_game_log(raw, season, season_type)
    player_id = raw["PLAYER_ID"] if "PLAYER_ID" in raw.columns else pd.Series(pd.NA, index=raw.index)
    player_name = (
        raw["PLAYER_NAME"] if "PLAYER_NAME" in raw.columns else pd.Series(pd.NA, index=raw.index)
    )
    teamish = teamish.copy()
    teamish["player_id"] = player_id.values
    teamish["player_name"] = player_name.values
    return teamish


def pull_boxscores(
    season: str,
    *,
    root: Path | None = None,
    level: str = "team",
    season_types: Sequence[str] | None = None,
    fetcher: BoxscoreFetcher | None = None,
    force: bool = False,
    also_csv: bool = True,
) -> Path:
    """Fetch season game logs and cache under ``data/raw/boxscores/``."""
    season = normalize_season(season)
    if level not in {"team", "player"}:
        raise ValueError(f"level must be 'team' or 'player', got {level!r}")

    out = boxscore_path(season, level=level, root=root, fmt="parquet")
    if out.exists() and not force:
        return out

    types = tuple(season_types) if season_types is not None else DEFAULT_SEASON_TYPES
    fetch = fetcher or _default_fetch_game_log
    frames: list[pd.DataFrame] = []
    for season_type in types:
        raw = fetch(season, season_type, level)
        if level == "team":
            frames.append(normalize_team_game_log(raw, season, season_type))
        else:
            frames.append(normalize_player_game_log(raw, season, season_type))

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=TEAM_COLUMNS if level == "team" else TEAM_COLUMNS + ["player_id", "player_name"])
    )
    return write_frame(combined, out, also_csv=also_csv)


def load_boxscores(
    season: str, *, level: str = "team", root: Path | None = None
) -> pd.DataFrame:
    """Load cached box / game-log table (offline)."""
    season = normalize_season(season)
    path = resolve_existing(boxscore_path(season, level=level, root=root, fmt="parquet"))
    return read_frame(path)


def boxscore_client_info() -> dict[str, Any]:
    return {"endpoint": "leaguegamelog", "provider": "nba_api"}
