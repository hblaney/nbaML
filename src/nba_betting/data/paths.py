"""Local data layout helpers.

Default layout (repo root)::

    data/
      raw/
        schedule/{season}.parquet
        schedule/{season}.csv
        boxscores/{season}_team.parquet
        boxscores/{season}_team.csv
        boxscores/{season}_player.parquet   # optional
      processed/
      blind_predictions.jsonl

Raw dumps under ``data/raw/`` are gitignored; only ``.gitkeep`` placeholders
are tracked. Override the root with env ``NBA_BETTING_DATA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

def data_root(root: Path | None = None) -> Path:
    """Return the data directory root (cwd ``./data`` unless overridden)."""
    if root is not None:
        return Path(root)
    env = os.environ.get("NBA_BETTING_DATA_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "data"


def raw_dir(root: Path | None = None) -> Path:
    return data_root(root) / "raw"


def processed_dir(root: Path | None = None) -> Path:
    return data_root(root) / "processed"


def schedule_path(season: str, *, root: Path | None = None, fmt: str = "parquet") -> Path:
    return raw_dir(root) / "schedule" / f"{season}.{fmt}"


def boxscore_path(
    season: str,
    *,
    level: str = "team",
    root: Path | None = None,
    fmt: str = "parquet",
) -> Path:
    if level not in {"team", "player"}:
        raise ValueError(f"level must be 'team' or 'player', got {level!r}")
    return raw_dir(root) / "boxscores" / f"{season}_{level}.{fmt}"


def blind_ledger_path(root: Path | None = None) -> Path:
    return data_root(root) / "blind_predictions.jsonl"
