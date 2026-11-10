"""Season string helpers (e.g. ``2024-25``)."""

from __future__ import annotations

import re

_SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")


def normalize_season(season: str) -> str:
    """Validate and return a stats.nba.com season string like ``2024-25``."""
    season = season.strip()
    match = _SEASON_RE.fullmatch(season)
    if not match:
        raise ValueError(
            f"invalid season {season!r}; expected format YYYY-YY (e.g. '2024-25')"
        )
    start = int(match.group(1))
    end_yy = int(match.group(2))
    expected_end = (start + 1) % 100
    if end_yy != expected_end:
        raise ValueError(
            f"invalid season {season!r}; second year should be {expected_end:02d}"
        )
    return season
