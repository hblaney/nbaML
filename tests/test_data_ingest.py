"""Fixture / mock unit tests for schedule + boxscore ingest (no nba.com)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_betting.data.boxscores import load_boxscores, normalize_team_game_log, pull_boxscores
from nba_betting.data.cache import read_frame
from nba_betting.data.paths import boxscore_path, schedule_path
from nba_betting.data.schedule import load_schedule, normalize_schedule, pull_schedule
from nba_betting.data.season import normalize_season

FIXTURES = Path(__file__).parent / "fixtures" / "data"


def test_normalize_season_ok() -> None:
    assert normalize_season("2024-25") == "2024-25"


def test_normalize_season_rejects_bad() -> None:
    with pytest.raises(ValueError):
        normalize_season("2024")
    with pytest.raises(ValueError):
        normalize_season("2024-26")


def test_pull_schedule_uses_fetcher_and_caches(tmp_path: Path) -> None:
    raw = pd.read_csv(FIXTURES / "schedule_raw.csv")

    def fake_fetch(season: str) -> pd.DataFrame:
        assert season == "2024-25"
        return raw

    path = pull_schedule("2024-25", root=tmp_path, fetcher=fake_fetch, force=True)
    assert path == schedule_path("2024-25", root=tmp_path)
    assert path.exists()
    assert path.with_suffix(".csv").exists()

    cached = load_schedule("2024-25", root=tmp_path)
    assert len(cached) == 2
    assert set(["game_id", "home_tricode", "away_tricode"]).issubset(cached.columns)
    assert cached.iloc[0]["home_tricode"] == "LAL"

    # Second pull without force must not call fetcher.
    def boom(_season: str) -> pd.DataFrame:
        raise AssertionError("should use cache")

    again = pull_schedule("2024-25", root=tmp_path, fetcher=boom, force=False)
    assert again == path


def test_normalize_schedule_empty() -> None:
    out = normalize_schedule(pd.DataFrame(), "2024-25")
    assert list(out.columns)
    assert out.empty


def test_pull_boxscores_uses_fetcher_and_caches(tmp_path: Path) -> None:
    raw = pd.read_csv(FIXTURES / "team_gamelog_raw.csv", dtype={"GAME_ID": str})
    calls: list[tuple[str, str, str]] = []

    def fake_fetch(season: str, season_type: str, level: str) -> pd.DataFrame:
        calls.append((season, season_type, level))
        assert level == "team"
        if season_type == "Regular Season":
            return raw
        return pd.DataFrame(columns=raw.columns)

    path = pull_boxscores(
        "2024-25",
        root=tmp_path,
        level="team",
        season_types=("Regular Season", "Playoffs"),
        fetcher=fake_fetch,
        force=True,
    )
    assert path == boxscore_path("2024-25", level="team", root=tmp_path)
    assert path.exists()

    df = load_boxscores("2024-25", level="team", root=tmp_path)
    assert len(df) == 2
    assert df["pts"].tolist() == [110, 103]
    assert {c[1] for c in calls} == {"Regular Season", "Playoffs"}

    # Offline re-load via csv fallback if parquet removed.
    path.unlink()
    csv_only = read_frame(path.with_suffix(".csv"))
    assert len(csv_only) == 2


def test_normalize_team_game_log() -> None:
    raw = pd.read_csv(FIXTURES / "team_gamelog_raw.csv", dtype={"GAME_ID": str})
    out = normalize_team_game_log(raw, "2024-25", "Regular Season")
    assert out.loc[0, "team_abbreviation"] == "LAL"
    assert out.loc[0, "season_type"] == "Regular Season"


@pytest.mark.live
def test_live_schedule_smoke() -> None:
    """Optional live hit to stats.nba.com — skipped unless RUN_LIVE_NBA_API=1."""
    import os

    if os.environ.get("RUN_LIVE_NBA_API") != "1":
        pytest.skip("set RUN_LIVE_NBA_API=1 to enable live nba_api smoke")

    from nba_betting.data.schedule import pull_schedule

    path = pull_schedule("2024-25", root=Path("/tmp/nba_betting_live_smoke"), force=True)
    df = read_frame(path)
    assert len(df) > 0
