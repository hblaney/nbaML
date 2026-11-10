"""Mocked Odds API + edge table tests (no real key / network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_betting.data.odds import (
    OddsApiConfigError,
    fetch_nba_odds,
    parse_event_odds,
    pull_odds,
)
from nba_betting.pricing.american import (
    american_to_implied,
    devig_two_way,
    implied_to_american,
)
from nba_betting.pricing.edge import build_edge_rows
from nba_betting.slate.edges import format_edges_report, slate_edges
from nba_betting.slate.score import score_date

FIXTURES = Path(__file__).parent / "fixtures"
ODDS_JSON = FIXTURES / "odds" / "odds_api_response.json"
MODEL = FIXTURES / "model"


@pytest.fixture
def odds_payload() -> list[dict]:
    return json.loads(ODDS_JSON.read_text(encoding="utf-8"))


@pytest.fixture
def schedule() -> pd.DataFrame:
    return pd.read_csv(MODEL / "schedule.csv", dtype={"game_id": str})


@pytest.fixture
def team_logs() -> pd.DataFrame:
    return pd.read_csv(MODEL / "team_logs.csv", dtype={"game_id": str})


def test_american_roundtrip() -> None:
    assert american_to_implied(-110) == pytest.approx(0.523809, rel=1e-4)
    assert american_to_implied(130) == pytest.approx(0.434783, rel=1e-4)
    fair = implied_to_american(0.6)
    assert fair < 0
    a, b = devig_two_way(-110, -110)
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.5)


def test_parse_event_odds(odds_payload: list[dict]) -> None:
    event = parse_event_odds(odds_payload[0])
    assert event is not None
    assert event.home_tricode == "LAL"
    assert event.away_tricode == "BOS"
    assert event.moneyline_home == -150
    assert event.spread_home_point == -3.5
    assert event.total_point == 220.5
    assert event.book == "draftkings"


def test_fetch_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsApiConfigError):
        fetch_nba_odds(http_get=lambda url, params: [])


def test_fetch_with_mock(monkeypatch: pytest.MonkeyPatch, odds_payload: list[dict]) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "test-key-not-real")
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_get(url: str, params: dict[str, str]):
        calls.append((url, params))
        assert "test-key-not-real" == params["apiKey"]
        assert params["markets"] == "h2h,spreads,totals"
        assert "basketball_nba" in url
        return odds_payload

    events = fetch_nba_odds(http_get=fake_get)
    assert len(events) == 2
    assert events[0].home_tricode == "LAL"
    assert len(calls) == 1


def test_pull_odds_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, odds_payload: list[dict]
) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "test-key-not-real")

    def fake_get(url: str, params: dict[str, str]):
        return odds_payload

    path = pull_odds(root=tmp_path, http_get=fake_get)
    assert path.exists()
    cached = json.loads(path.read_text(encoding="utf-8"))
    assert cached[0]["home_tricode"] == "LAL"


def test_build_edge_rows(
    schedule: pd.DataFrame, team_logs: pd.DataFrame, odds_payload: list[dict]
) -> None:
    preds = score_date(
        season="2024-25",
        game_date="2024-10-25",
        schedule=schedule,
        team_logs=team_logs,
        n_sims=2000,
        seed=3,
    )
    events = [parse_event_odds(e) for e in odds_payload]
    events = [e for e in events if e is not None]
    rows = build_edge_rows(preds, events, n_sims=2000)
    markets = {r.market for r in rows}
    assert {"moneyline", "spread", "total"} <= markets
    assert any(r.home_tricode == "LAL" and r.market == "moneyline" for r in rows)
    # Edge is model - de-vigged book
    ml_home = next(
        r for r in rows if r.market == "moneyline" and r.side == "home" and r.home_tricode == "LAL"
    )
    assert ml_home.edge == pytest.approx(ml_home.model_prob - ml_home.book_implied_fair)


def test_slate_edges_degrades_without_key(
    monkeypatch: pytest.MonkeyPatch,
    schedule: pd.DataFrame,
    team_logs: pd.DataFrame,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    rows, source = slate_edges(
        season="2024-25",
        game_date="2024-10-25",
        root=tmp_path,
        use_live=True,
        schedule=schedule,
        team_logs=team_logs,
        n_sims=500,
    )
    assert rows == []
    assert source.startswith("no-odds:")
    report = format_edges_report(rows, source)
    assert "ODDS_API_KEY" in report


def test_slate_edges_from_odds_file(
    schedule: pd.DataFrame,
    team_logs: pd.DataFrame,
    odds_payload: list[dict],
    tmp_path: Path,
) -> None:
    events = [parse_event_odds(e) for e in odds_payload]
    cache = tmp_path / "odds.json"
    cache.write_text(
        json.dumps([e.to_dict() for e in events if e is not None], indent=2),
        encoding="utf-8",
    )
    rows, source = slate_edges(
        season="2024-25",
        game_date="2024-10-25",
        odds_path=cache,
        schedule=schedule,
        team_logs=team_logs,
        n_sims=1500,
    )
    assert source.startswith("cache:")
    assert len(rows) > 0
