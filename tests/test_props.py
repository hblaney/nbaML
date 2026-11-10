"""Fixture tests for player prop projections and odds edges."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_betting.data.odds import OddsApiConfigError
from nba_betting.data.prop_odds import (
    fetch_event_prop_odds,
    parse_prop_markets,
    pull_prop_odds,
)
from nba_betting.features.player_priors import player_prior_asof, prepare_player_logs
from nba_betting.models.props import p_over_poisson, project_from_prior
from nba_betting.pricing.prop_edge import build_prop_edge_rows
from nba_betting.slate.props import (
    format_prop_edges_report,
    score_props_date,
    slate_prop_edges,
)

FIXTURES = Path(__file__).parent / "fixtures"
MODEL = FIXTURES / "model"
PROPS = FIXTURES / "props"


@pytest.fixture
def schedule() -> pd.DataFrame:
    return pd.read_csv(MODEL / "schedule.csv", dtype={"game_id": str})


@pytest.fixture
def player_logs() -> pd.DataFrame:
    return pd.read_csv(PROPS / "player_logs.csv", dtype={"game_id": str, "player_id": int})


@pytest.fixture
def prop_event() -> dict:
    return json.loads((PROPS / "prop_odds_event.json").read_text(encoding="utf-8"))


def test_player_prior_asof_no_leakage(player_logs: pd.DataFrame) -> None:
    prepared = prepare_player_logs(player_logs)
    prior = player_prior_asof(prepared, 2544, "2024-10-25", window=10)
    assert prior is not None
    assert prior.games == 2  # Oct 22 + Oct 24 only
    assert prior.exp_minutes > 0
    assert prior.pts_per_min > 0
    # Same-day / future rows must not be included.
    hist = prepared.loc[
        (prepared["player_id"] == 2544)
        & (prepared["game_date"] < pd.Timestamp("2024-10-25"))
    ]
    assert len(hist) == prior.games


def test_project_props_from_minutes(player_logs: pd.DataFrame) -> None:
    prepared = prepare_player_logs(player_logs)
    prior = player_prior_asof(prepared, 2544, "2024-10-25")
    assert prior is not None
    proj = project_from_prior(prior, game_id="0022400005", game_date="2024-10-25")
    assert proj.points == pytest.approx(proj.exp_minutes * prior.pts_per_min)
    assert proj.pra == pytest.approx(proj.points + proj.rebounds + proj.assists)
    assert 0.0 < p_over_poisson(proj.points, 25.5) < 1.0


def test_score_props_date(
    schedule: pd.DataFrame, player_logs: pd.DataFrame
) -> None:
    projections = score_props_date(
        season="2024-25",
        game_date="2024-10-25",
        schedule=schedule,
        player_logs=player_logs,
        min_exp_minutes=10.0,
    )
    assert len(projections) >= 2
    names = {p.player_name for p in projections}
    assert "LeBron James" in names
    assert "Jayson Tatum" in names


def test_parse_prop_markets(prop_event: dict) -> None:
    rows = parse_prop_markets(prop_event)
    markets = {r.market for r in rows}
    assert "player_points" in markets
    assert "player_points_rebounds_assists" in markets
    lebron_pts = next(
        r for r in rows if r.player_name == "LeBron James" and r.stat == "points"
    )
    assert lebron_pts.line == 25.5
    assert lebron_pts.over_price == -115


def test_fetch_prop_odds_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsApiConfigError):
        fetch_event_prop_odds("evt", http_get=lambda url, params: {})


def test_pull_prop_odds_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prop_event: dict
) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "test-key-not-real")

    def fake_get(url: str, params: dict[str, str]):
        if "/events/" in url and url.rstrip("/").endswith("/odds"):
            return prop_event
        # Featured events list used to discover event ids.
        return [
            {
                "id": "evt_lal_bos",
                "commence_time": "2024-10-25T23:30:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Lakers", "price": -150},
                                    {"name": "Boston Celtics", "price": 130},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

    path = pull_prop_odds(root=tmp_path, http_get=fake_get)
    assert path.exists()
    cached = json.loads(path.read_text(encoding="utf-8"))
    assert any(r["player_name"] == "LeBron James" for r in cached)


def test_prop_edges_and_degrade(
    schedule: pd.DataFrame,
    player_logs: pd.DataFrame,
    prop_event: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projections = score_props_date(
        season="2024-25",
        game_date="2024-10-25",
        schedule=schedule,
        player_logs=player_logs,
        min_exp_minutes=10.0,
    )
    odds = parse_prop_markets(prop_event)
    rows = build_prop_edge_rows(projections, odds)
    assert any(r.stat == "points" and r.player_name == "LeBron James" for r in rows)
    lebron = next(r for r in rows if r.player_name == "LeBron James" and r.stat == "points")
    assert lebron.edge_over == pytest.approx(
        lebron.model_p_over - (lebron.book_implied_over_fair or 0)
    )

    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    edge_rows, projs, source = slate_prop_edges(
        season="2024-25",
        game_date="2024-10-25",
        root=tmp_path,
        use_live=True,
        schedule=schedule,
        player_logs=player_logs,
    )
    assert edge_rows == []
    assert len(projs) >= 1
    assert source.startswith("no-odds:")
    assert "ODDS_API_KEY" in format_prop_edges_report(edge_rows, source)
