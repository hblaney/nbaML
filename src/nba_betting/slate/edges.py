"""Build today's edge table: model slate ⨝ cached / live odds."""

from __future__ import annotations

import json
from pathlib import Path

from nba_betting.data.odds import (
    EventOdds,
    OddsApiConfigError,
    fetch_nba_odds,
    latest_odds_cache,
    load_odds_cache,
)
from nba_betting.pricing.edge import EdgeRow, build_edge_rows, format_edge_table
from nba_betting.slate.score import score_date


def load_events_for_edges(
    *,
    root: Path | None = None,
    odds_path: Path | None = None,
    use_live: bool = False,
    http_get=None,
) -> tuple[list[EventOdds], str]:
    """Return (events, source_message). Degrades cleanly when odds unavailable."""
    if odds_path is not None:
        return load_odds_cache(odds_path), f"cache:{odds_path}"

    if use_live:
        try:
            events = fetch_nba_odds(http_get=http_get)
            return events, "live:the-odds-api"
        except OddsApiConfigError as exc:
            return [], f"no-odds:{exc}"

    cached = latest_odds_cache(root)
    if cached is not None:
        return load_odds_cache(cached), f"cache:{cached}"

    # Try live as a convenience when no cache exists.
    try:
        events = fetch_nba_odds(http_get=http_get)
        return events, "live:the-odds-api"
    except OddsApiConfigError as exc:
        return [], f"no-odds:{exc}"


def slate_edges(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    odds_path: Path | None = None,
    use_live: bool = False,
    min_edge: float = 0.0,
    n_sims: int = 10_000,
    http_get=None,
    schedule=None,
    team_logs=None,
) -> tuple[list[EdgeRow], str]:
    """Score the date and join to odds → edge rows + status message."""
    preds = score_date(
        season=season,
        game_date=game_date,
        root=root,
        n_sims=n_sims,
        schedule=schedule,
        team_logs=team_logs,
    )
    events, source = load_events_for_edges(
        root=root, odds_path=odds_path, use_live=use_live, http_get=http_get
    )
    if not events:
        return [], source
    rows = build_edge_rows(preds, events, n_sims=n_sims)
    if min_edge > 0:
        rows = [r for r in rows if r.edge >= min_edge]
    return rows, source


def write_edges_json(rows: list[EdgeRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in rows], indent=2) + "\n", encoding="utf-8")
    return path


def format_edges_report(rows: list[EdgeRow], source: str, *, min_edge: float = 0.0) -> str:
    header = f"odds source: {source}"
    if source.startswith("no-odds:"):
        return (
            f"{header}\n"
            "No Odds API key / cache available — model-only. "
            "Set ODDS_API_KEY or run `nba-betting odds pull` after configuring .env."
        )
    return f"{header}\n{format_edge_table(rows, min_edge=min_edge)}"
