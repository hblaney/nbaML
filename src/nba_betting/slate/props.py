"""Score a slate date's player props from cached player logs (+ optional odds)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_betting.data.boxscores import load_boxscores
from nba_betting.data.odds import OddsApiConfigError
from nba_betting.data.prop_odds import (
    PropOdds,
    fetch_nba_prop_odds,
    latest_prop_odds_cache,
    load_prop_odds_cache,
)
from nba_betting.data.schedule import load_schedule
from nba_betting.features.player_priors import (
    prepare_player_logs,
    priors_for_teams,
)
from nba_betting.models.props import PropProjection, project_players
from nba_betting.pricing.prop_edge import (
    PropEdgeRow,
    build_prop_edge_rows,
    format_prop_edge_table,
)


def _schedule_for_date(schedule: pd.DataFrame, game_date: str) -> pd.DataFrame:
    target = pd.Timestamp(game_date).normalize()
    dates = pd.to_datetime(schedule["game_date_est"]).dt.normalize()
    return schedule.loc[dates == target].copy()


def score_props_date(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    player_logs: pd.DataFrame | None = None,
    min_exp_minutes: float = 12.0,
    window: int = 10,
) -> list[PropProjection]:
    """Project PTS/REB/AST/3PM/PRA for slate teams using as-of minutes priors."""
    sched = schedule if schedule is not None else load_schedule(season, root=root)
    logs = (
        player_logs
        if player_logs is not None
        else load_boxscores(season, level="player", root=root)
    )
    prepared = prepare_player_logs(logs)
    day = _schedule_for_date(sched, game_date)
    out: list[PropProjection] = []
    for _, game in day.iterrows():
        team_ids = {int(game["home_team_id"]), int(game["away_team_id"])}
        priors = priors_for_teams(
            prepared,
            team_ids,
            game_date,
            window=window,
            min_exp_minutes=min_exp_minutes,
        )
        out.extend(
            project_players(
                priors,
                game_id=str(game["game_id"]),
                game_date=str(pd.Timestamp(game_date).date()),
            )
        )
    return out


def format_prop_projections(projections: list[PropProjection]) -> str:
    if not projections:
        return "(no player projections — need player box cache + slate games)"
    df = pd.DataFrame([p.to_dict() for p in projections])
    cols = [
        "game_id",
        "player_name",
        "team_abbreviation",
        "exp_minutes",
        "points",
        "rebounds",
        "assists",
        "threes",
        "pra",
    ]
    view = df[cols].sort_values(["game_id", "exp_minutes"], ascending=[True, False])
    for c in ["exp_minutes", "points", "rebounds", "assists", "threes", "pra"]:
        view[c] = view[c].map(lambda x: f"{x:.1f}")
    return view.to_string(index=False)


def load_prop_events_for_edges(
    *,
    root: Path | None = None,
    odds_path: Path | None = None,
    use_live: bool = False,
    http_get=None,
) -> tuple[list[PropOdds], str]:
    if odds_path is not None:
        return load_prop_odds_cache(odds_path), f"cache:{odds_path}"
    if use_live:
        try:
            return fetch_nba_prop_odds(http_get=http_get), "live:the-odds-api-props"
        except OddsApiConfigError as exc:
            return [], f"no-odds:{exc}"
    cached = latest_prop_odds_cache(root)
    if cached is not None:
        return load_prop_odds_cache(cached), f"cache:{cached}"
    try:
        return fetch_nba_prop_odds(http_get=http_get), "live:the-odds-api-props"
    except OddsApiConfigError as exc:
        return [], f"no-odds:{exc}"


def slate_prop_edges(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    odds_path: Path | None = None,
    use_live: bool = False,
    min_edge: float = 0.0,
    http_get=None,
    schedule: pd.DataFrame | None = None,
    player_logs: pd.DataFrame | None = None,
) -> tuple[list[PropEdgeRow], list[PropProjection], str]:
    projections = score_props_date(
        season=season,
        game_date=game_date,
        root=root,
        schedule=schedule,
        player_logs=player_logs,
    )
    events, source = load_prop_events_for_edges(
        root=root, odds_path=odds_path, use_live=use_live, http_get=http_get
    )
    if not events:
        return [], projections, source
    rows = build_prop_edge_rows(projections, events)
    if min_edge > 0:
        rows = [
            r
            for r in rows
            if (r.edge_over or 0) >= min_edge or (r.edge_under or 0) >= min_edge
        ]
    return rows, projections, source


def format_prop_edges_report(
    rows: list[PropEdgeRow], source: str, *, min_edge: float = 0.0
) -> str:
    header = f"prop odds source: {source}"
    if source.startswith("no-odds:"):
        return (
            f"{header}\n"
            "No Odds API key / prop cache — model projections only. "
            "Set ODDS_API_KEY or run `nba-betting odds pull-props`."
        )
    return f"{header}\n{format_prop_edge_table(rows, min_edge=min_edge)}"


def write_prop_projections_json(projections: list[PropProjection], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.to_dict() for p in projections], indent=2) + "\n", encoding="utf-8"
    )
    return path


def write_prop_edges_json(rows: list[PropEdgeRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in rows], indent=2) + "\n", encoding="utf-8"
    )
    return path
