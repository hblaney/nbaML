"""Score a slate date's player props from cached player logs (+ optional odds)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_betting.data.boxscores import load_boxscores
from nba_betting.data.odds import EventOdds, OddsApiConfigError, latest_odds_cache, load_odds_cache
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
from nba_betting.features.prop_redistribution import build_tracker_asof
from nba_betting.models.prop_distribution import (
    DistributionalPropProjection,
    project_distributional_props,
)
from nba_betting.models.props import PropProjection, project_players
from nba_betting.pricing.prop_edge import (
    AnyPropProjection,
    PropEdgeRow,
    build_prop_edge_rows,
    format_prop_edge_table,
)
from nba_betting.pricing.prop_parlay import (
    build_eligible_prop_legs,
    enumerate_prop_parlays,
    select_nonoverlapping_prop_parlays,
)


def _schedule_for_date(schedule: pd.DataFrame, game_date: str) -> pd.DataFrame:
    target = pd.Timestamp(game_date).normalize()
    dates = pd.to_datetime(schedule["game_date_est"]).dt.normalize()
    return schedule.loc[dates == target].copy()


def _lookup_game_lines(
    game: pd.Series,
    event_odds: list[EventOdds] | None,
    *,
    default_spread: float = 0.0,
    default_total: float = 226.5,
) -> tuple[float, float]:
    """Return ``(home_spread, total_line)`` for a scheduled game when odds cache exists."""
    sp = (
        float(game.get("spread_home"))
        if "spread_home" in game and pd.notna(game.get("spread_home"))
        else None
    )
    tot = (
        float(game.get("total_line"))
        if "total_line" in game and pd.notna(game.get("total_line"))
        else None
    )
    if sp is not None and tot is not None:
        return sp, tot

    if event_odds:
        htri = str(game.get("home_team_tricode") or game.get("home_tricode") or "").upper()
        atri = str(game.get("away_team_tricode") or game.get("away_tricode") or "").upper()
        for eo in event_odds:
            if (
                htri
                and atri
                and (eo.home_tricode or "").upper() == htri
                and (eo.away_tricode or "").upper() == atri
            ):
                s_val = (
                    float(eo.spread_home_point)
                    if eo.spread_home_point is not None
                    else default_spread
                )
                t_val = float(eo.total_point) if eo.total_point is not None else default_total
                return s_val, t_val
    return (
        sp if sp is not None else float(default_spread),
        tot if tot is not None else float(default_total),
    )


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
    """Project PTS/REB/AST/3PM/PRA for slate teams using as-of minutes priors (v1.5 baseline)."""
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


def score_distributional_props_date(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    schedule: pd.DataFrame | None = None,
    player_logs: pd.DataFrame | None = None,
    event_odds: list[EventOdds] | None = None,
    min_exp_minutes: float = 12.0,
    min_games: int = 3,
    default_spread: float = 0.0,
    default_total: float = 226.5,
) -> list[DistributionalPropProjection]:
    """Project PTS/REB/AST/3PM/PRA with non-linear redistribution + bimodal blowout mixtures."""
    sched = schedule if schedule is not None else load_schedule(season, root=root)
    logs = (
        player_logs
        if player_logs is not None
        else load_boxscores(season, level="player", root=root)
    )
    if event_odds is None:
        cached_odds = latest_odds_cache(root)
        if cached_odds is not None:
            event_odds = load_odds_cache(cached_odds)

    tracker = build_tracker_asof(logs, game_date)
    day = _schedule_for_date(sched, game_date)
    date_str = str(pd.Timestamp(game_date).date())
    out: list[DistributionalPropProjection] = []

    for _, game in day.iterrows():
        gid = str(game["game_id"])
        hid = int(game["home_team_id"])
        aid = int(game["away_team_id"])
        sp_home, tot_line = _lookup_game_lines(
            game, event_odds, default_spread=default_spread, default_total=default_total
        )

        for tid, opp_tid, tsp in ((hid, aid, sp_home), (aid, hid, -sp_home)):
            team_rates = tracker.project_team_pregame(
                tid,
                opp_tid,
                game_date,
                team_spread=tsp,
                total_line=tot_line,
                min_games=min_games,
            )
            for rates in team_rates.values():
                if rates.adj_minutes < float(min_exp_minutes):
                    continue
                out.append(
                    project_distributional_props(
                        rates,
                        game_id=gid,
                        game_date=date_str,
                        team_spread=tsp,
                        total_line=tot_line,
                    )
                )
    return out


def format_prop_projections(projections: list[AnyPropProjection]) -> str:
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
    consensus: bool = False,
    http_get=None,
) -> tuple[list[PropOdds], str]:
    if odds_path is not None:
        return load_prop_odds_cache(odds_path), f"cache:{odds_path}"
    if use_live:
        try:
            return (
                fetch_nba_prop_odds(http_get=http_get, consensus=consensus),
                "live:the-odds-api-props",
            )
        except OddsApiConfigError as exc:
            return [], f"no-odds:{exc}"
    cached = latest_prop_odds_cache(root)
    if cached is not None:
        return load_prop_odds_cache(cached), f"cache:{cached}"
    try:
        return (
            fetch_nba_prop_odds(http_get=http_get, consensus=consensus),
            "live:the-odds-api-props",
        )
    except OddsApiConfigError as exc:
        return [], f"no-odds:{exc}"


def slate_prop_edges(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    odds_path: Path | None = None,
    use_live: bool = False,
    consensus: bool = False,
    distributional: bool = False,
    min_edge: float = 0.0,
    http_get=None,
    schedule: pd.DataFrame | None = None,
    player_logs: pd.DataFrame | None = None,
) -> tuple[list[PropEdgeRow], list[AnyPropProjection], str]:
    if distributional:
        projections: list[AnyPropProjection] = list(
            score_distributional_props_date(
                season=season,
                game_date=game_date,
                root=root,
                schedule=schedule,
                player_logs=player_logs,
            )
        )
    else:
        projections = list(
            score_props_date(
                season=season,
                game_date=game_date,
                root=root,
                schedule=schedule,
                player_logs=player_logs,
            )
        )
    events, source = load_prop_events_for_edges(
        root=root,
        odds_path=odds_path,
        use_live=use_live,
        consensus=consensus,
        http_get=http_get,
    )
    if not events:
        return [], projections, source
    rows = build_prop_edge_rows(projections, events)
    if min_edge > 0:
        rows = [
            r for r in rows if (r.edge_over or 0) >= min_edge or (r.edge_under or 0) >= min_edge
        ]
    return rows, projections, source


def slate_prop_parlays(
    rows: list[PropEdgeRow],
    *,
    sizes: tuple[int, ...] = (2,),
    min_price_edge: float = 0.04,
    max_per_day: int = 2,
) -> pd.DataFrame:
    """Build cross-game 2/3-leg prop parlays from live or cached ``PropEdgeRow`` items."""
    legs = build_eligible_prop_legs(rows, y_over_col=None, require_vs_vigged=True)
    cands = enumerate_prop_parlays(legs, sizes=sizes, min_price_edge=min_price_edge)
    return select_nonoverlapping_prop_parlays(cands, max_per_day=max_per_day)


def format_prop_edges_report(rows: list[PropEdgeRow], source: str, *, min_edge: float = 0.0) -> str:
    header = f"prop odds source: {source}"
    if source.startswith("no-odds:"):
        return (
            f"{header}\n"
            "No Odds API key / prop cache — model projections only. "
            "Set ODDS_API_KEY or run `nba-betting odds pull-props`."
        )
    return f"{header}\n{format_prop_edge_table(rows, min_edge=min_edge)}"


def write_prop_projections_json(projections: list[AnyPropProjection], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.to_dict() for p in projections], indent=2) + "\n", encoding="utf-8"
    )
    return path


def write_prop_edges_json(rows: list[PropEdgeRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in rows], indent=2) + "\n", encoding="utf-8")
    return path
