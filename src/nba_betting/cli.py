"""CLI — data ingest, odds, model scoring, slate edges."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from nba_betting import __version__

app = typer.Typer(
    name="nba-betting",
    help="NBA game-market pricing and daily slate CLI.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Schedule and box-score ingest (nba_api).")
odds_app = typer.Typer(help="The Odds API adapter (NBA featured markets).")
model_app = typer.Typer(help="Game score model commands.")
slate_app = typer.Typer(help="Daily slate commands.")
backtest_app = typer.Typer(help="Walk-forward backtest.")
ledger_app = typer.Typer(help="Blind lock / grade ledger.")
props_app = typer.Typer(help="Player prop projections and edges (v1.5).")
app.add_typer(data_app, name="data")
app.add_typer(odds_app, name="odds")
app.add_typer(model_app, name="model")
app.add_typer(slate_app, name="slate")
app.add_typer(backtest_app, name="backtest")
app.add_typer(ledger_app, name="ledger")
app.add_typer(props_app, name="props")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


@app.callback()
def main_callback() -> None:
    """Load ``.env`` when present (never required)."""
    _load_dotenv()


@app.command("version")
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@data_app.command("pull-schedule")
def pull_schedule_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    force: bool = typer.Option(False, "--force", help="Re-fetch even if cache exists"),
) -> None:
    """Pull season schedule via nba_api ScheduleLeagueV2 and cache locally."""
    from nba_betting.data.schedule import pull_schedule

    path = pull_schedule(season, root=data_dir, force=force)
    typer.echo(f"schedule cached: {path}")


@data_app.command("pull-boxes")
def pull_boxes_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    level: str = typer.Option(
        "team", "--level", help="team (default) or player LeagueGameLog rows"
    ),
    force: bool = typer.Option(False, "--force", help="Re-fetch even if cache exists"),
) -> None:
    """Pull season team/player game logs (box lines) via nba_api LeagueGameLog."""
    from nba_betting.data.boxscores import pull_boxscores

    path = pull_boxscores(season, root=data_dir, level=level, force=force)
    typer.echo(f"boxscores cached: {path}")


@data_app.command("pull")
def pull_all_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    force: bool = typer.Option(False, "--force", help="Re-fetch even if cache exists"),
) -> None:
    """Pull schedule + team box/game logs for a season."""
    from nba_betting.data.boxscores import pull_boxscores
    from nba_betting.data.schedule import pull_schedule

    sched = pull_schedule(season, root=data_dir, force=force)
    boxes = pull_boxscores(season, root=data_dir, level="team", force=force)
    typer.echo(f"schedule cached: {sched}")
    typer.echo(f"boxscores cached: {boxes}")


@odds_app.command("pull")
def odds_pull_cmd(
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
) -> None:
    """Pull NBA featured odds (h2h, spreads, totals) from The Odds API."""
    from nba_betting.data.odds import OddsApiConfigError, pull_odds

    try:
        path = pull_odds(root=data_dir)
    except OddsApiConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=0) from exc
    typer.echo(f"odds cached: {path}")


@odds_app.command("pull-props")
def odds_pull_props_cmd(
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
) -> None:
    """Pull NBA player prop odds (per-event; paid Odds API tier may be required)."""
    from nba_betting.data.odds import OddsApiConfigError
    from nba_betting.data.prop_odds import pull_prop_odds

    try:
        path = pull_prop_odds(root=data_dir)
    except OddsApiConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=0) from exc
    typer.echo(f"prop odds cached: {path}")


def _score_date_impl(
    *,
    season: str,
    game_date: str,
    data_dir: Path | None,
    spread: float,
    total: float | None,
    n_sims: int,
    out: Path | None,
) -> None:
    from nba_betting.slate.score import (
        format_predictions_table,
        score_date,
        write_predictions_json,
    )

    preds = score_date(
        season=season,
        game_date=game_date,
        root=data_dir,
        market_spread=spread,
        market_total=total,
        n_sims=n_sims,
    )
    typer.echo(format_predictions_table(preds))
    if out is not None:
        path = write_predictions_json(preds, out)
        typer.echo(f"wrote {path}")


@model_app.command("score")
def model_score_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    game_date: str = typer.Option(..., "--date", help="Game date YYYY-MM-DD"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    spread: float = typer.Option(
        0.0, "--spread", help="Home market spread (e.g. -3.5); default 0"
    ),
    total: Optional[float] = typer.Option(
        None, "--total", help="Market total; default model expected total"
    ),
    n_sims: int = typer.Option(20000, "--n-sims", help="Monte Carlo simulations"),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path"),
) -> None:
    """Score a date from cached schedule/boxes (offline)."""
    _score_date_impl(
        season=season,
        game_date=game_date,
        data_dir=data_dir,
        spread=spread,
        total=total,
        n_sims=n_sims,
        out=out,
    )


@slate_app.command("score")
def slate_score_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    game_date: str = typer.Option(..., "--date", help="Game date YYYY-MM-DD"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    spread: float = typer.Option(0.0, "--spread", help="Home market spread; default 0"),
    total: Optional[float] = typer.Option(
        None, "--total", help="Market total; default model expected total"
    ),
    n_sims: int = typer.Option(20000, "--n-sims", help="Monte Carlo simulations"),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path"),
) -> None:
    """Score a slate date from cache (model board)."""
    _score_date_impl(
        season=season,
        game_date=game_date,
        data_dir=data_dir,
        spread=spread,
        total=total,
        n_sims=n_sims,
        out=out,
    )


@slate_app.command("edges")
def slate_edges_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    game_date: str = typer.Option(..., "--date", help="Game date YYYY-MM-DD"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    odds_file: Optional[Path] = typer.Option(
        None, "--odds-file", help="Odds JSON cache (default: latest under data/raw/odds)"
    ),
    live: bool = typer.Option(
        False, "--live", help="Fetch Odds API live (requires ODDS_API_KEY)"
    ),
    min_edge: float = typer.Option(
        0.0, "--min-edge", help="Only show edges >= this (e.g. 0.03)"
    ),
    n_sims: int = typer.Option(10000, "--n-sims", help="Monte Carlo simulations"),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path"),
) -> None:
    """Join model probs to book lines → edge table (degrades if no odds key/cache)."""
    from nba_betting.slate.edges import format_edges_report, slate_edges, write_edges_json

    rows, source = slate_edges(
        season=season,
        game_date=game_date,
        root=data_dir,
        odds_path=odds_file,
        use_live=live,
        min_edge=min_edge,
        n_sims=n_sims,
    )
    typer.echo(format_edges_report(rows, source, min_edge=min_edge))
    if out is not None:
        path = write_edges_json(rows, out)
        typer.echo(f"wrote {path}")


@slate_app.command("build")
def slate_build() -> None:
    """Build today's board artifact (use ``slate score`` / ``slate edges``)."""
    typer.echo("slate build: use `nba-betting slate score` or `slate edges`")
    raise typer.Exit(code=0)


@slate_app.callback()
def slate_callback() -> None:
    """Daily slate: score, edges, lock, grade."""


@backtest_app.command("run")
def backtest_run_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    start_date: Optional[str] = typer.Option(None, "--start", help="First date YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, "--end", help="Last date YYYY-MM-DD"),
    n_sims: int = typer.Option(5000, "--n-sims", help="Sims per game"),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Optional graded predictions CSV/JSON path"
    ),
) -> None:
    """Walk-forward backtest on cached schedule/boxes (Phase A: results-only)."""
    import json

    from nba_betting.backtest.walkforward import format_backtest_report, run_walkforward

    graded, metrics = run_walkforward(
        season=season,
        root=data_dir,
        start_date=start_date,
        end_date=end_date,
        n_sims=n_sims,
    )
    typer.echo(format_backtest_report(graded, metrics))
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            payload = {"metrics": metrics, "games": graded.to_dict(orient="records")}
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            graded.to_csv(out, index=False)
        typer.echo(f"wrote {out}")


@ledger_app.command("lock")
def ledger_lock_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    game_date: str = typer.Option(..., "--date", help="Game date YYYY-MM-DD"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    n_sims: int = typer.Option(10000, "--n-sims", help="Monte Carlo simulations"),
) -> None:
    """Lock model predictions for a date into the blind ledger (no overwrite)."""
    from nba_betting.slate.ledger import lock_date

    added, path = lock_date(
        season=season, game_date=game_date, root=data_dir, n_sims=n_sims
    )
    typer.echo(f"locked {added} new game(s) → {path}")


@ledger_app.command("grade")
def ledger_grade_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
) -> None:
    """Grade locked ledger rows from cached finals; never rewrite locks."""
    from nba_betting.slate.ledger import grade_ledger

    n, path = grade_ledger(season=season, root=data_dir)
    typer.echo(f"graded {n} row(s) → {path}")


@props_app.command("score")
def props_score_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    game_date: str = typer.Option(..., "--date", help="Game date YYYY-MM-DD"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path"),
) -> None:
    """Project PTS/REB/AST/3PM/PRA for a slate date from cached player logs."""
    from nba_betting.slate.props import (
        format_prop_projections,
        score_props_date,
        write_prop_projections_json,
    )

    projections = score_props_date(season=season, game_date=game_date, root=data_dir)
    typer.echo(format_prop_projections(projections))
    if out is not None:
        path = write_prop_projections_json(projections, out)
        typer.echo(f"wrote {path}")


@props_app.command("edges")
def props_edges_cmd(
    season: str = typer.Option(..., "--season", help="Season like 2024-25"),
    game_date: str = typer.Option(..., "--date", help="Game date YYYY-MM-DD"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data root (default: ./data)"
    ),
    odds_file: Optional[Path] = typer.Option(
        None, "--odds-file", help="Prop odds JSON cache (default: latest nba_props_*)"
    ),
    live: bool = typer.Option(
        False, "--live", help="Fetch prop odds live (requires ODDS_API_KEY)"
    ),
    min_edge: float = typer.Option(0.0, "--min-edge", help="Min edge to display"),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional JSON output path"),
) -> None:
    """Join prop projections to book lines → edge table (degrades without key)."""
    from nba_betting.slate.props import (
        format_prop_edges_report,
        slate_prop_edges,
        write_prop_edges_json,
    )

    rows, _projections, source = slate_prop_edges(
        season=season,
        game_date=game_date,
        root=data_dir,
        odds_path=odds_file,
        use_live=live,
        min_edge=min_edge,
    )
    typer.echo(format_prop_edges_report(rows, source, min_edge=min_edge))
    if out is not None:
        path = write_prop_edges_json(rows, out)
        typer.echo(f"wrote {path}")
