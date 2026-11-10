"""CLI — data ingest, model scoring, slate helpers."""

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
model_app = typer.Typer(help="Game score model commands.")
slate_app = typer.Typer(help="Daily slate commands.")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(slate_app, name="slate")


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
    """Alias: score today's (or any) slate date from cache."""
    _score_date_impl(
        season=season,
        game_date=game_date,
        data_dir=data_dir,
        spread=spread,
        total=total,
        n_sims=n_sims,
        out=out,
    )


@slate_app.command("build")
def slate_build() -> None:
    """Build today's board artifact (use ``slate score`` for model board)."""
    typer.echo("slate build: use `nba-betting slate score --season … --date …`")
    raise typer.Exit(code=0)


@slate_app.callback()
def slate_callback() -> None:
    """Daily slate: score, lock, grade."""
