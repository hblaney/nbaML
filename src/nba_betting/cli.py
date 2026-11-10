"""CLI — data ingest + slate stubs. No odds or models yet."""

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
slate_app = typer.Typer(help="Daily slate commands (stub).")
app.add_typer(data_app, name="data")
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


@slate_app.command("build")
def slate_build() -> None:
    """Build today's board (not implemented)."""
    typer.echo("slate build: not implemented yet")
    raise typer.Exit(code=0)


@slate_app.callback()
def slate_callback() -> None:
    """Daily slate: build board, lock, grade (stubs only)."""
