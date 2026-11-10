"""CLI stub — no live API calls or models yet."""

from __future__ import annotations

import typer

from nba_betting import __version__

app = typer.Typer(
    name="nba-betting",
    help="NBA game-market pricing and daily slate CLI (skeleton).",
    no_args_is_help=True,
)
slate_app = typer.Typer(help="Daily slate commands (stub).")
app.add_typer(slate_app, name="slate")


@app.command("version")
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@slate_app.command("build")
def slate_build() -> None:
    """Build today's board (not implemented)."""
    typer.echo("slate build: not implemented yet (skeleton)")
    raise typer.Exit(code=0)


@slate_app.callback()
def slate_callback() -> None:
    """Daily slate: build board, lock, grade (stubs only)."""
