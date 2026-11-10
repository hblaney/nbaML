"""Blind prediction ledger: lock pre-tip, grade after finals (never overwrite locks)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nba_betting.backtest.walkforward import grade_prediction_row
from nba_betting.data.paths import data_root
from nba_betting.data.schedule import load_schedule
from nba_betting.models.game import GamePrediction
from nba_betting.slate.score import score_date


def ledger_path(root: Path | None = None) -> Path:
    return data_root(root) / "blind_predictions.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ledger(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def lock_predictions(
    predictions: list[GamePrediction],
    *,
    root: Path | None = None,
    path: Path | None = None,
    locked_at: str | None = None,
) -> tuple[int, Path]:
    """Append new locks; skip game_ids already locked. Never mutates prior locks."""
    out = Path(path) if path is not None else ledger_path(root)
    existing = read_ledger(out)
    # Any existing game_id stays locked forever (grade may add fields later).
    locked_ids = {str(r.get("game_id")) for r in existing}
    stamp = locked_at or _now()
    added = 0
    for pred in predictions:
        if pred.game_id in locked_ids:
            continue
        row = pred.to_dict()
        row.update(
            {
                "status": "locked",
                "locked_at": stamp,
                "graded_at": None,
                "home_score": None,
                "away_score": None,
                "ml_correct": None,
                "spread_correct": None,
                "total_correct": None,
            }
        )
        existing.append(row)
        locked_ids.add(pred.game_id)
        added += 1
    _write_ledger(out, existing)
    return added, out


def lock_date(
    *,
    season: str,
    game_date: str,
    root: Path | None = None,
    path: Path | None = None,
    n_sims: int = 10_000,
    schedule: pd.DataFrame | None = None,
    team_logs: pd.DataFrame | None = None,
) -> tuple[int, Path]:
    preds = score_date(
        season=season,
        game_date=game_date,
        root=root,
        n_sims=n_sims,
        schedule=schedule,
        team_logs=team_logs,
    )
    return lock_predictions(preds, root=root, path=path)


def grade_ledger(
    *,
    season: str,
    root: Path | None = None,
    path: Path | None = None,
    schedule: pd.DataFrame | None = None,
) -> tuple[int, Path]:
    """Fill finals onto locked rows; never changes prediction fields / locked_at."""
    out = Path(path) if path is not None else ledger_path(root)
    rows = read_ledger(out)
    if not rows:
        return 0, out

    sched = schedule if schedule is not None else load_schedule(season, root=root)
    by_id = {
        str(r["game_id"]): r
        for _, r in sched.iterrows()
        if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score"))
    }

    graded = 0
    for row in rows:
        if row.get("graded_at"):
            continue
        game_id = str(row.get("game_id"))
        if game_id not in by_id:
            continue
        game = by_id[game_id]
        # Preserve original lock payload; only fill grade fields.
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        grade_input = pd.Series(
            {
                **row,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": float(home_score > away_score),
                "margin": home_score - away_score,
                "total_pts": home_score + away_score,
            }
        )
        result = grade_prediction_row(grade_input)
        row["home_score"] = home_score
        row["away_score"] = away_score
        row["home_win"] = float(home_score > away_score)
        row["margin"] = home_score - away_score
        row["total_pts"] = home_score + away_score
        row["ml_pick_home"] = result["ml_pick_home"]
        row["p_pick"] = result["p_pick"]
        row["ml_correct"] = result["ml_correct"]
        row["spread_correct"] = result["spread_correct"]
        row["total_correct"] = result["total_correct"]
        row["spread_line_used"] = result["spread_line_used"]
        row["total_line_used"] = result["total_line_used"]
        row["status"] = "graded"
        row["graded_at"] = _now()
        graded += 1

    _write_ledger(out, rows)
    return graded, out
