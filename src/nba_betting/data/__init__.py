"""Schedule, box scores, and (later) odds adapters."""

from nba_betting.data.boxscores import load_boxscores, pull_boxscores
from nba_betting.data.paths import boxscore_path, data_root, schedule_path
from nba_betting.data.schedule import load_schedule, pull_schedule

__all__ = [
    "boxscore_path",
    "data_root",
    "load_boxscores",
    "load_schedule",
    "pull_boxscores",
    "pull_schedule",
    "schedule_path",
]
