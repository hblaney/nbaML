"""Schedule, box scores, and odds adapters."""

from nba_betting.data.boxscores import load_boxscores, pull_boxscores
from nba_betting.data.odds import EventOdds, fetch_nba_odds, pull_odds
from nba_betting.data.paths import boxscore_path, data_root, schedule_path
from nba_betting.data.schedule import load_schedule, pull_schedule

__all__ = [
    "EventOdds",
    "boxscore_path",
    "data_root",
    "fetch_nba_odds",
    "load_boxscores",
    "load_schedule",
    "pull_boxscores",
    "pull_odds",
    "pull_schedule",
    "schedule_path",
]
