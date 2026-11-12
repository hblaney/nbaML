"""Schedule, box scores, and odds adapters."""

from nba_betting.data.boxscores import load_boxscores, pull_boxscores
from nba_betting.data.espn_closing import (
    fetch_event_closing_markets,
    fetch_event_closing_mls,
    grade_closing_markets,
    pull_closing_markets,
    pull_closing_moneylines,
)
from nba_betting.data.odds import EventOdds, fetch_nba_odds, pull_odds
from nba_betting.data.paths import boxscore_path, data_root, schedule_path
from nba_betting.data.prop_odds import PropOdds, fetch_nba_prop_odds, pull_prop_odds
from nba_betting.data.schedule import load_schedule, pull_schedule

__all__ = [
    "EventOdds",
    "PropOdds",
    "boxscore_path",
    "data_root",
    "fetch_event_closing_markets",
    "fetch_event_closing_mls",
    "fetch_nba_odds",
    "fetch_nba_prop_odds",
    "grade_closing_markets",
    "load_boxscores",
    "load_schedule",
    "pull_boxscores",
    "pull_closing_markets",
    "pull_closing_moneylines",
    "pull_odds",
    "pull_prop_odds",
    "pull_schedule",
    "schedule_path",
]
