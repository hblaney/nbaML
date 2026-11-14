"""24/7 Autonomous NBA Player Props & PrizePicks Scanner, Auto-Grader, Postmortem & R&D Lab."""

from nba_betting.props_autopilot.notifier import configured_channels, send_advancement_alert
from nba_betting.props_autopilot.postmortem import (
    LegAutopsy,
    LegSpec,
    analyze_leg_from_box,
    fetch_and_autopsy_leg,
    summarize_slip_autopsy,
)
from nba_betting.props_autopilot.prizepicks_math import (
    STAT_TAIL_CALIBRATION_22_24,
    apply_stat_tail_calibration,
    evaluate_live_cashout,
    exact_hits_pmf,
    settle_prizepicks_slip,
    slip_ev_and_log_growth,
)

__all__ = [
    "LegAutopsy",
    "LegSpec",
    "STAT_TAIL_CALIBRATION_22_24",
    "analyze_leg_from_box",
    "apply_stat_tail_calibration",
    "configured_channels",
    "evaluate_live_cashout",
    "exact_hits_pmf",
    "fetch_and_autopsy_leg",
    "send_advancement_alert",
    "settle_prizepicks_slip",
    "slip_ev_and_log_growth",
    "summarize_slip_autopsy",
]
