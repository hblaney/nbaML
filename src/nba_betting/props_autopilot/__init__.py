"""24/7 Autonomous NBA Player Props & PrizePicks Scanner, Auto-Grader, and R&D Lab."""

from nba_betting.props_autopilot.prizepicks_math import (
    STAT_TAIL_CALIBRATION_22_24,
    apply_stat_tail_calibration,
    evaluate_live_cashout,
    exact_hits_pmf,
    settle_prizepicks_slip,
    slip_ev_and_log_growth,
)

__all__ = [
    "STAT_TAIL_CALIBRATION_22_24",
    "apply_stat_tail_calibration",
    "evaluate_live_cashout",
    "exact_hits_pmf",
    "settle_prizepicks_slip",
    "slip_ev_and_log_growth",
]
