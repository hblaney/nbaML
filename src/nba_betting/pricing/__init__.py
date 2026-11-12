"""De-vig, fair odds, edge, Kelly sizing, parlay EV selector."""

from nba_betting.pricing.american import american_to_implied, implied_to_american
from nba_betting.pricing.copula_sgp import (
    sgp_prob_t_copula,
    simulate_blowout_ot_regime_sgp,
    t_copula_rect_prob,
    t_copula_tail_dependence,
)
from nba_betting.pricing.devig import devig_power, devig_shin, devig_shin_two_way
from nba_betting.pricing.edge import EdgeRow, build_edge_rows, format_edge_table
from nba_betting.pricing.kelly import kelly_fraction, stake_amount
from nba_betting.pricing.multimarket_parlay import (
    build_multimarket_legs,
    enumerate_multimarket_parlays,
    fast_seasonal_kelly_summary,
    kelly_parlay_stake_fraction,
    select_daily_multimarket_tickets,
    settle_parlay_legs,
    simulate_seasonal_kelly_bankroll,
)
from nba_betting.pricing.parlay_ev import (
    build_eligible_legs,
    clears_parlay_price,
    enumerate_parlays,
    enumerate_same_day_parlays,
    grade_tickets,
    parlay_decimal,
    parlay_ev_per_unit,
    run_parlay_selector,
    select_nonoverlapping_tickets,
    select_top_edge_tickets,
    summarize_ticket_pnl,
)
from nba_betting.pricing.prop_edge import PropEdgeRow, build_prop_edge_rows

__all__ = [
    "EdgeRow",
    "PropEdgeRow",
    "american_to_implied",
    "build_edge_rows",
    "build_eligible_legs",
    "build_multimarket_legs",
    "build_prop_edge_rows",
    "clears_parlay_price",
    "devig_power",
    "devig_shin",
    "devig_shin_two_way",
    "enumerate_multimarket_parlays",
    "enumerate_parlays",
    "enumerate_same_day_parlays",
    "fast_seasonal_kelly_summary",
    "format_edge_table",
    "grade_tickets",
    "implied_to_american",
    "kelly_fraction",
    "kelly_parlay_stake_fraction",
    "parlay_decimal",
    "parlay_ev_per_unit",
    "run_parlay_selector",
    "select_daily_multimarket_tickets",
    "select_nonoverlapping_tickets",
    "select_top_edge_tickets",
    "settle_parlay_legs",
    "sgp_prob_t_copula",
    "simulate_blowout_ot_regime_sgp",
    "simulate_seasonal_kelly_bankroll",
    "stake_amount",
    "summarize_ticket_pnl",
    "t_copula_rect_prob",
    "t_copula_tail_dependence",
]
