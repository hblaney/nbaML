"""Daily slate: build board, edges, lock, grade."""

from nba_betting.slate.edges import format_edges_report, slate_edges
from nba_betting.slate.ledger import grade_ledger, lock_date
from nba_betting.slate.props import format_prop_projections, score_props_date, slate_prop_edges
from nba_betting.slate.score import format_predictions_table, score_date

__all__ = [
    "format_edges_report",
    "format_predictions_table",
    "format_prop_projections",
    "grade_ledger",
    "lock_date",
    "score_date",
    "score_props_date",
    "slate_edges",
    "slate_prop_edges",
]
