"""Join prop projections to book prop lines and compute edges."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Union

import pandas as pd

from nba_betting.data.prop_odds import PropOdds
from nba_betting.models.prop_distribution import DistributionalPropProjection
from nba_betting.models.props import PropProjection, p_over_poisson
from nba_betting.pricing.american import devig_two_way, implied_to_american

AnyPropProjection = Union[PropProjection, DistributionalPropProjection]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(name: str) -> str:
    tokens = str(name or "").lower().strip().split()
    cleaned = [re.sub(r"[^a-z]", "", t) for t in tokens if re.sub(r"[^a-z]", "", t)]
    if cleaned and cleaned[-1] in _SUFFIXES:
        cleaned = cleaned[:-1]
    return " ".join(cleaned)


@dataclass(frozen=True)
class PropEdgeRow:
    game_id: str
    game_date: str
    player_name: str
    team_abbreviation: str
    market: str
    stat: str
    line: float
    model_exp: float
    model_p_over: float
    model_p_under: float
    fair_over_american: float
    fair_under_american: float
    book_over: float | None
    book_under: float | None
    book_implied_over_fair: float | None
    book_implied_under_fair: float | None
    edge_over: float | None
    edge_under: float | None
    book: str
    event_id: str
    player_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _match_prop(proj: AnyPropProjection, odds: list[PropOdds], *, stat: str) -> PropOdds | None:
    target = _norm_name(proj.player_name)
    for row in odds:
        if row.stat != stat:
            continue
        if _norm_name(row.player_name) == target:
            return row
    # Loose: last-name match when unambiguous in odds list
    last = target.split()[-1] if target else ""
    if last:
        candidates = [
            row for row in odds if row.stat == stat and last in _norm_name(row.player_name).split()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return candidates[0]
    return None


def _prob_over_for_projection(proj: AnyPropProjection, stat: str, line: float) -> float:
    if isinstance(proj, DistributionalPropProjection):
        return proj.p_over(stat, line)
    return p_over_poisson(proj.expected(stat), line)


def build_prop_edge_rows(
    projections: list[AnyPropProjection],
    odds: list[PropOdds],
) -> list[PropEdgeRow]:
    rows: list[PropEdgeRow] = []
    for proj in projections:
        for stat in ("points", "rebounds", "assists", "threes", "pra"):
            book = _match_prop(proj, odds, stat=stat)
            if book is None:
                continue
            if book.over_price is None or book.under_price is None:
                continue
            exp = proj.expected(stat)
            p_over = _prob_over_for_projection(proj, stat, book.line)
            p_under = 1.0 - p_over
            fair_over, fair_under = devig_two_way(book.over_price, book.under_price)
            rows.append(
                PropEdgeRow(
                    game_id=str(proj.game_id),
                    game_date=str(proj.game_date),
                    player_name=proj.player_name,
                    team_abbreviation=proj.team_abbreviation,
                    market=book.market,
                    stat=stat,
                    line=float(book.line),
                    model_exp=float(exp),
                    model_p_over=float(p_over),
                    model_p_under=float(p_under),
                    fair_over_american=float(implied_to_american(min(max(p_over, 1e-6), 1 - 1e-6))),
                    fair_under_american=float(
                        implied_to_american(min(max(p_under, 1e-6), 1 - 1e-6))
                    ),
                    book_over=float(book.over_price),
                    book_under=float(book.under_price),
                    book_implied_over_fair=float(fair_over),
                    book_implied_under_fair=float(fair_under),
                    edge_over=float(p_over - fair_over),
                    edge_under=float(p_under - fair_under),
                    book=book.book,
                    event_id=book.event_id,
                    player_id=str(proj.player_id),
                )
            )
    return rows


def prop_edges_to_frame(rows: list[PropEdgeRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in rows])


def format_prop_edge_table(rows: list[PropEdgeRow], *, min_edge: float = 0.0) -> str:
    if not rows:
        return "(no matched prop odds / edges)"
    records = []
    for r in rows:
        for side, edge, model_p, book_fair, fair_am, book_am in (
            (
                "over",
                r.edge_over,
                r.model_p_over,
                r.book_implied_over_fair,
                r.fair_over_american,
                r.book_over,
            ),
            (
                "under",
                r.edge_under,
                r.model_p_under,
                r.book_implied_under_fair,
                r.fair_under_american,
                r.book_under,
            ),
        ):
            if edge is None or edge < min_edge:
                continue
            records.append(
                {
                    "player": r.player_name,
                    "team": r.team_abbreviation,
                    "stat": r.stat,
                    "line": r.line,
                    "exp": r.model_exp,
                    "side": side,
                    "model_p": model_p,
                    "book_fair": book_fair,
                    "edge": edge,
                    "fair_am": fair_am,
                    "book_am": book_am,
                    "book": r.book,
                }
            )
    if not records:
        return f"(no prop edges >= {min_edge:.1%})"
    view = pd.DataFrame(records).sort_values("edge", ascending=False)
    view["exp"] = view["exp"].map(lambda x: f"{x:.1f}")
    view["model_p"] = view["model_p"].map(lambda x: f"{x:.3f}")
    view["book_fair"] = view["book_fair"].map(lambda x: f"{x:.3f}")
    view["edge"] = view["edge"].map(lambda x: f"{x:.3f}")
    view["fair_am"] = view["fair_am"].map(lambda x: f"{x:.0f}")
    view["book_am"] = view["book_am"].map(lambda x: f"{x:.0f}")
    return view.to_string(index=False)
