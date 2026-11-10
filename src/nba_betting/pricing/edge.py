"""Join model probabilities to book lines and compute edge rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from nba_betting.data.odds import EventOdds
from nba_betting.models.game import GamePrediction, derive_market_probs, simulate_scores
from nba_betting.pricing.american import (
    american_to_implied,
    devig_two_way,
    implied_to_american,
)


@dataclass(frozen=True)
class EdgeRow:
    game_id: str
    game_date: str
    home_tricode: str
    away_tricode: str
    market: str  # moneyline | spread | total
    side: str
    model_prob: float
    fair_american: float
    book_american: float
    book_implied_raw: float
    book_implied_fair: float
    edge: float  # model_prob - book_implied_fair
    book: str
    line: float | None
    event_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def _match_odds(
    pred: GamePrediction, events: Iterable[EventOdds]
) -> EventOdds | None:
    """Match on home/away tricodes (preferred) within the prediction set."""
    for event in events:
        if (
            event.home_tricode == pred.home_tricode
            and event.away_tricode == pred.away_tricode
        ):
            return event
    return None


def _resim_with_lines(
    pred: GamePrediction,
    *,
    market_spread: float,
    market_total: float,
    n_sims: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Re-derive cover/over probs at the book's lines from expected scores."""
    import numpy as np

    rng = np.random.default_rng(seed)
    home, away = simulate_scores(
        pred.exp_home_score, pred.exp_away_score, n_sims=n_sims, rng=rng
    )
    return derive_market_probs(
        home, away, market_spread=market_spread, market_total=market_total
    )


def build_edge_rows(
    predictions: list[GamePrediction],
    events: list[EventOdds],
    *,
    n_sims: int = 10_000,
) -> list[EdgeRow]:
    """Build edge rows for ML / spread / total where book lines exist."""
    rows: list[EdgeRow] = []
    for i, pred in enumerate(predictions):
        event = _match_odds(pred, events)
        if event is None:
            continue

        # Moneyline
        if event.moneyline_home is not None and event.moneyline_away is not None:
            fair_h, fair_a = devig_two_way(event.moneyline_home, event.moneyline_away)
            for side, model_p, book_price, book_fair in (
                ("home", pred.p_home_win, event.moneyline_home, fair_h),
                ("away", pred.p_away_win, event.moneyline_away, fair_a),
            ):
                rows.append(
                    EdgeRow(
                        game_id=pred.game_id,
                        game_date=pred.game_date,
                        home_tricode=pred.home_tricode,
                        away_tricode=pred.away_tricode,
                        market="moneyline",
                        side=side,
                        model_prob=float(model_p),
                        fair_american=float(implied_to_american(model_p)),
                        book_american=float(book_price),
                        book_implied_raw=float(american_to_implied(book_price)),
                        book_implied_fair=float(book_fair),
                        edge=float(model_p - book_fair),
                        book=event.book,
                        line=None,
                        event_id=event.event_id,
                    )
                )

        # Spread (home perspective line)
        if (
            event.spread_home_point is not None
            and event.spread_home_price is not None
            and event.spread_away_price is not None
        ):
            stats = _resim_with_lines(
                pred,
                market_spread=float(event.spread_home_point),
                market_total=float(event.total_point or pred.exp_total),
                n_sims=n_sims,
                seed=1000 + i,
            )
            fair_home, fair_away = devig_two_way(
                event.spread_home_price, event.spread_away_price
            )
            p_cover = stats["p_home_cover"]
            p_dog = 1.0 - p_cover
            for side, model_p, book_price, book_fair, line in (
                (
                    "home",
                    p_cover,
                    event.spread_home_price,
                    fair_home,
                    event.spread_home_point,
                ),
                (
                    "away",
                    p_dog,
                    event.spread_away_price,
                    fair_away,
                    event.spread_away_point,
                ),
            ):
                rows.append(
                    EdgeRow(
                        game_id=pred.game_id,
                        game_date=pred.game_date,
                        home_tricode=pred.home_tricode,
                        away_tricode=pred.away_tricode,
                        market="spread",
                        side=side,
                        model_prob=float(model_p),
                        fair_american=float(implied_to_american(min(max(model_p, 1e-6), 1 - 1e-6))),
                        book_american=float(book_price),
                        book_implied_raw=float(american_to_implied(book_price)),
                        book_implied_fair=float(book_fair),
                        edge=float(model_p - book_fair),
                        book=event.book,
                        line=None if line is None else float(line),
                        event_id=event.event_id,
                    )
                )

        # Totals
        if (
            event.total_point is not None
            and event.total_over_price is not None
            and event.total_under_price is not None
        ):
            stats = _resim_with_lines(
                pred,
                market_spread=float(event.spread_home_point or 0.0),
                market_total=float(event.total_point),
                n_sims=n_sims,
                seed=2000 + i,
            )
            fair_over, fair_under = devig_two_way(
                event.total_over_price, event.total_under_price
            )
            p_over = stats["p_over"]
            p_under = 1.0 - p_over
            for side, model_p, book_price, book_fair in (
                ("over", p_over, event.total_over_price, fair_over),
                ("under", p_under, event.total_under_price, fair_under),
            ):
                rows.append(
                    EdgeRow(
                        game_id=pred.game_id,
                        game_date=pred.game_date,
                        home_tricode=pred.home_tricode,
                        away_tricode=pred.away_tricode,
                        market="total",
                        side=side,
                        model_prob=float(model_p),
                        fair_american=float(implied_to_american(min(max(model_p, 1e-6), 1 - 1e-6))),
                        book_american=float(book_price),
                        book_implied_raw=float(american_to_implied(book_price)),
                        book_implied_fair=float(book_fair),
                        edge=float(model_p - book_fair),
                        book=event.book,
                        line=float(event.total_point),
                        event_id=event.event_id,
                    )
                )
    return rows


def edges_to_frame(rows: list[EdgeRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in rows])


def format_edge_table(rows: list[EdgeRow], *, min_edge: float = 0.0) -> str:
    if not rows:
        return "(no matched odds / edges)"
    df = edges_to_frame(rows)
    df = df.loc[df["edge"] >= min_edge].sort_values("edge", ascending=False)
    if df.empty:
        return f"(no edges >= {min_edge:.1%})"
    view = df[
        [
            "away_tricode",
            "home_tricode",
            "market",
            "side",
            "line",
            "model_prob",
            "book_implied_fair",
            "edge",
            "fair_american",
            "book_american",
            "book",
        ]
    ].copy()
    view["model_prob"] = view["model_prob"].map(lambda x: f"{x:.3f}")
    view["book_implied_fair"] = view["book_implied_fair"].map(lambda x: f"{x:.3f}")
    view["edge"] = view["edge"].map(lambda x: f"{x:.3f}")
    view["fair_american"] = view["fair_american"].map(lambda x: f"{x:.0f}")
    view["book_american"] = view["book_american"].map(lambda x: f"{x:.0f}")
    return view.to_string(index=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
