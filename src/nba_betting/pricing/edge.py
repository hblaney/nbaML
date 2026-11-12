"""Join model probabilities to multi-book lines and compute Shin / sharp-consensus edges."""

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
from nba_betting.pricing.consensus import (
    TwoWayQuote,
    compute_sharp_consensus,
    is_valid_pregame_book,
    normalize_book_key,
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


def _match_all_odds(pred: GamePrediction, events: Iterable[EventOdds]) -> list[EventOdds]:
    """Match ALL bookmaker rows for ``pred`` on home/away tricodes."""
    return [
        event
        for event in events
        if event.home_tricode == pred.home_tricode and event.away_tricode == pred.away_tricode
    ]


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
    home, away = simulate_scores(pred.exp_home_score, pred.exp_away_score, n_sims=n_sims, rng=rng)
    return derive_market_probs(home, away, market_spread=market_spread, market_total=market_total)


def build_edge_rows(
    predictions: list[GamePrediction],
    events: list[EventOdds],
    *,
    n_sims: int = 10_000,
    devig_method: str = "shin",
) -> list[EdgeRow]:
    """Build edge rows across ALL matched books using Shin sharp-consensus devigging."""
    rows: list[EdgeRow] = []
    for i, pred in enumerate(predictions):
        matched = _match_all_odds(pred, events)
        if not matched:
            continue

        # Precompute multi-book sharp consensus per market (LOO when >= 2 books exist).
        ml_quotes = [
            TwoWayQuote(
                book=ev.book,
                side_a_price=float(ev.moneyline_home),
                side_b_price=float(ev.moneyline_away),
                market="moneyline",
            )
            for ev in matched
            if ev.moneyline_home is not None
            and ev.moneyline_away is not None
            and is_valid_pregame_book(ev.book)
        ]
        sp_quotes = [
            TwoWayQuote(
                book=ev.book,
                side_a_price=float(ev.spread_home_price),
                side_b_price=float(ev.spread_away_price),
                line_a=float(ev.spread_home_point),
                market="spread",
            )
            for ev in matched
            if ev.spread_home_point is not None
            and ev.spread_home_price is not None
            and ev.spread_away_price is not None
            and is_valid_pregame_book(ev.book)
        ]
        tot_quotes = [
            TwoWayQuote(
                book=ev.book,
                side_a_price=float(ev.total_over_price),
                side_b_price=float(ev.total_under_price),
                line_a=float(ev.total_point),
                market="total",
            )
            for ev in matched
            if ev.total_point is not None
            and ev.total_over_price is not None
            and ev.total_under_price is not None
            and is_valid_pregame_book(ev.book)
        ]

        sim_cache: dict[tuple[float, float], dict[str, float]] = {}

        for event in matched:
            ck = normalize_book_key(event.book)

            # Moneyline
            if event.moneyline_home is not None and event.moneyline_away is not None:
                peers = [q for q in ml_quotes if q.canonical_book != ck]
                if peers:
                    cons = compute_sharp_consensus(peers, devig_method=devig_method)
                    fair_h, fair_a = cons.prob_a, cons.prob_b
                else:
                    fair_h, fair_a = devig_two_way(
                        event.moneyline_home, event.moneyline_away, method=devig_method
                    )
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
                sp_key = (
                    float(event.spread_home_point),
                    float(event.total_point or pred.exp_total),
                )
                if sp_key not in sim_cache:
                    sim_cache[sp_key] = _resim_with_lines(
                        pred,
                        market_spread=sp_key[0],
                        market_total=sp_key[1],
                        n_sims=n_sims,
                        seed=1000 + i,
                    )
                stats = sim_cache[sp_key]
                peers_sp = [q for q in sp_quotes if q.canonical_book != ck]
                if peers_sp:
                    cons_sp = compute_sharp_consensus(peers_sp, devig_method=devig_method)
                    fair_home = cons_sp.prob_side_a_at_line(float(event.spread_home_point))
                    fair_away = 1.0 - fair_home
                else:
                    fair_home, fair_away = devig_two_way(
                        event.spread_home_price,
                        event.spread_away_price,
                        method=devig_method,
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
                            fair_american=float(
                                implied_to_american(min(max(model_p, 1e-6), 1 - 1e-6))
                            ),
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
                tot_key = (
                    float(event.spread_home_point or 0.0),
                    float(event.total_point),
                )
                if tot_key not in sim_cache:
                    sim_cache[tot_key] = _resim_with_lines(
                        pred,
                        market_spread=tot_key[0],
                        market_total=tot_key[1],
                        n_sims=n_sims,
                        seed=2000 + i,
                    )
                stats = sim_cache[tot_key]
                peers_tot = [q for q in tot_quotes if q.canonical_book != ck]
                if peers_tot:
                    cons_tot = compute_sharp_consensus(peers_tot, devig_method=devig_method)
                    fair_over = cons_tot.prob_side_a_at_line(float(event.total_point))
                    fair_under = 1.0 - fair_over
                else:
                    fair_over, fair_under = devig_two_way(
                        event.total_over_price,
                        event.total_under_price,
                        method=devig_method,
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
                            fair_american=float(
                                implied_to_american(min(max(model_p, 1e-6), 1 - 1e-6))
                            ),
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
