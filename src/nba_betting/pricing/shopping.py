"""Cross-book market-dispersion analyzer and multi-book parlay line-shopping router.

Supports:
1. Multi-book ingestion from The Odds API raw JSON payloads AND captured markdown
   slates (e.g. ``docs/nba-live-slate.md``).
2. Sharp-weighted consensus pricing (Shin / Power / Probit / Multiplicative / Additive)
   with leave-one-out stale-outlier detection across moneyline, spread, and total markets.
3. Executable single-ticket cross-book parlay routing: evaluates every candidate parlay
   at each sportsbook's actual offered lines and prices, routing the ticket to ``b*``
   that maximizes parlay expected value (plus staggered-tip best-leg rollover comparison).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nba_betting.data.teams import team_to_tricode
from nba_betting.pricing.consensus import (
    DEFAULT_MARGIN_SIGMA,
    DEFAULT_TOTAL_SIGMA,
    ConsensusResult,
    StaleOutlier,
    TwoWayQuote,
    compute_sharp_consensus,
    detect_stale_outliers,
    is_valid_pregame_book,
    normalize_book_key,
)
from nba_betting.pricing.devig import normalize_devig_method
from nba_betting.pricing.kelly import (
    KELLY_FRACTION,
    MAX_BET_FRAC,
    american_to_decimal,
    kelly_fraction,
)

DEFAULT_ROUTING_BOOKS: tuple[str, ...] = (
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "espnbet",
    "betrivers",
    "pinnacle",
    "lowvig",
    "bovada",
)

# Odds API v4 bookmaker keys corresponding to DEFAULT_ROUTING_BOOKS (Caesars = williamhill_us).
ODDS_API_BOOKMAKER_KEYS: dict[str, str] = {
    "draftkings": "draftkings",
    "fanduel": "fanduel",
    "betmgm": "betmgm",
    "caesars": "williamhill_us",
    "espnbet": "espnbet",
    "betrivers": "betrivers",
    "pinnacle": "pinnacle",
    "lowvig": "lowvig",
    "bovada": "bovada",
}

_ESPN_TRICODE_MAP: dict[str, str] = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "WSH": "WAS",
    "UTAH": "UTA",
}


def canonical_tricode(code_or_name: str | None) -> str | None:
    if not code_or_name:
        return None
    cleaned = str(code_or_name).strip()
    upper = cleaned.upper()
    if upper in _ESPN_TRICODE_MAP:
        return _ESPN_TRICODE_MAP[upper]
    return team_to_tricode(cleaned)


@dataclass(frozen=True)
class BookLine:
    """One sportsbook's pregame quotes for a single NBA event."""

    book: str
    canonical_book: str
    moneyline_home: float | None = None
    moneyline_away: float | None = None
    spread_home_point: float | None = None
    spread_home_price: float | None = None
    spread_away_point: float | None = None
    spread_away_price: float | None = None
    total_point: float | None = None
    total_over_price: float | None = None
    total_under_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiBookEvent:
    """All sportsbook quotes captured for one NBA game."""

    event_id: str
    commence_time: str
    game_date: str
    home: str
    away: str
    home_tricode: str | None
    away_tricode: str | None
    books: list[BookLine] = field(default_factory=list)
    asof_ts: str = ""

    @property
    def matchup_key(self) -> str:
        at = self.away_tricode or self.away
        ht = self.home_tricode or self.home
        return f"{at}@{ht}"

    def quotes_for_market(self, market: str) -> list[TwoWayQuote]:
        mkt = market.lower()
        out: list[TwoWayQuote] = []
        for b in self.books:
            if not is_valid_pregame_book(b.book):
                continue
            if mkt in {"moneyline", "ml", "h2h"}:
                if b.moneyline_home is not None and b.moneyline_away is not None:
                    out.append(
                        TwoWayQuote(
                            book=b.book,
                            side_a_price=float(b.moneyline_home),
                            side_b_price=float(b.moneyline_away),
                            line_a=None,
                            market="moneyline",
                        )
                    )
            elif mkt in {"spread", "spreads"}:
                if (
                    b.spread_home_point is not None
                    and b.spread_home_price is not None
                    and b.spread_away_price is not None
                ):
                    out.append(
                        TwoWayQuote(
                            book=b.book,
                            side_a_price=float(b.spread_home_price),
                            side_b_price=float(b.spread_away_price),
                            line_a=float(b.spread_home_point),
                            market="spread",
                        )
                    )
            elif mkt in {"total", "totals"}:
                if (
                    b.total_point is not None
                    and b.total_over_price is not None
                    and b.total_under_price is not None
                ):
                    out.append(
                        TwoWayQuote(
                            book=b.book,
                            side_a_price=float(b.total_over_price),
                            side_b_price=float(b.total_under_price),
                            line_a=float(b.total_point),
                            market="total",
                        )
                    )
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutedParlayLeg:
    event_id: str
    matchup: str
    commence_time: str
    market: str
    side: str
    line: float | None
    american: float
    decimal: float
    fair_prob: float
    leg_ev_at_book: float
    best_single_book: str
    best_single_american: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutedParlayTicket:
    """Executable single-book parlay routed across sportsbooks to maximize EV."""

    game_date: str
    n_legs: int
    routed_book: str
    routed_canonical_book: str
    legs: list[RoutedParlayLeg]
    p_joint_fair: float
    routed_parlay_decimal: float
    routed_book_implied: float
    routed_price_edge: float
    routed_ev_per_unit: float
    n_books_offering_all_legs: int
    median_book_decimal: float
    worst_book_decimal: float
    median_book_ev: float
    worst_book_ev: float
    ev_gain_vs_median: float
    ev_gain_vs_worst: float
    rollover_best_leg_decimal: float
    rollover_best_leg_ev: float
    rollover_books: list[str]
    kelly_stake_fraction: float
    recommended_stake: float
    bankroll: float
    book_ev_table: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_float_num(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace("−", "-").replace(",", "")
    if not s:
        return None
    if s.upper() in {"PK", "PICK", "EVEN"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _parse_american_num(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace("−", "-").replace(",", "")
    if not s:
        return None
    if s.upper() == "EVEN":
        return 100.0
    try:
        v = float(s)
        return None if v == 0.0 else v
    except ValueError:
        return None


def _date_from_commence(commence_time: str) -> str:
    if not commence_time:
        return ""
    try:
        # Convert UTC tip to US Eastern date (~5h offset) so 01:00Z tips stay on slate date.
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        from datetime import timedelta

        est = dt - timedelta(hours=5)
        return str(est.date())
    except ValueError:
        return commence_time[:10]


def parse_odds_api_multi_book(
    payload: Sequence[Mapping[str, Any]],
    *,
    asof_ts: str | None = None,
) -> list[MultiBookEvent]:
    """Parse an Odds API ``/v4/sports/basketball_nba/odds`` response preserving all books."""
    stamp = asof_ts or datetime.now(timezone.utc).isoformat()
    events: list[MultiBookEvent] = []
    for raw_ev in payload:
        home = str(raw_ev.get("home_team") or "")
        away = str(raw_ev.get("away_team") or "")
        commence = str(raw_ev.get("commence_time") or "")
        eid = str(raw_ev.get("id") or f"{away}@{home}:{commence}")
        books: list[BookLine] = []

        for bm in raw_ev.get("bookmakers") or []:
            bkey = str(bm.get("key") or bm.get("title") or "").strip()
            if not bkey or not is_valid_pregame_book(bkey):
                continue
            ml_h = ml_a = None
            sp_h_pt = sp_h_pr = sp_a_pt = sp_a_pr = None
            tot_pt = over_pr = under_pr = None

            for mkt in bm.get("markets") or []:
                mkey = str(mkt.get("key") or "")
                outcomes = mkt.get("outcomes") or []
                if mkey == "h2h":
                    for o in outcomes:
                        if o.get("name") == home:
                            ml_h = _parse_american_num(o.get("price"))
                        elif o.get("name") == away:
                            ml_a = _parse_american_num(o.get("price"))
                elif mkey == "spreads":
                    for o in outcomes:
                        if o.get("name") == home:
                            sp_h_pt = _parse_float_num(o.get("point"))
                            sp_h_pr = _parse_american_num(o.get("price"))
                        elif o.get("name") == away:
                            sp_a_pt = _parse_float_num(o.get("point"))
                            sp_a_pr = _parse_american_num(o.get("price"))
                elif mkey == "totals":
                    for o in outcomes:
                        oname = str(o.get("name") or "").lower()
                        if oname == "over":
                            tot_pt = _parse_float_num(o.get("point"))
                            over_pr = _parse_american_num(o.get("price"))
                        elif oname == "under":
                            if tot_pt is None:
                                tot_pt = _parse_float_num(o.get("point"))
                            under_pr = _parse_american_num(o.get("price"))

            books.append(
                BookLine(
                    book=bkey,
                    canonical_book=normalize_book_key(bkey),
                    moneyline_home=ml_h,
                    moneyline_away=ml_a,
                    spread_home_point=sp_h_pt,
                    spread_home_price=sp_h_pr,
                    spread_away_point=sp_a_pt,
                    spread_away_price=sp_a_pr,
                    total_point=tot_pt,
                    total_over_price=over_pr,
                    total_under_price=under_pr,
                )
            )

        events.append(
            MultiBookEvent(
                event_id=eid,
                commence_time=commence,
                game_date=_date_from_commence(commence),
                home=home,
                away=away,
                home_tricode=canonical_tricode(home),
                away_tricode=canonical_tricode(away),
                books=books,
                asof_ts=stamp,
            )
        )
    return events


_MD_GAME_HEADER_RE = re.compile(r"^##\s+(.+?)\s+@\s+(.+?)\s*$")
_MD_BOOK_HEADER_RE = re.compile(r"^###\s+(.+?)\s*$")
_MD_COMMENCE_RE = re.compile(r"^\-\s+\*\*Commence \(UTC\):\*\*\s*(\S+)")
_MD_FETCHED_RE = re.compile(r"^\-\s+\*\*Fetched:\*\*\s*(.+)$")
_MD_ML_SIDE_RE = re.compile(r"^(.+?)\s+([+\-−]?\d+(?:\.\d+)?|EVEN)$", re.IGNORECASE)
_MD_SPREAD_SIDE_RE = re.compile(
    r"^(.+?)\s+([+\-−]?\d+(?:\.\d+)?|PK)\s*\(\s*([+\-−]?\d+(?:\.\d+)?|EVEN)\s*\)$",
    re.IGNORECASE,
)
_MD_TOTAL_RE = re.compile(
    r"^o\s*(\d+(?:\.\d+)?)\s*\(\s*([+\-−]?\d+(?:\.\d+)?|EVEN)\s*\)\s*/\s*u\s*(\d+(?:\.\d+)?)\s*\(\s*([+\-−]?\d+(?:\.\d+)?|EVEN)\s*\)$",
    re.IGNORECASE,
)


def parse_live_slate_markdown(source: str | Path) -> list[MultiBookEvent]:
    """Parse captured multi-book Odds API markdown (e.g. ``docs/nba-live-slate.md``)."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)

    events: list[MultiBookEvent] = []
    fetched_ts = ""
    cur_away = ""
    cur_home = ""
    cur_commence = ""
    cur_books: list[BookLine] = []
    cur_book_name: str | None = None
    cur_fields: dict[str, float | None] = {}

    def flush_book() -> None:
        nonlocal cur_book_name, cur_fields
        if cur_book_name and is_valid_pregame_book(cur_book_name):
            cur_books.append(
                BookLine(
                    book=cur_book_name,
                    canonical_book=normalize_book_key(cur_book_name),
                    moneyline_home=cur_fields.get("ml_home"),
                    moneyline_away=cur_fields.get("ml_away"),
                    spread_home_point=cur_fields.get("sp_home_pt"),
                    spread_home_price=cur_fields.get("sp_home_pr"),
                    spread_away_point=cur_fields.get("sp_away_pt"),
                    spread_away_price=cur_fields.get("sp_away_pr"),
                    total_point=cur_fields.get("tot_pt"),
                    total_over_price=cur_fields.get("over_pr"),
                    total_under_price=cur_fields.get("under_pr"),
                )
            )
        cur_book_name = None
        cur_fields = {}

    def flush_game() -> None:
        nonlocal cur_away, cur_home, cur_commence, cur_books
        flush_book()
        if cur_away and cur_home:
            ht = canonical_tricode(cur_home)
            at = canonical_tricode(cur_away)
            gdate = _date_from_commence(cur_commence)
            eid = f"{at or cur_away}@{ht or cur_home}:{gdate or cur_commence}"
            events.append(
                MultiBookEvent(
                    event_id=eid,
                    commence_time=cur_commence,
                    game_date=gdate,
                    home=cur_home,
                    away=cur_away,
                    home_tricode=ht,
                    away_tricode=at,
                    books=list(cur_books),
                    asof_ts=fetched_ts,
                )
            )
        cur_away = ""
        cur_home = ""
        cur_commence = ""
        cur_books = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_fetch = _MD_FETCHED_RE.match(line)
        if m_fetch and not fetched_ts:
            fetched_ts = m_fetch.group(1).strip()
            continue

        m_game = _MD_GAME_HEADER_RE.match(line)
        if m_game:
            flush_game()
            cur_away = m_game.group(1).strip()
            cur_home = m_game.group(2).strip()
            continue

        m_comm = _MD_COMMENCE_RE.match(line)
        if m_comm:
            cur_commence = m_comm.group(1).strip()
            continue

        m_book = _MD_BOOK_HEADER_RE.match(line)
        if m_book and cur_home:
            flush_book()
            cur_book_name = m_book.group(1).strip()
            continue

        if cur_book_name and line.startswith("- **Moneyline:**"):
            body = line.split("**Moneyline:**", 1)[1].strip()
            for part in body.split("·"):
                sm = _MD_ML_SIDE_RE.match(part.strip())
                if not sm:
                    continue
                tname, price_s = sm.group(1).strip(), sm.group(2).strip()
                price = _parse_american_num(price_s)
                if tname == cur_home:
                    cur_fields["ml_home"] = price
                elif tname == cur_away:
                    cur_fields["ml_away"] = price
        elif cur_book_name and line.startswith("- **Spread:**"):
            body = line.split("**Spread:**", 1)[1].strip()
            for part in body.split("·"):
                sm = _MD_SPREAD_SIDE_RE.match(part.strip())
                if not sm:
                    continue
                tname, pt_s, pr_s = sm.group(1).strip(), sm.group(2).strip(), sm.group(3).strip()
                pt = _parse_float_num(pt_s)
                pr = _parse_american_num(pr_s)
                if tname == cur_home:
                    cur_fields["sp_home_pt"] = pt
                    cur_fields["sp_home_pr"] = pr
                elif tname == cur_away:
                    cur_fields["sp_away_pt"] = pt
                    cur_fields["sp_away_pr"] = pr
        elif cur_book_name and line.startswith("- **Total:**"):
            body = line.split("**Total:**", 1)[1].strip()
            tm = _MD_TOTAL_RE.match(body)
            if tm:
                cur_fields["tot_pt"] = _parse_float_num(tm.group(1))
                cur_fields["over_pr"] = _parse_american_num(tm.group(2))
                cur_fields["under_pr"] = _parse_american_num(tm.group(4))

    flush_game()
    return events


def load_multi_book_file(path: Path) -> list[MultiBookEvent]:
    """Load ``MultiBookEvent`` list from Odds API JSON, cached MultiBook JSON, or Markdown."""
    p = Path(path)
    if p.suffix.lower() == ".md":
        return parse_live_slate_markdown(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return []
    first = payload[0]
    if "bookmakers" in first:
        return parse_odds_api_multi_book(payload)
    if "books" in first:
        out: list[MultiBookEvent] = []
        for row in payload:
            books = [BookLine(**b) for b in (row.get("books") or [])]
            r_copy = {k: v for k, v in row.items() if k != "books"}
            out.append(MultiBookEvent(**r_copy, books=books))
        return out
    if "book" in first and "moneyline_home" in first:
        # Backward-compat with flat single-book EventOdds JSON cache
        grouped: dict[str, MultiBookEvent] = {}
        for row in payload:
            eid = str(row.get("event_id") or "")
            bline = BookLine(
                book=str(row.get("book") or ""),
                canonical_book=normalize_book_key(str(row.get("book") or "")),
                moneyline_home=row.get("moneyline_home"),
                moneyline_away=row.get("moneyline_away"),
                spread_home_point=row.get("spread_home_point"),
                spread_home_price=row.get("spread_home_price"),
                spread_away_point=row.get("spread_away_point"),
                spread_away_price=row.get("spread_away_price"),
                total_point=row.get("total_point"),
                total_over_price=row.get("total_over_price"),
                total_under_price=row.get("total_under_price"),
            )
            if eid not in grouped:
                comm = str(row.get("commence_time") or "")
                grouped[eid] = MultiBookEvent(
                    event_id=eid,
                    commence_time=comm,
                    game_date=_date_from_commence(comm),
                    home=str(row.get("home") or ""),
                    away=str(row.get("away") or ""),
                    home_tricode=canonical_tricode(row.get("home_tricode") or row.get("home")),
                    away_tricode=canonical_tricode(row.get("away_tricode") or row.get("away")),
                    books=[bline],
                    asof_ts=str(row.get("asof_ts") or ""),
                )
            else:
                grouped[eid].books.append(bline)
        return list(grouped.values())
    raise ValueError(f"Unrecognized multi-book file format in {p}")


def analyze_slate_markets(
    events: Sequence[MultiBookEvent],
    *,
    devig_method: str = "shin",
    markets: Sequence[str] = ("moneyline", "spread", "total"),
    min_books: int = 2,
    min_outlier_ev: float = 0.0,
    max_outlier_ev: float = 0.15,
    min_peer_weight: float = 4.0,
    min_outlier_z: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute sharp consensus and stale outliers across all events and markets."""
    canon_devig = normalize_devig_method(devig_method)
    consensus_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []

    for ev in events:
        for mkt in markets:
            quotes = ev.quotes_for_market(mkt)
            if len(quotes) < min_books:
                continue
            cons = compute_sharp_consensus(quotes, devig_method=canon_devig)
            side_a_label = "over" if cons.market == "total" else (ev.home_tricode or ev.home)
            side_b_label = "under" if cons.market == "total" else (ev.away_tricode or ev.away)
            consensus_rows.append(
                {
                    "event_id": ev.event_id,
                    "game_date": ev.game_date,
                    "matchup": ev.matchup_key,
                    "market": cons.market,
                    "devig_method": cons.devig_method,
                    "n_books": cons.n_books,
                    "consensus_line": cons.consensus_line,
                    "side_a": side_a_label,
                    "prob_a": cons.prob_a,
                    "fair_american_a": cons.fair_american_a,
                    "best_book_a": cons.best_book_a,
                    "best_price_a": cons.best_price_a,
                    "best_line_a": cons.best_line_a,
                    "side_b": side_b_label,
                    "prob_b": cons.prob_b,
                    "fair_american_b": cons.fair_american_b,
                    "best_book_b": cons.best_book_b,
                    "best_price_b": cons.best_price_b,
                    "best_line_b": cons.best_line_b,
                    "mean_hold_pct": (cons.mean_overround - 1.0) * 100.0,
                    "min_hold_pct": (cons.min_overround - 1.0) * 100.0,
                    "max_hold_pct": (cons.max_overround - 1.0) * 100.0,
                    "synthetic_hold_pct": (cons.synthetic_overround - 1.0) * 100.0,
                    "is_arb": cons.is_arb,
                    "dispersion_std": cons.dispersion_std_a,
                }
            )
            outliers: list[StaleOutlier] = detect_stale_outliers(
                quotes,
                devig_method=canon_devig,
                min_ev=min_outlier_ev,
                max_ev=max_outlier_ev,
                min_peer_weight=min_peer_weight,
                min_z_score=min_outlier_z,
            )
            for o in outliers:
                side_name = side_a_label if o.side == "a" else side_b_label
                outlier_rows.append(
                    {
                        "event_id": ev.event_id,
                        "game_date": ev.game_date,
                        "matchup": ev.matchup_key,
                        "market": o.market,
                        "side": side_name,
                        "book": o.book,
                        "canonical_book": o.canonical_book,
                        "line": o.book_line,
                        "consensus_line": o.consensus_line,
                        "line_out_pts": o.line_out_pts,
                        "book_american": o.book_american,
                        "consensus_fair_american": o.consensus_fair_american,
                        "consensus_prob": o.consensus_prob,
                        "book_implied_vigged": o.book_implied_vigged,
                        "ev_pct": o.ev_per_unit * 100.0,
                        "vig_free_edge_pct": o.vig_free_edge * 100.0,
                        "z_score": o.z_score,
                        "n_peer_books": o.n_peer_books,
                        "peer_sharp_weight_sum": o.peer_sharp_weight_sum,
                    }
                )
    outlier_rows.sort(key=lambda r: (-float(r["ev_pct"]), -float(r["z_score"])))
    return consensus_rows, outlier_rows


@dataclass(frozen=True)
class _CandidateSideSpec:
    """Market-level candidate leg before evaluating per-book execution prices."""

    event_id: str
    game_date: str
    commence_time: str
    matchup: str
    market: str
    side_code: str  # "a" or "b"
    side_label: str
    consensus: ConsensusResult
    loo_consensus_by_book: dict[str, ConsensusResult]
    model_prob_a_override: float | None
    model_weight: float
    book_quotes: dict[
        str, tuple[str, float, float | None]
    ]  # canon_book -> (display_book, price, line)
    best_book_display: str
    best_book_price: float
    best_book_ev: float


def _prob_for_book_quote(
    spec: _CandidateSideSpec,
    line_at_book: float | None,
    *,
    canon_book: str | None = None,
    margin_sigma: float = DEFAULT_MARGIN_SIGMA,
    total_sigma: float = DEFAULT_TOTAL_SIGMA,
) -> float:
    """Fair win probability of ``spec`` evaluated at ``line_at_book`` using LOO consensus."""
    bench = (
        spec.loo_consensus_by_book.get(canon_book, spec.consensus)
        if canon_book is not None
        else spec.consensus
    )
    if spec.market == "spread" and line_at_book is not None:
        home_line = float(line_at_book) if spec.side_code == "a" else -float(line_at_book)
        pa_cons = bench.prob_side_a_at_line(home_line, margin_sigma=margin_sigma)
    elif spec.market == "total" and line_at_book is not None:
        pa_cons = bench.prob_side_a_at_line(float(line_at_book), total_sigma=total_sigma)
    else:
        pa_cons = bench.prob_a

    if spec.model_prob_a_override is not None and spec.model_weight > 0.0:
        w = float(np.clip(spec.model_weight, 0.0, 1.0))
        pa = w * float(spec.model_prob_a_override) + (1.0 - w) * pa_cons
    else:
        pa = pa_cons
    pa = float(np.clip(pa, 1e-6, 1.0 - 1.0e-6))
    return pa if spec.side_code == "a" else (1.0 - pa)


def route_parlays_across_books(
    events: Sequence[MultiBookEvent],
    *,
    devig_method: str = "shin",
    markets: Sequence[str] = ("moneyline", "spread", "total"),
    execution_books: Sequence[str] | None = DEFAULT_ROUTING_BOOKS,
    include_all_books_when_unmatched: bool = False,
    model_home_probs: Mapping[str, float] | None = None,
    model_weight: float = 0.0,
    sizes: Sequence[int] = (2, 3),
    min_books_for_consensus: int = 2,
    min_peer_weight: float = 4.0,
    min_best_leg_ev: float = -0.025,
    max_leg_ev: float = 0.15,
    min_parlay_ev: float = 0.0,
    min_price_edge: float = 0.0,
    max_parlay_decimal: float | None = 25.0,
    max_per_day: int = 5,
    allow_same_day_only: bool = True,
    bankroll: float = 50.0,
    kelly_frac: float = KELLY_FRACTION,
    max_bet_frac: float = MAX_BET_FRAC,
) -> list[RoutedParlayTicket]:
    """Route single bets (``n=1``) and multi-leg parlays (``n>=2``) to the highest-EV book ``b*``.

    Key realism guarantees:
    - Leave-one-out (LOO) sharp consensus anchored on ``LowVig.ag`` / ``BetOnline.ag`` /
      ``Pinnacle`` excludes target book ``b*`` so a stale retail number never shifts its
      own fair probability benchmark.
    - Enforces ``sum_{b != b*} w_b >= min_peer_weight`` (default ``4.0``) and suppresses
      corrupt feed spikes above ``max_leg_ev`` (default ``+15%``).
    - Each returned ``RoutedParlayTicket`` is executable at a **single sportsbook**
      ``routed_book`` at that book's exact offered lines and prices.
    """
    from nba_betting.pricing.consensus import _deduplicate_quotes, book_sharpness_weight

    canon_devig = normalize_devig_method(devig_method)
    allowed_exec: set[str] | None = (
        {normalize_book_key(b) for b in execution_books} if execution_books is not None else None
    )

    specs: list[_CandidateSideSpec] = []
    for ev in events:
        m_override: float | None = None
        if model_home_probs:
            if ev.event_id in model_home_probs:
                m_override = float(model_home_probs[ev.event_id])
            elif ev.matchup_key in model_home_probs:
                m_override = float(model_home_probs[ev.matchup_key])

        for mkt in markets:
            quotes = ev.quotes_for_market(mkt)
            if len(quotes) < min_books_for_consensus:
                continue
            cons = compute_sharp_consensus(quotes, devig_method=canon_devig)
            side_a_label = "over" if cons.market == "total" else f"{ev.home_tricode or ev.home}"
            side_b_label = "under" if cons.market == "total" else f"{ev.away_tricode or ev.away}"

            # Precompute LOO consensus per canonical book + check min_peer_weight
            loo_map: dict[str, ConsensusResult] = {}
            valid_cks: set[str] = set()
            for q in quotes:
                ck = q.canonical_book
                if ck in loo_map:
                    continue
                peers = [p for p in quotes if p.canonical_book != ck]
                dedup_peers = _deduplicate_quotes(peers)
                pw_sum = float(sum(book_sharpness_weight(p.book) for p in dedup_peers))
                if pw_sum < float(min_peer_weight):
                    continue
                valid_cks.add(ck)
                loo_map[ck] = (
                    compute_sharp_consensus(peers, devig_method=canon_devig) if peers else cons
                )

            if not valid_cks:
                continue

            for side_code, side_label in (("a", side_a_label), ("b", side_b_label)):
                bmap: dict[str, tuple[str, float, float | None]] = {}
                for q in quotes:
                    ck = q.canonical_book
                    if ck not in valid_cks:
                        continue
                    if allowed_exec is not None and ck not in allowed_exec:
                        continue
                    price = float(q.side_a_price if side_code == "a" else q.side_b_price)
                    line_val = (
                        q.line_a
                        if side_code == "a"
                        else (
                            -float(q.line_a)
                            if (q.line_a is not None and cons.market == "spread")
                            else q.line_a
                        )
                    )
                    prev = bmap.get(ck)
                    if prev is None or price > prev[1]:
                        bmap[ck] = (q.book, price, line_val)

                if not bmap and include_all_books_when_unmatched:
                    for q in quotes:
                        ck = q.canonical_book
                        if ck not in valid_cks:
                            continue
                        price = float(q.side_a_price if side_code == "a" else q.side_b_price)
                        line_val = (
                            q.line_a
                            if side_code == "a"
                            else (
                                -float(q.line_a)
                                if (q.line_a is not None and cons.market == "spread")
                                else q.line_a
                            )
                        )
                        bmap[ck] = (q.book, price, line_val)

                if not bmap:
                    continue

                tmp_spec = _CandidateSideSpec(
                    event_id=ev.event_id,
                    game_date=ev.game_date or "slate",
                    commence_time=ev.commence_time,
                    matchup=ev.matchup_key,
                    market=cons.market,
                    side_code=side_code,
                    side_label=side_label,
                    consensus=cons,
                    loo_consensus_by_book=loo_map,
                    model_prob_a_override=(m_override if cons.market == "moneyline" else None),
                    model_weight=float(model_weight),
                    book_quotes=bmap,
                    best_book_display="",
                    best_book_price=0.0,
                    best_book_ev=float("-inf"),
                )
                # Filter out corrupt > max_leg_ev quotes at individual books
                filtered_bmap: dict[str, tuple[str, float, float | None]] = {}
                best_ev = float("-inf")
                best_disp = ""
                best_pr = 0.0
                for ck, (disp, pr, ln) in bmap.items():
                    p_leg = _prob_for_book_quote(tmp_spec, ln, canon_book=ck)
                    ev_leg = p_leg * american_to_decimal(pr) - 1.0
                    if ev_leg > float(max_leg_ev):
                        continue
                    filtered_bmap[ck] = (disp, pr, ln)
                    if ev_leg > best_ev:
                        best_ev = ev_leg
                        best_disp = disp
                        best_pr = pr

                if not filtered_bmap or best_ev < float(min_best_leg_ev):
                    continue

                specs.append(
                    _CandidateSideSpec(
                        event_id=tmp_spec.event_id,
                        game_date=tmp_spec.game_date,
                        commence_time=tmp_spec.commence_time,
                        matchup=tmp_spec.matchup,
                        market=tmp_spec.market,
                        side_code=tmp_spec.side_code,
                        side_label=tmp_spec.side_label,
                        consensus=tmp_spec.consensus,
                        loo_consensus_by_book=loo_map,
                        model_prob_a_override=tmp_spec.model_prob_a_override,
                        model_weight=tmp_spec.model_weight,
                        book_quotes=filtered_bmap,
                        best_book_display=best_disp,
                        best_book_price=best_pr,
                        best_book_ev=best_ev,
                    )
                )

    if not specs:
        return []

    by_day: dict[str, list[_CandidateSideSpec]] = {}
    for sp in specs:
        key = sp.game_date if allow_same_day_only else "all"
        by_day.setdefault(key, []).append(sp)

    all_candidates: list[RoutedParlayTicket] = []
    for gdate, day_specs in sorted(by_day.items()):
        for n in sizes:
            if len(day_specs) < n:
                continue
            for idxs in combinations(range(len(day_specs)), n):
                picked = [day_specs[i] for i in idxs]
                eids = {p.event_id for p in picked}
                if len(eids) != n:
                    continue

                common_books = set(picked[0].book_quotes.keys())
                for p in picked[1:]:
                    common_books &= set(p.book_quotes.keys())
                if not common_books:
                    continue

                book_evals: list[tuple[float, float, float, str, str, list[RoutedParlayLeg]]] = []
                for ck in sorted(common_books):
                    leg_objs: list[RoutedParlayLeg] = []
                    p_joint = 1.0
                    dec_prod = 1.0
                    disp_book = picked[0].book_quotes[ck][0]
                    for sp in picked:
                        _, pr, ln = sp.book_quotes[ck]
                        dec = american_to_decimal(pr)
                        p_leg = _prob_for_book_quote(sp, ln, canon_book=ck)
                        p_joint *= p_leg
                        dec_prod *= dec
                        leg_objs.append(
                            RoutedParlayLeg(
                                event_id=sp.event_id,
                                matchup=sp.matchup,
                                commence_time=sp.commence_time,
                                market=sp.market,
                                side=sp.side_label,
                                line=ln,
                                american=pr,
                                decimal=dec,
                                fair_prob=p_leg,
                                leg_ev_at_book=p_leg * dec - 1.0,
                                best_single_book=sp.best_book_display,
                                best_single_american=sp.best_book_price,
                            )
                        )
                    ev_b = p_joint * dec_prod - 1.0
                    book_evals.append((ev_b, dec_prod, p_joint, ck, disp_book, leg_objs))

                book_evals.sort(key=lambda x: (-x[0], -x[1]))
                best_ev, best_dec, best_p_joint, best_ck, best_disp, best_legs = book_evals[0]
                if max_parlay_decimal is not None and best_dec > float(max_parlay_decimal):
                    continue
                book_imp = 1.0 / best_dec
                price_edge = best_p_joint / book_imp - 1.0
                if best_ev < float(min_parlay_ev) or price_edge < float(min_price_edge):
                    continue

                evs = np.array([x[0] for x in book_evals], dtype=float)
                decs = np.array([x[1] for x in book_evals], dtype=float)
                med_ev = float(np.median(evs))
                worst_ev = float(np.min(evs))
                med_dec = float(np.median(decs))
                worst_dec = float(np.min(decs))

                rollover_dec = 1.0
                rollover_p = 1.0
                rollover_books: list[str] = []
                for sp in picked:
                    best_ck_leg, best_leg_tuple = max(
                        sp.book_quotes.items(),
                        key=lambda kv: (
                            _prob_for_book_quote(sp, kv[1][2], canon_book=kv[0])
                            * american_to_decimal(kv[1][1])
                        ),
                    )
                    rollover_books.append(best_leg_tuple[0])
                    rollover_dec *= american_to_decimal(best_leg_tuple[1])
                    rollover_p *= _prob_for_book_quote(
                        sp, best_leg_tuple[2], canon_book=best_ck_leg
                    )
                rollover_ev = rollover_p * rollover_dec - 1.0

                parlay_american = (
                    (best_dec - 1.0) * 100.0 if best_dec >= 2.0 else -100.0 / (best_dec - 1.0)
                )
                k_frac = kelly_fraction(
                    best_p_joint,
                    parlay_american,
                    fraction=kelly_frac,
                    max_bet_frac=max_bet_frac,
                )
                stake_dollars = math.floor(float(bankroll) * k_frac * 100.0) / 100.0

                all_candidates.append(
                    RoutedParlayTicket(
                        game_date=gdate,
                        n_legs=n,
                        routed_book=best_disp,
                        routed_canonical_book=best_ck,
                        legs=best_legs,
                        p_joint_fair=float(best_p_joint),
                        routed_parlay_decimal=float(best_dec),
                        routed_book_implied=float(book_imp),
                        routed_price_edge=float(price_edge),
                        routed_ev_per_unit=float(best_ev),
                        n_books_offering_all_legs=len(book_evals),
                        median_book_decimal=med_dec,
                        worst_book_decimal=worst_dec,
                        median_book_ev=med_ev,
                        worst_book_ev=worst_ev,
                        ev_gain_vs_median=float(best_ev - med_ev),
                        ev_gain_vs_worst=float(best_ev - worst_ev),
                        rollover_best_leg_decimal=float(rollover_dec),
                        rollover_best_leg_ev=float(rollover_ev),
                        rollover_books=rollover_books,
                        kelly_stake_fraction=float(k_frac),
                        recommended_stake=float(stake_dollars),
                        bankroll=float(bankroll),
                        book_ev_table={x[3]: float(x[0]) for x in book_evals},
                    )
                )

    if not all_candidates:
        return []

    # Greedy non-overlapping selection per (day, leg_bucket: singles vs parlays)
    selected: list[RoutedParlayTicket] = []
    by_day_tickets: dict[tuple[str, bool], list[RoutedParlayTicket]] = {}
    for t in all_candidates:
        by_day_tickets.setdefault((t.game_date, t.n_legs == 1), []).append(t)

    for (_gdate, _is_single), tickets in sorted(by_day_tickets.items()):
        tickets.sort(key=lambda t: (-t.routed_ev_per_unit, -t.routed_price_edge))
        used_events: set[str] = set()
        kept = 0
        for t in tickets:
            t_events = {leg.event_id for leg in t.legs}
            if t_events & used_events:
                continue
            selected.append(t)
            used_events |= t_events
            kept += 1
            if kept >= int(max_per_day):
                break

    selected.sort(key=lambda t: (t.game_date, t.n_legs, -t.routed_ev_per_unit))
    return selected


def route_single_bets_across_books(
    events: Sequence[MultiBookEvent],
    *,
    devig_method: str = "shin",
    markets: Sequence[str] = ("moneyline", "spread", "total"),
    execution_books: Sequence[str] | None = DEFAULT_ROUTING_BOOKS,
    min_peer_weight: float = 4.0,
    min_ev: float = 0.0,
    max_ev: float = 0.15,
    max_per_day: int = 10,
    bankroll: float = 50.0,
    kelly_frac: float = KELLY_FRACTION,
    max_bet_frac: float = 0.025,
) -> list[RoutedParlayTicket]:
    """Route top-down +EV single bets (`n=1`) anchored on sharp Shin consensus."""
    return route_parlays_across_books(
        events,
        devig_method=devig_method,
        markets=markets,
        execution_books=execution_books,
        sizes=(1,),
        min_peer_weight=min_peer_weight,
        min_best_leg_ev=min_ev,
        max_leg_ev=max_ev,
        min_parlay_ev=min_ev,
        max_per_day=max_per_day,
        bankroll=bankroll,
        kelly_frac=kelly_frac,
        max_bet_frac=max_bet_frac,
    )


def format_market_shopping_report(
    consensus_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
    tickets: Sequence[RoutedParlayTicket],
    *,
    source: str = "",
) -> str:
    """Format human-readable CLI report for sharp consensus, outliers, and routed parlays."""
    lines: list[str] = []
    if source:
        lines.append(f"source: {source}")
    lines.append(
        f"markets_analyzed: {len(consensus_rows)} | stale_outliers: {len(outlier_rows)} "
        f"| routed_parlays: {len(tickets)}"
    )

    if outlier_rows:
        lines.append("\n=== STALE OUTLIER QUOTES (+EV VS SHARP LOO CONSENSUS) ===")
        for r in outlier_rows[:20]:
            ln_str = (
                f" {r['line']:+g}"
                if (r["line"] is not None and r["market"] == "spread")
                else (f" {r['line']:g}" if r["line"] is not None else "")
            )
            lo_pts = r.get("line_out_pts")
            lo_str = (
                f" [stale {lo_pts:+.1f}pt vs {r.get('consensus_line'):g}]"
                if (
                    lo_pts is not None
                    and abs(float(lo_pts)) >= 0.25
                    and r.get("consensus_line") is not None
                )
                else ""
            )
            fair_am = r["consensus_fair_american"]
            lines.append(
                f"  {r['game_date']} {r['matchup']:9s} {r['market']:9s} "
                f"{r['side']}{ln_str}{lo_str} "
                f"@ {r['book']} ({r['book_american']:+.0f}) | fair {fair_am:+.0f} "
                f"(p={r['consensus_prob']:.3f}) | EV={r['ev_pct']:+.2f}% | z={r['z_score']:+.2f}"
            )
    else:
        lines.append("\n(no single-leg +EV outliers cleared threshold)")

    singles = [t for t in tickets if t.n_legs == 1]
    parlays = [t for t in tickets if t.n_legs >= 2]

    if singles:
        lines.append("\n=== ROUTED +EV SINGLE PLAYS (TOP-DOWN SHARP CONSENSUS) ===")
        for i, t in enumerate(singles, 1):
            L = t.legs[0]
            ln_s = (
                f"({L.line:+g})"
                if (L.line is not None and L.market == "spread")
                else (f"({L.line:g})" if L.line is not None else "")
            )
            b0_ev = t.book_ev_table.get("betmgm")
            b0_str = f", vs BetMGM[0] {b0_ev * 100.0:+.2f}%" if b0_ev is not None else ""
            lines.append(
                f"  #{i} [{t.game_date}] {L.matchup} {L.market}:{L.side}{ln_s} "
                f"-> {t.routed_book} [{L.american:+.0f}] "
                f"(p_sharp={t.p_joint_fair:.3f}, EV={t.routed_ev_per_unit * 100.0:+.2f}%, "
                f"median {t.median_book_ev * 100.0:+.2f}%{b0_str}, "
                f"stake=${t.recommended_stake:.2f})"
            )

    if parlays:
        lines.append("\n=== BEST CROSS-BOOK ROUTED PARLAYS ===")
        for i, t in enumerate(parlays, 1):
            leg_desc = " + ".join(
                f"{L.matchup} {L.market}:{L.side}"
                + (
                    f"({L.line:+g})"
                    if (L.line is not None and L.market == "spread")
                    else (f"({L.line:g})" if L.line is not None else "")
                )
                + f" [{L.american:+.0f}]"
                for L in t.legs
            )
            b0_ev = t.book_ev_table.get("betmgm")
            diff_b0 = (t.routed_ev_per_unit - b0_ev) * 100.0 if b0_ev is not None else 0.0
            b0_str = (
                f", vs BetMGM[0] {b0_ev * 100.0:+.2f}% (+{diff_b0:.2f}%)"
                if b0_ev is not None
                else ""
            )
            lines.append(
                f"  #{i} [{t.game_date}] ROUTE -> {t.routed_book} "
                f"({t.n_legs}-leg, dec={t.routed_parlay_decimal:.3f}x, "
                f"p_joint={t.p_joint_fair:.3f})"
            )
            lines.append(f"     legs: {leg_desc}")
            lines.append(
                f"     EV={t.routed_ev_per_unit * 100.0:+.2f}% "
                f"(median book {t.median_book_ev * 100.0:+.2f}%, "
                f"worst book {t.worst_book_ev * 100.0:+.2f}%{b0_str})"
            )
            lines.append(
                f"     1/4-Kelly stake on ${t.bankroll:.2f}: ${t.recommended_stake:.2f} "
                f"({t.kelly_stake_fraction * 100.0:.2f}%) | "
                f"staggered rollover ceiling: {t.rollover_best_leg_decimal:.3f}x "
                f"({t.rollover_best_leg_ev * 100.0:+.2f}%)"
            )
    elif not singles:
        lines.append("\n(no +EV cross-book tickets cleared threshold)")

    return "\n".join(lines)
