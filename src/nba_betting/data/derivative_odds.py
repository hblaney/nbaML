"""The Odds API v4 adapter for NBA 1st Quarter (1Q) and 1st Half (1H) derivative markets.

Supported market keys:
- ``h2h_q1``
- ``spreads_q1``
- ``totals_q1``
- ``h2h_h1``
- ``spreads_h1``
- ``totals_h1``

Supports both per-event Odds API v4 payloads (``/events/{eventId}/odds``) and bulk
event lists when bookmakers include period markets.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nba_betting.data.cache import write_frame
from nba_betting.data.odds import (
    ODDS_API_BASE,
    ODDS_SPORT_KEY,
    OddsApiConfigError,
    OddsHttpGetter,
    _default_http_get,
    _market_by_key,
    _pick_bookmaker,
    fetch_nba_odds,
    get_odds_api_key,
    odds_env_settings,
)
from nba_betting.data.paths import raw_dir
from nba_betting.data.teams import team_to_tricode

DERIVATIVE_MARKETS: tuple[str, ...] = (
    "h2h_q1",
    "spreads_q1",
    "totals_q1",
    "h2h_h1",
    "spreads_h1",
    "totals_h1",
)
DEFAULT_DERIVATIVE_MARKETS_CSV = ",".join(DERIVATIVE_MARKETS)


def derivative_markets_from_env() -> tuple[str, ...]:
    raw = os.environ.get("ODDS_DERIVATIVE_MARKETS", "").strip()
    if not raw:
        return DERIVATIVE_MARKETS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class DerivativeEventOdds:
    """Normalized 1Q and 1H derivative lines for one NBA game."""

    event_id: str
    commence_time: str
    home: str
    away: str
    home_tricode: str | None
    away_tricode: str | None
    # 1st Quarter (Q1)
    q1_moneyline_home: float | None
    q1_moneyline_away: float | None
    q1_spread_home_point: float | None
    q1_spread_home_price: float | None
    q1_spread_away_point: float | None
    q1_spread_away_price: float | None
    q1_total_point: float | None
    q1_total_over_price: float | None
    q1_total_under_price: float | None
    # 1st Half (H1)
    h1_moneyline_home: float | None
    h1_moneyline_away: float | None
    h1_spread_home_point: float | None
    h1_spread_home_price: float | None
    h1_spread_away_point: float | None
    h1_spread_away_price: float | None
    h1_total_point: float | None
    h1_total_over_price: float | None
    h1_total_under_price: float | None
    book: str
    asof_ts: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_any_derivative(self) -> bool:
        return any(
            val is not None
            for val in (
                self.q1_moneyline_home,
                self.q1_spread_home_point,
                self.q1_total_point,
                self.h1_moneyline_home,
                self.h1_spread_home_point,
                self.h1_total_point,
            )
        )


def derivative_odds_cache_path(root: Path | None = None, *, asof: str | None = None) -> Path:
    stamp = asof or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return raw_dir(root) / "odds" / f"nba_derivatives_{stamp}.json"


def _parse_period_h2h(
    book: dict[str, Any], market_key: str, home: str, away: str
) -> tuple[float | None, float | None]:
    ml_home = ml_away = None
    mkt = _market_by_key(book, market_key)
    if mkt:
        for outcome in mkt.get("outcomes") or []:
            name = outcome.get("name")
            price = outcome.get("price")
            if price is None:
                continue
            if name == home:
                ml_home = float(price)
            elif name == away:
                ml_away = float(price)
    return ml_home, ml_away


def _parse_period_spreads(
    book: dict[str, Any], market_key: str, home: str, away: str
) -> tuple[float | None, float | None, float | None, float | None]:
    sp_h_pt = sp_h_pr = sp_a_pt = sp_a_pr = None
    mkt = _market_by_key(book, market_key)
    if mkt:
        for outcome in mkt.get("outcomes") or []:
            name = outcome.get("name")
            pt = outcome.get("point")
            pr = outcome.get("price")
            if pt is None or pr is None:
                continue
            if name == home:
                sp_h_pt = float(pt)
                sp_h_pr = float(pr)
            elif name == away:
                sp_a_pt = float(pt)
                sp_a_pr = float(pr)
    return sp_h_pt, sp_h_pr, sp_a_pt, sp_a_pr


def _parse_period_totals(
    book: dict[str, Any], market_key: str
) -> tuple[float | None, float | None, float | None]:
    tot_pt = ov_pr = un_pr = None
    mkt = _market_by_key(book, market_key)
    if mkt:
        for outcome in mkt.get("outcomes") or []:
            side = str(outcome.get("name") or "").strip().lower()
            pt = outcome.get("point")
            pr = outcome.get("price")
            if pt is None or pr is None:
                continue
            if side == "over":
                tot_pt = float(pt)
                ov_pr = float(pr)
            elif side == "under":
                tot_pt = float(pt) if tot_pt is None else tot_pt
                un_pr = float(pr)
    return tot_pt, ov_pr, un_pr


def parse_derivative_event_odds(
    event_payload: dict[str, Any],
    *,
    book_key: str | None = None,
    asof_ts: str | None = None,
) -> DerivativeEventOdds | None:
    """Parse one Odds API event payload into ``DerivativeEventOdds``."""
    books = event_payload.get("bookmakers") or []
    if not books:
        return None

    # Prefer requested book if it carries at least one derivative market; otherwise pick first book with derivative markets
    chosen_book: dict[str, Any] | None = None
    if book_key:
        cand = _pick_bookmaker(event_payload, preferred=book_key)
        if cand is not None:
            mkeys = {str(m.get("key") or "") for m in (cand.get("markets") or [])}
            if mkeys & set(DERIVATIVE_MARKETS):
                chosen_book = cand
    if chosen_book is None:
        for b in books:
            mkeys = {str(m.get("key") or "") for m in (b.get("markets") or [])}
            if mkeys & set(DERIVATIVE_MARKETS):
                chosen_book = b
                break
    if chosen_book is None:
        chosen_book = _pick_bookmaker(event_payload, preferred=book_key)
    if chosen_book is None:
        return None

    home = str(event_payload.get("home_team") or event_payload.get("home") or "")
    away = str(event_payload.get("away_team") or event_payload.get("away") or "")
    asof = asof_ts or datetime.now(timezone.utc).isoformat()

    q1_ml_h, q1_ml_a = _parse_period_h2h(chosen_book, "h2h_q1", home, away)
    q1_sp_h_pt, q1_sp_h_pr, q1_sp_a_pt, q1_sp_a_pr = _parse_period_spreads(
        chosen_book, "spreads_q1", home, away
    )
    q1_tot_pt, q1_ov_pr, q1_un_pr = _parse_period_totals(chosen_book, "totals_q1")

    h1_ml_h, h1_ml_a = _parse_period_h2h(chosen_book, "h2h_h1", home, away)
    h1_sp_h_pt, h1_sp_h_pr, h1_sp_a_pt, h1_sp_a_pr = _parse_period_spreads(
        chosen_book, "spreads_h1", home, away
    )
    h1_tot_pt, h1_ov_pr, h1_un_pr = _parse_period_totals(chosen_book, "totals_h1")

    return DerivativeEventOdds(
        event_id=str(event_payload.get("id") or event_payload.get("event_id") or ""),
        commence_time=str(event_payload.get("commence_time") or ""),
        home=home,
        away=away,
        home_tricode=team_to_tricode(home),
        away_tricode=team_to_tricode(away),
        q1_moneyline_home=q1_ml_h,
        q1_moneyline_away=q1_ml_a,
        q1_spread_home_point=q1_sp_h_pt,
        q1_spread_home_price=q1_sp_h_pr,
        q1_spread_away_point=q1_sp_a_pt,
        q1_spread_away_price=q1_sp_a_pr,
        q1_total_point=q1_tot_pt,
        q1_total_over_price=q1_ov_pr,
        q1_total_under_price=q1_un_pr,
        h1_moneyline_home=h1_ml_h,
        h1_moneyline_away=h1_ml_a,
        h1_spread_home_point=h1_sp_h_pt,
        h1_spread_home_price=h1_sp_h_pr,
        h1_spread_away_point=h1_sp_a_pt,
        h1_spread_away_price=h1_sp_a_pr,
        h1_total_point=h1_tot_pt,
        h1_total_over_price=h1_ov_pr,
        h1_total_under_price=h1_un_pr,
        book=str(chosen_book.get("key") or chosen_book.get("title") or ""),
        asof_ts=asof,
    )


def fetch_event_derivative_odds(
    event_id: str,
    *,
    api_key: str | None = None,
    regions: str | None = None,
    markets: Sequence[str] | None = None,
    http_get: OddsHttpGetter | None = None,
    book_key: str | None = None,
    event_meta: dict[str, Any] | None = None,
) -> DerivativeEventOdds | None:
    """Fetch 1Q/1H derivative markets for one event via ``/events/{eventId}/odds``."""
    key = api_key if api_key is not None else get_odds_api_key()
    if not key:
        raise OddsApiConfigError(
            "ODDS_API_KEY is not set. Add it to .env or the environment, "
            "or run derivatives in model-only mode."
        )
    settings = odds_env_settings()
    market_csv = ",".join(markets or derivative_markets_from_env())
    params = {
        "apiKey": key,
        "regions": regions or settings["regions"],
        "markets": market_csv,
        "oddsFormat": "american",
    }
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/events/{event_id}/odds"
    getter = http_get or _default_http_get
    payload = getter(url, params)
    if not isinstance(payload, dict):
        raise RuntimeError("unexpected Odds API derivative payload (expected event object)")
    merged = {**(event_meta or {}), **payload}
    if "id" not in merged:
        merged["id"] = event_id
    return parse_derivative_event_odds(
        merged,
        book_key=book_key,
        asof_ts=datetime.now(timezone.utc).isoformat(),
    )


def fetch_nba_derivative_odds(
    *,
    api_key: str | None = None,
    regions: str | None = None,
    markets: Sequence[str] | None = None,
    event_ids: Sequence[str] | None = None,
    http_get: OddsHttpGetter | None = None,
    book_key: str | None = None,
) -> list[DerivativeEventOdds]:
    """Fetch 1Q/1H derivative odds across active NBA events."""
    key = api_key if api_key is not None else get_odds_api_key()
    if not key:
        raise OddsApiConfigError(
            "ODDS_API_KEY is not set. Add it to .env or the environment, "
            "or run derivatives in model-only mode."
        )
    getter = http_get or _default_http_get
    if event_ids is None:
        featured = fetch_nba_odds(api_key=key, regions=regions, http_get=getter, book_key=book_key)
        event_metas = [
            {
                "id": e.event_id,
                "commence_time": e.commence_time,
                "home_team": e.home,
                "away_team": e.away,
            }
            for e in featured
        ]
    else:
        event_metas = [{"id": str(eid)} for eid in event_ids]

    out: list[DerivativeEventOdds] = []
    for meta in event_metas:
        eid = str(meta["id"])
        try:
            parsed = fetch_event_derivative_odds(
                eid,
                api_key=key,
                regions=regions,
                markets=markets,
                http_get=getter,
                book_key=book_key,
                event_meta=meta,
            )
            if parsed is not None and parsed.has_any_derivative():
                out.append(parsed)
        except Exception:
            continue
    return out


def pull_derivative_odds(
    *,
    root: Path | None = None,
    api_key: str | None = None,
    regions: str | None = None,
    markets: Sequence[str] | None = None,
    event_ids: Sequence[str] | None = None,
    http_get: OddsHttpGetter | None = None,
    book_key: str | None = None,
) -> Path:
    """Fetch 1Q/1H derivative odds and cache under ``data/raw/odds/nba_derivatives_*.json``."""
    events = fetch_nba_derivative_odds(
        api_key=api_key,
        regions=regions,
        markets=markets,
        event_ids=event_ids,
        http_get=http_get,
        book_key=book_key,
    )
    path = derivative_odds_cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.to_dict() for e in events], indent=2) + "\n",
        encoding="utf-8",
    )
    if events:
        import pandas as pd

        write_frame(pd.DataFrame([e.to_dict() for e in events]), path.with_suffix(".parquet"))
    return path


def load_derivative_odds_cache(path: Path) -> list[DerivativeEventOdds]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DerivativeEventOdds(**row) for row in payload]


def latest_derivative_odds_cache(root: Path | None = None) -> Path | None:
    folder = raw_dir(root) / "odds"
    if not folder.exists():
        return None
    files = sorted(folder.glob("nba_derivatives_*.json"))
    return files[-1] if files else None
