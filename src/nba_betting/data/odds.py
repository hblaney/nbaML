"""Thin The Odds API adapter for NBA featured markets (h2h / spreads / totals)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from nba_betting.data.cache import write_frame
from nba_betting.data.paths import raw_dir
from nba_betting.data.teams import team_to_tricode

ODDS_SPORT_KEY = "basketball_nba"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"

OddsHttpGetter = Callable[[str, dict[str, str]], Any]


class OddsApiConfigError(RuntimeError):
    """Raised when the Odds API cannot be used (missing key, etc.)."""


@dataclass(frozen=True)
class EventOdds:
    """Stable odds contract (provider-agnostic)."""

    event_id: str
    commence_time: str
    home: str
    away: str
    home_tricode: str | None
    away_tricode: str | None
    moneyline_home: float | None
    moneyline_away: float | None
    spread_home_point: float | None
    spread_home_price: float | None
    spread_away_point: float | None
    spread_away_price: float | None
    total_point: float | None
    total_over_price: float | None
    total_under_price: float | None
    book: str
    asof_ts: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_odds_api_key() -> str | None:
    """Return ``ODDS_API_KEY`` if set and non-empty."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    return key or None


def odds_env_settings() -> dict[str, str]:
    return {
        "regions": os.environ.get("ODDS_REGIONS", DEFAULT_REGIONS).strip() or DEFAULT_REGIONS,
        "markets": os.environ.get("ODDS_MARKETS", DEFAULT_MARKETS).strip() or DEFAULT_MARKETS,
    }


def odds_cache_path(root: Path | None = None, *, asof: str | None = None) -> Path:
    stamp = asof or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return raw_dir(root) / "odds" / f"nba_{stamp}.json"


def _default_http_get(url: str, params: dict[str, str]) -> Any:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _pick_bookmaker(event: dict[str, Any], preferred: str | None = None) -> dict[str, Any] | None:
    books = event.get("bookmakers") or []
    if not books:
        return None
    if preferred:
        for book in books:
            if book.get("key") == preferred:
                return book
    return books[0]


def _market_by_key(book: dict[str, Any], key: str) -> dict[str, Any] | None:
    for market in book.get("markets") or []:
        if market.get("key") == key:
            return market
    return None


def parse_event_odds(
    event: dict[str, Any],
    *,
    book_key: str | None = None,
    asof_ts: str | None = None,
) -> EventOdds | None:
    """Flatten one Odds API event into ``EventOdds`` (first / preferred book)."""
    book = _pick_bookmaker(event, preferred=book_key)
    if book is None:
        return None

    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    asof = asof_ts or datetime.now(timezone.utc).isoformat()

    ml_home = ml_away = None
    h2h = _market_by_key(book, "h2h")
    if h2h:
        for outcome in h2h.get("outcomes") or []:
            name = outcome.get("name")
            price = outcome.get("price")
            if name == home:
                ml_home = float(price)
            elif name == away:
                ml_away = float(price)

    sp_home_pt = sp_home_price = sp_away_pt = sp_away_price = None
    spreads = _market_by_key(book, "spreads")
    if spreads:
        for outcome in spreads.get("outcomes") or []:
            name = outcome.get("name")
            if name == home:
                sp_home_pt = float(outcome.get("point"))
                sp_home_price = float(outcome.get("price"))
            elif name == away:
                sp_away_pt = float(outcome.get("point"))
                sp_away_price = float(outcome.get("price"))

    total_pt = over_price = under_price = None
    totals = _market_by_key(book, "totals")
    if totals:
        for outcome in totals.get("outcomes") or []:
            name = str(outcome.get("name") or "").lower()
            if name == "over":
                total_pt = float(outcome.get("point"))
                over_price = float(outcome.get("price"))
            elif name == "under":
                total_pt = float(outcome.get("point")) if total_pt is None else total_pt
                under_price = float(outcome.get("price"))

    return EventOdds(
        event_id=str(event.get("id") or ""),
        commence_time=str(event.get("commence_time") or ""),
        home=home,
        away=away,
        home_tricode=team_to_tricode(home),
        away_tricode=team_to_tricode(away),
        moneyline_home=ml_home,
        moneyline_away=ml_away,
        spread_home_point=sp_home_pt,
        spread_home_price=sp_home_price,
        spread_away_point=sp_away_pt,
        spread_away_price=sp_away_price,
        total_point=total_pt,
        total_over_price=over_price,
        total_under_price=under_price,
        book=str(book.get("key") or book.get("title") or ""),
        asof_ts=asof,
    )


def fetch_nba_odds(
    *,
    api_key: str | None = None,
    regions: str | None = None,
    markets: str | None = None,
    http_get: OddsHttpGetter | None = None,
    book_key: str | None = None,
) -> list[EventOdds]:
    """Fetch featured NBA odds. Raises ``OddsApiConfigError`` if key missing."""
    key = api_key if api_key is not None else get_odds_api_key()
    if not key:
        raise OddsApiConfigError(
            "ODDS_API_KEY is not set. Add it to .env or the environment, "
            "or run without odds (model-only mode)."
        )
    settings = odds_env_settings()
    params = {
        "apiKey": key,
        "regions": regions or settings["regions"],
        "markets": markets or settings["markets"],
        "oddsFormat": "american",
    }
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds"
    getter = http_get or _default_http_get
    payload = getter(url, params)
    if not isinstance(payload, list):
        raise RuntimeError("unexpected Odds API payload (expected a list of events)")
    asof = datetime.now(timezone.utc).isoformat()
    out: list[EventOdds] = []
    for event in payload:
        parsed = parse_event_odds(event, book_key=book_key, asof_ts=asof)
        if parsed is not None:
            out.append(parsed)
    return out


def pull_odds(
    *,
    root: Path | None = None,
    api_key: str | None = None,
    http_get: OddsHttpGetter | None = None,
    book_key: str | None = None,
) -> Path:
    """Fetch odds and cache JSON under ``data/raw/odds/``."""
    events = fetch_nba_odds(api_key=api_key, http_get=http_get, book_key=book_key)
    path = odds_cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.to_dict() for e in events], indent=2) + "\n",
        encoding="utf-8",
    )
    # Also write a flat parquet/csv for quick inspection when non-empty.
    if events:
        import pandas as pd

        write_frame(pd.DataFrame([e.to_dict() for e in events]), path.with_suffix(".parquet"))
    return path


def load_odds_cache(path: Path) -> list[EventOdds]:
    """Load ``EventOdds`` list from a cached JSON pull."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EventOdds(**row) for row in payload]


def latest_odds_cache(root: Path | None = None) -> Path | None:
    folder = raw_dir(root) / "odds"
    if not folder.exists():
        return None
    files = sorted(folder.glob("nba_*.json"))
    return files[-1] if files else None
