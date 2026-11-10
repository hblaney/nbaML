"""Odds API extension for NBA player prop markets (v1.5).

Per-event endpoint (paid tier may be required)::

    GET /v4/sports/basketball_nba/events/{eventId}/odds
      ?regions=us
      &markets=player_points,player_rebounds,player_assists,player_threes,
               player_points_rebounds_assists
      &oddsFormat=american
      &apiKey=…

Market keys (documented):
- ``player_points``
- ``player_rebounds``
- ``player_assists``
- ``player_threes``
- ``player_points_rebounds_assists`` (PRA)

No DFS / PrizePicks markets in v1.5.
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
    _pick_bookmaker,
    fetch_nba_odds,
    get_odds_api_key,
    odds_env_settings,
)
from nba_betting.data.paths import raw_dir

DEFAULT_PROP_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
)


def prop_markets_from_env() -> tuple[str, ...]:
    raw = os.environ.get("ODDS_PROP_MARKETS", "").strip()
    if not raw:
        return DEFAULT_PROP_MARKETS
    return tuple(part.strip() for part in raw.split(",") if part.strip())

PROP_MARKET_TO_STAT = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pra",
}


@dataclass(frozen=True)
class PropOdds:
    event_id: str
    commence_time: str
    home: str
    away: str
    player_name: str
    market: str  # Odds API market key
    stat: str  # points|rebounds|assists|threes|pra
    line: float
    over_price: float | None
    under_price: float | None
    book: str
    asof_ts: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prop_odds_cache_path(root: Path | None = None, *, asof: str | None = None) -> Path:
    stamp = asof or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return raw_dir(root) / "odds" / f"nba_props_{stamp}.json"


def parse_prop_markets(
    event_payload: dict[str, Any],
    *,
    asof_ts: str | None = None,
    book_key: str | None = None,
) -> list[PropOdds]:
    """Parse an event odds payload (with ``bookmakers``) into ``PropOdds`` rows."""
    book = _pick_bookmaker(event_payload, preferred=book_key)
    if book is None:
        return []

    asof = asof_ts or datetime.now(timezone.utc).isoformat()
    home = str(event_payload.get("home_team") or event_payload.get("home") or "")
    away = str(event_payload.get("away_team") or event_payload.get("away") or "")
    event_id = str(event_payload.get("id") or event_payload.get("event_id") or "")
    commence = str(event_payload.get("commence_time") or "")
    book_name = str(book.get("key") or book.get("title") or "")

    rows: list[PropOdds] = []
    for market in book.get("markets") or []:
        mkey = str(market.get("key") or "")
        if mkey not in PROP_MARKET_TO_STAT:
            continue
        by_player: dict[tuple[str, float], dict[str, float]] = {}
        for outcome in market.get("outcomes") or []:
            player = str(outcome.get("description") or "").strip()
            side = str(outcome.get("name") or "").strip().lower()
            if side not in {"over", "under"}:
                continue
            if not player:
                continue
            point = float(outcome.get("point"))
            price = float(outcome.get("price"))
            by_player.setdefault((player, point), {})[side] = price
        for (player, point), prices in by_player.items():
            rows.append(
                PropOdds(
                    event_id=event_id,
                    commence_time=commence,
                    home=home,
                    away=away,
                    player_name=player,
                    market=mkey,
                    stat=PROP_MARKET_TO_STAT[mkey],
                    line=float(point),
                    over_price=prices.get("over"),
                    under_price=prices.get("under"),
                    book=book_name,
                    asof_ts=asof,
                )
            )
    return rows


def fetch_event_prop_odds(
    event_id: str,
    *,
    api_key: str | None = None,
    regions: str | None = None,
    markets: Sequence[str] | None = None,
    http_get: OddsHttpGetter | None = None,
    event_meta: dict[str, Any] | None = None,
) -> list[PropOdds]:
    """Fetch prop markets for a single event. Raises if key missing."""
    key = api_key if api_key is not None else get_odds_api_key()
    if not key:
        raise OddsApiConfigError(
            "ODDS_API_KEY is not set. Add it to .env or the environment, "
            "or run props in model-only mode."
        )
    settings = odds_env_settings()
    market_list = ",".join(markets or prop_markets_from_env())
    params = {
        "apiKey": key,
        "regions": regions or settings["regions"],
        "markets": market_list,
        "oddsFormat": "american",
    }
    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/events/{event_id}/odds"
    getter = http_get or _default_http_get
    payload = getter(url, params)
    if not isinstance(payload, dict):
        raise RuntimeError("unexpected Odds API prop payload (expected object)")
    # Merge meta so home/away survive if the endpoint omits them.
    merged = {**(event_meta or {}), **payload}
    if "id" not in merged:
        merged["id"] = event_id
    return parse_prop_markets(merged, asof_ts=datetime.now(timezone.utc).isoformat())


def fetch_nba_prop_odds(
    *,
    api_key: str | None = None,
    event_ids: Sequence[str] | None = None,
    http_get: OddsHttpGetter | None = None,
    markets: Sequence[str] | None = None,
) -> list[PropOdds]:
    """Fetch props for featured events (or an explicit event id list)."""
    key = api_key if api_key is not None else get_odds_api_key()
    if not key:
        raise OddsApiConfigError(
            "ODDS_API_KEY is not set. Add it to .env or the environment, "
            "or run props in model-only mode."
        )
    getter = http_get or _default_http_get
    if event_ids is None:
        featured = fetch_nba_odds(api_key=key, http_get=getter)
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
        event_metas = [{"id": eid} for eid in event_ids]

    rows: list[PropOdds] = []
    for meta in event_metas:
        eid = str(meta["id"])
        try:
            rows.extend(
                fetch_event_prop_odds(
                    eid,
                    api_key=key,
                    http_get=getter,
                    markets=markets,
                    event_meta=meta,
                )
            )
        except Exception:
            # Degrade per-event (tier / empty books) without failing the slate.
            continue
    return rows


def pull_prop_odds(
    *,
    root: Path | None = None,
    api_key: str | None = None,
    http_get: OddsHttpGetter | None = None,
) -> Path:
    """Fetch prop odds and cache under ``data/raw/odds/nba_props_*.json``."""
    rows = fetch_nba_prop_odds(api_key=api_key, http_get=http_get)
    path = prop_odds_cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in rows], indent=2) + "\n", encoding="utf-8")
    if rows:
        import pandas as pd

        write_frame(pd.DataFrame([r.to_dict() for r in rows]), path.with_suffix(".parquet"))
    return path


def load_prop_odds_cache(path: Path) -> list[PropOdds]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PropOdds(**row) for row in payload]


def latest_prop_odds_cache(root: Path | None = None) -> Path | None:
    folder = raw_dir(root) / "odds"
    if not folder.exists():
        return None
    files = sorted(folder.glob("nba_props_*.json"))
    return files[-1] if files else None
