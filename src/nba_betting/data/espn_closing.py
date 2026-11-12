"""Historical opening & closing moneylines, spreads, and totals via ESPN's public sports.core API.

Extracts both ``open`` and ``close`` blocks for:
- Moneyline (`open_home_ml`, `open_away_ml`, `home_ml`, `away_ml`)
- Point Spread (`open_spread_home`, `open_spread_home_odds`, `open_spread_away_odds`,
  `spread_home`, `spread_home_odds`, `spread_away_odds`)
- Game Total (`open_total_line`, `open_over_odds`, `open_under_odds`,
  `total_line`, `over_odds`, `under_odds`)
along with both proportional (`novig`) and Shin (`shin`) de-vigged probabilities
and pregame open-to-close line steam features. Strictly rejects in-game "Live Odds"
feeds so zero post-tipoff state can leak into pregame prices.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
import math
import ssl
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from nba_betting.pricing.american import american_to_implied, devig_two_way
from nba_betting.pricing.devig import devig_shin_two_way

ESPN_ODDS_TMPL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds"
)
DEFAULT_UA = "Mozilla/5.0 (compatible; nbaML-backtest/0.4)"
SCHEMA_VERSION = 3

_LIVE_PROVIDER_IDS = {"59", "46"}
_NON_BOOK_PROVIDER_IDS = {"1001", "1002"}  # accuscore, teamrankings
_TLS_LOCAL = threading.local()
_SSL_CTX = ssl.create_default_context()


def _get_keepalive_conn(host: str) -> http.client.HTTPSConnection:
    conns: dict[str, http.client.HTTPSConnection] = getattr(_TLS_LOCAL, "conns", None) or {}
    _TLS_LOCAL.conns = conns
    conn = conns.get(host)
    if conn is None:
        conn = http.client.HTTPSConnection(host, timeout=25, context=_SSL_CTX)
        conns[host] = conn
    return conn


def _reset_keepalive_conn(host: str) -> None:
    conns: dict[str, http.client.HTTPSConnection] = getattr(_TLS_LOCAL, "conns", None) or {}
    old = conns.pop(host, None)
    if old is not None:
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass


def _http_json(url: str, *, retries: int = 4, pause: float = 0.5) -> Any:
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    last: Exception | None = None
    for i in range(retries):
        try:
            conn = _get_keepalive_conn(host)
            conn.request("GET", path, headers={"User-Agent": DEFAULT_UA, "Connection": "keep-alive"})
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 404:
                raise HTTPError(url, 404, "Not Found", resp.headers, None)
            if resp.status >= 400:
                _reset_keepalive_conn(host)
                raise RuntimeError(f"HTTP {resp.status} on {url}")
            return json.loads(body.decode())
        except HTTPError:
            raise
        except Exception as exc:  # noqa: BLE001
            _reset_keepalive_conn(host)
            last = exc
            time.sleep(pause * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _clean_text(raw: Any) -> str | None:
    if raw is None or isinstance(raw, (dict, list, bool)):
        return None
    s = str(raw).strip().replace("−", "-").replace(",", "")
    return s if s != "" else None


def _parse_american_price(
    raw: Any,
    *,
    min_abs: float = 100.0,
    max_abs: float = 25000.0,
) -> float | None:
    """Parse an American odds price (e.g. '-110', '+135', 'EVEN' -> +100.0)."""
    s = _clean_text(raw)
    if s is None:
        return None
    up = s.upper()
    if up in {"EVEN", "EV"}:
        return 100.0
    try:
        val = float(s)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    if abs(val) < min_abs or abs(val) > max_abs:
        return None
    return float(val)


def _parse_line_number(
    raw: Any,
    *,
    is_total: bool = False,
) -> float | None:
    """Parse a point spread or total line (e.g. '-4.5', 'PK' -> 0.0, '224.5')."""
    s = _clean_text(raw)
    if s is None:
        return None
    up = s.upper()
    if not is_total and up in {"PK", "PICK", "EVEN", "EV"}:
        return 0.0
    if is_total and up[:1] in {"O", "U"}:
        s = s[1:].strip()
    try:
        val = float(s)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    if is_total:
        if not (160.0 <= val <= 310.0):
            return None
    else:
        if not (-40.0 <= val <= 40.0):
            return None
    return float(val)


def _extract_price_from_quote(
    quote: Any,
    *,
    max_abs: float = 25000.0,
) -> float | None:
    if not isinstance(quote, dict):
        return None
    for key in ("american", "alternateDisplayValue"):
        val = _parse_american_price(quote.get(key), max_abs=max_abs)
        if val is not None:
            return val
    return None


def _extract_line_from_quote(
    quote: Any,
    *,
    is_total: bool = False,
) -> float | None:
    if not isinstance(quote, dict):
        return None
    for key in ("american", "alternateDisplayValue", "displayValue", "value"):
        val = _parse_line_number(quote.get(key), is_total=is_total)
        if val is not None:
            return val
    return None


def _parse_american_ml(side: dict[str, Any], odds_obj: dict[str, Any]) -> float | None:
    close = side.get("close") or {}
    val = _extract_price_from_quote(close.get("moneyLine"), max_abs=25000.0)
    if val is not None:
        return val
    cur = side.get("current") or {}
    val_cur = _extract_price_from_quote(cur.get("moneyLine"), max_abs=25000.0)
    if odds_obj.get("close") is not None and val_cur is not None:
        return val_cur
    top = _parse_american_price(side.get("moneyLine"), max_abs=25000.0)
    if top is not None:
        return top
    return val_cur


def _parse_open_ml(side: dict[str, Any]) -> float | None:
    op = side.get("open") or {}
    return _extract_price_from_quote(op.get("moneyLine"), max_abs=25000.0)


def _parse_spread_juice(side: dict[str, Any], odds_obj: dict[str, Any]) -> float | None:
    close = side.get("close") or {}
    val = _extract_price_from_quote(close.get("spread"), max_abs=400.0)
    if val is not None:
        return val
    cur = side.get("current") or {}
    val_cur = _extract_price_from_quote(cur.get("spread"), max_abs=400.0)
    if odds_obj.get("close") is not None and val_cur is not None:
        return val_cur
    top = _parse_american_price(side.get("spreadOdds"), max_abs=400.0)
    if top is not None:
        return top
    return val_cur


def _parse_open_spread_juice(side: dict[str, Any]) -> float | None:
    op = side.get("open") or {}
    return _extract_price_from_quote(op.get("spread"), max_abs=400.0)


def _parse_pair_spread_home(
    h_block: dict[str, Any],
    a_block: dict[str, Any],
) -> float | None:
    hp = _extract_line_from_quote(h_block.get("pointSpread"), is_total=False)
    ap = _extract_line_from_quote(a_block.get("pointSpread"), is_total=False)
    if hp is not None and ap is not None:
        if abs(hp + ap) > 1e-6:
            return None
        return hp
    if hp is not None:
        return hp
    if ap is not None:
        return -ap
    return None


def _parse_spread_home(
    home: dict[str, Any],
    away: dict[str, Any],
    odds_obj: dict[str, Any],
) -> float | None:
    """Extract closing home point spread (negative when home is favored)."""
    close_sp = _parse_pair_spread_home(home.get("close") or {}, away.get("close") or {})
    if close_sp is not None:
        return close_sp
    cur_sp = _parse_pair_spread_home(home.get("current") or {}, away.get("current") or {})
    if cur_sp is not None:
        return cur_sp
    return _parse_line_number(odds_obj.get("spread"), is_total=False)


def _parse_total_close(odds_obj: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Extract ``(total_line, over_odds, under_odds)`` for closing total."""
    close = odds_obj.get("close") or {}
    cur = odds_obj.get("current") or {}

    line = _extract_line_from_quote(close.get("total"), is_total=True)
    if line is None and odds_obj.get("close") is not None:
        line = _extract_line_from_quote(cur.get("total"), is_total=True)
    if line is None:
        line = _parse_line_number(odds_obj.get("overUnder"), is_total=True)
    if line is None:
        line = _extract_line_from_quote(cur.get("total"), is_total=True)

    over_odds = _extract_price_from_quote(close.get("over"), max_abs=400.0)
    if over_odds is None and odds_obj.get("close") is not None:
        over_odds = _extract_price_from_quote(cur.get("over"), max_abs=400.0)
    if over_odds is None:
        over_odds = _parse_american_price(odds_obj.get("overOdds"), max_abs=400.0)
    if over_odds is None:
        over_odds = _extract_price_from_quote(cur.get("over"), max_abs=400.0)

    under_odds = _extract_price_from_quote(close.get("under"), max_abs=400.0)
    if under_odds is None and odds_obj.get("close") is not None:
        under_odds = _extract_price_from_quote(cur.get("under"), max_abs=400.0)
    if under_odds is None:
        under_odds = _parse_american_price(odds_obj.get("underOdds"), max_abs=400.0)
    if under_odds is None:
        under_odds = _extract_price_from_quote(cur.get("under"), max_abs=400.0)

    return line, over_odds, under_odds


def _parse_total_open(odds_obj: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Extract ``(open_total_line, open_over_odds, open_under_odds)``."""
    op = odds_obj.get("open") or {}
    line = _extract_line_from_quote(op.get("total"), is_total=True)
    over_o = _extract_price_from_quote(op.get("over"), max_abs=400.0)
    under_o = _extract_price_from_quote(op.get("under"), max_abs=400.0)
    return line, over_o, under_o


def _spread_matches_ml_direction(
    spread_home: float | None,
    home_ml: float | None,
    away_ml: float | None,
) -> bool:
    """Reject corrupted/alt spreads whose favorite side contradicts a clear ML favorite."""
    if spread_home is None or home_ml is None or away_ml is None:
        return True
    if home_ml <= -145 and away_ml >= 115 and spread_home > 0.0:
        return False
    if away_ml <= -145 and home_ml >= 115 and spread_home < 0.0:
        return False
    return True


def _extract_provider_mls(odds_obj: dict[str, Any]) -> dict[str, Any]:
    """Extract full open & close ML, spread, and total fields from one ESPN provider object."""
    provider = odds_obj.get("provider") or {}
    home = odds_obj.get("homeTeamOdds") or {}
    away = odds_obj.get("awayTeamOdds") or {}

    home_ml = _parse_american_ml(home, odds_obj)
    away_ml = _parse_american_ml(away, odds_obj)
    open_home_ml = _parse_open_ml(home)
    open_away_ml = _parse_open_ml(away)

    spread_home = _parse_spread_home(home, away, odds_obj)
    spread_home_odds = _parse_spread_juice(home, odds_obj)
    spread_away_odds = _parse_spread_juice(away, odds_obj)
    if not _spread_matches_ml_direction(spread_home, home_ml, away_ml):
        spread_home = None
        spread_home_odds = None
        spread_away_odds = None

    open_spread_home = _parse_pair_spread_home(home.get("open") or {}, away.get("open") or {})
    open_spread_home_odds = _parse_open_spread_juice(home)
    open_spread_away_odds = _parse_open_spread_juice(away)
    if not _spread_matches_ml_direction(open_spread_home, open_home_ml, open_away_ml):
        open_spread_home = None
        open_spread_home_odds = None
        open_spread_away_odds = None

    total_line, over_odds, under_odds = _parse_total_close(odds_obj)
    open_total_line, open_over_odds, open_under_odds = _parse_total_open(odds_obj)

    has_close = bool(
        odds_obj.get("close")
        or (home.get("close") if isinstance(home, dict) else None)
        or (away.get("close") if isinstance(away, dict) else None)
    )
    has_open = bool(
        odds_obj.get("open")
        or (home.get("open") if isinstance(home, dict) else None)
        or (away.get("open") if isinstance(away, dict) else None)
    )
    return {
        "provider_id": str(provider.get("id")) if provider.get("id") is not None else None,
        "provider_name": provider.get("name"),
        "has_close_block": has_close,
        "has_open_block": has_open,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "open_home_ml": open_home_ml,
        "open_away_ml": open_away_ml,
        "spread_home": spread_home,
        "spread_home_odds": spread_home_odds,
        "spread_away_odds": spread_away_odds,
        "open_spread_home": open_spread_home,
        "open_spread_home_odds": open_spread_home_odds,
        "open_spread_away_odds": open_spread_away_odds,
        "total_line": total_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "open_total_line": open_total_line,
        "open_over_odds": open_over_odds,
        "open_under_odds": open_under_odds,
    }


# Explicit alias matching frontier spec naming
_extract_provider_full_odds = _extract_provider_mls


def _is_disallowed_provider(p: dict[str, Any]) -> bool:
    pid = str(p.get("provider_id") or "")
    name = str(p.get("provider_name") or "").lower()
    if "live" in name or pid in _LIVE_PROVIDER_IDS:
        return True
    if pid in _NON_BOOK_PROVIDER_IDS or name in {"accuscore", "teamrankings"}:
        return True
    return False


def _provider_priority_score(p: dict[str, Any]) -> int:
    pid = str(p.get("provider_id") or "")
    name = str(p.get("provider_name") or "")
    score = 0
    if pid == "58" or ("ESPN BET" in name and "Live" not in name):
        score += 30
    elif pid in {"100", "40", "41"} or "Draft" in name or "SugarHouse" in name:
        score += 22
    elif pid in {"45", "52", "31", "57"} or "Caesars" in name:
        score += 18
    elif pid == "47" or "MGM" in name:
        score += 16
    elif pid == "48" or "PointsBet" in name:
        score += 14
    elif pid == "50" or "Betfair" in name:
        score += 12
    elif pid == "1004" or name.lower() == "consensus":
        score += 8
    else:
        score += 4
    if p.get("has_close_block"):
        score += 5
    if p.get("has_open_block"):
        score += 2
    return score


def _has_valid_ml(p: dict[str, Any]) -> bool:
    return p.get("home_ml") is not None and p.get("away_ml") is not None


def _has_valid_spread(p: dict[str, Any]) -> bool:
    return (
        p.get("spread_home") is not None
        and p.get("spread_home_odds") is not None
        and p.get("spread_away_odds") is not None
    )


def _has_valid_total(p: dict[str, Any]) -> bool:
    return (
        p.get("total_line") is not None
        and p.get("over_odds") is not None
        and p.get("under_odds") is not None
    )


def _pick_best_provider(
    providers: list[dict[str, Any]],
    *,
    market: str = "ml",
) -> dict[str, Any] | None:
    """Pick highest-priority pregame sportsbook provider for ``market`` ('ml'|'spread'|'total')."""
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in providers:
        if not isinstance(p, dict) or _is_disallowed_provider(p):
            continue
        if market == "ml" and not _has_valid_ml(p):
            continue
        if market == "spread" and not _has_valid_spread(p):
            continue
        if market == "total" and not _has_valid_total(p):
            continue
        score = _provider_priority_score(p)
        if _has_valid_ml(p) and _has_valid_spread(p) and _has_valid_total(p):
            score += 3
        scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def _assemble_event_markets(eid: str, providers: list[dict[str, Any]]) -> dict[str, Any]:
    best_ml = _pick_best_provider(providers, market="ml")
    best_sp = _pick_best_provider(providers, market="spread")
    best_tot = _pick_best_provider(providers, market="total")

    if best_ml is not None and _has_valid_spread(best_ml):
        best_sp = best_ml
    if best_ml is not None and _has_valid_total(best_ml):
        best_tot = best_ml

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": eid,
        "providers": providers,
        "best": best_ml,
    }
    if best_ml is not None:
        home_ml = float(best_ml["home_ml"])
        away_ml = float(best_ml["away_ml"])
        fair_home, fair_away = devig_two_way(home_ml, away_ml)
        shin_home, shin_away = devig_shin_two_way(home_ml, away_ml)
        out.update(
            {
                "provider": best_ml.get("provider_name"),
                "has_close_block": bool(best_ml.get("has_close_block")),
                "has_open_block": bool(best_ml.get("has_open_block")),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "p_home_implied": american_to_implied(home_ml),
                "p_away_implied": american_to_implied(away_ml),
                "p_home_novig": fair_home,
                "p_away_novig": fair_away,
                "p_home_shin": shin_home,
                "p_away_shin": shin_away,
                "open_home_ml": best_ml.get("open_home_ml"),
                "open_away_ml": best_ml.get("open_away_ml"),
            }
        )
        if best_ml.get("open_home_ml") is not None and best_ml.get("open_away_ml") is not None:
            oh_ml = float(best_ml["open_home_ml"])
            oa_ml = float(best_ml["open_away_ml"])
            op_shin_h, op_shin_a = devig_shin_two_way(oh_ml, oa_ml)
            out["p_open_home_shin"] = op_shin_h
            out["p_open_away_shin"] = op_shin_a

    if best_sp is not None:
        sp_h = float(best_sp["spread_home"])
        sp_ho = float(best_sp["spread_home_odds"])
        sp_ao = float(best_sp["spread_away_odds"])
        fair_sp_h, fair_sp_a = devig_two_way(sp_ho, sp_ao)
        shin_sp_h, shin_sp_a = devig_shin_two_way(sp_ho, sp_ao)
        osp_h = best_sp.get("open_spread_home")
        osp_ho = best_sp.get("open_spread_home_odds")
        osp_ao = best_sp.get("open_spread_away_odds")
        out.update(
            {
                "spread_provider": best_sp.get("provider_name"),
                "spread_home": sp_h,
                "spread_away": -sp_h,
                "spread_home_odds": sp_ho,
                "spread_away_odds": sp_ao,
                "p_spread_home_implied": american_to_implied(sp_ho),
                "p_spread_away_implied": american_to_implied(sp_ao),
                "p_spread_home_novig": fair_sp_h,
                "p_spread_away_novig": fair_sp_a,
                "p_spread_home_shin": shin_sp_h,
                "p_spread_away_shin": shin_sp_a,
                "open_spread_home": osp_h,
                "open_spread_away": (-float(osp_h)) if osp_h is not None else None,
                "open_spread_home_odds": osp_ho,
                "open_spread_away_odds": osp_ao,
            }
        )
        if osp_ho is not None and osp_ao is not None:
            osh, osa = devig_shin_two_way(float(osp_ho), float(osp_ao))
            out["p_open_spread_home_shin"] = osh
            out["p_open_spread_away_shin"] = osa

    if best_tot is not None:
        t_line = float(best_tot["total_line"])
        ov_o = float(best_tot["over_odds"])
        un_o = float(best_tot["under_odds"])
        fair_ov, fair_un = devig_two_way(ov_o, un_o)
        shin_ov, shin_un = devig_shin_two_way(ov_o, un_o)
        ot_line = best_tot.get("open_total_line")
        oov_o = best_tot.get("open_over_odds")
        oun_o = best_tot.get("open_under_odds")
        out.update(
            {
                "total_provider": best_tot.get("provider_name"),
                "total_line": t_line,
                "over_odds": ov_o,
                "under_odds": un_o,
                "p_over_implied": american_to_implied(ov_o),
                "p_under_implied": american_to_implied(un_o),
                "p_over_novig": fair_ov,
                "p_under_novig": fair_un,
                "p_over_shin": shin_ov,
                "p_under_shin": shin_un,
                "open_total_line": ot_line,
                "open_over_odds": oov_o,
                "open_under_odds": oun_o,
            }
        )
        if oov_o is not None and oun_o is not None:
            oov_s, oun_s = devig_shin_two_way(float(oov_o), float(oun_o))
            out["p_open_over_shin"] = oov_s
            out["p_open_under_shin"] = oun_s

    return out


def fetch_event_closing_markets(
    event_id: str,
    *,
    cache_dir: Path | None = None,
    require_schema_v2: bool = True,
) -> dict[str, Any]:
    """Fetch opening & closing ML, spread, and total markets for one ESPN event id."""
    eid = str(event_id)
    cache_path = (cache_dir / f"{eid}.json") if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        is_transient_err = bool(cached.get("error")) and not bool(cached.get("not_found"))
        if not is_transient_err and (
            not require_schema_v2 or cached.get("schema_version") == SCHEMA_VERSION
        ):
            return cached

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    url = ESPN_ODDS_TMPL.format(eid=eid)
    try:
        idx = _http_json(url)
    except HTTPError as exc:
        out = {
            "schema_version": SCHEMA_VERSION,
            "event_id": eid,
            "error": str(exc),
            "not_found": exc.code == 404,
            "providers": [],
        }
        if cache_path is not None and exc.code == 404:
            cache_path.write_text(json.dumps(out))
        return out
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": eid,
            "error": str(exc),
            "providers": [],
        }

    providers: list[dict[str, Any]] = []
    for item in idx.get("items") or []:
        if not isinstance(item, dict):
            continue
        if "provider" in item and ("homeTeamOdds" in item or "overUnder" in item or "spread" in item):
            odds_obj = item
        else:
            ref = str(item.get("$ref") or "").replace("http://", "https://")
            if not ref:
                continue
            try:
                odds_obj = _http_json(ref)
            except Exception as exc:  # noqa: BLE001
                providers.append({"ref": ref, "error": str(exc)})
                continue
        providers.append(_extract_provider_mls(odds_obj))

    out = _assemble_event_markets(eid, providers)
    if cache_path is not None:
        cache_path.write_text(json.dumps(out))
    return out


def fetch_event_closing_mls(
    event_id: str,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper returning closing markets (legacy cache accepted)."""
    return fetch_event_closing_markets(event_id, cache_dir=cache_dir, require_schema_v2=False)


def _record_to_row(eid: str, rec: dict[str, Any]) -> dict[str, Any] | None:
    if (
        rec.get("home_ml") is None
        and rec.get("spread_home") is None
        and rec.get("total_line") is None
    ):
        return None
    p_hi = rec.get("p_home_implied")
    p_ai = rec.get("p_away_implied")
    p_shi = rec.get("p_spread_home_implied")
    p_sai = rec.get("p_spread_away_implied")
    p_oi = rec.get("p_over_implied")
    p_ui = rec.get("p_under_implied")

    sp_c = rec.get("spread_home")
    sp_o = rec.get("open_spread_home")
    tl_c = rec.get("total_line")
    tl_o = rec.get("open_total_line")
    ml_sc = rec.get("p_home_shin")
    ml_so = rec.get("p_open_home_shin")

    return {
        "game_id": str(eid),
        "provider": rec.get("provider") or rec.get("spread_provider") or rec.get("total_provider"),
        "spread_provider": rec.get("spread_provider"),
        "total_provider": rec.get("total_provider"),
        "has_close_block": bool(rec.get("has_close_block")),
        "has_open_block": bool(rec.get("has_open_block")),
        # Closing Moneyline
        "home_ml": rec.get("home_ml"),
        "away_ml": rec.get("away_ml"),
        "p_home_implied": p_hi,
        "p_away_implied": p_ai,
        "p_home_novig": rec.get("p_home_novig"),
        "p_away_novig": rec.get("p_away_novig"),
        "p_home_shin": ml_sc,
        "p_away_shin": rec.get("p_away_shin"),
        "overround": (float(p_hi) + float(p_ai)) if (p_hi is not None and p_ai is not None) else None,
        # Opening Moneyline
        "open_home_ml": rec.get("open_home_ml"),
        "open_away_ml": rec.get("open_away_ml"),
        "p_open_home_shin": ml_so,
        "p_open_away_shin": rec.get("p_open_away_shin"),
        # Closing Spread
        "spread_home": sp_c,
        "spread_away": rec.get("spread_away"),
        "spread_home_odds": rec.get("spread_home_odds"),
        "spread_away_odds": rec.get("spread_away_odds"),
        "p_spread_home_implied": p_shi,
        "p_spread_away_implied": p_sai,
        "p_spread_home_novig": rec.get("p_spread_home_novig"),
        "p_spread_away_novig": rec.get("p_spread_away_novig"),
        "p_spread_home_shin": rec.get("p_spread_home_shin"),
        "p_spread_away_shin": rec.get("p_spread_away_shin"),
        "spread_overround": (
            (float(p_shi) + float(p_sai)) if (p_shi is not None and p_sai is not None) else None
        ),
        # Opening Spread
        "open_spread_home": sp_o,
        "open_spread_away": rec.get("open_spread_away"),
        "open_spread_home_odds": rec.get("open_spread_home_odds"),
        "open_spread_away_odds": rec.get("open_spread_away_odds"),
        "p_open_spread_home_shin": rec.get("p_open_spread_home_shin"),
        "p_open_spread_away_shin": rec.get("p_open_spread_away_shin"),
        # Closing Total
        "total_line": tl_c,
        "over_odds": rec.get("over_odds"),
        "under_odds": rec.get("under_odds"),
        "p_over_implied": p_oi,
        "p_under_implied": p_ui,
        "p_over_novig": rec.get("p_over_novig"),
        "p_under_novig": rec.get("p_under_novig"),
        "p_over_shin": rec.get("p_over_shin"),
        "p_under_shin": rec.get("p_under_shin"),
        "total_overround": (
            (float(p_oi) + float(p_ui)) if (p_oi is not None and p_ui is not None) else None
        ),
        # Opening Total
        "open_total_line": tl_o,
        "open_over_odds": rec.get("open_over_odds"),
        "open_under_odds": rec.get("open_under_odds"),
        "p_open_over_shin": rec.get("p_open_over_shin"),
        "p_open_under_shin": rec.get("p_open_under_shin"),
        # Open-to-Close Steam (§3.5)
        "steam_margin_pts": (
            ((-float(sp_c)) - (-float(sp_o))) if (sp_c is not None and sp_o is not None) else None
        ),
        "steam_total_pts": (
            (float(tl_c) - float(tl_o)) if (tl_c is not None and tl_o is not None) else None
        ),
        "steam_home_ml_shin": (
            (float(ml_sc) - float(ml_so)) if (ml_sc is not None and ml_so is not None) else None
        ),
    }


def pull_closing_markets(
    event_ids: list[str],
    *,
    cache_dir: Path | None = None,
    max_workers: int = 28,
) -> pd.DataFrame:
    """Pull opening & closing ML, spread, and total markets concurrently."""
    ordered_ids = [str(e) for e in event_ids]
    results: dict[str, dict[str, Any]] = {}
    if max_workers <= 1:
        for eid in ordered_ids:
            results[eid] = fetch_event_closing_markets(eid, cache_dir=cache_dir)
    else:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            futs = {
                pool.submit(fetch_event_closing_markets, eid, cache_dir=cache_dir): eid
                for eid in ordered_ids
            }
            for fut in as_completed(futs):
                eid = futs[fut]
                results[eid] = fut.result()

    rows: list[dict[str, Any]] = []
    for eid in ordered_ids:
        row = _record_to_row(eid, results[eid])
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def pull_closing_moneylines(
    event_ids: list[str],
    *,
    cache_dir: Path | None = None,
    sleep_s: float = 0.015,
) -> pd.DataFrame:
    """Pull closing MLs for many ESPN event ids; return one row per success."""
    rows: list[dict[str, Any]] = []
    for eid in event_ids:
        rec = fetch_event_closing_mls(eid, cache_dir=cache_dir)
        if rec.get("home_ml") is None or rec.get("away_ml") is None:
            continue
        rows.append(
            {
                "game_id": str(eid),
                "provider": rec.get("provider"),
                "home_ml": rec["home_ml"],
                "away_ml": rec["away_ml"],
                "p_home_implied": rec["p_home_implied"],
                "p_away_implied": rec["p_away_implied"],
                "p_home_novig": rec["p_home_novig"],
                "p_away_novig": rec["p_away_novig"],
                "overround": float(rec["p_home_implied"]) + float(rec["p_away_implied"]),
            }
        )
        if sleep_s:
            time.sleep(sleep_s)
    return pd.DataFrame(rows)


def grade_closing_markets(
    df: pd.DataFrame,
    *,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    spread_home_col: str = "spread_home",
    total_line_col: str = "total_line",
) -> pd.DataFrame:
    """Grade ML, ATS (``home_margin + spread_home``), and totals with exact push flags."""
    out = df.copy()
    hs = pd.to_numeric(out[home_score_col], errors="coerce").to_numpy(dtype=float)
    as_ = pd.to_numeric(out[away_score_col], errors="coerce").to_numpy(dtype=float)
    margin = hs - as_
    tot_pts = hs + as_

    out["home_margin"] = margin
    out["actual_total"] = tot_pts
    out["home_win"] = np.where(np.isfinite(margin), (margin > 0).astype(float), np.nan)

    sp = (
        pd.to_numeric(out[spread_home_col], errors="coerce").to_numpy(dtype=float)
        if spread_home_col in out.columns
        else np.full(len(out), np.nan)
    )
    ats_diff = margin + sp
    valid_sp = np.isfinite(ats_diff)
    sp_push = valid_sp & (np.abs(ats_diff) < 1e-9)
    out["ats_margin_vs_spread"] = ats_diff
    out["spread_push"] = sp_push
    out["home_cover"] = np.where(~valid_sp | sp_push, np.nan, (ats_diff > 0).astype(float))
    out["away_cover"] = np.where(~valid_sp | sp_push, np.nan, (ats_diff < 0).astype(float))
    ats_res = np.full(len(out), "missing", dtype=object)
    ats_res[valid_sp & (ats_diff > 1e-9)] = "home"
    ats_res[valid_sp & (ats_diff < -1e-9)] = "away"
    ats_res[sp_push] = "push"
    out["ats_result"] = ats_res

    tl = (
        pd.to_numeric(out[total_line_col], errors="coerce").to_numpy(dtype=float)
        if total_line_col in out.columns
        else np.full(len(out), np.nan)
    )
    tot_diff = tot_pts - tl
    valid_tot = np.isfinite(tot_diff)
    tot_push = valid_tot & (np.abs(tot_diff) < 1e-9)
    out["total_diff_vs_line"] = tot_diff
    out["total_push"] = tot_push
    out["over_hit"] = np.where(~valid_tot | tot_push, np.nan, (tot_diff > 0).astype(float))
    out["under_hit"] = np.where(~valid_tot | tot_push, np.nan, (tot_diff < 0).astype(float))
    tot_res = np.full(len(out), "missing", dtype=object)
    tot_res[valid_tot & (tot_diff > 1e-9)] = "over"
    tot_res[valid_tot & (tot_diff < -1e-9)] = "under"
    tot_res[tot_push] = "push"
    out["total_result"] = tot_res

    # Also grade against opening spread & opening total when present (for open-line CLV / backtests)
    if "open_spread_home" in out.columns:
        osp = pd.to_numeric(out["open_spread_home"], errors="coerce").to_numpy(dtype=float)
        o_ats = margin + osp
        vo = np.isfinite(o_ats)
        op_push = vo & (np.abs(o_ats) < 1e-9)
        out["open_spread_push"] = op_push
        out["open_home_cover"] = np.where(~vo | op_push, np.nan, (o_ats > 0).astype(float))

    if "open_total_line" in out.columns:
        otl = pd.to_numeric(out["open_total_line"], errors="coerce").to_numpy(dtype=float)
        o_tot = tot_pts - otl
        vot = np.isfinite(o_tot)
        ot_push = vot & (np.abs(o_tot) < 1e-9)
        out["open_total_push"] = ot_push
        out["open_over_hit"] = np.where(~vot | ot_push, np.nan, (o_tot > 0).astype(float))

    return out
