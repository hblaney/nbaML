"""Historical closing & opening odds via ESPN's public sports.core odds API.

Supports single-book (ESPN BET / DraftKings) and multi-book historical snapshots
(2022-23 & 2023-24 archived 11–16 providers per game; 2023-24..2025-26 include
inline ``open`` and ``close`` blocks; 2022-23 includes ``teamrankings`` line
movement history under ``odds/1002/history/0``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from nba_betting.pricing.american import american_to_implied, devig_two_way
from nba_betting.pricing.consensus import (
    TwoWayQuote,
    compute_sharp_consensus,
    is_valid_pregame_book,
    normalize_book_key,
)

ESPN_ODDS_TMPL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds?limit=100"
)
ESPN_HISTORY_TMPL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds/{pid}/history/0"
)
DEFAULT_UA = "Mozilla/5.0 (compatible; nbaML-backtest/0.3)"


def _http_json(url: str, *, retries: int = 4, pause: float = 0.8) -> Any:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": DEFAULT_UA})
            with urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(pause * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _clean_american(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace("−", "-").replace(",", "")
    if not s:
        return None
    if s.upper() == "EVEN":
        return 100.0
    try:
        out = float(s)
        return None if out == 0.0 else out
    except (TypeError, ValueError):
        return None


def _clean_line(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace("−", "-").replace(",", "")
    if not s:
        return None
    if s.upper() in {"PK", "PICK", "EVEN"}:
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_american_ml(side: dict[str, Any], odds_obj: dict[str, Any]) -> float | None:
    close = side.get("close") or {}
    ml_obj = close.get("moneyLine")
    if isinstance(ml_obj, dict):
        am = _clean_american(ml_obj.get("american") or ml_obj.get("alternateDisplayValue"))
        if am is not None:
            return am
    # When a close block exists at event level, current ML is often the freeze.
    if odds_obj.get("close") is not None:
        cur = (side.get("current") or {}).get("moneyLine")
        if isinstance(cur, dict):
            am = _clean_american(cur.get("american") or cur.get("alternateDisplayValue"))
            if am is not None:
                return am
    cur_ml = (side.get("current") or {}).get("moneyLine")
    if isinstance(cur_ml, dict):
        am = _clean_american(cur_ml.get("american") or cur_ml.get("alternateDisplayValue"))
        if am is not None:
            return am
    top = side.get("moneyLine")
    if top is not None and not isinstance(top, dict):
        return _clean_american(top)
    return None


def _parse_open_american_ml(side: dict[str, Any]) -> float | None:
    op = side.get("open") or {}
    ml_obj = op.get("moneyLine")
    if isinstance(ml_obj, dict):
        return _clean_american(ml_obj.get("american") or ml_obj.get("alternateDisplayValue"))
    return None


def _parse_spread_side(
    side: dict[str, Any], phase: str = "close"
) -> tuple[float | None, float | None]:
    block = side.get(phase) or {}
    ps = block.get("pointSpread") or {}
    sp = block.get("spread") or {}
    pt = _clean_line(ps.get("alternateDisplayValue") or ps.get("american"))
    # Guard against ESPN quirk where close.pointSpread.american occasionally holds juice (-110)
    if pt is not None and abs(pt) >= 50.0:
        pt = None
    pr = _clean_american(sp.get("american") or sp.get("alternateDisplayValue"))
    if phase == "close":
        if pt is None:
            cur_ps = (side.get("current") or {}).get("pointSpread") or {}
            cand = _clean_line(cur_ps.get("alternateDisplayValue") or cur_ps.get("american"))
            if cand is not None and abs(cand) < 50.0:
                pt = cand
        if pr is None:
            pr = _clean_american(side.get("spreadOdds"))
    return pt, pr


def _extract_provider_mls(odds_obj: dict[str, Any]) -> dict[str, Any]:
    provider = odds_obj.get("provider") or {}
    pid = provider.get("id")
    pname = provider.get("name")
    home = odds_obj.get("homeTeamOdds") or {}
    away = odds_obj.get("awayTeamOdds") or {}
    sp_home_close, sp_home_odds = _parse_spread_side(home, "close")
    sp_away_close, sp_away_odds = _parse_spread_side(away, "close")
    if sp_home_close is None and odds_obj.get("spread") is not None:
        top_sp = _clean_line(odds_obj.get("spread"))
        if top_sp is not None and abs(top_sp) < 50.0:
            sp_home_close = top_sp
            sp_away_close = -top_sp
    sp_home_open, sp_home_open_odds = _parse_spread_side(home, "open")
    _, sp_away_open_odds = _parse_spread_side(away, "open")

    return {
        "provider_id": pid,
        "provider_name": pname,
        "canonical_book": normalize_book_key(str(pname or "")),
        "is_pregame_book": is_valid_pregame_book(pname, pid),
        "home_ml": _parse_american_ml(home, odds_obj),
        "away_ml": _parse_american_ml(away, odds_obj),
        "open_home_ml": _parse_open_american_ml(home),
        "open_away_ml": _parse_open_american_ml(away),
        "spread_home": sp_home_close,
        "spread_away": sp_away_close,
        "spread_home_odds": sp_home_odds,
        "spread_away_odds": sp_away_odds,
        "open_spread_home": sp_home_open,
        "open_spread_home_odds": sp_home_open_odds,
        "open_spread_away_odds": sp_away_open_odds,
        "has_close_block": bool(odds_obj.get("close")),
        "has_open_block": bool(odds_obj.get("open") or home.get("open")),
    }


def _pick_best_provider(providers: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in providers:
        if p.get("home_ml") is None or p.get("away_ml") is None:
            continue
        name = str(p.get("provider_name") or "")
        pid = str(p.get("provider_id") or "")
        score = 0
        if pid == "58" or ("ESPN BET" in name and "Live" not in name):
            score += 10
        elif pid in {"40", "100"} or ("Draft" in name and "Live" not in name):
            score += 8
        if p.get("has_close_block"):
            score += 2
        if "Live" in name or pid in {"46", "59"}:
            score -= 20
        scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def summarize_event_multi_book(
    providers: list[dict[str, Any]],
    *,
    devig_method: str = "shin",
) -> dict[str, Any]:
    """Summarize multi-book consensus, best executable prices, and open->close CLV."""
    pregame = [
        p
        for p in providers
        if p.get(
            "is_pregame_book", is_valid_pregame_book(p.get("provider_name"), p.get("provider_id"))
        )
        and pd.notna(p.get("home_ml"))
        and pd.notna(p.get("away_ml"))
        and float(p["home_ml"]) != 0.0
        and float(p["away_ml"]) != 0.0
    ]
    if not pregame:
        return {"n_pregame_books": 0}

    close_quotes = [
        TwoWayQuote(
            book=str(p.get("provider_name") or ""),
            side_a_price=float(p["home_ml"]),
            side_b_price=float(p["away_ml"]),
            market="moneyline",
        )
        for p in pregame
    ]
    cons_close = compute_sharp_consensus(close_quotes, devig_method=devig_method)

    open_quotes = [
        TwoWayQuote(
            book=str(p.get("provider_name") or ""),
            side_a_price=float(p["open_home_ml"]),
            side_b_price=float(p["open_away_ml"]),
            market="moneyline",
        )
        for p in pregame
        if pd.notna(p.get("open_home_ml"))
        and pd.notna(p.get("open_away_ml"))
        and float(p["open_home_ml"]) != 0.0
        and float(p["open_away_ml"]) != 0.0
    ]
    open_prob_home: float | None = None
    open_home_ml_ref: float | None = None
    open_away_ml_ref: float | None = None
    if open_quotes:
        cons_open = compute_sharp_consensus(open_quotes, devig_method=devig_method)
        open_prob_home = cons_open.prob_a
        open_home_ml_ref = cons_open.best_price_a
        open_away_ml_ref = cons_open.best_price_b

    return {
        "n_pregame_books": cons_close.n_books,
        "consensus_prob_home": cons_close.prob_a,
        "consensus_prob_away": cons_close.prob_b,
        "best_home_ml": cons_close.best_price_a,
        "best_home_book": cons_close.best_book_a,
        "best_away_ml": cons_close.best_price_b,
        "best_away_book": cons_close.best_book_b,
        "synthetic_overround": cons_close.synthetic_overround,
        "mean_overround": cons_close.mean_overround,
        "min_overround": cons_close.min_overround,
        "max_overround": cons_close.max_overround,
        "dispersion_std_home": cons_close.dispersion_std_a,
        "n_open_books": len(open_quotes),
        "open_prob_home": open_prob_home,
        "open_home_ml_best": open_home_ml_ref,
        "open_away_ml_best": open_away_ml_ref,
        "steam_home_prob_delta": (
            (cons_close.prob_a - open_prob_home) if open_prob_home is not None else None
        ),
    }


def fetch_event_closing_mls(
    event_id: str,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch closing/opening home/away American moneylines across providers for one ESPN event."""
    eid = str(event_id)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{eid}.json"
        if path.exists():
            return json.loads(path.read_text())

    url = ESPN_ODDS_TMPL.format(eid=eid)
    try:
        idx = _http_json(url)
    except Exception as exc:  # noqa: BLE001 - callers want soft failure per game
        out: dict[str, Any] = {"event_id": eid, "error": str(exc), "providers": []}
        if cache_dir is not None:
            (cache_dir / f"{eid}.json").write_text(json.dumps(out))
        return out

    providers: list[dict[str, Any]] = []
    for item in idx.get("items") or []:
        if "homeTeamOdds" in item and "provider" in item:
            providers.append(_extract_provider_mls(item))
            continue
        ref = str(item.get("$ref") or "").replace("http://", "https://")
        if not ref:
            continue
        try:
            odds_obj = _http_json(ref)
        except Exception as exc:  # noqa: BLE001
            providers.append({"ref": ref, "error": str(exc)})
            continue
        providers.append(_extract_provider_mls(odds_obj))
        time.sleep(0.02)

    best = _pick_best_provider(providers)
    out = {
        "event_id": eid,
        "providers": providers,
        "best": best,
        "multi_book": summarize_event_multi_book(providers, devig_method="shin"),
    }
    if best is not None:
        home_ml = float(best["home_ml"])
        away_ml = float(best["away_ml"])
        fair_home, fair_away = devig_two_way(home_ml, away_ml)
        out.update(
            {
                "provider": best.get("provider_name"),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "open_home_ml": best.get("open_home_ml"),
                "open_away_ml": best.get("open_away_ml"),
                "p_home_implied": american_to_implied(home_ml),
                "p_away_implied": american_to_implied(away_ml),
                "p_home_novig": fair_home,
                "p_away_novig": fair_away,
            }
        )
    if cache_dir is not None:
        (cache_dir / f"{eid}.json").write_text(json.dumps(out))
    return out


def pull_closing_moneylines(
    event_ids: list[str],
    *,
    cache_dir: Path | None = None,
    sleep_s: float = 0.015,
) -> pd.DataFrame:
    """Pull closing MLs for many ESPN event ids; return one row per success."""
    rows: list[dict[str, Any]] = []
    for i, eid in enumerate(event_ids):
        rec = fetch_event_closing_mls(eid, cache_dir=cache_dir)
        if rec.get("home_ml") is None or rec.get("away_ml") is None:
            continue
        mb = rec.get("multi_book") or {}
        rows.append(
            {
                "game_id": str(eid),
                "provider": rec.get("provider"),
                "home_ml": rec["home_ml"],
                "away_ml": rec["away_ml"],
                "open_home_ml": rec.get("open_home_ml"),
                "open_away_ml": rec.get("open_away_ml"),
                "p_home_implied": rec["p_home_implied"],
                "p_away_implied": rec["p_away_implied"],
                "p_home_novig": rec["p_home_novig"],
                "p_away_novig": rec["p_away_novig"],
                "overround": float(rec["p_home_implied"]) + float(rec["p_away_implied"]),
                "n_pregame_books": mb.get("n_pregame_books", 0),
                "best_home_ml": mb.get("best_home_ml"),
                "best_away_ml": mb.get("best_away_ml"),
                "consensus_prob_home_shin": mb.get("consensus_prob_home"),
                "steam_home_prob_delta": mb.get("steam_home_prob_delta"),
            }
        )
        if sleep_s:
            time.sleep(sleep_s)
        if (i + 1) % 200 == 0:
            pass
    return pd.DataFrame(rows)
