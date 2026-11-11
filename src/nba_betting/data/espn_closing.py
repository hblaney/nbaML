"""Historical closing moneylines via ESPN's public sports.core odds API.

Uses ESPN BET (provider 58) open/close blocks when present. No Odds API key
required. Intended for walk-forward market baselines / blends.
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

ESPN_ODDS_TMPL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds"
)
DEFAULT_UA = "Mozilla/5.0 (compatible; nbaML-backtest/0.2)"


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


def _parse_american_ml(side: dict[str, Any], odds_obj: dict[str, Any]) -> float | None:
    close = side.get("close") or {}
    ml_obj = close.get("moneyLine")
    if isinstance(ml_obj, dict):
        am = ml_obj.get("american")
        if am is not None and str(am).strip() != "":
            return float(str(am).replace("−", "-").replace(",", ""))
    # When a close block exists at event level, current ML is often the freeze.
    if odds_obj.get("close") is not None:
        cur = (side.get("current") or {}).get("moneyLine")
        if isinstance(cur, dict):
            am = cur.get("american")
            if am is not None and str(am).strip() != "":
                return float(str(am).replace("−", "-").replace(",", ""))
    top = side.get("moneyLine")
    if top is not None and not isinstance(top, dict):
        try:
            return float(top)
        except (TypeError, ValueError):
            return None
    return None


def _extract_provider_mls(odds_obj: dict[str, Any]) -> dict[str, Any]:
    provider = odds_obj.get("provider") or {}
    home = odds_obj.get("homeTeamOdds") or {}
    away = odds_obj.get("awayTeamOdds") or {}
    return {
        "provider_id": provider.get("id"),
        "provider_name": provider.get("name"),
        "home_ml": _parse_american_ml(home, odds_obj),
        "away_ml": _parse_american_ml(away, odds_obj),
        "has_close_block": bool(odds_obj.get("close")),
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
        if p.get("has_close_block"):
            score += 2
        if "Live" in name or pid == "59":
            score -= 5
        scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def fetch_event_closing_mls(
    event_id: str,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch closing home/away American moneylines for one ESPN event id."""
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
    out = {"event_id": eid, "providers": providers, "best": best}
    if best is not None:
        home_ml = float(best["home_ml"])
        away_ml = float(best["away_ml"])
        fair_home, fair_away = devig_two_way(home_ml, away_ml)
        out.update(
            {
                "provider": best.get("provider_name"),
                "home_ml": home_ml,
                "away_ml": away_ml,
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
        if (i + 1) % 200 == 0:
            # progress breadcrumb for long pulls
            pass
    return pd.DataFrame(rows)
