"""ESPN NBA period linescores, PBP period possessions/stagger, boxscores, and 1Q/1H odds.

Extracts strictly factual per-game observations from ESPN's public ``summary`` and
``sports.core`` ``propBets`` endpoints so downstream expanding/walk-forward models
can build zero-leakage pregame features across 2022-23 through 2025-26.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ESPN_SUMMARY_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={eid}"
)
ESPN_ODDS_INDEX_TMPL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds"
)
ESPN_PROPBETS_TMPL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds/{pid}/propBets?limit=1000&page={page}"
)
DEFAULT_UA = "Mozilla/5.0 (compatible; nbaML-derivatives/0.1)"

#Normalize ESPN tricodes (e.g. GS -> GSW, NO -> NOP, NY -> NYK, SA -> SAS, UTAH -> UTA, WSH -> WAS)
ESPN_TRICODE_MAP: dict[str, str] = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def normalize_espn_tricode(abbr: str | None) -> str:
    if not abbr:
        return ""
    up = str(abbr).strip().upper()
    return ESPN_TRICODE_MAP.get(up, up)


_TLS = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_TLS, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": DEFAULT_UA})
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _TLS.session = sess
    return sess


def _http_json(url: str, *, retries: int = 5, pause: float = 0.6, timeout: float = 25.0) -> Any:
    sess = _get_session()
    last: Exception | None = None
    for i in range(retries):
        try:
            resp = sess.get(url, timeout=timeout)
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            resp.raise_for_status()
            return resp.json()
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _parse_clock_remaining_sec(display_clock: str | None) -> float:
    """Parse ESPN period clock like '6:11' or '42.3' into seconds remaining in period."""
    if not display_clock:
        return 0.0
    s = str(display_clock).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(s)
    except ValueError:
        return 0.0


def _parse_minutes_val(val: Any) -> float:
    if val is None:
        return 0.0
    s = str(val).strip()
    if s in ("", "--", "DNP"):
        return 0.0
    try:
        if ":" in s:
            p = s.split(":")
            return float(p[0]) + float(p[1]) / 60.0
        return float(s)
    except ValueError:
        return 0.0


def _parse_float_val(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().replace("+", "").replace("−", "-").replace(",", "")
    if s in ("", "--"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parse_made_att(val: Any) -> tuple[float, float]:
    if not val:
        return 0.0, 0.0
    s = str(val).strip()
    if "-" not in s:
        return 0.0, 0.0
    parts = s.split("-", 1)
    return _parse_float_val(parts[0], 0.0), _parse_float_val(parts[1], 0.0)


def parse_summary_payload(
    summary: dict[str, Any],
    *,
    fallback_game_id: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse an ESPN summary JSON into (game_period_row, player_box_rows)."""
    header_comps = (summary.get("header") or {}).get("competitions") or []
    comp = header_comps[0] if header_comps else {}
    gid = str(comp.get("id") or fallback_game_id)
    tipoff_utc = str(comp.get("date") or "")
    is_neutral = bool(comp.get("neutralSite", False))

    venue = (summary.get("gameInfo") or {}).get("venue") or {}
    addr = venue.get("address") or {}
    venue_name = str(venue.get("fullName") or "")
    venue_city = str(addr.get("city") or "")
    venue_state = str(addr.get("state") or "")

    home_comp: dict[str, Any] = {}
    away_comp: dict[str, Any] = {}
    for c in comp.get("competitors") or []:
        if c.get("homeAway") == "home":
            home_comp = c
        elif c.get("homeAway") == "away":
            away_comp = c

    home_tid = int(home_comp.get("id") or (home_comp.get("team") or {}).get("id") or 0)
    away_tid = int(away_comp.get("id") or (away_comp.get("team") or {}).get("id") or 0)
    home_tri = normalize_espn_tricode((home_comp.get("team") or {}).get("abbreviation"))
    away_tri = normalize_espn_tricode((away_comp.get("team") or {}).get("abbreviation"))

    home_ls = [_parse_float_val(x.get("displayValue"), np.nan) for x in (home_comp.get("linescores") or [])]
    away_ls = [_parse_float_val(x.get("displayValue"), np.nan) for x in (away_comp.get("linescores") or [])]

    def _q(ls: list[float], idx: int) -> float:
        return float(ls[idx]) if idx < len(ls) else float("nan")

    home_q1, home_q2, home_q3, home_q4 = _q(home_ls, 0), _q(home_ls, 1), _q(home_ls, 2), _q(home_ls, 3)
    away_q1, away_q2, away_q3, away_q4 = _q(away_ls, 0), _q(away_ls, 1), _q(away_ls, 2), _q(away_ls, 3)
    n_ot = max(0, max(len(home_ls), len(away_ls)) - 4)
    home_ot = float(np.nansum(home_ls[4:])) if len(home_ls) > 4 else 0.0
    away_ot = float(np.nansum(away_ls[4:])) if len(away_ls) > 4 else 0.0

    home_h1 = home_q1 + home_q2
    away_h1 = away_q1 + away_q2
    home_h2 = home_q3 + home_q4
    away_h2 = away_q3 + away_q4
    home_score = _parse_float_val(home_comp.get("score"), home_h1 + home_h2 + home_ot)
    away_score = _parse_float_val(away_comp.get("score"), away_h1 + away_h2 + away_ot)

    # Team boxscores
    box_teams: dict[int, dict[str, float]] = {}
    for bt in (summary.get("boxscore") or {}).get("teams") or []:
        tid = int((bt.get("team") or {}).get("id") or 0)
        smap = {str(s.get("name")): s.get("displayValue") for s in (bt.get("statistics") or [])}
        fgm, fga = _parse_made_att(smap.get("fieldGoalsMade-fieldGoalsAttempted"))
        fg3m, fg3a = _parse_made_att(smap.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
        ftm, fta = _parse_made_att(smap.get("freeThrowsMade-freeThrowsAttempted"))
        oreb = _parse_float_val(smap.get("offensiveRebounds"), 0.0)
        dreb = _parse_float_val(smap.get("defensiveRebounds"), 0.0)
        reb = _parse_float_val(smap.get("totalRebounds"), oreb + dreb)
        ast = _parse_float_val(smap.get("assists"), 0.0)
        tov = _parse_float_val(smap.get("totalTurnovers", smap.get("turnovers")), 0.0)
        box_teams[tid] = {
            "fgm": fgm,
            "fga": fga,
            "fg3m": fg3m,
            "fg3a": fg3a,
            "ftm": ftm,
            "fta": fta,
            "oreb": oreb,
            "dreb": dreb,
            "reb": reb,
            "ast": ast,
            "tov": tov,
        }

    # Player boxscores (starter vs bench splits + player rows)
    starter_ids_by_team: dict[int, set[str]] = {home_tid: set(), away_tid: set()}
    player_splits: dict[int, dict[str, float]] = {}
    player_rows: list[dict[str, Any]] = []

    for pteam in (summary.get("boxscore") or {}).get("players") or []:
        tid = int((pteam.get("team") or {}).get("id") or 0)
        stat_blocks = pteam.get("statistics") or []
        if not stat_blocks:
            continue
        s0 = stat_blocks[0]
        labels = [str(x).upper() for x in (s0.get("labels") or [])]
        idx_min = labels.index("MIN") if "MIN" in labels else -1
        idx_pts = labels.index("PTS") if "PTS" in labels else -1
        idx_pm = labels.index("+/-") if "+/-" in labels else -1

        s_min = b_min = s_pts = b_pts = s_pm = b_pm = 0.0
        for ath in s0.get("athletes") or []:
            athlete_info = ath.get("athlete") or {}
            pid = str(athlete_info.get("id") or "")
            pname = str(athlete_info.get("displayName") or "")
            is_starter = bool(ath.get("starter", False))
            dnp = bool(ath.get("didNotPlay", False))
            st_vals = ath.get("stats") or []
            if dnp or not st_vals:
                mins = pts = pm = 0.0
            else:
                mins = _parse_minutes_val(st_vals[idx_min]) if 0 <= idx_min < len(st_vals) else 0.0
                pts = _parse_float_val(st_vals[idx_pts], 0.0) if 0 <= idx_pts < len(st_vals) else 0.0
                pm = _parse_float_val(st_vals[idx_pm], 0.0) if 0 <= idx_pm < len(st_vals) else 0.0

            if is_starter and pid:
                starter_ids_by_team.setdefault(tid, set()).add(pid)
            if is_starter:
                s_min += mins
                s_pts += pts
                s_pm += pm
            else:
                b_min += mins
                b_pts += pts
                b_pm += pm

            if pid:
                player_rows.append(
                    {
                        "game_id": gid,
                        "team_id": tid,
                        "player_id": pid,
                        "player_name": pname,
                        "starter": is_starter,
                        "dnp": dnp or mins <= 0.0,
                        "minutes": mins,
                        "pts": pts,
                        "plus_minus": pm,
                    }
                )

        tot_min = s_min + b_min
        player_splits[tid] = {
            "starter_min": s_min,
            "bench_min": b_min,
            "team_min": tot_min,
            "starter_min_share": (s_min / tot_min) if tot_min > 0 else np.nan,
            "starter_pts": s_pts,
            "bench_pts": b_pts,
            "starter_pm": s_pm,
            "bench_pm": b_pm,
        }

    # Play-by-play period possessions, 6-minute stint scores, and starter stagger timing
    plays = summary.get("plays") or []
    pbp_counts: dict[tuple[int, int], dict[str, float]] = {}
    for tid in (home_tid, away_tid):
        for per in (1, 2, 3, 4):
            pbp_counts[(tid, per)] = {"fga": 0.0, "fta": 0.0, "oreb": 0.0, "tov": 0.0}

    # Track Q1 first-6-min score (clock >= 6:00 -> last score before clock drops < 360s)
    q1_mid_home = 0.0
    q1_mid_away = 0.0
    q2_mid_home = home_q1 if np.isfinite(home_q1) else 0.0
    q2_mid_away = away_q1 if np.isfinite(away_q1) else 0.0

    # Track starter on-court player-seconds in Q1 and Q2 via substitution clock integration
    on_court: dict[int, set[str]] = {
        home_tid: set(starter_ids_by_team.get(home_tid, set())),
        away_tid: set(starter_ids_by_team.get(away_tid, set())),
    }
    last_rem_sec: dict[int, float] = {1: 720.0, 2: 720.0}
    starter_player_sec: dict[tuple[int, int], float] = {
        (home_tid, 1): 0.0,
        (away_tid, 1): 0.0,
        (home_tid, 2): 0.0,
        (away_tid, 2): 0.0,
    }
    first_sub_elapsed: dict[int, float] = {home_tid: 720.0, away_tid: 720.0}

    prev_period = 0
    for p in plays:
        per = int((p.get("period") or {}).get("number") or 0)
        if per not in (1, 2, 3, 4):
            continue
        rem = _parse_clock_remaining_sec((p.get("clock") or {}).get("displayValue"))
        rem = float(np.clip(rem, 0.0, 720.0))

        if per in (1, 2):
            if per != prev_period:
                if prev_period == 1 and last_rem_sec[1] > 0.0:
                    dt_end = last_rem_sec[1]
                    for tid in (home_tid, away_tid):
                        n_st = len(on_court[tid] & starter_ids_by_team.get(tid, set()))
                        starter_player_sec[(tid, 1)] += dt_end * n_st
                    last_rem_sec[1] = 0.0
                prev_period = per
            dt = max(0.0, last_rem_sec[per] - rem)
            if dt > 0.0:
                for tid in (home_tid, away_tid):
                    n_st = len(on_court[tid] & starter_ids_by_team.get(tid, set()))
                    starter_player_sec[(tid, per)] += dt * n_st
                last_rem_sec[per] = rem

        # Track mid-quarter scores (at 6:00 mark = 360s remaining)
        h_sc = _parse_float_val(p.get("homeScore"), np.nan)
        a_sc = _parse_float_val(p.get("awayScore"), np.nan)
        if np.isfinite(h_sc) and np.isfinite(a_sc):
            if per == 1 and rem >= 360.0:
                q1_mid_home, q1_mid_away = h_sc, a_sc
            elif per == 2 and rem >= 360.0:
                q2_mid_home, q2_mid_away = h_sc, a_sc

        tid = int((p.get("team") or {}).get("id") or 0)
        ptype = str((p.get("type") or {}).get("text") or "")
        txt = str(p.get("text") or "")

        if per in (1, 2) and tid in (home_tid, away_tid) and (
            "Substitution" in ptype or "enters the game" in txt
        ):
            parts = p.get("participants") or []
            if len(parts) >= 2:
                p_in = str((parts[0].get("athlete") or {}).get("id") or "")
                p_out = str((parts[1].get("athlete") or {}).get("id") or "")
                if per == 1 and p_out in starter_ids_by_team.get(tid, set()):
                    elapsed = 720.0 - rem
                    if elapsed < first_sub_elapsed[tid]:
                        first_sub_elapsed[tid] = elapsed
                if p_out:
                    on_court[tid].discard(p_out)
                if p_in:
                    on_court[tid].add(p_in)

        if tid not in (home_tid, away_tid):
            continue

        bucket = pbp_counts[(tid, per)]
        if bool(p.get("shootingPlay")):
            pts_att = int(p.get("pointsAttempted") or 0)
            if "Free Throw" in ptype or pts_att == 1:
                if "technical" not in ptype.lower() and "technical" not in txt.lower():
                    bucket["fta"] += 1.0
            elif pts_att in (2, 3):
                bucket["fga"] += 1.0
        elif "Offensive Rebound" in ptype:
            if p.get("participants"):
                bucket["oreb"] += 1.0
        elif "Turnover" in ptype:
            bucket["tov"] += 1.0

    # Flush remaining clock in period 1 and 2
    for per in (1, 2):
        if last_rem_sec[per] > 0.0 and plays:
            dt_end = last_rem_sec[per]
            for tid in (home_tid, away_tid):
                n_st = len(on_court[tid] & starter_ids_by_team.get(tid, set()))
                starter_player_sec[(tid, per)] += dt_end * n_st

    def _poss(tid: int, periods: tuple[int, ...]) -> float:
        if not plays:
            return float("nan")
        fga = sum(pbp_counts[(tid, pr)]["fga"] for pr in periods)
        fta = sum(pbp_counts[(tid, pr)]["fta"] for pr in periods)
        oreb = sum(pbp_counts[(tid, pr)]["oreb"] for pr in periods)
        tov = sum(pbp_counts[(tid, pr)]["tov"] for pr in periods)
        raw = fga + 0.44 * fta - oreb + tov
        return float(max(raw, 1.0)) if (fga + fta + tov) > 0 else float("nan")

    home_q1_poss = _poss(home_tid, (1,))
    away_q1_poss = _poss(away_tid, (1,))
    home_h1_poss = _poss(home_tid, (1, 2))
    away_h1_poss = _poss(away_tid, (1, 2))

    # Full-game possessions from official team boxscore (fallback PBP)
    def _fg_poss(tid: int) -> float:
        bt = box_teams.get(tid)
        if bt and bt["fga"] > 0:
            return float(max(bt["fga"] + 0.44 * bt["fta"] - bt["oreb"] + bt["tov"], 1.0))
        return _poss(tid, (1, 2, 3, 4))

    home_fg_poss = _fg_poss(home_tid)
    away_fg_poss = _fg_poss(away_tid)

    q1_poss_avg = 0.5 * (home_q1_poss + away_q1_poss)
    h1_poss_avg = 0.5 * (home_h1_poss + away_h1_poss)
    fg_poss_avg = 0.5 * (home_fg_poss + away_fg_poss)
    reg_plus_ot_min = 48.0 + 5.0 * n_ot

    q1_pace_48 = 4.0 * q1_poss_avg
    h1_pace_48 = 2.0 * h1_poss_avg
    fg_pace_48 = fg_poss_avg * (48.0 / reg_plus_ot_min)

    home_q1_ortg = 100.0 * home_q1 / q1_poss_avg if np.isfinite(q1_poss_avg) and q1_poss_avg > 0 else np.nan
    away_q1_ortg = 100.0 * away_q1 / q1_poss_avg if np.isfinite(q1_poss_avg) and q1_poss_avg > 0 else np.nan
    home_h1_ortg = 100.0 * home_h1 / h1_poss_avg if np.isfinite(h1_poss_avg) and h1_poss_avg > 0 else np.nan
    away_h1_ortg = 100.0 * away_h1 / h1_poss_avg if np.isfinite(h1_poss_avg) and h1_poss_avg > 0 else np.nan
    home_fg_ortg = 100.0 * home_score / fg_poss_avg if np.isfinite(fg_poss_avg) and fg_poss_avg > 0 else np.nan
    away_fg_ortg = 100.0 * away_score / fg_poss_avg if np.isfinite(fg_poss_avg) and fg_poss_avg > 0 else np.nan

    hs = player_splits.get(home_tid, {})
    as_ = player_splits.get(away_tid, {})
    h_bt = box_teams.get(home_tid, {})
    a_bt = box_teams.get(away_tid, {})

    has_starters_pbp = bool(plays and starter_ids_by_team.get(home_tid) and starter_ids_by_team.get(away_tid))
    h_q1_st_share = starter_player_sec[(home_tid, 1)] / 3600.0 if has_starters_pbp else np.nan
    a_q1_st_share = starter_player_sec[(away_tid, 1)] / 3600.0 if has_starters_pbp else np.nan
    h_h1_st_share = (
        (starter_player_sec[(home_tid, 1)] + starter_player_sec[(home_tid, 2)]) / 7200.0
        if has_starters_pbp
        else np.nan
    )
    a_h1_st_share = (
        (starter_player_sec[(away_tid, 1)] + starter_player_sec[(away_tid, 2)]) / 7200.0
        if has_starters_pbp
        else np.nan
    )

    game_row: dict[str, Any] = {
        "game_id": gid,
        "tipoff_utc": tipoff_utc,
        "venue_name": venue_name,
        "venue_city": venue_city,
        "venue_state": venue_state,
        "summary_is_neutral": is_neutral,
        "espn_home_team_id": home_tid,
        "espn_away_team_id": away_tid,
        "home_tricode_norm": home_tri,
        "away_tricode_norm": away_tri,
        "home_q1": home_q1,
        "home_q2": home_q2,
        "home_q3": home_q3,
        "home_q4": home_q4,
        "home_ot": home_ot,
        "home_h1": home_h1,
        "home_h2": home_h2,
        "away_q1": away_q1,
        "away_q2": away_q2,
        "away_q3": away_q3,
        "away_q4": away_q4,
        "away_ot": away_ot,
        "away_h1": away_h1,
        "away_h2": away_h2,
        "n_ot_periods": int(n_ot),
        "q1_margin": home_q1 - away_q1,
        "q2_margin": home_q2 - away_q2,
        "q3_margin": home_q3 - away_q3,
        "q4_margin": home_q4 - away_q4,
        "h1_margin": home_h1 - away_h1,
        "h2_margin": home_h2 - away_h2,
        "fg_margin": home_score - away_score,
        "q1_total": home_q1 + away_q1,
        "q2_total": home_q2 + away_q2,
        "q3_total": home_q3 + away_q3,
        "q4_total": home_q4 + away_q4,
        "h1_total": home_h1 + away_h1,
        "h2_total": home_h2 + away_h2,
        "fg_total": home_score + away_score,
        "q1_first6_margin": (q1_mid_home - q1_mid_away) if plays else np.nan,
        "q1_last6_margin": ((home_q1 - away_q1) - (q1_mid_home - q1_mid_away)) if plays else np.nan,
        "q2_first6_margin": ((q2_mid_home - q2_mid_away) - (home_q1 - away_q1)) if plays else np.nan,
        "home_q1_poss": home_q1_poss,
        "away_q1_poss": away_q1_poss,
        "q1_pace_48": q1_pace_48,
        "home_h1_poss": home_h1_poss,
        "away_h1_poss": away_h1_poss,
        "h1_pace_48": h1_pace_48,
        "home_fg_poss": home_fg_poss,
        "away_fg_poss": away_fg_poss,
        "fg_pace_48": fg_pace_48,
        "home_q1_ortg": home_q1_ortg,
        "home_q1_drtg": away_q1_ortg,
        "away_q1_ortg": away_q1_ortg,
        "away_q1_drtg": home_q1_ortg,
        "home_h1_ortg": home_h1_ortg,
        "home_h1_drtg": away_h1_ortg,
        "away_h1_ortg": away_h1_ortg,
        "away_h1_drtg": home_h1_ortg,
        "home_fg_ortg": home_fg_ortg,
        "home_fg_drtg": away_fg_ortg,
        "away_fg_ortg": away_fg_ortg,
        "away_fg_drtg": home_fg_ortg,
        "home_fga": h_bt.get("fga", np.nan),
        "home_fgm": h_bt.get("fgm", np.nan),
        "home_fg3a": h_bt.get("fg3a", np.nan),
        "home_fg3m": h_bt.get("fg3m", np.nan),
        "home_fta": h_bt.get("fta", np.nan),
        "home_ftm": h_bt.get("ftm", np.nan),
        "home_oreb": h_bt.get("oreb", np.nan),
        "home_dreb": h_bt.get("dreb", np.nan),
        "home_tov": h_bt.get("tov", np.nan),
        "away_fga": a_bt.get("fga", np.nan),
        "away_fgm": a_bt.get("fgm", np.nan),
        "away_fg3a": a_bt.get("fg3a", np.nan),
        "away_fg3m": a_bt.get("fg3m", np.nan),
        "away_fta": a_bt.get("fta", np.nan),
        "away_ftm": a_bt.get("ftm", np.nan),
        "away_oreb": a_bt.get("oreb", np.nan),
        "away_dreb": a_bt.get("dreb", np.nan),
        "away_tov": a_bt.get("tov", np.nan),
        "home_starter_min": hs.get("starter_min", np.nan),
        "home_bench_min": hs.get("bench_min", np.nan),
        "home_starter_min_share": hs.get("starter_min_share", np.nan),
        "home_starter_pts": hs.get("starter_pts", np.nan),
        "home_bench_pts": hs.get("bench_pts", np.nan),
        "home_starter_pm": hs.get("starter_pm", np.nan),
        "home_bench_pm": hs.get("bench_pm", np.nan),
        "away_starter_min": as_.get("starter_min", np.nan),
        "away_bench_min": as_.get("bench_min", np.nan),
        "away_starter_min_share": as_.get("starter_min_share", np.nan),
        "away_starter_pts": as_.get("starter_pts", np.nan),
        "away_bench_pts": as_.get("bench_pts", np.nan),
        "away_starter_pm": as_.get("starter_pm", np.nan),
        "away_bench_pm": as_.get("bench_pm", np.nan),
        "home_q1_starter_min_share": float(np.clip(h_q1_st_share, 0.0, 1.0)) if np.isfinite(h_q1_st_share) else np.nan,
        "away_q1_starter_min_share": float(np.clip(a_q1_st_share, 0.0, 1.0)) if np.isfinite(a_q1_st_share) else np.nan,
        "home_h1_starter_min_share": float(np.clip(h_h1_st_share, 0.0, 1.0)) if np.isfinite(h_h1_st_share) else np.nan,
        "away_h1_starter_min_share": float(np.clip(a_h1_st_share, 0.0, 1.0)) if np.isfinite(a_h1_st_share) else np.nan,
        "home_q1_first_sub_sec": first_sub_elapsed[home_tid] if has_starters_pbp else np.nan,
        "away_q1_first_sub_sec": first_sub_elapsed[away_tid] if has_starters_pbp else np.nan,
    }
    return game_row, player_rows


def _team_id_from_ref(ref: str | None) -> int | None:
    if not ref or "/teams/" not in ref:
        return None
    tail = ref.split("/teams/", 1)[1].split("?", 1)[0].split("/", 1)[0]
    try:
        return int(tail)
    except ValueError:
        return None


def parse_espn_period_propbets(
    items: list[dict[str, Any]],
    *,
    home_team_id: int,
    away_team_id: int,
    provider_name: str,
) -> dict[str, Any]:
    """Extract 1Q and 1H moneyline, spread, and total lines from ESPN propBets items."""
    out: dict[str, Any] = {
        "deriv_odds_provider": None,
        "book_q1_home_ml": np.nan,
        "book_q1_away_ml": np.nan,
        "book_q1_spread_home": np.nan,
        "book_q1_spread_away": np.nan,
        "book_q1_spread_home_odds": np.nan,
        "book_q1_spread_away_odds": np.nan,
        "book_q1_open_spread_home": np.nan,
        "book_q1_total_line": np.nan,
        "book_q1_over_odds": np.nan,
        "book_q1_under_odds": np.nan,
        "book_q1_open_total_line": np.nan,
        "book_h1_home_ml": np.nan,
        "book_h1_away_ml": np.nan,
        "book_h1_spread_home": np.nan,
        "book_h1_spread_away": np.nan,
        "book_h1_spread_home_odds": np.nan,
        "book_h1_spread_away_odds": np.nan,
        "book_h1_open_spread_home": np.nan,
        "book_h1_total_line": np.nan,
        "book_h1_over_odds": np.nan,
        "book_h1_under_odds": np.nan,
        "book_h1_open_total_line": np.nan,
    }
    found_any = False
    for it in items:
        if "athlete" in it:
            continue
        tname = str((it.get("type") or {}).get("name") or "")
        if tname not in {
            "1st Quarter Moneyline",
            "1st Quarter Spread",
            "1st Quarter Total",
            "1st Half Moneyline",
            "1st Half Spread",
            "1st Half Total",
        }:
            continue

        odds_blk = it.get("odds") or {}
        am_blk = odds_blk.get("american") or {}
        tot_blk = odds_blk.get("total") or {}
        cur_blk = it.get("current") or {}

        am_close = _parse_float_val(am_blk.get("value"), np.nan)
        line_close = _parse_float_val(
            tot_blk.get("value", (cur_blk.get("target") or {}).get("value")),
            np.nan,
        )
        line_open = _parse_float_val(tot_blk.get("open"), np.nan)
        tid = _team_id_from_ref((it.get("team") or {}).get("$ref"))
        prefix = "book_q1" if "1st Quarter" in tname else "book_h1"

        if "Moneyline" in tname:
            if tid == home_team_id and np.isfinite(am_close):
                out[f"{prefix}_home_ml"] = am_close
                found_any = True
            elif tid == away_team_id and np.isfinite(am_close):
                out[f"{prefix}_away_ml"] = am_close
                found_any = True
        elif "Spread" in tname:
            if tid == home_team_id and np.isfinite(line_close):
                out[f"{prefix}_spread_home"] = line_close
                out[f"{prefix}_spread_home_odds"] = am_close
                out[f"{prefix}_open_spread_home"] = line_open
                found_any = True
            elif tid == away_team_id and np.isfinite(line_close):
                out[f"{prefix}_spread_away"] = line_close
                out[f"{prefix}_spread_away_odds"] = am_close
                found_any = True
        elif "Total" in tname:
            if np.isfinite(line_close):
                out[f"{prefix}_total_line"] = line_close
                if np.isfinite(line_open):
                    out[f"{prefix}_open_total_line"] = line_open
                found_any = True
                # Record Over/Under juice only when explicitly labeled (e.g. ESPN BET current.over / current.under)
                if "over" in cur_blk:
                    ov_am = _parse_float_val((cur_blk.get("over") or {}).get("american", am_close), am_close)
                    out[f"{prefix}_over_odds"] = ov_am
                elif "under" in cur_blk:
                    un_am = _parse_float_val((cur_blk.get("under") or {}).get("american", am_close), am_close)
                    out[f"{prefix}_under_odds"] = un_am

    if found_any:
        out["deriv_odds_provider"] = provider_name
    return out


def fetch_event_period_propbets(
    event_id: str,
    *,
    home_team_id: int,
    away_team_id: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch archived 1Q/1H closing odds from ESPN sports.core propBets if present."""
    eid = str(event_id)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"propbets_{eid}.json"
        if cpath.exists():
            return json.loads(cpath.read_text(encoding="utf-8"))

    empty = parse_espn_period_propbets([], home_team_id=home_team_id, away_team_id=away_team_id, provider_name="")
    # Try primary non-live bookmaker IDs in priority order: 58 (ESPN BET), 100 (DraftKings)
    for pid, pname in (("58", "ESPN BET"), ("100", "DraftKings")):
        try:
            d1 = _http_json(ESPN_PROPBETS_TMPL.format(eid=eid, pid=pid, page=1), retries=2, pause=0.3, timeout=15.0)
        except Exception:
            continue
        cnt = int(d1.get("count") or 0)
        if cnt <= 0:
            continue
        items = list(d1.get("items") or [])
        n_pages = int(d1.get("pageCount") or 1)
        for pg in range(2, min(n_pages + 1, 4)):
            try:
                d_pg = _http_json(
                    ESPN_PROPBETS_TMPL.format(eid=eid, pid=pid, page=pg),
                    retries=2,
                    pause=0.3,
                    timeout=15.0,
                )
                items.extend(d_pg.get("items") or [])
            except Exception:
                break
        parsed = parse_espn_period_propbets(
            items,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            provider_name=pname,
        )
        if parsed["deriv_odds_provider"] is not None:
            if cache_dir is not None:
                (cache_dir / f"propbets_{eid}.json").write_text(json.dumps(parsed), encoding="utf-8")
            return parsed

    if cache_dir is not None:
        (cache_dir / f"propbets_{eid}.json").write_text(json.dumps(empty), encoding="utf-8")
    return empty


def fetch_and_parse_summary(
    event_id: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fetch one ESPN summary JSON (or load parsed cache) and return (game_row, player_rows)."""
    eid = str(event_id)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / f"summary_parsed_{eid}.json"
        if cpath.exists():
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            return cached["game"], cached["players"]

    raw = _http_json(ESPN_SUMMARY_TMPL.format(eid=eid))
    game_row, player_rows = parse_summary_payload(raw, fallback_game_id=eid)
    if cache_dir is not None:
        (cache_dir / f"summary_parsed_{eid}.json").write_text(
            json.dumps({"game": game_row, "players": player_rows}),
            encoding="utf-8",
        )
    return game_row, player_rows


def build_period_linescores_dataset(
    closing_markets_df: pd.DataFrame,
    *,
    cache_dir: Path | None = None,
    max_workers: int = 24,
    pull_period_odds_from_date: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull ESPN period linescores, boxscores, PBP stats, and 1Q/1H odds for all games."""
    base = closing_markets_df.copy()
    base["game_id"] = base["game_id"].astype(str)
    eids = base["game_id"].tolist()

    game_rows: dict[str, dict[str, Any]] = {}
    all_players: list[dict[str, Any]] = []
    meta_by_gid = {
        str(r["game_id"]): (str(r["season"]), str(r["game_date"]))
        for _, r in base[["game_id", "season", "game_date"]].iterrows()
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fetch_and_parse_summary, eid, cache_dir=cache_dir): eid for eid in eids}
        done = 0
        for fut in as_completed(futs):
            eid = futs[fut]
            grow, prows = fut.result()
            game_rows[eid] = grow
            s_val, d_val = meta_by_gid[eid]
            for pr in prows:
                pr["season"] = s_val
                pr["game_date"] = d_val
            all_players.extend(prows)
            done += 1
            if done % 1000 == 0:
                print(f"[espn_periods] summaries parsed: {done}/{len(eids)}", flush=True)

    summary_df = pd.DataFrame([game_rows[eid] for eid in eids])

    # Pull archived 1Q/1H propBets odds for games >= pull_period_odds_from_date
    odds_candidates = base.loc[
        pd.to_datetime(base["game_date"]) >= pd.Timestamp(pull_period_odds_from_date),
        ["game_id", "home_team_id", "away_team_id"],
    ]
    period_odds_by_gid: dict[str, dict[str, Any]] = {}
    if not odds_candidates.empty:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            ofuts = {
                pool.submit(
                    fetch_event_period_propbets,
                    str(r["game_id"]),
                    home_team_id=int(r["home_team_id"]),
                    away_team_id=int(r["away_team_id"]),
                    cache_dir=cache_dir,
                ): str(r["game_id"])
                for _, r in odds_candidates.iterrows()
            }
            odone = 0
            for fut in as_completed(ofuts):
                gid = ofuts[fut]
                period_odds_by_gid[gid] = fut.result()
                odone += 1
                if odone % 500 == 0:
                    print(f"[espn_periods] period propBets checked: {odone}/{len(odds_candidates)}", flush=True)

    empty_odds = parse_espn_period_propbets([], home_team_id=0, away_team_id=0, provider_name="")
    odds_df = pd.DataFrame(
        [{"game_id": eid, **period_odds_by_gid.get(eid, empty_odds)} for eid in eids]
    )

    keep_base_cols = [
        "season",
        "season_type",
        "game_id",
        "game_date",
        "home_team_id",
        "home_tricode",
        "away_team_id",
        "away_tricode",
        "home_score",
        "away_score",
        "is_neutral",
        "provider",
        "spread_provider",
        "total_provider",
        "home_ml",
        "away_ml",
        "p_home_novig",
        "p_away_novig",
        "spread_home",
        "spread_away",
        "spread_home_odds",
        "spread_away_odds",
        "total_line",
        "over_odds",
        "under_odds",
        "open_spread_home",
        "open_total_line",
    ]
    merged = (
        base[keep_base_cols]
        .merge(summary_df, on="game_id", how="left")
        .merge(odds_df, on="game_id", how="left")
    )
    merged["home_tricode"] = merged["home_tricode_norm"].where(
        merged["home_tricode_norm"].astype(str).str.len() > 0,
        merged["home_tricode"].map(normalize_espn_tricode),
    )
    merged["away_tricode"] = merged["away_tricode_norm"].where(
        merged["away_tricode_norm"].astype(str).str.len() > 0,
        merged["away_tricode"].map(normalize_espn_tricode),
    )
    merged = merged.drop(columns=["home_tricode_norm", "away_tricode_norm"], errors="ignore")
    merged = merged.sort_values(["game_date", "tipoff_utc", "game_id"]).reset_index(drop=True)

    players_df = pd.DataFrame(all_players)
    if not players_df.empty:
        players_df = players_df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)
    return merged, players_df
