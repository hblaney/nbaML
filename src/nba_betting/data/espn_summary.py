"""ESPN NBA game summary parser for linescores, non-garbage stints, and officials.

Fetches ``https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<id>``
and extracts strictly structured game-level, player-level, stint-level, and
pregame official crew records without touching live/current roster endpoints
such as ``summary.injuries``.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ESPN_SUMMARY_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={eid}"
)
DEFAULT_UA = "Mozilla/5.0 (compatible; nbaML-stint-officials/0.1)"

_OFFICIAL_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_OFFICIAL_WS_RE = re.compile(r"\s+")


def normalize_official_name(name: str | None) -> str:
    """Canonicalize referee names across punctuation/diacritic variants."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = s.replace(".", "").replace("-", " ").replace("'", "")
    s = _OFFICIAL_PUNCT_RE.sub(" ", s)
    s = _OFFICIAL_WS_RE.sub(" ", s).strip()
    # Canonicalize common NBA official suffixes/nicknames if needed
    aliases = {
        "jt orr": "jt orr",
        "cj washington": "cj washington",
        "jb derosa": "jb derosa",
        "joe derosa": "jb derosa",
    }
    return aliases.get(s, s)


def parse_clock_seconds(display_value: str | None, period: int = 1) -> float:
    """Parse ESPN period clock displayValue (e.g. '11:46' or '42.3') into seconds."""
    if display_value is None:
        return 0.0
    s = str(display_value).strip()
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":", 1)
        try:
            return max(0.0, float(parts[0]) * 60.0 + float(parts[1]))
        except ValueError:
            return 0.0
    try:
        return max(0.0, float(s))
    except ValueError:
        return 0.0


def is_garbage_time_state(
    period: int,
    rem_sec: float,
    abs_margin: float,
    already_garbage_q4: bool = False,
) -> bool:
    """Cleaning-the-Glass style garbage time rule on regulationblowouts.

    Overtime (period >= 5) is never garbage time. Once garbage time triggers in
    Q4, it stays active unless the margin closes back under 10 points.
    """
    if period >= 5:
        return False
    m = float(abs_margin)
    rem = float(rem_sec)
    if period == 3:
        return bool(m >= 32.0 or (rem <= 360.0 and m >= 28.0))
    if period < 4:
        return False
    if already_garbage_q4 and m >= 10.0:
        return True
    if m >= 25.0:
        return True
    if rem <= 540.0 and m >= 22.0:
        return True
    if rem <= 360.0 and m >= 18.0:
        return True
    if rem <= 180.0 and m >= 14.0:
        return True
    return bool(rem <= 60.0 and m >= 10.0)


def competitive_leverage_weight(period: int, rem_sec: float, abs_margin: float) -> float:
    """Smooth competitive leverage weight in [0.05, 1.0].

    Uses Brownian-bridge remaining-margin scale so competitive play throughout
    Q1-Q4/OT gets ~1.0 while lopsided blowouts smoothly decay toward 0.05.
    """
    m = float(abs_margin)
    if m <= 6.0:
        return 1.0
    if period >= 5:
        t_eff = max(120.0, float(rem_sec))
    else:
        t_eff = max(120.0, float(4 - min(int(period), 4)) * 720.0 + float(rem_sec))
    sigma_rem = 12.5 * math.sqrt(t_eff / 2880.0)
    z = (m - 6.0) / max(sigma_rem, 1.5)
    w = math.exp(-0.5 * z * z)
    return float(max(0.05, min(1.0, w)))


def _parse_made_att(val: str | None) -> tuple[float, float]:
    if not val or "-" not in str(val):
        return 0.0, 0.0
    a, b = str(val).split("-", 1)
    try:
        return float(a), float(b)
    except ValueError:
        return 0.0, 0.0


def _parse_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    s = str(val).strip().replace("+", "").replace("−", "-")
    if not s or s == "--":
        return default
    if ":" in s:
        parts = s.split(":", 1)
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except ValueError:
            return default
    try:
        return float(s)
    except ValueError:
        return default


def _infer_period_starters(
    period_plays: list[dict[str, Any]],
    team_ids: tuple[str, str],
    player_team: dict[str, str],
    prev_on_court: dict[str, set[str]],
    game_starters: dict[str, set[str]],
    period_num: int,
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    """Infer exact 5-man opening lineups for a period from PBP + 12:00 subs."""
    start_clk = "12:00" if period_num <= 4 else "5:00"
    prior = {t: set(prev_on_court.get(t, set())) for t in team_ids}
    idx = 0
    while idx < len(period_plays):
        p = period_plays[idx]
        clk = str((p.get("clock") or {}).get("displayValue") or "")
        ttext = str((p.get("type") or {}).get("text") or "")
        tid_type = str((p.get("type") or {}).get("id") or "")
        is_sub = tid_type == "584" or ttext == "Substitution"
        if clk == start_clk and is_sub:
            parts = p.get("participants") or []
            tid = str((p.get("team") or {}).get("id") or "")
            if len(parts) >= 2 and tid in prior:
                pin = str(((parts[0] or {}).get("athlete") or {}).get("id") or "")
                pout = str(((parts[1] or {}).get("athlete") or {}).get("id") or "")
                if pout:
                    prior[tid].discard(pout)
                if pin:
                    prior[tid].add(pin)
            idx += 1
        else:
            break

    rem_plays = period_plays[idx:]
    starters: dict[str, set[str]] = {t: set() for t in team_ids}
    subbed_in: dict[str, set[str]] = {t: set() for t in team_ids}

    for p in rem_plays:
        ttext = str((p.get("type") or {}).get("text") or "")
        tid_type = str((p.get("type") or {}).get("id") or "")
        is_sub = tid_type == "584" or ttext == "Substitution"
        if is_sub:
            parts = p.get("participants") or []
            tid = str((p.get("team") or {}).get("id") or "")
            if len(parts) >= 2 and tid in starters:
                pin = str(((parts[0] or {}).get("athlete") or {}).get("id") or "")
                pout = str(((parts[1] or {}).get("athlete") or {}).get("id") or "")
                if pout and pout not in subbed_in[tid]:
                    starters[tid].add(pout)
                if pin:
                    subbed_in[tid].add(pin)
        else:
            if "Technical" in ttext or "Ejection" in ttext or "Timeout" in ttext:
                continue
            for part in p.get("participants") or []:
                aid = str(((part or {}).get("athlete") or {}).get("id") or "")
                tid = player_team.get(aid, "")
                if tid in starters and aid and aid not in subbed_in[tid]:
                    starters[tid].add(aid)

    for t in team_ids:
        if len(starters[t]) < 5:
            for cand in list(prior[t]) + list(game_starters.get(t, set())):
                if len(starters[t]) >= 5:
                    break
                if cand and cand not in subbed_in[t]:
                    starters[t].add(cand)
        if len(starters[t]) > 5:
            ordered = sorted(starters[t], key=lambda a: (0 if a in prior[t] else 1, a))
            starters[t] = set(ordered[:5])
    return starters, rem_plays


def extract_pickcenter_fallback(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract fallback closing odds from summary.pickcenter when needed."""
    items = summary.get("pickcenter") or []
    best: dict[str, Any] | None = None
    best_score = -999
    for item in items:
        prov = (item.get("provider") or {}).get("name") or ""
        pid = str((item.get("provider") or {}).get("id") or "")
        if "live" in prov.lower() or pid in {"59", "46", "1001", "1002"}:
            continue
        score = 10 if (pid == "58" or "espn" in prov.lower()) else 5
        if score > best_score:
            best_score = score
            best = item
    if not best:
        return {}
    home_odds = best.get("homeTeamOdds") or {}
    away_odds = best.get("awayTeamOdds") or {}
    return {
        "pc_provider": (best.get("provider") or {}).get("name"),
        "pc_spread_home": _parse_float(best.get("spread"), default=float("nan")),
        "pc_total_line": _parse_float(best.get("overUnder"), default=float("nan")),
        "pc_home_ml": _parse_float(home_odds.get("moneyLine"), default=float("nan")),
        "pc_away_ml": _parse_float(away_odds.get("moneyLine"), default=float("nan")),
    }


def parse_espn_summary(summary: dict[str, Any], event_id: str) -> dict[str, Any]:
    """Parse one ESPN summary JSON into game, official, player, and stint records."""
    eid = str(event_id)
    hdr = summary.get("header") or {}
    comps = hdr.get("competitions") or [{}]
    comp = comps[0] if comps else {}
    competitors = comp.get("competitors") or []

    home_comp: dict[str, Any] = {}
    away_comp: dict[str, Any] = {}
    for c in competitors:
        if c.get("homeAway") == "home":
            home_comp = c
        elif c.get("homeAway") == "away":
            away_comp = c

    home_id = str(home_comp.get("id") or "")
    away_id = str(away_comp.get("id") or "")
    home_tri = str((home_comp.get("team") or {}).get("abbreviation") or "")
    away_tri = str((away_comp.get("team") or {}).get("abbreviation") or "")

    h_ls = [_parse_float(x.get("displayValue"), 0.0) for x in (home_comp.get("linescores") or [])]
    a_ls = [_parse_float(x.get("displayValue"), 0.0) for x in (away_comp.get("linescores") or [])]
    while len(h_ls) < 4:
        h_ls.append(0.0)
    while len(a_ls) < 4:
        a_ls.append(0.0)
    n_ot = max(0, max(len(h_ls), len(a_ls)) - 4)
    h_reg = float(sum(h_ls[:4]))
    a_reg = float(sum(a_ls[:4]))
    h_ot = float(sum(h_ls[4:]))
    a_ot = float(sum(a_ls[4:]))
    h_final = _parse_float(home_comp.get("score"), h_reg + h_ot)
    a_final = _parse_float(away_comp.get("score"), a_reg + a_ot)

    margin_q1 = float(h_ls[0] - a_ls[0])
    margin_1h = float((h_ls[0] + h_ls[1]) - (a_ls[0] + a_ls[1]))
    margin_3q = float(sum(h_ls[:3]) - sum(a_ls[:3]))
    margin_q4 = float(h_ls[3] - a_ls[3])
    final_margin = float(h_final - a_final)

    # Officials (top 3 on-court officials ordered by `order`; `order == 4` is playoff alternate)
    gi = summary.get("gameInfo") or {}
    raw_offs = sorted(
        gi.get("officials") or [],
        key=lambda o: int(o.get("order") or 99),
    )
    on_court_offs = raw_offs[:3]
    off_names = [
        str(o.get("displayName") or o.get("fullName") or "").strip() for o in on_court_offs
    ]
    off_norm = [normalize_official_name(n) for n in off_names if normalize_official_name(n)]
    attendance = int(_parse_float(gi.get("attendance"), 0.0))

    # Team boxscores
    bs = summary.get("boxscore") or {}
    team_box: dict[str, dict[str, float]] = {}
    for tb in bs.get("teams") or []:
        tid = str((tb.get("team") or {}).get("id") or "")
        smap = {
            str(s.get("name") or ""): str(s.get("displayValue") or "")
            for s in (tb.get("statistics") or [])
        }
        fgm, fga = _parse_made_att(smap.get("fieldGoalsMade-fieldGoalsAttempted"))
        fg3m, fg3a = _parse_made_att(
            smap.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        )
        ftm, fta = _parse_made_att(smap.get("freeThrowsMade-freeThrowsAttempted"))
        oreb = _parse_float(smap.get("offensiveRebounds"))
        dreb = _parse_float(smap.get("defensiveRebounds"))
        tov = _parse_float(smap.get("totalTurnovers") or smap.get("turnovers"))
        pf = _parse_float(smap.get("fouls"))
        tech = _parse_float(smap.get("totalTechnicalFouls") or smap.get("technicalFouls"))
        pip = _parse_float(smap.get("pointsInPaint"))
        fbp = _parse_float(smap.get("fastBreakPoints"))
        lead = _parse_float(smap.get("largestLead"))
        poss = fga - oreb + tov + 0.44 * fta
        team_box[tid] = {
            "fgm": fgm,
            "fga": fga,
            "fg3m": fg3m,
            "fg3a": fg3a,
            "ftm": ftm,
            "fta": fta,
            "oreb": oreb,
            "dreb": dreb,
            "tov": tov,
            "pf": pf,
            "tech": tech,
            "pip": pip,
            "fbp": fbp,
            "largest_lead": lead,
            "box_poss": poss,
        }

    # Player boxscores + starter sets
    game_starters: dict[str, set[str]] = {home_id: set(), away_id: set()}
    player_team: dict[str, str] = {}
    player_records: dict[str, dict[str, Any]] = {}
    for tp in bs.get("players") or []:
        tid = str((tp.get("team") or {}).get("id") or "")
        if tid not in game_starters:
            game_starters[tid] = set()
        for grp in tp.get("statistics") or []:
            keys = [str(k) for k in (grp.get("keys") or [])]
            k_idx = {k: idx for idx, k in enumerate(keys)}
            for ath in grp.get("athletes") or []:
                ainfo = ath.get("athlete") or {}
                aid = str(ainfo.get("id") or "")
                if not aid:
                    continue
                player_team[aid] = tid
                starter = bool(ath.get("starter"))
                if starter:
                    game_starters[tid].add(aid)
                dnp = bool(ath.get("didNotPlay"))
                stats = ath.get("stats") or []

                def _stat(
                    name: str,
                    _k: dict[str, int] = k_idx,
                    _s: list[Any] = stats,
                ) -> str | None:
                    i = _k.get(name, -1)
                    return str(_s[i]) if 0 <= i < len(_s) else None

                mins = 0.0 if dnp or not stats else _parse_float(_stat("minutes"))
                pts = 0.0 if dnp or not stats else _parse_float(_stat("points"))
                pm = 0.0 if dnp or not stats else _parse_float(_stat("plusMinus"))
                pf_p = 0.0 if dnp or not stats else _parse_float(_stat("fouls"))
                _fgm_p, fga_p = (
                    (0.0, 0.0)
                    if dnp or not stats
                    else _parse_made_att(_stat("fieldGoalsMade-fieldGoalsAttempted"))
                )
                _ftm_p, fta_p = (
                    (0.0, 0.0)
                    if dnp or not stats
                    else _parse_made_att(_stat("freeThrowsMade-freeThrowsAttempted"))
                )
                player_records[aid] = {
                    "game_id": eid,
                    "team_id": int(tid) if tid.isdigit() else -1,
                    "player_id": aid,
                    "player_name": str(ainfo.get("displayName") or ""),
                    "starter": starter,
                    "dnp": dnp,
                    "box_minutes": mins,
                    "box_pts": pts,
                    "box_pm": pm,
                    "box_fga": fga_p,
                    "box_fta": fta_p,
                    "box_pf": pf_p,
                    "clean_sec": 0.0,
                    "clean_pm": 0.0,
                    "clean_poss": 0.0,
                    "lev_pm": 0.0,
                    "lev_poss": 0.0,
                }

    # Walk play-by-play for garbage-time filtering, leverage weights, and 5-v-5 stints
    plays = summary.get("plays") or []
    pbp_acc: dict[str, dict[str, float]] = {
        t: {
            "clean_pts": 0.0,
            "clean_poss": 0.0,
            "clean_fga": 0.0,
            "clean_fta": 0.0,
            "clean_pf": 0.0,
            "clean_tov": 0.0,
            "clean_oreb": 0.0,
            "pace_clean_poss": 0.0,
            "lev_pts": 0.0,
            "lev_poss": 0.0,
            "garb_pts": 0.0,
            "garb_poss": 0.0,
            "endgame_fta": 0.0,
        }
        for t in (home_id, away_id)
    }
    clean_sec_total = 0.0
    garb_sec_total = 0.0
    pace_clean_sec_total = 0.0
    stint_acc: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, float]] = {}

    if plays and home_id and away_id:
        team_ids = (home_id, away_id)
        on_court = {t: set(game_starters.get(t, set())) for t in team_ids}
        prev_h = 0.0
        prev_a = 0.0
        garb_q4_latch = False

        for per, per_iter in groupby(
            plays, key=lambda p: int((p.get("period") or {}).get("number") or 1)
        ):
            p_plays = list(per_iter)
            if per == 1 and len(on_court[home_id]) == 5 and len(on_court[away_id]) == 5:
                rem_plays = p_plays
            else:
                on_court, rem_plays = _infer_period_starters(
                    p_plays, team_ids, player_team, on_court, game_starters, per
                )
            prev_rem = 720.0 if per <= 4 else 300.0
            if per != 4:
                garb_q4_latch = False

            for clk, clk_iter in groupby(
                rem_plays, key=lambda p: str((p.get("clock") or {}).get("displayValue") or "")
            ):
                clist = list(clk_iter)
                rem_sec = parse_clock_seconds(clk, per)
                dt = max(0.0, prev_rem - rem_sec)
                prev_rem = rem_sec

                abs_m = abs(prev_h - prev_a)
                is_garb = is_garbage_time_state(per, rem_sec, abs_m, garb_q4_latch)
                if per == 4 and is_garb:
                    garb_q4_latch = True
                elif per == 4 and abs_m < 10.0:
                    garb_q4_latch = False
                is_foul_fest = bool(per >= 4 and rem_sec <= 90.0 and 1.0 <= abs_m <= 9.0)
                w_lev = competitive_leverage_weight(per, rem_sec, abs_m)

                if is_garb:
                    garb_sec_total += dt
                else:
                    clean_sec_total += dt
                    if per <= 4 and not is_foul_fest:
                        pace_clean_sec_total += dt
                    for tid in team_ids:
                        for aid in on_court[tid]:
                            if aid in player_records:
                                player_records[aid]["clean_sec"] += dt

                h_key = tuple(sorted(on_court[home_id]))
                a_key = tuple(sorted(on_court[away_id]))
                valid_5v5 = len(h_key) == 5 and len(a_key) == 5
                if valid_5v5 and not is_garb:
                    srec = stint_acc.setdefault(
                        (h_key, a_key),
                        {
                            "clean_sec": 0.0,
                            "clean_pts_home": 0.0,
                            "clean_pts_away": 0.0,
                            "clean_poss_home": 0.0,
                            "clean_poss_away": 0.0,
                            "lev_pts_home": 0.0,
                            "lev_pts_away": 0.0,
                            "lev_poss_home": 0.0,
                            "lev_poss_away": 0.0,
                        },
                    )
                    srec["clean_sec"] += dt

                has_ft = any(
                    "Free Throw" in str((x.get("type") or {}).get("text") or "")
                    and "Technical" not in str((x.get("type") or {}).get("text") or "")
                    for x in clist
                )
                if has_ft:
                    subs = [
                        x
                        for x in clist
                        if str((x.get("type") or {}).get("id") or "") == "584"
                        or str((x.get("type") or {}).get("text") or "") == "Substitution"
                    ]
                    nonsubs = [x for x in clist if x not in subs]
                    ordered = nonsubs + subs
                else:
                    ordered = clist

                for p in ordered:
                    ttext = str((p.get("type") or {}).get("text") or "")
                    tid_type = str((p.get("type") or {}).get("id") or "")
                    txt = str(p.get("text") or "")
                    tid = str((p.get("team") or {}).get("id") or "")

                    h_key = tuple(sorted(on_court[home_id]))
                    a_key = tuple(sorted(on_court[away_id]))
                    valid_5v5 = len(h_key) == 5 and len(a_key) == 5

                    hs = prev_h
                    as_ = prev_a
                    dh = 0.0
                    da = 0.0
                    if bool(p.get("scoringPlay")):
                        sv = max(0.0, _parse_float(p.get("scoreValue"), 0.0))
                        if sv > 0.0:
                            if tid == home_id:
                                dh = sv
                                hs = prev_h + dh
                            elif tid == away_id:
                                da = sv
                                as_ = prev_a + da
                    prev_h, prev_a = hs, as_

                    # Possession delta for acting team
                    d_poss = 0.0
                    d_fga = 0.0
                    d_fta = 0.0
                    d_pf = 0.0
                    d_tov = 0.0
                    d_oreb = 0.0
                    if tid in pbp_acc:
                        if p.get("shootingPlay") and "Free Throw" not in ttext:
                            d_fga = 1.0
                            d_poss += 1.0
                        elif "Free Throw" in ttext:
                            d_fta = 1.0
                            if "Technical" not in ttext:
                                d_poss += 0.44
                        elif ttext == "Offensive Rebound" and "team rebound" not in txt.lower():
                            d_oreb = 1.0
                            d_poss -= 1.0
                        elif "Turnover" in ttext or "Shot Clock" in ttext:
                            d_tov = 1.0
                            d_poss += 1.0
                        if "Foul" in ttext and "Technical" not in ttext:
                            d_pf = 1.0

                    # Credit team & stint accumulators
                    if dh > 0.0 or da > 0.0:
                        net_h = dh - da
                        if is_garb:
                            pbp_acc[home_id]["garb_pts"] += dh
                            pbp_acc[away_id]["garb_pts"] += da
                        else:
                            pbp_acc[home_id]["clean_pts"] += dh
                            pbp_acc[away_id]["clean_pts"] += da
                            for aid in on_court[home_id]:
                                if aid in player_records:
                                    player_records[aid]["clean_pm"] += net_h
                            for aid in on_court[away_id]:
                                if aid in player_records:
                                    player_records[aid]["clean_pm"] -= net_h
                            if valid_5v5:
                                srec = stint_acc.setdefault(
                                    (h_key, a_key),
                                    {
                                        "clean_sec": 0.0,
                                        "clean_pts_home": 0.0,
                                        "clean_pts_away": 0.0,
                                        "clean_poss_home": 0.0,
                                        "clean_poss_away": 0.0,
                                        "lev_pts_home": 0.0,
                                        "lev_pts_away": 0.0,
                                        "lev_poss_home": 0.0,
                                        "lev_poss_away": 0.0,
                                    },
                                )
                                srec["clean_pts_home"] += dh
                                srec["clean_pts_away"] += da
                                srec["lev_pts_home"] += w_lev * dh
                                srec["lev_pts_away"] += w_lev * da
                        pbp_acc[home_id]["lev_pts"] += w_lev * dh
                        pbp_acc[away_id]["lev_pts"] += w_lev * da
                        for aid in on_court[home_id]:
                            if aid in player_records:
                                player_records[aid]["lev_pm"] += w_lev * net_h
                        for aid in on_court[away_id]:
                            if aid in player_records:
                                player_records[aid]["lev_pm"] -= w_lev * net_h

                    if tid in pbp_acc:
                        if is_foul_fest and d_fta > 0:
                            pbp_acc[tid]["endgame_fta"] += d_fta
                        if is_garb:
                            pbp_acc[tid]["garb_poss"] += d_poss
                        else:
                            pbp_acc[tid]["clean_poss"] += d_poss
                            pbp_acc[tid]["clean_fga"] += d_fga
                            pbp_acc[tid]["clean_fta"] += d_fta
                            pbp_acc[tid]["clean_pf"] += d_pf
                            pbp_acc[tid]["clean_tov"] += d_tov
                            pbp_acc[tid]["clean_oreb"] += d_oreb
                            if per <= 4 and not is_foul_fest:
                                pbp_acc[tid]["pace_clean_poss"] += d_poss
                            if d_poss != 0.0:
                                for aid in on_court[home_id] | on_court[away_id]:
                                    if aid in player_records:
                                        player_records[aid]["clean_poss"] += 0.5 * d_poss
                            if valid_5v5 and d_poss != 0.0:
                                srec = stint_acc.setdefault(
                                    (h_key, a_key),
                                    {
                                        "clean_sec": 0.0,
                                        "clean_pts_home": 0.0,
                                        "clean_pts_away": 0.0,
                                        "clean_poss_home": 0.0,
                                        "clean_poss_away": 0.0,
                                        "lev_pts_home": 0.0,
                                        "lev_pts_away": 0.0,
                                        "lev_poss_home": 0.0,
                                        "lev_poss_away": 0.0,
                                    },
                                )
                                if tid == home_id:
                                    srec["clean_poss_home"] += d_poss
                                    srec["lev_poss_home"] += w_lev * d_poss
                                else:
                                    srec["clean_poss_away"] += d_poss
                                    srec["lev_poss_away"] += w_lev * d_poss
                        pbp_acc[tid]["lev_poss"] += w_lev * d_poss
                        if d_poss != 0.0:
                            for aid in on_court[home_id] | on_court[away_id]:
                                if aid in player_records:
                                    player_records[aid]["lev_poss"] += 0.5 * w_lev * d_poss

                    # Apply substitution
                    if tid_type == "584" or ttext == "Substitution":
                        parts = p.get("participants") or []
                        if len(parts) >= 2 and tid in on_court:
                            pin = str(((parts[0] or {}).get("athlete") or {}).get("id") or "")
                            pout = str(((parts[1] or {}).get("athlete") or {}).get("id") or "")
                            if pout in on_court[tid]:
                                on_court[tid].discard(pout)
                                if pin:
                                    on_court[tid].add(pin)
                            elif pin and pin not in on_court[tid] and len(on_court[tid]) < 5:
                                on_court[tid].add(pin)

            # Close any remaining clock at end of period
            if prev_rem > 0.0:
                abs_m = abs(prev_h - prev_a)
                is_garb = is_garbage_time_state(per, 0.0, abs_m, garb_q4_latch)
                if is_garb:
                    garb_sec_total += prev_rem
                else:
                    clean_sec_total += prev_rem
                    if per <= 4 and not (per >= 4 and 1.0 <= abs_m <= 9.0):
                        pace_clean_sec_total += prev_rem
                    for tid in team_ids:
                        for aid in on_court[tid]:
                            if aid in player_records:
                                player_records[aid]["clean_sec"] += prev_rem

    hb = team_box.get(home_id, {})
    ab = team_box.get(away_id, {})
    hp = pbp_acc.get(home_id, {})
    ap = pbp_acc.get(away_id, {})

    total_sec = float(2880.0 + 300.0 * n_ot)
    box_poss_home = float(hb.get("box_poss", 0.0))
    box_poss_away = float(ab.get("box_poss", 0.0))
    box_poss_avg = 0.5 * (box_poss_home + box_poss_away)
    raw_pace = 2880.0 * box_poss_avg / max(total_sec, 1.0) if box_poss_avg > 0 else 0.0

    clean_poss_h = max(float(hp.get("clean_poss", 0.0)), 0.0)
    clean_poss_a = max(float(ap.get("clean_poss", 0.0)), 0.0)
    clean_poss_avg = 0.5 * (clean_poss_h + clean_poss_a)
    clean_pts_h = float(hp.get("clean_pts", 0.0))
    clean_pts_a = float(ap.get("clean_pts", 0.0))

    pace_poss_avg = 0.5 * (
        max(float(hp.get("pace_clean_poss", 0.0)), 0.0)
        + max(float(ap.get("pace_clean_poss", 0.0)), 0.0)
    )
    clean_pace = (
        2880.0 * pace_poss_avg / max(pace_clean_sec_total, 300.0)
        if pace_clean_sec_total >= 300.0 and pace_poss_avg > 0
        else raw_pace
    )

    lev_poss_h = max(float(hp.get("lev_poss", 0.0)), 0.0)
    lev_poss_a = max(float(ap.get("lev_poss", 0.0)), 0.0)
    lev_pts_h = float(hp.get("lev_pts", 0.0))
    lev_pts_a = float(ap.get("lev_pts", 0.0))

    game_row: dict[str, Any] = {
        "game_id": eid,
        "espn_home_id": int(home_id) if home_id.isdigit() else -1,
        "espn_away_id": int(away_id) if away_id.isdigit() else -1,
        "home_tricode_espn": home_tri,
        "away_tricode_espn": away_tri,
        "attendance": attendance,
        "n_plays": len(plays),
        "n_ot_periods": n_ot,
        "q1_home": h_ls[0],
        "q2_home": h_ls[1],
        "q3_home": h_ls[2],
        "q4_home": h_ls[3],
        "ot_home": h_ot,
        "q1_away": a_ls[0],
        "q2_away": a_ls[1],
        "q3_away": a_ls[2],
        "q4_away": a_ls[3],
        "ot_away": a_ot,
        "reg_home_pts": h_reg,
        "reg_away_pts": a_reg,
        "reg_total_pts": h_reg + a_reg,
        "margin_q1": margin_q1,
        "margin_1h": margin_1h,
        "margin_3q": margin_3q,
        "margin_q4": margin_q4,
        "q4_giveback_home": margin_3q - final_margin,
        # Officials
        "n_officials": len(off_norm),
        "ref_1": off_norm[0] if len(off_norm) > 0 else "",
        "ref_2": off_norm[1] if len(off_norm) > 1 else "",
        "ref_3": off_norm[2] if len(off_norm) > 2 else "",
        "ref_1_display": off_names[0] if len(off_names) > 0 else "",
        "ref_2_display": off_names[1] if len(off_names) > 1 else "",
        "ref_3_display": off_names[2] if len(off_names) > 2 else "",
        # Full-game boxscore stats
        "home_fga": float(hb.get("fga", 0.0)),
        "away_fga": float(ab.get("fga", 0.0)),
        "home_fgm": float(hb.get("fgm", 0.0)),
        "away_fgm": float(ab.get("fgm", 0.0)),
        "home_fg3a": float(hb.get("fg3a", 0.0)),
        "away_fg3a": float(ab.get("fg3a", 0.0)),
        "home_fg3m": float(hb.get("fg3m", 0.0)),
        "away_fg3m": float(ab.get("fg3m", 0.0)),
        "home_fta": float(hb.get("fta", 0.0)),
        "away_fta": float(ab.get("fta", 0.0)),
        "home_ftm": float(hb.get("ftm", 0.0)),
        "away_ftm": float(ab.get("ftm", 0.0)),
        "home_oreb": float(hb.get("oreb", 0.0)),
        "away_oreb": float(ab.get("oreb", 0.0)),
        "home_tov": float(hb.get("tov", 0.0)),
        "away_tov": float(ab.get("tov", 0.0)),
        "home_pf": float(hb.get("pf", 0.0)),
        "away_pf": float(ab.get("pf", 0.0)),
        "home_tech": float(hb.get("tech", 0.0)),
        "away_tech": float(ab.get("tech", 0.0)),
        "home_pip": float(hb.get("pip", 0.0)),
        "away_pip": float(ab.get("pip", 0.0)),
        "home_largest_lead": float(hb.get("largest_lead", 0.0)),
        "away_largest_lead": float(ab.get("largest_lead", 0.0)),
        "home_box_poss": box_poss_home,
        "away_box_poss": box_poss_away,
        "raw_pace": raw_pace,
        "home_raw_ortg": 100.0 * h_final / max(box_poss_home, 1.0) if box_poss_home > 0 else 114.0,
        "away_raw_ortg": 100.0 * a_final / max(box_poss_away, 1.0) if box_poss_away > 0 else 114.0,
        # Non-garbage-time & leverage-weighted stats
        "clean_sec": clean_sec_total,
        "garb_sec": garb_sec_total,
        "garb_frac": garb_sec_total / max(clean_sec_total + garb_sec_total, 1.0),
        "clean_pace": clean_pace,
        "home_clean_pts": clean_pts_h,
        "away_clean_pts": clean_pts_a,
        "home_clean_poss": clean_poss_h,
        "away_clean_poss": clean_poss_a,
        "home_clean_ortg": (
            100.0 * clean_pts_h / max(clean_poss_h, 1.0)
            if clean_poss_avg >= 20.0
            else (100.0 * h_final / max(box_poss_home, 1.0) if box_poss_home > 0 else 114.0)
        ),
        "away_clean_ortg": (
            100.0 * clean_pts_a / max(clean_poss_a, 1.0)
            if clean_poss_avg >= 20.0
            else (100.0 * a_final / max(box_poss_away, 1.0) if box_poss_away > 0 else 114.0)
        ),
        "home_lev_pts": lev_pts_h,
        "away_lev_pts": lev_pts_a,
        "home_lev_poss": lev_poss_h,
        "away_lev_poss": lev_poss_a,
        "home_lev_ortg": (
            100.0 * lev_pts_h / max(lev_poss_h, 1.0)
            if 0.5 * (lev_poss_h + lev_poss_a) >= 20.0
            else (100.0 * h_final / max(box_poss_home, 1.0) if box_poss_home > 0 else 114.0)
        ),
        "away_lev_ortg": (
            100.0 * lev_pts_a / max(lev_poss_a, 1.0)
            if 0.5 * (lev_poss_h + lev_poss_a) >= 20.0
            else (100.0 * a_final / max(box_poss_away, 1.0) if box_poss_away > 0 else 114.0)
        ),
        "home_endgame_fta": float(hp.get("endgame_fta", 0.0)),
        "away_endgame_fta": float(ap.get("endgame_fta", 0.0)),
        **extract_pickcenter_fallback(summary),
    }

    stint_rows: list[dict[str, Any]] = []
    for (h_key, a_key), sr in stint_acc.items():
        poss_avg = 0.5 * (sr["clean_poss_home"] + sr["clean_poss_away"])
        if poss_avg <= 0.0 and sr["clean_pts_home"] == 0.0 and sr["clean_pts_away"] == 0.0:
            continue
        stint_rows.append(
            {
                "game_id": eid,
                "home_p1": h_key[0],
                "home_p2": h_key[1],
                "home_p3": h_key[2],
                "home_p4": h_key[3],
                "home_p5": h_key[4],
                "away_p1": a_key[0],
                "away_p2": a_key[1],
                "away_p3": a_key[2],
                "away_p4": a_key[3],
                "away_p5": a_key[4],
                **sr,
            }
        )

    player_rows: list[dict[str, Any]] = []
    for rec in player_records.values():
        rec["clean_minutes"] = float(rec.pop("clean_sec")) / 60.0
        player_rows.append(rec)

    return {
        "game": game_row,
        "players": player_rows,
        "stints": stint_rows,
    }


def fetch_and_parse_summary(
    event_id: str,
    *,
    session: requests.Session | None = None,
    raw_cache_dir: Path | None = None,
    retries: int = 4,
    backoff_s: float = 0.7,
) -> dict[str, Any]:
    """Fetch one ESPN summary (with optional .json.gz disk cache) and parse it."""
    eid = str(event_id)
    gz_path: Path | None = None
    if raw_cache_dir is not None:
        raw_cache_dir.mkdir(parents=True, exist_ok=True)
        gz_path = raw_cache_dir / f"{eid}.json.gz"
        if gz_path.exists():
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                    return parse_espn_summary(json.load(fh), eid)
            except (OSError, json.JSONDecodeError, EOFError):
                gz_path.unlink(missing_ok=True)

    url = ESPN_SUMMARY_TMPL.format(eid=eid)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            if session is not None:
                resp = session.get(url, timeout=35)
            else:
                resp = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=35)
            resp.raise_for_status()
            payload = resp.json()
            if gz_path is not None:
                tmp_path = raw_cache_dir / f"{eid}.json.gz.tmp"  # type: ignore[operator]
                with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=1) as fh:
                    json.dump(payload, fh, separators=(",", ":"))
                tmp_path.replace(gz_path)
            return parse_espn_summary(payload, eid)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"Failed ESPN summary fetch for {eid}: {last_exc}")


def pull_espn_summaries_dataset(
    event_ids: list[str],
    *,
    out_dir: Path,
    raw_cache_dir: Path | None = None,
    max_workers: int = 24,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull & parse ESPN summaries concurrently into game, player, and stint parquets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "espn_summary_games.parquet"
    players_path = out_dir / "espn_summary_players.parquet"
    stints_path = out_dir / "espn_summary_stints.parquet"
    if not force and games_path.exists() and players_path.exists() and stints_path.exists():
        g_df = pd.read_parquet(games_path)
        if len(g_df) >= len({str(x) for x in event_ids}):
            return (
                g_df,
                pd.read_parquet(players_path),
                pd.read_parquet(stints_path),
            )

    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers + 8,
        pool_maxsize=max_workers + 8,
    )
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": DEFAULT_UA})

    unique_ids = list(dict.fromkeys(str(x) for x in event_ids))
    game_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    stint_rows: list[dict[str, Any]] = []

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                fetch_and_parse_summary,
                eid,
                session=sess,
                raw_cache_dir=raw_cache_dir,
            ): eid
            for eid in unique_ids
        }
        for fut in as_completed(futs):
            parsed = fut.result()
            game_rows.append(parsed["game"])
            player_rows.extend(parsed["players"])
            stint_rows.extend(parsed["stints"])
            done += 1
            if done % 500 == 0 or done == len(unique_ids):
                dt = time.time() - t0
                print(
                    f"[espn_summary] {done}/{len(unique_ids)} parsed in {dt:.1f}s "
                    f"({done / max(dt, 1e-3):.1f} g/s)",
                    flush=True,
                )

    games_df = pd.DataFrame(game_rows).sort_values("game_id").reset_index(drop=True)
    players_df = pd.DataFrame(player_rows).sort_values(["game_id", "team_id", "player_id"]).reset_index(drop=True)
    stints_df = pd.DataFrame(stint_rows).sort_values(["game_id"]).reset_index(drop=True)

    games_df.to_parquet(games_path, index=False)
    players_df.to_parquet(players_path, index=False)
    stints_df.to_parquet(stints_path, index=False)
    return games_df, players_df, stints_df
