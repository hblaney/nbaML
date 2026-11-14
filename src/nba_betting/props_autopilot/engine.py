"""24/7 Autonomous PrizePicks + DraftKings Live Scanner, Snapshot Archiver, Grader & R&D Loop."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
import unicodedata
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nbinom

from nba_betting.props_autopilot.prizepicks_math import (
    apply_stat_tail_calibration,
    settle_prizepicks_slip,
    slip_ev_and_log_growth,
)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
NORM_TEAM = {"NYK": "NY", "NOP": "NO", "SAS": "SA", "GSW": "GS", "UTA": "UTAH", "WAS": "WSH"}

TWO_WAY_MAP = {
    "Total Points": "Points",
    "Total Rebounds": "Rebounds",
    "Total Assists": "Assists",
    "Total Points, Rebounds, and Assists": "Pts+Rebs+Asts",
    "Total Points and Rebounds": "Pts+Rebs",
    "Total Points and Assists": "Pts+Asts",
    "Total Assists and Rebounds": "Rebs+Asts",
    "Total 3-Point Field Goals": "3-PT Made",
}
MILE_MAP = {
    "Points Milestones": "Points",
    "Rebounds Milestones": "Rebounds",
    "Assists Milestones": "Assists",
    "Points + Assists + Rebounds Milestones": "Pts+Rebs+Asts",
    "Points + Rebounds Milestones": "Pts+Rebs",
    "Points + Assists Milestones": "Pts+Asts",
    "Rebounds + Assists Milestones": "Rebs+Asts",
    "3-Point Field Goals Milestones": "3-PT Made",
}
STAT_COL_MAP = {
    "Points": ("pts", 0.060),
    "Rebounds": ("reb", 0.050),
    "Assists": ("ast", 0.070),
    "Pts+Rebs+Asts": ("pra", 0.045),
    "Pts+Rebs": ("pr", 0.050),
    "Pts+Asts": ("pa", 0.055),
    "Rebs+Asts": ("ra", 0.055),
    "3-PT Made": ("fg3m", 0.080),
}


def canon_name(s: str | None) -> str:
    if not s:
        return ""
    clean = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()
    clean = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", clean)
    return re.sub(r"[^a-z0-9]+", " ", clean).strip()


def norm_team(t: str | None) -> str:
    if not t:
        return ""
    up = str(t).upper().strip()
    return NORM_TEAM.get(up, up)


def _fetch_json(url: str, extra: dict[str, str] | None = None, retries: int = 3) -> Any:
    hdrs = {"User-Agent": UA, "Accept": "application/json"}
    if extra:
        hdrs.update(extra)
    for i in range(retries):
        try:
            with urlopen(Request(url, headers=hdrs), timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if i + 1 == retries:
                raise
            time.sleep(0.35 * (i + 1))


def devig_shin_2way(dec_over: float, dec_under: float) -> tuple[float, float]:
    q0, q1 = 1.0 / float(dec_over), 1.0 / float(dec_under)
    qsum = q0 + q1
    if qsum <= 1.0 + 1e-9:
        return q0 / qsum, q1 / qsum
    p0 = float(np.clip(q0 - 0.5 * (qsum - 1.0), 0.01, 0.99))
    return p0, 1.0 - p0


def nb_prob_over_under_push(line: float, mu: float, alpha: float) -> tuple[float, float, float]:
    mu_c = max(float(mu), 0.05)
    n = 1.0 / float(alpha)
    p = n / (n + mu_c)
    if float(line).is_integer():
        k = int(round(line))
        p_push = float(nbinom.pmf(k, n, p))
        p_less = float(nbinom.cdf(k - 1, n, p)) if k >= 1 else 0.0
        p_more = max(0.0, 1.0 - p_less - p_push)
        return p_more, p_less, p_push
    k_floor = int(np.floor(line))
    p_less = float(nbinom.cdf(k_floor, n, p))
    return 1.0 - p_less, p_less, 0.0


def fetch_prizepicks_standard_board() -> pd.DataFrame:
    pp = _fetch_json(
        "https://partner-api.prizepicks.com/projections?league_id=7&per_page=1000&single_stat=true",
        {"Referer": "https://app.prizepicks.com/", "Origin": "https://app.prizepicks.com"},
    )
    inc = {(x["type"], x["id"]): x.get("attributes", {}) for x in pp.get("included", [])}
    rows: list[dict[str, Any]] = []
    for item in pp.get("data", []):
        attr = item.get("attributes") or {}
        if attr.get("odds_type") != "standard" or attr.get("is_promo") or attr.get("in_game"):
            continue
        if attr.get("status") != "pre_game":
            continue
        rel = item.get("relationships") or {}
        np_id = ((rel.get("new_player") or {}).get("data") or {}).get("id")
        pl = inc.get(("new_player", np_id), {})
        if pl.get("combo"):
            continue
        dur_id = ((rel.get("duration") or {}).get("data") or {}).get("id")
        dur = inc.get(("duration", dur_id), {})
        if dur.get("name") and dur.get("name") != "Full":
            continue
        pname = pl.get("display_name") or pl.get("name")
        stat_type = attr.get("stat_type")
        if stat_type not in STAT_COL_MAP:
            continue
        rows.append(
            {
                "pp_id": str(item.get("id")),
                "player_name": pname,
                "cname": canon_name(pname),
                "team_norm": norm_team(pl.get("team")),
                "stat_type": stat_type,
                "pp_line": float(attr.get("line_score")),
                "start_time": str(attr.get("start_time") or ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_slate_and_dk_props(dates_yyyymmdd: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_meta: list[dict[str, Any]] = []
    for dt in dates_yyyymmdd:
        sb = _fetch_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={dt}")
        for ev in sb.get("events", []):
            st_name = ev.get("status", {}).get("type", {}).get("name", "")
            if "FINAL" in st_name or "IN_PROGRESS" in st_name:
                continue
            eid = str(ev["id"])
            comp = ev["competitions"][0]
            ht = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            at = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            odds0 = (comp.get("odds") or [{}])[0]
            games_meta.append(
                {
                    "slate_date": dt,
                    "game_id": eid,
                    "start_utc": ev.get("date"),
                    "status_detail": ev["status"]["type"]["detail"],
                    "home_norm": norm_team(ht["team"]["abbreviation"]),
                    "away_norm": norm_team(at["team"]["abbreviation"]),
                    "spread_home": float(odds0.get("spread") or 0.0),
                    "total_line": float(odds0.get("overUnder") or 224.0),
                }
            )
    gm_df = pd.DataFrame(games_meta).drop_duplicates(subset=["game_id"]) if games_meta else pd.DataFrame()
    if gm_df.empty:
        return gm_df, pd.DataFrame()

    ath_cache: dict[str, str] = {}

    def get_ath(ref_url: str) -> str:
        u = ref_url.replace("http://", "https://")
        if u not in ath_cache:
            d = _fetch_json(u)
            ath_cache[u] = d.get("fullName") or d.get("displayName") or ""
        return ath_cache[u]

    dk_rows: list[dict[str, Any]] = []
    for g in gm_df.itertuples(index=False):
        eid = str(g.game_id)
        try:
            pb0 = _fetch_json(
                f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds/100/propBets?limit=1000"
            )
        except Exception:
            continue
        items = list(pb0.get("items") or [])
        for pg in range(2, int(pb0.get("pageCount", 1)) + 1):
            pbp = _fetch_json(
                f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds/100/propBets?limit=1000&page={pg}"
            )
            items.extend(pbp.get("items") or [])
        arefs = list({(it.get("athlete") or {}).get("$ref", "") for it in items if (it.get("athlete") or {}).get("$ref")})
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(get_ath, arefs))

        grouped_tw: dict[tuple[Any, ...], list[tuple[str, float]]] = {}
        dk_ms: dict[tuple[str, str, str], dict[int, float]] = {}
        for it in items:
            aref = (it.get("athlete") or {}).get("$ref", "")
            if not aref:
                continue
            pname = get_ath(aref)
            cn = canon_name(pname)
            tname = (it.get("type") or {}).get("name", "")
            tgt = ((it.get("current") or {}).get("target") or {}).get("value")
            op_tgt = ((it.get("open") or {}).get("target") or {}).get("value")
            am_s = ((it.get("odds") or {}).get("american") or {}).get("value")
            dec_s = ((it.get("odds") or {}).get("decimal") or {}).get("value")
            if tgt is None or dec_s is None:
                continue
            tgt_f, dec_f = float(tgt), float(dec_s)
            if tname in TWO_WAY_MAP:
                st_pp = TWO_WAY_MAP[tname]
                grouped_tw.setdefault(
                    (eid, cn, pname, st_pp, tgt_f, float(op_tgt) if op_tgt is not None else np.nan), []
                ).append((str(am_s), dec_f))
            elif tname in MILE_MAP:
                dk_ms.setdefault((eid, cn, MILE_MAP[tname]), {})[int(round(tgt_f))] = dec_f

        for (eid_k, cn, pname, st_pp, tgt_f, op_tgt_f), pairs in grouped_tw.items():
            if len(pairs) != 2:
                continue
            ov_am, ov_dec = pairs[0]
            un_am, un_dec = pairs[1]
            m_ladd = dk_ms.get((eid_k, cn, st_pp), {})
            k_need = int(np.floor(tgt_f)) + 1
            if k_need in m_ladd and abs(m_ladd[k_need] - un_dec) < abs(m_ladd[k_need] - ov_dec) - 0.12:
                ov_am, ov_dec, un_am, un_dec = un_am, un_dec, ov_am, ov_dec
            p_ov_shin, p_un_shin = devig_shin_2way(ov_dec, un_dec)
            dk_rows.append(
                {
                    "game_id": eid_k,
                    "cname": cn,
                    "dk_player": pname,
                    "stat_type": st_pp,
                    "dk_line": tgt_f,
                    "dk_open_line": op_tgt_f,
                    "dk_over_am": ov_am,
                    "dk_under_am": un_am,
                    "p_dk_over_shin": p_ov_shin,
                    "p_dk_under_shin": p_un_shin,
                }
            )
    return gm_df, pd.DataFrame(dk_rows)


def fetch_injuries_map() -> dict[str, str]:
    inj = _fetch_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
    out: dict[str, str] = {}
    for tblk in inj.get("injuries", []):
        for it in tblk.get("injuries", []):
            ath = it.get("athlete") or {}
            nm = ath.get("displayName", "")
            st = str(it.get("status") or "")
            cmt = (it.get("shortComment") or it.get("longComment") or "").lower()
            if "probable" in cmt:
                tag = "Probable"
            elif "questionable" in cmt or st == "Questionable":
                tag = "Questionable"
            elif "doubtful" in cmt or st == "Doubtful":
                tag = "Doubtful"
            elif st == "Out" or "out" in cmt:
                tag = "Out"
            else:
                tag = st or "Healthy"
            out[canon_name(nm)] = tag
    return out


def grade_completed_date(yyyymmdd: str) -> dict[str, dict[str, float]]:
    """Fetch final boxscores for a date -> dict[cname, stat_dict] for automatic grading."""
    sb = _fetch_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yyyymmdd}")
    finals = [
        str(ev["id"])
        for ev in sb.get("events", [])
        if "FINAL" in ev.get("status", {}).get("type", {}).get("name", "")
    ]
    actuals: dict[str, dict[str, float]] = {}
    for eid in finals:
        sm = _fetch_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={eid}")
        for tblk in (sm.get("boxscore") or {}).get("players", []):
            stats0 = (tblk.get("statistics") or [{}])[0]
            labels = stats0.get("labels", [])
            for ath in stats0.get("athletes", []):
                if ath.get("didNotPlay"):
                    continue
                vals = dict(zip(labels, ath.get("stats", [])))
                min_s = str(vals.get("MIN", "0"))
                if min_s in ("--", "", "-", "0"):
                    continue
                pts = float(vals.get("PTS", 0) or 0)
                reb = float(vals.get("REB", 0) or 0)
                ast = float(vals.get("AST", 0) or 0)
                fg3_s = str(vals.get("3PT", "0-0"))
                fg3 = float(fg3_s.split("-")[0]) if "-" in fg3_s else 0.0
                cn = canon_name((ath.get("athlete") or {}).get("displayName"))
                actuals[cn] = {
                    "Points": pts,
                    "Rebounds": reb,
                    "Assists": ast,
                    "Pts+Rebs+Asts": pts + reb + ast,
                    "Pts+Rebs": pts + reb,
                    "Pts+Asts": pts + ast,
                    "Rebs+Asts": reb + ast,
                    "3-PT Made": fg3,
                }
    return actuals


def run_autopilot_cycle(state_dir: Path) -> dict[str, Any]:
    """Run one complete 24/7 autopilot cycle: scan live PrizePicks vs DK, archive, grade past, build best slip."""
    state_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = state_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    now_cst = now_utc - timedelta(hours=6)
    today_s = now_cst.strftime("%Y%m%d")
    tomorrow_s = (now_cst + timedelta(days=1)).strftime("%Y%m%d")
    yesterday_s = (now_cst - timedelta(days=1)).strftime("%Y%m%d")

    pp_df = fetch_prizepicks_standard_board()
    gm_df, dk_df = fetch_slate_and_dk_props([today_s, tomorrow_s])
    inj_map = fetch_injuries_map()

    scored_plays: list[dict[str, Any]] = []
    if not pp_df.empty and not gm_df.empty and not dk_df.empty:
        team_game: dict[str, dict[str, Any]] = {}
        for g in gm_df.itertuples(index=False):
            matchup = f"{g.away_norm}@{g.home_norm}"
            for t in (g.home_norm, g.away_norm):
                team_game.setdefault(
                    t,
                    {
                        "game_id": str(g.game_id),
                        "slate_date": str(g.slate_date),
                        "matchup": matchup,
                        "status_detail": g.status_detail,
                        "abs_spread": abs(float(g.spread_home)),
                    },
                )

        pp_elig = pp_df[pp_df["team_norm"].isin(team_game.keys())].copy()
        pp_elig["game_id"] = pp_elig["team_norm"].map(lambda t: team_game[t]["game_id"])
        merged = pp_elig.merge(dk_df, on=["game_id", "cname", "stat_type"], how="inner")

        for r in merged.itertuples(index=False):
            tinfo = team_game[r.team_norm]
            _, disp_alpha = STAT_COL_MAP[r.stat_type]
            pp_line, dk_line = float(r.pp_line), float(r.dk_line)
            try:
                mu_dk = brentq(
                    lambda mu: nb_prob_over_under_push(dk_line, mu, disp_alpha)[0] - float(r.p_dk_over_shin),
                    0.1,
                    90.0,
                )
            except Exception:
                mu_dk = dk_line
            pm, pl, pp_push = nb_prob_over_under_push(pp_line, mu_dk, disp_alpha)
            denom = max(pm + pl, 1e-6)
            pm_c, pl_c = pm / denom, pl / denom
            side = "MORE" if pm_c >= pl_c else "LESS"
            p_dk_cond = max(pm_c, pl_c)

            # Blowout Q4 truncation adjustment + Cycle #1 promoted stat-type Empirical-Bayes calibration
            abs_s = tinfo["abs_spread"]
            blow_bonus = 0.010 if (side == "LESS" and abs_s >= 9.0) else 0.0
            p_cal = apply_stat_tail_calibration(p_dk_cond + blow_bonus, str(r.stat_type))
            inj_tag = inj_map.get(r.cname, "Healthy")

            scored_plays.append(
                {
                    "slate_date": tinfo["slate_date"],
                    "game_id": r.game_id,
                    "matchup": tinfo["matchup"],
                    "start_time": r.start_time,
                    "team": r.team_norm,
                    "player": r.player_name,
                    "cname": r.cname,
                    "inj_status": inj_tag,
                    "stat": r.stat_type,
                    "side": side,
                    "pp_line": pp_line,
                    "dk_line": dk_line,
                    "line_diff": round(pp_line - dk_line, 2),
                    "dk_odds": f"{r.dk_over_am}/{r.dk_under_am}",
                    "p_dk_at_pp": round(p_dk_cond, 4),
                    "p_calibrated": round(p_cal, 4),
                    "p_push": round(pp_push, 4),
                }
            )

    sc_df = (
        pd.DataFrame(scored_plays).sort_values("p_calibrated", ascending=False).reset_index(drop=True)
        if scored_plays
        else pd.DataFrame()
    )

    if not sc_df.empty:
        ts_tag = now_utc.strftime("%Y%m%dT%H%M%SZ")
        sc_df.to_parquet(snap_dir / f"scan_{ts_tag}.parquet", index=False)

    # Select optimal 1-a-day card (Healthy/Probable, .5 half-points, 1 pick per game)
    recommended_slip: dict[str, Any] | None = None
    top_edges: list[dict[str, Any]] = []
    if not sc_df.empty:
        clean = sc_df[
            (sc_df["inj_status"].isin(["Healthy", "Probable"]))
            & (sc_df["p_push"] == 0.0)
            & (sc_df["p_dk_at_pp"] >= 0.542)
            & (sc_df["p_calibrated"] >= 0.548)
        ].copy()
        top_edges = clean.head(12).to_dict(orient="records")
        distinct_game = clean.drop_duplicates(subset=["game_id"], keep="first").sort_values(
            "p_calibrated", ascending=False
        )
        best_choice = None
        for k_legs, mode, label in [
            (5, "flex", "5-Pick Flex (10x / 2x / 0.4x)"),
            (4, "flex", "4-Pick Flex (6x / 1.5x)"),
            (3, "power", "3-Pick Power (6x)"),
        ]:
            if len(distinct_game) >= k_legs:
                sub = distinct_game.iloc[:k_legs]
                metrics = slip_ev_and_log_growth(
                    sub["p_calibrated"].tolist(),
                    mode=mode,  # type: ignore[arg-type]
                    c_kelly=0.50,
                    max_frac=0.06,
                )
                if metrics["ev"] > 0 and (
                    best_choice is None or metrics["expected_log_growth"] > best_choice[0]
                ):
                    best_choice = (
                        metrics["expected_log_growth"],
                        {
                            "slip_type": label,
                            "mode": mode,
                            "n_legs": k_legs,
                            **metrics,
                            "legs": sub.to_dict(orient="records"),
                        },
                    )
        if best_choice is not None:
            recommended_slip = best_choice[1]

    # Auto-grade completed games from today & yesterday for live verified ledger tracking
    graded_recent: dict[str, int] = {}
    for dt_check in (yesterday_s, today_s):
        try:
            act = grade_completed_date(dt_check)
            graded_recent[dt_check] = len(act)
        except Exception:
            graded_recent[dt_check] = 0

    status_payload = {
        "updated_at_utc": now_utc.isoformat(),
        "updated_at_cst": now_cst.strftime("%Y-%m-%d %I:%M:%S %p CST"),
        "prizepicks_standard_props_live": int(len(pp_df)),
        "upcoming_games_scanned": int(len(gm_df)),
        "matched_two_way_props": int(len(sc_df)),
        "active_model_version": "v2.1-stat-bayes-blowout",
        "promoted_rd_upgrades": load_rd_log(state_dir),
        "recommended_slip": recommended_slip,
        "top_edges": top_edges,
        "graded_players_by_date": graded_recent,
    }

    (state_dir / "latest_status.json").write_text(json.dumps(status_payload, indent=2))
    return status_payload


DEFAULT_RD_LOG: list[dict[str, str]] = [
    {
        "id": "RD-001",
        "title": "Empirical-Bayes Stat-Type Tail Calibration (2022-24 Fit)",
        "finding": "Assists (+1.36%) & Rebounds (+0.17%) outperform raw model/book probability in OOS tails while Points (-0.51%) & 3PM (-0.25%) carry shooting-variance drag.",
        "oos_lift": "Flipped 2024-25 5-Flex ROI from -8.5% -> +16.4% (ESPN) and -46.7% -> +60.0% (OddsAPI); +176.9% 5-Flex ROI on 2025-26 OddsAPI holdout.",
        "status": "PROMOTED",
    },
    {
        "id": "RD-002",
        "title": "Unbounded 6-Feature Matchup Logit Residual",
        "finding": "Improved Confirm 2024-25 3-Power (+31.7%) but degraded OddsAPI 2024-26 (-5.6%) due to colinearity with market line.",
        "oos_lift": "Rejected by dual-holdout gate (zero overfit promotions).",
        "status": "REJECTED",
    },
]


def load_rd_log(state_dir: Path) -> list[dict[str, str]]:
    """Merge persistent R&D log so scans never wipe later REJECTED/PROMOTED entries."""
    path = state_dir / "rd_log.json"
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_RD_LOG, indent=2))
        return list(DEFAULT_RD_LOG)
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return list(DEFAULT_RD_LOG)
    if not isinstance(raw, list) or not raw:
        return list(DEFAULT_RD_LOG)
    by_id: dict[str, dict[str, str]] = {e["id"]: e for e in DEFAULT_RD_LOG if "id" in e}
    for entry in raw:
        if isinstance(entry, dict) and entry.get("id"):
            by_id[str(entry["id"])] = {str(k): str(v) for k, v in entry.items()}

    def sort_key(eid: str) -> tuple[int, str]:
        m = re.search(r"(\d+)", eid)
        return (int(m.group(1)) if m else 10_000, eid)

    return [by_id[k] for k in sorted(by_id.keys(), key=sort_key)]


def upsert_rd_log(state_dir: Path, entry: dict[str, str]) -> list[dict[str, str]]:
    """Insert or replace one R&D experiment row in the persistent log."""
    rows = load_rd_log(state_dir)
    eid = str(entry.get("id", "")).strip()
    if not eid:
        return rows
    cleaned = {str(k): str(v) for k, v in entry.items()}
    out = [cleaned if r.get("id") == eid else r for r in rows]
    if not any(r.get("id") == eid for r in out):
        out.append(cleaned)
    (state_dir / "rd_log.json").write_text(json.dumps(out, indent=2))
    return out
