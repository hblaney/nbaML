"""Strictly-pregame expanding features for 1st Quarter (1Q) and 1st Half (1H) NBA derivatives.

All team, rotation, pace, availability, travel, and schedule features for date ``d`` are
computed exclusively from completed games with ``game_date < d`` (zero future or same-day
leakage). Full-game pregame closing spread and total act as baseline anchors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nba_betting.data.espn_periods import normalize_espn_tricode
from nba_betting.features.player_impact import (
    PlayerState,
    learn_play_rate,
    prepare_player_box,
    update_player_states_after_game,
)

# (IANA timezone, elevation_feet, latitude, longitude)
NBA_ARENA_META: dict[str, tuple[str, float, float, float]] = {
    "ATL": ("America/New_York", 1050.0, 33.7573, -84.3963),
    "BOS": ("America/New_York", 20.0, 42.3662, -71.0621),
    "BKN": ("America/New_York", 30.0, 40.6826, -73.9754),
    "CHA": ("America/New_York", 751.0, 35.2251, -80.8392),
    "CHI": ("America/Chicago", 594.0, 41.8807, -87.6742),
    "CLE": ("America/New_York", 653.0, 41.4965, -81.6882),
    "DAL": ("America/Chicago", 430.0, 32.7905, -96.8103),
    "DEN": ("America/Denver", 5280.0, 39.7487, -105.0077),
    "DET": ("America/New_York", 600.0, 42.3410, -83.0551),
    "GSW": ("America/Los_Angeles", 15.0, 37.7680, -122.3877),
    "HOU": ("America/Chicago", 50.0, 29.7508, -95.3621),
    "IND": ("America/New_York", 715.0, 39.7640, -86.1555),
    "LAC": ("America/Los_Angeles", 285.0, 33.9450, -118.3426),
    "LAL": ("America/Los_Angeles", 285.0, 34.0430, -118.2673),
    "MEM": ("America/Chicago", 337.0, 35.1382, -90.0505),
    "MIA": ("America/New_York", 10.0, 25.7814, -80.1870),
    "MIL": ("America/Chicago", 617.0, 43.0451, -87.9172),
    "MIN": ("America/Chicago", 830.0, 44.9795, -93.2760),
    "NOP": ("America/Chicago", 10.0, 29.9490, -90.0821),
    "NYK": ("America/New_York", 35.0, 40.7505, -73.9934),
    "OKC": ("America/Chicago", 1201.0, 35.4634, -97.5151),
    "ORL": ("America/New_York", 82.0, 28.5392, -81.3839),
    "PHI": ("America/New_York", 39.0, 39.9012, -75.1720),
    "PHX": ("America/Phoenix", 1086.0, 33.4457, -112.0712),
    "POR": ("America/Los_Angeles", 50.0, 45.5316, -122.6668),
    "SAC": ("America/Los_Angeles", 30.0, 38.5802, -121.4997),
    "SAS": ("America/Chicago", 650.0, 29.4270, -98.4375),
    "TOR": ("America/Toronto", 249.0, 43.6435, -79.3791),
    "UTA": ("America/Denver", 4226.0, 40.7683, -111.9011),
    "WAS": ("America/New_York", 25.0, 38.8981, -77.0209),
}

NEUTRAL_CITY_META: dict[str, tuple[str, float, float, float]] = {
    "mexico city": ("America/Mexico_City", 7350.0, 19.4326, -99.1332),
    "paris": ("Europe/Paris", 115.0, 48.8386, 2.3785),
    "london": ("Europe/London", 36.0, 51.5030, 0.0032),
    "abu dhabi": ("Asia/Dubai", 20.0, 24.4539, 54.3773),
    "las vegas": ("America/Los_Angeles", 2001.0, 36.1028, -115.1783),
    "seattle": ("America/Los_Angeles", 175.0, 47.6221, -122.3540),
    "montreal": ("America/Toronto", 118.0, 45.4961, -73.5693),
}

# Canonical mechanical benchmark multipliers stated in mission specification
MECH_Q1_PROP = 0.25
MECH_H1_PROP = 0.505


def snap_to_half_point(val: float, *, zero_to: float = -0.5) -> float:
    """Snap a continuous spread/total to the nearest sportsbook half-point (x.5)."""
    if not np.isfinite(val):
        return float("nan")
    if abs(val) < 1e-12:
        return float(zero_to)
    low = math.floor(val - 0.5) + 0.5
    high = low + 1.0
    if abs(val - low) < abs(high - val) - 1e-12:
        return float(low)
    if abs(high - val) < abs(val - low) - 1e-12:
        return float(high)
    return float(low if val < 0 else high)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return float(2.0 * r * math.asin(min(1.0, math.sqrt(a))))


def _venue_meta(home_tricode: str, venue_city: str, is_neutral: bool) -> tuple[str, float, float, float]:
    city_key = str(venue_city or "").strip().lower()
    if is_neutral and city_key in NEUTRAL_CITY_META:
        return NEUTRAL_CITY_META[city_key]
    if city_key == "mexico city":
        return NEUTRAL_CITY_META["mexico city"]
    return NBA_ARENA_META.get(home_tricode, ("America/New_York", 200.0, 39.8, -98.5))


def _parse_tipoff_utc(tipoff_utc: str, fallback_date: str) -> datetime:
    if tipoff_utc:
        try:
            s = str(tipoff_utc).strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except ValueError:
            pass
    # Default 7:30 PM ET = 00:30 UTC next day
    dt = pd.Timestamp(fallback_date).to_pydatetime()
    return datetime(dt.year, dt.month, dt.day, 23, 30, tzinfo=timezone.utc)


def _local_decimal_hour(utc_dt: datetime, tz_name: str) -> tuple[float, float]:
    tz = ZoneInfo(tz_name)
    loc = utc_dt.astimezone(tz)
    dec_hour = loc.hour + loc.minute / 60.0
    # Treat 00:00-03:59 local (never happens except timezone wrap) as 24+
    if dec_hour < 5.0:
        dec_hour += 24.0
    utcoff = (loc.utcoffset().total_seconds() / 3600.0) if loc.utcoffset() else -5.0
    return float(dec_hour), float(utcoff)


def _ema_step(old: float, obs: float, alpha: float) -> float:
    if not np.isfinite(obs):
        return old
    if not np.isfinite(old):
        return float(obs)
    return float((1.0 - alpha) * old + alpha * obs)


@dataclass
class TeamDerivativeState:
    games_season: int = 0
    games_total: int = 0
    ortg_q1: float = np.nan
    drtg_q1: float = np.nan
    ortg_h1: float = np.nan
    drtg_h1: float = np.nan
    ortg_fg: float = np.nan
    drtg_fg: float = np.nan
    pace_q1: float = np.nan
    pace_h1: float = np.nan
    pace_fg: float = np.nan
    starter_min_share: float = np.nan
    q1_starter_min_share: float = np.nan
    h1_starter_min_share: float = np.nan
    q1_first_sub_sec: float = np.nan
    starter_net48: float = np.nan
    bench_net48: float = np.nan
    q1_first6_margin: float = np.nan
    q1_last6_margin: float = np.nan
    q2_first6_margin: float = np.nan
    q1_margin_resid: float = np.nan
    h1_margin_resid: float = np.nan
    q1_total_share: float = np.nan
    h1_total_share: float = np.nan
    # Fast (8-game half-life) form trackers
    fast_net_q1: float = np.nan
    fast_net_h1: float = np.nan
    fast_net_fg: float = np.nan
    # Schedule & travel history (strictly < asof_date)
    prior_dates: list[pd.Timestamp] = field(default_factory=list)
    last_was_away: bool = False
    last_lat: float = np.nan
    last_lon: float = np.nan

    def apply_offseason_shrinkage(
        self,
        league: dict[str, float],
        *,
        retention: float = 0.35,
    ) -> None:
        """Partially regress team state toward league baseline across offseasons."""
        self.games_season = 0
        r = float(retention)
        for attr, lg_key in (
            ("ortg_q1", "ortg_q1"),
            ("drtg_q1", "ortg_q1"),
            ("ortg_h1", "ortg_h1"),
            ("drtg_h1", "ortg_h1"),
            ("ortg_fg", "ortg_fg"),
            ("drtg_fg", "ortg_fg"),
            ("pace_q1", "pace_q1"),
            ("pace_h1", "pace_h1"),
            ("pace_fg", "pace_fg"),
            ("starter_min_share", "starter_min_share"),
            ("q1_starter_min_share", "q1_starter_min_share"),
            ("h1_starter_min_share", "h1_starter_min_share"),
            ("q1_first_sub_sec", "q1_first_sub_sec"),
            ("starter_net48", "zero"),
            ("bench_net48", "zero"),
            ("q1_first6_margin", "zero"),
            ("q1_last6_margin", "zero"),
            ("q2_first6_margin", "zero"),
            ("q1_margin_resid", "zero"),
            ("h1_margin_resid", "zero"),
            ("q1_total_share", "q1_total_share"),
            ("h1_total_share", "h1_total_share"),
            ("fast_net_q1", "zero"),
            ("fast_net_h1", "zero"),
            ("fast_net_fg", "zero"),
        ):
            cur = getattr(self, attr)
            base = league.get(lg_key, 0.0)
            if np.isfinite(cur) and np.isfinite(base):
                setattr(self, attr, float(r * cur + (1.0 - r) * base))
        self.prior_dates.clear()
        self.last_was_away = False
        self.last_lat = np.nan
        self.last_lon = np.nan


def _shrink(val: float, base: float, eff_games: float, k: float = 8.0) -> float:
    if not np.isfinite(val):
        return float(base)
    w = float(eff_games) / (float(eff_games) + float(k))
    return float(w * val + (1.0 - w) * base)


def _player_rotation_features_asof(
    player_states: dict[str, PlayerState],
    team_id: int,
    play_rates: dict[int, float],
) -> dict[str, float]:
    """Compute pregame active starter vs bench impact & missing starter impact from prior boxes."""
    members = [
        (pid, st)
        for pid, st in player_states.items()
        if st.last_team_id == int(team_id) and st.games > 0 and np.isfinite(st.ema_minutes)
    ]
    if not members:
        return {
            "rot_top5_pm": 0.0,
            "rot_bench_pm": 0.0,
            "rot_top5_minus_bench_pm": 0.0,
            "rot_missing_starter_pm": 0.0,
            "rot_missing_starter_min": 0.0,
        }
    members.sort(key=lambda x: -float(x[1].ema_minutes))
    top5 = members[:5]
    bench = members[5:10]

    top5_pm = 0.0
    miss_starter_pm = 0.0
    miss_starter_min = 0.0
    for _, st in top5:
        streak = int(st.miss_streak)
        p_play = float(np.clip(play_rates.get(streak, 0.88 if streak == 0 else 0.30), 0.0, 1.0))
        mins = float(st.ema_minutes)
        pm_rate = float(st.ema_pm_per_min) if np.isfinite(st.ema_pm_per_min) else 0.0
        top5_pm += p_play * mins * pm_rate
        if streak >= 1:
            miss_starter_min += (1.0 - p_play) * mins
            miss_starter_pm += (1.0 - p_play) * mins * pm_rate

    bench_pm = 0.0
    for _, st in bench:
        streak = int(st.miss_streak)
        p_play = float(np.clip(play_rates.get(streak, 0.85 if streak == 0 else 0.30), 0.0, 1.0))
        mins = float(st.ema_minutes)
        pm_rate = float(st.ema_pm_per_min) if np.isfinite(st.ema_pm_per_min) else 0.0
        bench_pm += p_play * mins * pm_rate

    return {
        "rot_top5_pm": float(top5_pm),
        "rot_bench_pm": float(bench_pm),
        "rot_top5_minus_bench_pm": float(top5_pm - bench_pm),
        "rot_missing_starter_pm": float(miss_starter_pm),
        "rot_missing_starter_min": float(miss_starter_min),
    }


def build_pregame_derivative_features(
    games_df: pd.DataFrame,
    *,
    player_boxes_df: pd.DataFrame | None = None,
    ema_half_life: float = 18.0,
    fast_half_life: float = 8.0,
    prior_k: float = 8.0,
    offseason_retention: float = 0.35,
) -> pd.DataFrame:
    """Build strictly-pregame expanding features day-by-day across ``games_df``.

    Guarantees that every row on ``game_date == d`` uses ONLY completed games
    from ``game_date < d``.
    """
    df = games_df.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["game_date_dt"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df.sort_values(["game_date_dt", "tipoff_utc", "game_id"]).reset_index(drop=True)

    alpha = 1.0 - math.exp(math.log(0.5) / float(ema_half_life))
    alpha_fast = 1.0 - math.exp(math.log(0.5) / float(fast_half_life))
    alpha_lg = 1.0 - math.exp(math.log(0.5) / 150.0)

    # Expanding league baselines (initialized to neutral priors, updated strictly < d)
    league: dict[str, float] = {
        "ortg_q1": 114.0,
        "ortg_h1": 114.0,
        "ortg_fg": 114.0,
        "pace_q1": 100.5,
        "pace_h1": 100.0,
        "pace_fg": 99.5,
        "starter_min_share": 0.65,
        "q1_starter_min_share": 0.76,
        "h1_starter_min_share": 0.68,
        "q1_first_sub_sec": 350.0,
        "q1_total_share": 0.2515,
        "h1_total_share": 0.5026,
        "zero": 0.0,
    }

    team_states: dict[str, TeamDerivativeState] = {}
    player_states: dict[str, PlayerState] = {}
    streak_history: list[tuple[int, int]] = []
    pbox_by_gid: dict[str, pd.DataFrame] = {}
    if player_boxes_df is not None and not player_boxes_df.empty:
        pbox_prep = prepare_player_box(player_boxes_df)
        pbox_by_gid = {str(gid): grp for gid, grp in pbox_prep.groupby("game_id", sort=False)}

    current_season: str | None = None
    feature_rows: list[dict[str, Any]] = []

    for gdate, day_df in df.groupby("game_date_dt", sort=True):
        day_season = str(day_df.iloc[0]["season"])
        if current_season is not None and day_season != current_season:
            for st in team_states.values():
                st.apply_offseason_shrinkage(league, retention=offseason_retention)
            # Decay player miss streaks across offseason
            for pst in player_states.values():
                pst.miss_streak = 0
        current_season = day_season

        play_rates = learn_play_rate(streak_history[-25000:]) if streak_history else {0: 0.90, 1: 0.45, 2: 0.30}

        # STEP 1: Score every game on gdate using ONLY prior state (game_date < gdate)
        for _, row in day_df.iterrows():
            gid = str(row["game_id"])
            h_tri = normalize_espn_tricode(row["home_tricode"])
            a_tri = normalize_espn_tricode(row["away_tricode"])
            h_tid = int(row.get("espn_home_team_id", row["home_team_id"]))
            a_tid = int(row.get("espn_away_team_id", row["away_team_id"]))
            is_neut = bool(row.get("is_neutral", False) or row.get("summary_is_neutral", False))

            hst = team_states.get(h_tri) or TeamDerivativeState()
            ast = team_states.get(a_tri) or TeamDerivativeState()

            # Effective sample size (includes discounted prior-season information)
            h_eff = float(hst.games_season) + offseason_retention * max(0, hst.games_total - hst.games_season)
            a_eff = float(ast.games_season) + offseason_retention * max(0, ast.games_total - ast.games_season)

            def _team_feats(st: TeamDerivativeState, eff: float) -> dict[str, float]:
                ortg_q1 = _shrink(st.ortg_q1, league["ortg_q1"], eff, prior_k)
                drtg_q1 = _shrink(st.drtg_q1, league["ortg_q1"], eff, prior_k)
                ortg_h1 = _shrink(st.ortg_h1, league["ortg_h1"], eff, prior_k)
                drtg_h1 = _shrink(st.drtg_h1, league["ortg_h1"], eff, prior_k)
                ortg_fg = _shrink(st.ortg_fg, league["ortg_fg"], eff, prior_k)
                drtg_fg = _shrink(st.drtg_fg, league["ortg_fg"], eff, prior_k)

                net_q1 = ortg_q1 - drtg_q1
                net_h1 = ortg_h1 - drtg_h1
                net_fg = ortg_fg - drtg_fg

                delta_ortg_q1 = (ortg_q1 - league["ortg_q1"]) - (ortg_fg - league["ortg_fg"])
                delta_drtg_q1 = (drtg_q1 - league["ortg_q1"]) - (drtg_fg - league["ortg_fg"])
                delta_net_q1 = net_q1 - net_fg

                delta_ortg_h1 = (ortg_h1 - league["ortg_h1"]) - (ortg_fg - league["ortg_fg"])
                delta_drtg_h1 = (drtg_h1 - league["ortg_h1"]) - (drtg_fg - league["ortg_fg"])
                delta_net_h1 = net_h1 - net_fg

                pace_q1 = _shrink(st.pace_q1, league["pace_q1"], eff, prior_k)
                pace_h1 = _shrink(st.pace_h1, league["pace_h1"], eff, prior_k)
                pace_fg = _shrink(st.pace_fg, league["pace_fg"], eff, prior_k)
                lg_q1_ratio = league["pace_q1"] / max(league["pace_fg"], 1.0)
                lg_h1_ratio = league["pace_h1"] / max(league["pace_fg"], 1.0)
                q1_pace_ratio = (pace_q1 / max(pace_fg, 1.0)) - lg_q1_ratio
                h1_pace_ratio = (pace_h1 / max(pace_fg, 1.0)) - lg_h1_ratio

                s_min_sh = _shrink(st.starter_min_share, league["starter_min_share"], eff, prior_k)
                q1_s_min_sh = _shrink(st.q1_starter_min_share, league["q1_starter_min_share"], eff, prior_k)
                h1_s_min_sh = _shrink(st.h1_starter_min_share, league["h1_starter_min_share"], eff, prior_k)
                first_sub = _shrink(st.q1_first_sub_sec, league["q1_first_sub_sec"], eff, prior_k)
                s_net48 = _shrink(st.starter_net48, 0.0, eff, prior_k)
                b_net48 = _shrink(st.bench_net48, 0.0, eff, prior_k)
                sb_div = s_net48 - b_net48

                q1_f6 = _shrink(st.q1_first6_margin, 0.0, eff, prior_k)
                q1_l6 = _shrink(st.q1_last6_margin, 0.0, eff, prior_k)
                q2_f6 = _shrink(st.q2_first6_margin, 0.0, eff, prior_k)
                q1_m_res = _shrink(st.q1_margin_resid, 0.0, eff, prior_k * 1.5)
                h1_m_res = _shrink(st.h1_margin_resid, 0.0, eff, prior_k * 1.5)
                q1_t_sh = _shrink(st.q1_total_share, league["q1_total_share"], eff, prior_k) - league["q1_total_share"]
                h1_t_sh = _shrink(st.h1_total_share, league["h1_total_share"], eff, prior_k) - league["h1_total_share"]

                fast_d_q1 = _shrink(st.fast_net_q1 - st.fast_net_fg, 0.0, eff, prior_k)
                fast_d_h1 = _shrink(st.fast_net_h1 - st.fast_net_fg, 0.0, eff, prior_k)

                return {
                    "ortg_q1": ortg_q1,
                    "drtg_q1": drtg_q1,
                    "net_q1": net_q1,
                    "ortg_h1": ortg_h1,
                    "drtg_h1": drtg_h1,
                    "net_h1": net_h1,
                    "ortg_fg": ortg_fg,
                    "drtg_fg": drtg_fg,
                    "net_fg": net_fg,
                    "delta_ortg_q1": delta_ortg_q1,
                    "delta_drtg_q1": delta_drtg_q1,
                    "delta_net_q1": delta_net_q1,
                    "delta_ortg_h1": delta_ortg_h1,
                    "delta_drtg_h1": delta_drtg_h1,
                    "delta_net_h1": delta_net_h1,
                    "pace_q1": pace_q1,
                    "pace_h1": pace_h1,
                    "pace_fg": pace_fg,
                    "q1_pace_ratio": q1_pace_ratio,
                    "h1_pace_ratio": h1_pace_ratio,
                    "starter_min_share": s_min_sh,
                    "q1_starter_min_share": q1_s_min_sh,
                    "h1_starter_min_share": h1_s_min_sh,
                    "q1_first_sub_sec": first_sub,
                    "starter_net48": s_net48,
                    "bench_net48": b_net48,
                    "starter_bench_div": sb_div,
                    "q1_first6_margin": q1_f6,
                    "q1_last6_margin": q1_l6,
                    "q2_first6_margin": q2_f6,
                    "q1_margin_resid": q1_m_res,
                    "h1_margin_resid": h1_m_res,
                    "q1_total_share_excess": q1_t_sh,
                    "h1_total_share_excess": h1_t_sh,
                    "fast_delta_q1": fast_d_q1,
                    "fast_delta_h1": fast_d_h1,
                }

            hf = _team_feats(hst, h_eff)
            af = _team_feats(ast, a_eff)

            # Schedule / rest / B2B / travel / timezone / altitude
            def _sched_feats(
                st: TeamDerivativeState,
                is_away_tonight: bool,
                v_lat: float,
                v_lon: float,
            ) -> dict[str, float]:
                if not st.prior_dates:
                    return {
                        "rest_days": 4.0,
                        "is_b2b": 0.0,
                        "road_b2b": 0.0,
                        "road_road_b2b": 0.0,
                        "three_in_four": 0.0,
                        "games_last7": 0.0,
                        "travel_km": 0.0,
                        "log_travel_km": 0.0,
                    }
                last_dt = st.prior_dates[-1]
                rest_d = float(max(1, (gdate - last_dt).days))
                b2b = 1.0 if rest_d == 1.0 else 0.0
                r_b2b = 1.0 if (b2b == 1.0 and is_away_tonight) else 0.0
                rr_b2b = 1.0 if (r_b2b == 1.0 and st.last_was_away) else 0.0
                in4 = sum(1 for d0 in st.prior_dates if (gdate - d0).days <= 3)
                in7 = sum(1 for d0 in st.prior_dates if (gdate - d0).days <= 7)
                t_km = (
                    _haversine_km(st.last_lat, st.last_lon, v_lat, v_lon)
                    if np.isfinite(st.last_lat) and np.isfinite(st.last_lon)
                    else 0.0
                )
                return {
                    "rest_days": float(min(rest_d, 5.0)),
                    "is_b2b": b2b,
                    "road_b2b": r_b2b,
                    "road_road_b2b": rr_b2b,
                    "three_in_four": 1.0 if in4 >= 2 else 0.0,
                    "games_last7": float(in7),
                    "travel_km": t_km,
                    "log_travel_km": math.log1p(t_km),
                }

            v_tz, v_elev, v_lat, v_lon = _venue_meta(h_tri, str(row.get("venue_city", "")), is_neut)
            h_home_tz, h_home_elev, _, _ = NBA_ARENA_META.get(h_tri, (v_tz, v_elev, v_lat, v_lon))
            a_home_tz, a_home_elev, _, _ = NBA_ARENA_META.get(a_tri, (v_tz, v_elev, v_lat, v_lon))

            hs_sched = _sched_feats(hst, False, v_lat, v_lon)
            as_sched = _sched_feats(ast, True, v_lat, v_lon)

            utc_dt = _parse_tipoff_utc(str(row.get("tipoff_utc", "")), str(row["game_date"]))
            venue_hour, venue_off = _local_decimal_hour(utc_dt, v_tz)
            home_body_hour, home_off = _local_decimal_hour(utc_dt, h_home_tz)
            away_body_hour, away_off = _local_decimal_hour(utc_dt, a_home_tz)

            is_early_matinee = 1.0 if venue_hour < 16.0 else 0.0
            away_morning_body = float(max(0.0, 17.0 - away_body_hour))
            away_late_body = float(max(0.0, away_body_hour - 22.0))
            tz_diff_away = float(venue_off - away_off)  # +3 means West team playing in East

            alt_adv_kft = (
                float(max(0.0, v_elev - a_home_elev) / 1000.0)
                if (not is_neut and v_elev >= 3500.0)
                else 0.0
            )
            venue_high_alt = 1.0 if v_elev >= 3500.0 else 0.0

            h_rot = _player_rotation_features_asof(player_states, h_tid, play_rates)
            a_rot = _player_rotation_features_asof(player_states, a_tid, play_rates)

            # Closing full-game market baseline anchors
            fg_sp_home = float(row["spread_home"]) if pd.notna(row.get("spread_home")) else np.nan
            fg_tot = float(row["total_line"]) if pd.notna(row.get("total_line")) else np.nan
            fg_exp_margin = -fg_sp_home if np.isfinite(fg_sp_home) else 0.0
            open_sp = float(row["open_spread_home"]) if pd.notna(row.get("open_spread_home")) else fg_sp_home
            open_tot = float(row["open_total_line"]) if pd.notna(row.get("open_total_line")) else fg_tot
            fg_sp_move = (fg_sp_home - open_sp) if (np.isfinite(fg_sp_home) and np.isfinite(open_sp)) else 0.0
            fg_tot_move = (fg_tot - open_tot) if (np.isfinite(fg_tot) and np.isfinite(open_tot)) else 0.0

            # Mechanical baseline targets (25% 1Q, 50.5% 1H)
            mech_q1_exp_margin = MECH_Q1_PROP * fg_exp_margin
            mech_h1_exp_margin = MECH_H1_PROP * fg_exp_margin
            mech_q1_exp_total = MECH_Q1_PROP * fg_tot if np.isfinite(fg_tot) else np.nan
            mech_h1_exp_total = MECH_H1_PROP * fg_tot if np.isfinite(fg_tot) else np.nan

            q1_m = float(row["q1_margin"])
            h1_m = float(row["h1_margin"])
            q1_t = float(row["q1_total"])
            h1_t = float(row["h1_total"])

            feature_rows.append(
                {
                    "season": str(row["season"]),
                    "season_type": str(row["season_type"]),
                    "game_id": gid,
                    "game_date": str(row["game_date"]),
                    "tipoff_utc": str(row.get("tipoff_utc", "")),
                    "home_tricode": h_tri,
                    "away_tricode": a_tri,
                    "is_playoffs": 1.0 if str(row["season_type"]) == "Playoffs" else 0.0,
                    "is_neutral": 1.0 if is_neut else 0.0,
                    "home_season_games": int(hst.games_season),
                    "away_season_games": int(ast.games_season),
                    # Full-game closing market anchors
                    "fg_spread_home": fg_sp_home,
                    "fg_exp_margin": fg_exp_margin,
                    "abs_fg_spread": abs(fg_sp_home) if np.isfinite(fg_sp_home) else 0.0,
                    "fg_total_line": fg_tot,
                    "fg_p_home_novig": float(row.get("p_home_novig", 0.5)),
                    "fg_spread_move": fg_sp_move,
                    "fg_total_move": fg_tot_move,
                    # 1Q & 1H efficiency differentials vs full-game rating (delta_1Q, delta_1H)
                    "diff_delta_net_q1": hf["delta_net_q1"] - af["delta_net_q1"],
                    "diff_delta_net_h1": hf["delta_net_h1"] - af["delta_net_h1"],
                    "diff_fast_delta_q1": hf["fast_delta_q1"] - af["fast_delta_q1"],
                    "diff_fast_delta_h1": hf["fast_delta_h1"] - af["fast_delta_h1"],
                    "diff_net_q1": hf["net_q1"] - af["net_q1"],
                    "diff_net_h1": hf["net_h1"] - af["net_h1"],
                    "diff_net_fg": hf["net_fg"] - af["net_fg"],
                    "sum_delta_ortg_q1": hf["delta_ortg_q1"] + af["delta_ortg_q1"],
                    "sum_delta_drtg_q1": hf["delta_drtg_q1"] + af["delta_drtg_q1"],
                    "sum_delta_ortg_h1": hf["delta_ortg_h1"] + af["delta_ortg_h1"],
                    "sum_delta_drtg_h1": hf["delta_drtg_h1"] + af["delta_drtg_h1"],
                    "diff_q1_margin_resid": hf["q1_margin_resid"] - af["q1_margin_resid"],
                    "diff_h1_margin_resid": hf["h1_margin_resid"] - af["h1_margin_resid"],
                    "sum_q1_total_share_excess": hf["q1_total_share_excess"] + af["q1_total_share_excess"],
                    "sum_h1_total_share_excess": hf["h1_total_share_excess"] + af["h1_total_share_excess"],
                    # Starter minutes share, stagger timing, & starter-vs-bench net rating divergence
                    "diff_starter_min_share": hf["starter_min_share"] - af["starter_min_share"],
                    "sum_starter_min_share": hf["starter_min_share"] + af["starter_min_share"] - 1.30,
                    "diff_q1_starter_min_share": hf["q1_starter_min_share"] - af["q1_starter_min_share"],
                    "sum_q1_starter_min_share": hf["q1_starter_min_share"] + af["q1_starter_min_share"] - 1.52,
                    "diff_h1_starter_min_share": hf["h1_starter_min_share"] - af["h1_starter_min_share"],
                    "diff_q1_first_sub_sec": (hf["q1_first_sub_sec"] - af["q1_first_sub_sec"]) / 60.0,
                    "diff_starter_bench_div": hf["starter_bench_div"] - af["starter_bench_div"],
                    "diff_starter_net48": hf["starter_net48"] - af["starter_net48"],
                    "diff_bench_net48": hf["bench_net48"] - af["bench_net48"],
                    "diff_q1_first6_margin": hf["q1_first6_margin"] - af["q1_first6_margin"],
                    "diff_q1_last6_margin": hf["q1_last6_margin"] - af["q1_last6_margin"],
                    "diff_q2_first6_margin": hf["q2_first6_margin"] - af["q2_first6_margin"],
                    "diff_rot_top5_minus_bench": h_rot["rot_top5_minus_bench_pm"] - a_rot["rot_top5_minus_bench_pm"],
                    "diff_missing_starter_pm": h_rot["rot_missing_starter_pm"] - a_rot["rot_missing_starter_pm"],
                    "diff_missing_starter_min": h_rot["rot_missing_starter_min"] - a_rot["rot_missing_starter_min"],
                    # 1Q & 1H pace profiles vs full-game pace
                    "sum_q1_pace_ratio": hf["q1_pace_ratio"] + af["q1_pace_ratio"],
                    "diff_q1_pace_ratio": hf["q1_pace_ratio"] - af["q1_pace_ratio"],
                    "sum_h1_pace_ratio": hf["h1_pace_ratio"] + af["h1_pace_ratio"],
                    "matchup_pace_q1_minus_fg": 0.5 * (hf["pace_q1"] + af["pace_q1"]) - 0.5 * (hf["pace_fg"] + af["pace_fg"]),
                    "matchup_pace_h1_minus_fg": 0.5 * (hf["pace_h1"] + af["pace_h1"]) - 0.5 * (hf["pace_fg"] + af["pace_fg"]),
                    # Rest, road B2B (1H vs 2H fade), altitude, travel, & timezone early-tip
                    "home_b2b": hs_sched["is_b2b"],
                    "away_b2b": as_sched["is_b2b"],
                    "away_road_b2b": as_sched["road_b2b"],
                    "away_road_road_b2b": as_sched["road_road_b2b"],
                    "b2b_diff": hs_sched["is_b2b"] - as_sched["is_b2b"],
                    "rest_diff": hs_sched["rest_days"] - as_sched["rest_days"],
                    "three_in_four_diff": hs_sched["three_in_four"] - as_sched["three_in_four"],
                    "three_in_four_sum": hs_sched["three_in_four"] + as_sched["three_in_four"],
                    "away_log_travel_km": as_sched["log_travel_km"],
                    "travel_log_diff": hs_sched["log_travel_km"] - as_sched["log_travel_km"],
                    "altitude_adv_kft": alt_adv_kft,
                    "altitude_x_away_b2b": alt_adv_kft * as_sched["is_b2b"],
                    "venue_high_alt": venue_high_alt,
                    "venue_local_hour": venue_hour,
                    "is_early_matinee": is_early_matinee,
                    "away_morning_body": away_morning_body,
                    "away_late_body": away_late_body,
                    "tz_diff_away": tz_diff_away,
                    # Targets & actuals (for post-prediction evaluation / strict past-only training)
                    "q1_margin": q1_m,
                    "h1_margin": h1_m,
                    "h2_margin": float(row["h2_margin"]),
                    "fg_margin": float(row["fg_margin"]),
                    "q1_total": q1_t,
                    "h1_total": h1_t,
                    "fg_total": float(row["fg_total"]),
                    "mech_q1_exp_margin": mech_q1_exp_margin,
                    "mech_h1_exp_margin": mech_h1_exp_margin,
                    "mech_q1_exp_total": mech_q1_exp_total,
                    "mech_h1_exp_total": mech_h1_exp_total,
                    "resid_q1_margin_mech": q1_m - mech_q1_exp_margin if np.isfinite(fg_sp_home) else np.nan,
                    "resid_h1_margin_mech": h1_m - mech_h1_exp_margin if np.isfinite(fg_sp_home) else np.nan,
                    "resid_q1_total_mech": q1_t - mech_q1_exp_total if np.isfinite(fg_tot) else np.nan,
                    "resid_h1_total_mech": h1_t - mech_h1_exp_total if np.isfinite(fg_tot) else np.nan,
                    # Archived real sportsbook 1Q/1H lines (when present on ESPN)
                    "deriv_odds_provider": row.get("deriv_odds_provider"),
                    "book_q1_home_ml": row.get("book_q1_home_ml", np.nan),
                    "book_q1_away_ml": row.get("book_q1_away_ml", np.nan),
                    "book_q1_spread_home": row.get("book_q1_spread_home", np.nan),
                    "book_q1_spread_away": row.get("book_q1_spread_away", np.nan),
                    "book_q1_spread_home_odds": row.get("book_q1_spread_home_odds", np.nan),
                    "book_q1_spread_away_odds": row.get("book_q1_spread_away_odds", np.nan),
                    "book_q1_total_line": row.get("book_q1_total_line", np.nan),
                    "book_q1_over_odds": row.get("book_q1_over_odds", np.nan),
                    "book_q1_under_odds": row.get("book_q1_under_odds", np.nan),
                    "book_h1_home_ml": row.get("book_h1_home_ml", np.nan),
                    "book_h1_away_ml": row.get("book_h1_away_ml", np.nan),
                    "book_h1_spread_home": row.get("book_h1_spread_home", np.nan),
                    "book_h1_spread_away": row.get("book_h1_spread_away", np.nan),
                    "book_h1_spread_home_odds": row.get("book_h1_spread_home_odds", np.nan),
                    "book_h1_spread_away_odds": row.get("book_h1_spread_away_odds", np.nan),
                    "book_h1_total_line": row.get("book_h1_total_line", np.nan),
                    "book_h1_over_odds": row.get("book_h1_over_odds", np.nan),
                    "book_h1_under_odds": row.get("book_h1_under_odds", np.nan),
                }
            )

        # STEP 2: AFTER all games on gdate are scored, update team/league/player states
        for _, row in day_df.iterrows():
            gid = str(row["game_id"])
            h_tri = normalize_espn_tricode(row["home_tricode"])
            a_tri = normalize_espn_tricode(row["away_tricode"])
            is_neut = bool(row.get("is_neutral", False) or row.get("summary_is_neutral", False))
            _, _, v_lat, v_lon = _venue_meta(h_tri, str(row.get("venue_city", "")), is_neut)

            hst = team_states.setdefault(h_tri, TeamDerivativeState())
            ast = team_states.setdefault(a_tri, TeamDerivativeState())

            fg_sp_home = float(row["spread_home"]) if pd.notna(row.get("spread_home")) else 0.0
            fg_exp_m = -fg_sp_home
            q1_m = float(row["q1_margin"])
            h1_m = float(row["h1_margin"])
            fg_t = float(row["fg_total"])
            q1_t = float(row["q1_total"])
            h1_t = float(row["h1_total"])
            q1_t_sh = (q1_t / fg_t) if fg_t > 0 else np.nan
            h1_t_sh = (h1_t / fg_t) if fg_t > 0 else np.nan

            # Update league EMAs
            league["ortg_q1"] = _ema_step(
                league["ortg_q1"],
                0.5 * (float(row["home_q1_ortg"]) + float(row["away_q1_ortg"])),
                alpha_lg,
            )
            league["ortg_h1"] = _ema_step(
                league["ortg_h1"],
                0.5 * (float(row["home_h1_ortg"]) + float(row["away_h1_ortg"])),
                alpha_lg,
            )
            league["ortg_fg"] = _ema_step(
                league["ortg_fg"],
                0.5 * (float(row["home_fg_ortg"]) + float(row["away_fg_ortg"])),
                alpha_lg,
            )
            league["pace_q1"] = _ema_step(league["pace_q1"], float(row["q1_pace_48"]), alpha_lg)
            league["pace_h1"] = _ema_step(league["pace_h1"], float(row["h1_pace_48"]), alpha_lg)
            league["pace_fg"] = _ema_step(league["pace_fg"], float(row["fg_pace_48"]), alpha_lg)
            if np.isfinite(q1_t_sh):
                league["q1_total_share"] = _ema_step(league["q1_total_share"], q1_t_sh, alpha_lg)
            if np.isfinite(h1_t_sh):
                league["h1_total_share"] = _ema_step(league["h1_total_share"], h1_t_sh, alpha_lg)

            def _upd_team(
                st: TeamDerivativeState,
                *,
                side: str,
                sign: float,
                is_away_game: bool,
            ) -> None:
                ortg_q1 = float(row[f"{side}_q1_ortg"])
                drtg_q1 = float(row[f"{side}_q1_drtg"])
                ortg_h1 = float(row[f"{side}_h1_ortg"])
                drtg_h1 = float(row[f"{side}_h1_drtg"])
                ortg_fg = float(row[f"{side}_fg_ortg"])
                drtg_fg = float(row[f"{side}_fg_drtg"])

                st.ortg_q1 = _ema_step(st.ortg_q1, ortg_q1, alpha)
                st.drtg_q1 = _ema_step(st.drtg_q1, drtg_q1, alpha)
                st.ortg_h1 = _ema_step(st.ortg_h1, ortg_h1, alpha)
                st.drtg_h1 = _ema_step(st.drtg_h1, drtg_h1, alpha)
                st.ortg_fg = _ema_step(st.ortg_fg, ortg_fg, alpha)
                st.drtg_fg = _ema_step(st.drtg_fg, drtg_fg, alpha)

                st.fast_net_q1 = _ema_step(st.fast_net_q1, ortg_q1 - drtg_q1, alpha_fast)
                st.fast_net_h1 = _ema_step(st.fast_net_h1, ortg_h1 - drtg_h1, alpha_fast)
                st.fast_net_fg = _ema_step(st.fast_net_fg, ortg_fg - drtg_fg, alpha_fast)

                st.pace_q1 = _ema_step(st.pace_q1, float(row["q1_pace_48"]), alpha)
                st.pace_h1 = _ema_step(st.pace_h1, float(row["h1_pace_48"]), alpha)
                st.pace_fg = _ema_step(st.pace_fg, float(row["fg_pace_48"]), alpha)

                st.starter_min_share = _ema_step(
                    st.starter_min_share, float(row[f"{side}_starter_min_share"]), alpha
                )
                st.q1_starter_min_share = _ema_step(
                    st.q1_starter_min_share, float(row[f"{side}_q1_starter_min_share"]), alpha
                )
                st.h1_starter_min_share = _ema_step(
                    st.h1_starter_min_share, float(row[f"{side}_h1_starter_min_share"]), alpha
                )
                st.q1_first_sub_sec = _ema_step(
                    st.q1_first_sub_sec, float(row[f"{side}_q1_first_sub_sec"]), alpha
                )

                s_min = float(row[f"{side}_starter_min"])
                b_min = float(row[f"{side}_bench_min"])
                s_pm = float(row[f"{side}_starter_pm"])
                b_pm = float(row[f"{side}_bench_pm"])
                if np.isfinite(s_min) and s_min > 10.0 and np.isfinite(s_pm):
                    st.starter_net48 = _ema_step(st.starter_net48, (s_pm / s_min) * 48.0, alpha)
                if np.isfinite(b_min) and b_min > 5.0 and np.isfinite(b_pm):
                    st.bench_net48 = _ema_step(st.bench_net48, (b_pm / b_min) * 48.0, alpha)

                st.q1_first6_margin = _ema_step(
                    st.q1_first6_margin, sign * float(row["q1_first6_margin"]), alpha
                )
                st.q1_last6_margin = _ema_step(
                    st.q1_last6_margin, sign * float(row["q1_last6_margin"]), alpha
                )
                st.q2_first6_margin = _ema_step(
                    st.q2_first6_margin, sign * float(row["q2_first6_margin"]), alpha
                )
                st.q1_margin_resid = _ema_step(
                    st.q1_margin_resid, sign * (q1_m - 0.315 * fg_exp_m), alpha
                )
                st.h1_margin_resid = _ema_step(
                    st.h1_margin_resid, sign * (h1_m - 0.570 * fg_exp_m), alpha
                )
                st.q1_total_share = _ema_step(st.q1_total_share, q1_t_sh, alpha)
                st.h1_total_share = _ema_step(st.h1_total_share, h1_t_sh, alpha)

                st.games_season += 1
                st.games_total += 1
                st.prior_dates.append(gdate)
                st.last_was_away = is_away_game
                st.last_lat = v_lat
                st.last_lon = v_lon

            _upd_team(hst, side="home", sign=1.0, is_away_game=False)
            _upd_team(ast, side="away", sign=-1.0, is_away_game=True)

            # Update player rotation states strictly after gdate
            g_pbox = pbox_by_gid.get(gid)
            if g_pbox is not None and not g_pbox.empty:
                tids_in_g = set(int(x) for x in g_pbox["team_id"].unique())
                seen_pids = set(
                    str(r0["player_id"]) for _, r0 in g_pbox.iterrows() if float(r0["minutes"]) > 0.0
                )
                for pid, pst in player_states.items():
                    if pst.last_team_id in tids_in_g and pst.games > 0:
                        streak_history.append((int(pst.miss_streak), 1 if pid in seen_pids else 0))
                update_player_states_after_game(player_states, g_pbox)

    return pd.DataFrame(feature_rows)
