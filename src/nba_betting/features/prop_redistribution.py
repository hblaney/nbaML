"""Non-linear pregame teammate-absence redistribution for player props.

Tracks dual-horizon (fast/slow EMA) minutes, usage, assist, rebound, and three-point
rates alongside online covariance against missing-teammate load. All features use
strictly prior box scores (zero same-day or postgame leakage):

1. Availability probability ``P(play | miss_streak)`` is learned online via
   Beta-smoothed transitions from strictly prior team games.
2. Expected missing teammate load (minutes, usage, assists, rebounds) is computed
   from rotation teammates currently carrying ``miss_streak >= 1``.
3. Absorption across active teammates is **non-linear**:
   - Minutes absorption is concave / headroom-capped ``(37.5 - m)^1.15`` with a
     promotion multiplier for recent spot-starters, shrunk with each player's
     empirical covariance ``Cov(m_i, missing_min) / Var(missing_min)``.
   - Usage, assist, and rebound rate shares scale super-linearly with skill rate
     (``rate^1.35`` / ``rate^1.40``), shrunk with player-specific usage covariance,
     and act only on the **innovation** ``missing_today - ema_missing_in_baseline``
     so multi-game absences are never double-counted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FAST_HALF_LIFE = 8.0
SLOW_HALF_LIFE = 24.0
OPP_HALF_LIFE = 18.0
COV_DECAY = 0.96

LEAGUE_AVG_TEAM_TOTAL = 113.5
LEAGUE_AVG_OPP_ALLOWED = {
    "pts": 114.5,
    "reb": 43.5,
    "ast": 26.0,
    "fg3m": 12.6,
}
LEAGUE_PRIOR_PER_MIN = {
    "points": 0.46,
    "rebounds": 0.18,
    "assists": 0.11,
    "threes": 0.052,
}


def _ema_alpha(half_life: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / float(half_life))


@dataclass
class PlayerPropState:
    """Strictly-prior online state for one player."""

    player_id: str
    player_name: str = ""
    team_id: int = -1
    team_abbreviation: str = ""
    games: int = 0
    miss_streak: int = 0
    min_fast: float = 0.0
    min_slow: float = 0.0
    pts_fast: float = 0.0
    pts_slow: float = 0.0
    reb_fast: float = 0.0
    reb_slow: float = 0.0
    ast_fast: float = 0.0
    ast_slow: float = 0.0
    fg3_fast: float = 0.0
    fg3_slow: float = 0.0
    usg_slow: float = 0.0
    start_slow: float = 0.0
    last_start: float = 0.0
    miss_min_fast: float = 0.0
    miss_min_slow: float = 0.0
    miss_usg_fast: float = 0.0
    miss_usg_slow: float = 0.0
    miss_ast_fast: float = 0.0
    miss_ast_slow: float = 0.0
    miss_reb_fast: float = 0.0
    miss_reb_slow: float = 0.0
    cov_min_miss: float = 0.0
    var_miss_min: float = 0.0
    cov_pts_miss: float = 0.0
    var_miss_usg: float = 0.0

    @property
    def base_minutes(self) -> float:
        return float(0.60 * self.min_fast + 0.40 * self.min_slow)


@dataclass
class OpponentAllowedState:
    pts: float = LEAGUE_AVG_OPP_ALLOWED["pts"]
    reb: float = LEAGUE_AVG_OPP_ALLOWED["reb"]
    ast: float = LEAGUE_AVG_OPP_ALLOWED["ast"]
    fg3m: float = LEAGUE_AVG_OPP_ALLOWED["fg3m"]
    games: int = 0


@dataclass(frozen=True)
class RedistributedRates:
    """Pregame adjusted player baseline before game-script / blowout mixture."""

    player_id: str
    player_name: str
    team_id: int
    team_abbreviation: str
    games: int
    base_minutes: float
    adj_minutes: float
    pts_per_min: float
    reb_per_min: float
    ast_per_min: float
    fg3m_per_min: float
    pra_per_min: float
    delta_missing_min: float
    delta_missing_usg: float
    spot_starter_flag: float
    bench_demotion_flag: float
    is_b2b: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RedistributionTracker:
    """Chronological state machine for zero-leakage prop feature generation."""

    states: dict[str, PlayerPropState] = field(default_factory=dict)
    streak_counts: dict[int, list[float]] = field(default_factory=dict)
    opp_allowed: dict[int, OpponentAllowedState] = field(default_factory=dict)
    team_last_date: dict[int, pd.Timestamp] = field(default_factory=dict)
    rolling_logs: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    def play_prob(self, miss_streak: int) -> float:
        s = int(min(max(miss_streak, 0), 8))
        ab = self.streak_counts.get(s)
        if ab is None:
            # Uninformative Beta prior before observations accumulate
            return 0.86 if s == 0 else max(0.08, 0.45 * (0.75 ** (s - 1)))
        return float(ab[0] / (ab[0] + ab[1]))

    def rolling10_expectation(
        self, player_id: str, stat: str, window: int = 10
    ) -> tuple[float, float] | None:
        """Return ``(rolling_exp_minutes, rolling_stat_mean)`` from prior played games."""
        logs = self.rolling_logs.get(str(player_id))
        if not logs:
            return None
        tail = logs[-int(window) :]
        tot_min = sum(x["min"] for x in tail)
        if tot_min <= 0:
            return None
        exp_m = tot_min / len(tail)
        col = {
            "points": "pts",
            "rebounds": "reb",
            "assists": "ast",
            "threes": "fg3m",
            "pra": "pra",
        }[stat]
        rate = sum(x[col] for x in tail) / tot_min
        return float(exp_m), float(exp_m * rate)

    def project_team_pregame(
        self,
        team_id: int,
        opp_team_id: int | None,
        game_date: str | pd.Timestamp,
        *,
        team_spread: float = 0.0,
        total_line: float = 226.5,
        min_games: int = 3,
    ) -> dict[str, RedistributedRates]:
        """Compute redistributed pregame minutes & per-minute rates for ``team_id``."""
        tid = int(team_id)
        gd = pd.Timestamp(game_date).normalize()
        rest_days = (gd - self.team_last_date[tid]).days if tid in self.team_last_date else 3
        is_b2b = 1.0 if rest_days == 1 else 0.0

        team_imp_tot = 0.5 * (float(total_line) - float(team_spread))
        pace_env = float(np.clip(team_imp_tot / LEAGUE_AVG_TEAM_TOTAL, 0.88, 1.14))
        ost = self.opp_allowed.get(int(opp_team_id)) if opp_team_id is not None else None
        if ost is None or ost.games == 0:
            ost = OpponentAllowedState()

        opp_pts_mult = float(np.clip((ost.pts / LEAGUE_AVG_OPP_ALLOWED["pts"]) ** 0.45, 0.93, 1.07))
        opp_reb_mult = float(np.clip((ost.reb / LEAGUE_AVG_OPP_ALLOWED["reb"]) ** 0.50, 0.92, 1.08))
        opp_ast_mult = float(np.clip((ost.ast / LEAGUE_AVG_OPP_ALLOWED["ast"]) ** 0.50, 0.92, 1.08))
        opp_fg3_mult = float(
            np.clip((ost.fg3m / LEAGUE_AVG_OPP_ALLOWED["fg3m"]) ** 0.55, 0.90, 1.10)
        )

        roster = [
            st for st in self.states.values() if st.team_id == tid and st.games >= int(min_games)
        ]
        if not roster:
            return {}
        roster.sort(key=lambda s: -s.min_slow)
        rot = roster[:10]

        miss_min = miss_usg = miss_ast = miss_reb = 0.0
        cap_min_sum = cap_usg_sum = cap_ast_sum = cap_reb_sum = 1e-6
        caps: dict[str, tuple[float, float, float, float]] = {}

        for st in rot:
            pp = self.play_prob(st.miss_streak)
            bm = float(0.65 * st.min_fast + 0.35 * st.min_slow)
            if st.miss_streak >= 1:
                p_abs = 1.0 - pp
                miss_min += p_abs * bm
                miss_usg += p_abs * bm * st.usg_slow
                miss_ast += p_abs * bm * st.ast_slow
                miss_reb += p_abs * bm * st.reb_slow
            headroom = max(0.5, 37.5 - bm)
            promo_mult = 1.30 if (st.last_start > 0.5 and st.start_slow < 0.45) else 1.0
            w_min = pp * (headroom**1.15) * min(1.0, bm / 18.0) * promo_mult
            w_usg = pp * bm * (max(st.usg_slow, 0.05) ** 1.35)
            w_ast = pp * bm * (max(st.ast_slow, 0.02) ** 1.40)
            w_reb = pp * bm * (max(st.reb_slow, 0.03) ** 1.35)
            caps[st.player_id] = (w_min, w_usg, w_ast, w_reb)
            cap_min_sum += w_min
            cap_usg_sum += w_usg
            cap_ast_sum += w_ast
            cap_reb_sum += w_reb

        out: dict[str, RedistributedRates] = {}
        for st in roster:
            n_g = st.games
            shrink = n_g / (n_g + 6.0)
            base_m = st.base_minutes
            w_min, w_usg, w_ast, w_reb = caps.get(st.player_id, (0.0, 0.0, 0.0, 0.0))

            role_min_elast = 0.55 * (w_min / cap_min_sum)
            if st.var_miss_min > 50.0 and n_g >= 12:
                emp_min_elast = float(np.clip(st.cov_min_miss / st.var_miss_min, -0.05, 0.35))
                pw = min(0.50, st.var_miss_min / (st.var_miss_min + 400.0))
                min_elast = pw * emp_min_elast + (1.0 - pw) * role_min_elast
            else:
                min_elast = role_min_elast

            d_miss_min = miss_min - (0.60 * st.miss_min_fast + 0.40 * st.miss_min_slow)
            d_miss_usg = miss_usg - (0.60 * st.miss_usg_fast + 0.40 * st.miss_usg_slow)
            d_miss_ast = miss_ast - (0.60 * st.miss_ast_fast + 0.40 * st.miss_ast_slow)
            d_miss_reb = miss_reb - (0.60 * st.miss_reb_fast + 0.40 * st.miss_reb_slow)

            b2b_drag = 0.35 * is_b2b if base_m >= 30.0 else 0.0
            adj_min = float(np.clip(base_m + min_elast * d_miss_min - b2b_drag, 4.0, 40.0))

            pts_base = (
                shrink * (0.45 * st.pts_fast + 0.55 * st.pts_slow)
                + (1.0 - shrink) * LEAGUE_PRIOR_PER_MIN["points"]
            )
            reb_base = (
                shrink * (0.45 * st.reb_fast + 0.55 * st.reb_slow)
                + (1.0 - shrink) * LEAGUE_PRIOR_PER_MIN["rebounds"]
            )
            ast_base = (
                shrink * (0.45 * st.ast_fast + 0.55 * st.ast_slow)
                + (1.0 - shrink) * LEAGUE_PRIOR_PER_MIN["assists"]
            )
            fg3_base = (
                shrink * (0.45 * st.fg3_fast + 0.55 * st.fg3_slow)
                + (1.0 - shrink) * LEAGUE_PRIOR_PER_MIN["threes"]
            )

            role_usg_share = 0.35 * (w_usg / cap_usg_sum)
            usg_bump = role_usg_share * d_miss_usg / max(adj_min * max(st.usg_slow, 0.15), 1.0)
            if st.var_miss_usg > 20.0 and n_g >= 12:
                emp_pts_elast = float(np.clip(st.cov_pts_miss / st.var_miss_usg, -0.02, 0.12))
                pw_u = min(0.45, st.var_miss_usg / (st.var_miss_usg + 150.0))
                pts_pm_raw = (1.0 - pw_u) * (pts_base * (1.0 + usg_bump)) + pw_u * (
                    pts_base + emp_pts_elast * d_miss_usg / max(adj_min, 10.0)
                )
            else:
                pts_pm_raw = pts_base * (1.0 + usg_bump)

            ast_bump = (
                0.40
                * d_miss_ast
                * (w_ast / cap_ast_sum)
                / max(adj_min * max(st.ast_slow, 0.04), 1.0)
            )
            reb_bump = (
                0.35
                * d_miss_reb
                * (w_reb / cap_reb_sum)
                / max(adj_min * max(st.reb_slow, 0.05), 1.0)
            )

            pts_pm = float(np.clip(pts_pm_raw, 0.05, 1.35)) * opp_pts_mult * (pace_env**0.70)
            reb_pm = (
                reb_base
                * float(np.clip(1.0 + reb_bump, 0.88, 1.18))
                * opp_reb_mult
                * (pace_env**0.40)
            )
            ast_pm = (
                ast_base
                * float(np.clip(1.0 + ast_bump, 0.86, 1.22))
                * opp_ast_mult
                * (pace_env**0.65)
            )
            fg3_pm = (
                fg3_base
                * float(np.clip(1.0 + 0.70 * usg_bump, 0.88, 1.16))
                * opp_fg3_mult
                * (pace_env**0.60)
            )

            spot_flag = 1.0 if (st.last_start > 0.5 and st.start_slow < 0.45) else 0.0
            demote_flag = 1.0 if (st.last_start < 0.5 and st.start_slow > 0.65) else 0.0

            out[st.player_id] = RedistributedRates(
                player_id=st.player_id,
                player_name=st.player_name,
                team_id=tid,
                team_abbreviation=st.team_abbreviation,
                games=n_g,
                base_minutes=float(base_m),
                adj_minutes=float(adj_min),
                pts_per_min=float(pts_pm),
                reb_per_min=float(reb_pm),
                ast_per_min=float(ast_pm),
                fg3m_per_min=float(fg3_pm),
                pra_per_min=float(pts_pm + reb_pm + ast_pm),
                delta_missing_min=float(d_miss_min),
                delta_missing_usg=float(d_miss_usg),
                spot_starter_flag=spot_flag,
                bench_demotion_flag=demote_flag,
                is_b2b=is_b2b,
            )
        return out

    def update_team_postgame(
        self,
        team_rows: pd.DataFrame,
        *,
        game_date: str | pd.Timestamp,
        opp_team_id: int | None = None,
    ) -> None:
        """Update player states, empirical elasticities, and play-rates after a game finishes."""
        if team_rows.empty:
            return
        af = _ema_alpha(FAST_HALF_LIFE)
        aslow = _ema_alpha(SLOW_HALF_LIFE)
        aopp = _ema_alpha(OPP_HALF_LIFE)
        gd = pd.Timestamp(game_date).normalize()
        tid = int(team_rows["team_id"].iloc[0])
        self.team_last_date[tid] = gd

        min_col = "minutes" if "minutes" in team_rows.columns else "min"
        roster_prev = [st for st in self.states.values() if st.team_id == tid and st.games >= 3]
        roster_prev.sort(key=lambda s: -s.min_slow)
        played_ids = {
            str(r["player_id"])
            for _, r in team_rows.iterrows()
            if float(r.get(min_col, 0.0) or 0.0) > 0.0
        }

        real_miss_min = real_miss_usg = real_miss_ast = real_miss_reb = 0.0
        for st in roster_prev[:9]:
            if st.player_id not in played_ids:
                bm = 0.65 * st.min_fast + 0.35 * st.min_slow
                real_miss_min += bm
                real_miss_usg += bm * st.usg_slow
                real_miss_ast += bm * st.ast_slow
                real_miss_reb += bm * st.reb_slow

        if opp_team_id is not None:
            otid = int(opp_team_id)
            ost = self.opp_allowed.get(otid) or OpponentAllowedState()
            t_pts = float(team_rows["pts"].sum())
            t_reb = float(team_rows["reb"].sum())
            t_ast = float(team_rows["ast"].sum())
            t_fg3 = float(team_rows["fg3m"].sum())
            if ost.games == 0:
                ost = OpponentAllowedState(pts=t_pts, reb=t_reb, ast=t_ast, fg3m=t_fg3, games=1)
            else:
                ost.pts = (1.0 - aopp) * ost.pts + aopp * t_pts
                ost.reb = (1.0 - aopp) * ost.reb + aopp * t_reb
                ost.ast = (1.0 - aopp) * ost.ast + aopp * t_ast
                ost.fg3m = (1.0 - aopp) * ost.fg3m + aopp * t_fg3
                ost.games += 1
            self.opp_allowed[otid] = ost

        seen: set[str] = set()
        for _, r in team_rows.iterrows():
            pid = str(r["player_id"])
            if not pid:
                continue
            seen.add(pid)
            mins = float(r.get(min_col, 0.0) or 0.0)
            played = 1 if mins > 0.0 else 0
            st = self.states.get(pid)
            if st is not None and st.team_id == tid and st.games >= 1:
                ms = int(min(max(st.miss_streak, 0), 8))
                bucket = self.streak_counts.setdefault(ms, [2.0, 2.0])
                bucket[0 if played else 1] += 1.0

            if mins <= 0.0:
                if st is not None:
                    st.team_id = tid
                    st.miss_streak += 1
                continue

            pts_v = float(r.get("pts", 0.0) or 0.0)
            reb_v = float(r.get("reb", 0.0) or 0.0)
            ast_v = float(r.get("ast", 0.0) or 0.0)
            fg3_v = float(r.get("fg3m", 0.0) or 0.0)
            fga_v = float(r.get("fga", pts_v * 0.75) or 0.0)
            fta_v = float(r.get("fta", pts_v * 0.22) or 0.0)
            tov_v = float(r.get("tov", 1.0) or 0.0)
            starter_raw = r.get("starter")
            if starter_raw is None or pd.isna(starter_raw):
                strt = 1.0 if mins >= 27.0 else 0.0
            else:
                strt = 1.0 if bool(starter_raw) else 0.0

            self.rolling_logs.setdefault(pid, []).append(
                {
                    "min": mins,
                    "pts": pts_v,
                    "reb": reb_v,
                    "ast": ast_v,
                    "fg3m": fg3_v,
                    "pra": pts_v + reb_v + ast_v,
                }
            )
            pts_pm = pts_v / mins
            reb_pm = reb_v / mins
            ast_pm = ast_v / mins
            fg3_pm = fg3_v / mins
            usg_pm = (fga_v + 0.44 * fta_v + tov_v) / mins
            pname = str(r.get("player_name") or (st.player_name if st else ""))
            tri = str(
                r.get("team_tricode")
                or r.get("team_abbreviation")
                or (st.team_abbreviation if st else "")
            )

            if st is None or st.games == 0:
                self.states[pid] = PlayerPropState(
                    player_id=pid,
                    player_name=pname,
                    team_id=tid,
                    team_abbreviation=tri,
                    games=1,
                    miss_streak=0,
                    min_fast=mins,
                    min_slow=mins,
                    pts_fast=pts_pm,
                    pts_slow=pts_pm,
                    reb_fast=reb_pm,
                    reb_slow=reb_pm,
                    ast_fast=ast_pm,
                    ast_slow=ast_pm,
                    fg3_fast=fg3_pm,
                    fg3_slow=fg3_pm,
                    usg_slow=usg_pm,
                    start_slow=strt,
                    last_start=strt,
                    miss_min_fast=real_miss_min,
                    miss_min_slow=real_miss_min,
                    miss_usg_fast=real_miss_usg,
                    miss_usg_slow=real_miss_usg,
                    miss_ast_fast=real_miss_ast,
                    miss_ast_slow=real_miss_ast,
                    miss_reb_fast=real_miss_reb,
                    miss_reb_slow=real_miss_reb,
                )
            else:
                dm = real_miss_min - st.miss_min_slow
                du = real_miss_usg - st.miss_usg_slow
                st.cov_min_miss = COV_DECAY * st.cov_min_miss + (mins - st.min_slow) * dm
                st.var_miss_min = COV_DECAY * st.var_miss_min + dm * dm
                st.cov_pts_miss = (
                    COV_DECAY * st.cov_pts_miss + (pts_v - st.min_slow * st.pts_slow) * du
                )
                st.var_miss_usg = COV_DECAY * st.var_miss_usg + du * du

                st.player_name = pname or st.player_name
                st.team_id = tid
                st.team_abbreviation = tri or st.team_abbreviation
                st.games += 1
                st.miss_streak = 0
                st.last_start = strt

                st.min_fast = (1.0 - af) * st.min_fast + af * mins
                st.pts_fast = (1.0 - af) * st.pts_fast + af * pts_pm
                st.reb_fast = (1.0 - af) * st.reb_fast + af * reb_pm
                st.ast_fast = (1.0 - af) * st.ast_fast + af * ast_pm
                st.fg3_fast = (1.0 - af) * st.fg3_fast + af * fg3_pm
                st.miss_min_fast = (1.0 - af) * st.miss_min_fast + af * real_miss_min
                st.miss_usg_fast = (1.0 - af) * st.miss_usg_fast + af * real_miss_usg
                st.miss_ast_fast = (1.0 - af) * st.miss_ast_fast + af * real_miss_ast
                st.miss_reb_fast = (1.0 - af) * st.miss_reb_fast + af * real_miss_reb

                st.min_slow = (1.0 - aslow) * st.min_slow + aslow * mins
                st.pts_slow = (1.0 - aslow) * st.pts_slow + aslow * pts_pm
                st.reb_slow = (1.0 - aslow) * st.reb_slow + aslow * reb_pm
                st.ast_slow = (1.0 - aslow) * st.ast_slow + aslow * ast_pm
                st.fg3_slow = (1.0 - aslow) * st.fg3_slow + aslow * fg3_pm
                st.usg_slow = (1.0 - aslow) * st.usg_slow + aslow * usg_pm
                st.start_slow = (1.0 - aslow) * st.start_slow + aslow * strt
                st.miss_min_slow = (1.0 - aslow) * st.miss_min_slow + aslow * real_miss_min
                st.miss_usg_slow = (1.0 - aslow) * st.miss_usg_slow + aslow * real_miss_usg
                st.miss_ast_slow = (1.0 - aslow) * st.miss_ast_slow + aslow * real_miss_ast
                st.miss_reb_slow = (1.0 - aslow) * st.miss_reb_slow + aslow * real_miss_reb

        for pid, st in list(self.states.items()):
            if pid not in seen and st.team_id == tid and st.games >= 1:
                ms = int(min(max(st.miss_streak, 0), 8))
                bucket = self.streak_counts.setdefault(ms, [2.0, 2.0])
                bucket[1] += 1.0
                st.miss_streak += 1


def build_tracker_asof(
    player_logs: pd.DataFrame,
    asof_date: str | pd.Timestamp,
) -> RedistributionTracker:
    """Replay historical player box scores strictly prior to ``asof_date``."""
    tracker = RedistributionTracker()
    if player_logs.empty:
        return tracker
    df = player_logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    asof = pd.Timestamp(asof_date).normalize()
    hist = df.loc[df["game_date"] < asof].sort_values(["game_date", "game_id", "team_id"])
    if hist.empty:
        return tracker
    for (gd, _gid), gdf in hist.groupby(["game_date", "game_id"], sort=True):
        tids = [int(t) for t in gdf["team_id"].unique()]
        for tid in tids:
            opp = next((t for t in tids if t != tid), None)
            if "opp_team_id" in gdf.columns:
                sub_opp = gdf.loc[gdf["team_id"] == tid, "opp_team_id"].dropna()
                if not sub_opp.empty:
                    opp = int(sub_opp.iloc[0])
            tracker.update_team_postgame(
                gdf.loc[gdf["team_id"] == tid],
                game_date=gd,
                opp_team_id=opp,
            )
    return tracker
