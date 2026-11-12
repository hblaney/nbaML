"""Garbage-time-filtered / leverage-weighted efficiency & regularized Stint RAPM.

All features for games on ``game_date`` are computed strictly from completed games
with ``prior_game_date < game_date`` (zero same-day or postgame leakage).

Extracts:
1. Opponent-adjusted non-garbage-time ORTG / DRTG / NetRtg and continuous
   leverage-weighted NetRtg vs raw boxscore NetRtg ("garbage-time distortion
   residual").
2. First-3Q linescore dominance & 4Q blowout giveback residuals.
3. Clean regulation non-foul-fest pace & OT / foul-fest total-points inflation
   biases (key Over/Under signal).
4. Expanding Ridge-regularized Two-Way Stint RAPM (ORAPM + DRAPM = Net RAPM,
   ORAPM - DRAPM = Scoring Tilt) solved from exact 5-v-5 non-garbage stint
   normal equations and projected onto expected active team rotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import solve

from nba_betting.features.player_impact import learn_play_rate
from nba_betting.features.schedule_context import schedule_density_asof

DEFAULT_TEAM_HL = 14.0
DEFAULT_PLAYER_HL = 18.0
DEFAULT_OFFSEASON_TEAM_CARRY = 0.65
DEFAULT_OFFSEASON_RAPM_DECAY = 0.70
DEFAULT_DAILY_RAPM_DECAY = 0.9985
DEFAULT_ORAPM_LAMBDA = 2000.0
DEFAULT_DRAPM_LAMBDA = 2500.0
PRIOR_ORAPM_REPLACEMENT = -0.60
PRIOR_DRAPM_REPLACEMENT = -0.40
REPLACEMENT_FILL_ORAPM = -1.20
REPLACEMENT_FILL_DRAPM = -0.80


def _ema_alpha(half_life: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / float(half_life))


def _ema_step(old: float, obs: float, n_prev: int, alpha: float) -> float:
    if n_prev <= 0 or math.isnan(old):
        return float(obs)
    return float((1.0 - alpha) * old + alpha * obs)


@dataclass
class CleanPlayerState:
    games: int = 0
    ema_clean_min: float = float("nan")
    ema_box_min: float = float("nan")
    ema_clean_pm_per_min: float = float("nan")
    ema_box_pm_per_min: float = float("nan")
    miss_streak: int = 0
    last_team_id: int = -1


@dataclass
class TeamCleanState:
    games: int = 0
    season_games: int = 0
    raw_ortg: float = 114.5
    raw_drtg: float = 114.5
    adj_raw_ortg: float = 114.5
    adj_raw_drtg: float = 114.5
    clean_ortg: float = 114.5
    clean_drtg: float = 114.5
    adj_clean_ortg: float = 114.5
    adj_clean_drtg: float = 114.5
    lev_ortg: float = 114.5
    lev_drtg: float = 114.5
    adj_lev_ortg: float = 114.5
    adj_lev_drtg: float = 114.5
    raw_pace: float = 99.0
    clean_pace: float = 99.0
    margin_3q_ema: float = 0.0
    final_margin_ema: float = 0.0
    q4_giveback_ema: float = 0.0
    raw_total_ema: float = 228.0
    reg_total_ema: float = 226.5
    ot_rate_ema: float = 0.05
    endgame_fta_ema: float = 2.0
    off_ftr_ema: float = 0.26
    def_ftr_ema: float = 0.26
    pf_rate_ema: float = 20.0
    opp_pf_rate_ema: float = 20.0
    fg3_rate_ema: float = 0.39


class ExpandingTwoWayRapm:
    """Exact normal-equation tracker for weighted Ridge 2-way (O/D) Stint RAPM."""

    def __init__(
        self,
        max_players: int = 1200,
        lambda_off: float = DEFAULT_ORAPM_LAMBDA,
        lambda_def: float = DEFAULT_DRAPM_LAMBDA,
        prior_off: float = PRIOR_ORAPM_REPLACEMENT,
        prior_def: float = PRIOR_DRAPM_REPLACEMENT,
    ) -> None:
        self.cap = int(max_players)
        self.lambda_off = float(lambda_off)
        self.lambda_def = float(lambda_def)
        self.prior_off = float(prior_off)
        self.prior_def = float(prior_def)
        self.player_to_idx: dict[str, int] = {}
        self.n_players = 0
        dim = 2 * self.cap
        self.G = np.zeros((dim, dim), dtype=np.float64)
        self.b = np.zeros(dim, dtype=np.float64)
        self._dirty = True
        self._orapm: dict[str, float] = {}
        self._drapm: dict[str, float] = {}

    def _ensure_capacity(self, needed_players: int) -> None:
        if needed_players <= self.cap:
            return
        new_cap = max(self.cap * 2, needed_players + 256)
        new_dim = 2 * new_cap
        old_dim = 2 * self.cap
        new_G = np.zeros((new_dim, new_dim), dtype=np.float64)
        new_b = np.zeros(new_dim, dtype=np.float64)
        new_G[:old_dim, :old_dim] = self.G
        new_b[:old_dim] = self.b
        self.G = new_G
        self.b = new_b
        self.cap = new_cap

    def _idx(self, pid: str) -> int:
        idx = self.player_to_idx.get(pid)
        if idx is not None:
            return idx
        self._ensure_capacity(self.n_players + 1)
        idx = self.n_players
        self.player_to_idx[pid] = idx
        self.n_players += 1
        return idx

    def decay(self, factor: float) -> None:
        if self.n_players == 0 or factor == 1.0:
            return
        d = 2 * self.n_players
        self.G[:d, :d] *= float(factor)
        self.b[:d] *= float(factor)
        self._dirty = True

    def add_game_stints(
        self,
        stint_rows: pd.DataFrame,
        *,
        league_ortg: float,
        hca_per_100: float = 1.35,
        is_neutral: bool = False,
    ) -> None:
        if stint_rows.empty:
            return
        hca = 0.0 if is_neutral else float(hca_per_100)
        base_home_off = float(league_ortg) + hca
        base_away_off = float(league_ortg) - hca

        for row in stint_rows.itertuples(index=False):
            h_pids = (row.home_p1, row.home_p2, row.home_p3, row.home_p4, row.home_p5)
            a_pids = (row.away_p1, row.away_p2, row.away_p3, row.away_p4, row.away_p5)
            h_idx = [self._idx(str(p)) for p in h_pids]
            a_idx = [self._idx(str(p)) for p in a_pids]

            # Blend binary non-garbage and continuous leverage possessions/points
            wh = 0.5 * float(row.clean_poss_home) + 0.5 * float(row.lev_poss_home)
            wa = 0.5 * float(row.clean_poss_away) + 0.5 * float(row.lev_poss_away)
            ph = 0.5 * float(row.clean_pts_home) + 0.5 * float(row.lev_pts_home)
            pa = 0.5 * float(row.clean_pts_away) + 0.5 * float(row.lev_pts_away)

            if wh > 0.01 or ph > 0.0:
                w_eff = max(wh, 0.25)
                wy = 100.0 * ph - w_eff * base_home_off
                inds = [2 * i for i in h_idx] + [2 * j + 1 for j in a_idx]
                signs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
                ix = np.ix_(inds, inds)
                self.G[ix] += w_eff * np.outer(signs, signs)
                for k_i, s_i in zip(inds, signs):
                    self.b[k_i] += wy * s_i

            if wa > 0.01 or pa > 0.0:
                w_eff = max(wa, 0.25)
                wy = 100.0 * pa - w_eff * base_away_off
                inds = [2 * j for j in a_idx] + [2 * i + 1 for i in h_idx]
                signs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
                ix = np.ix_(inds, inds)
                self.G[ix] += w_eff * np.outer(signs, signs)
                for k_i, s_i in zip(inds, signs):
                    self.b[k_i] += wy * s_i

        self._dirty = True

    def current_ratings(self) -> tuple[dict[str, float], dict[str, float]]:
        if not self._dirty:
            return self._orapm, self._drapm
        if self.n_players == 0:
            self._dirty = False
            return {}, {}
        d = 2 * self.n_players
        A = self.G[:d, :d].copy()
        rhs = self.b[:d].copy()
        diag_pen = np.empty(d, dtype=np.float64)
        diag_pen[0::2] = self.lambda_off
        diag_pen[1::2] = self.lambda_def
        A[np.diag_indices(d)] += diag_pen
        rhs[0::2] += self.lambda_off * self.prior_off
        rhs[1::2] += self.lambda_def * self.prior_def

        sol = solve(A, rhs, assume_a="pos", check_finite=False)
        inv_map = {idx: pid for pid, idx in self.player_to_idx.items()}
        self._orapm = {inv_map[i]: float(sol[2 * i]) for i in range(self.n_players)}
        self._drapm = {inv_map[i]: float(sol[2 * i + 1]) for i in range(self.n_players)}
        self._dirty = False
        return self._orapm, self._drapm


def _project_team_rotation_rapm(
    pstates: dict[str, CleanPlayerState],
    team_id: int,
    *,
    orapm_map: dict[str, float],
    drapm_map: dict[str, float],
    play_rates: dict[int, float],
    top_n: int = 9,
) -> dict[str, float]:
    """Project 2-way RAPM & clean +/- onto expected active 240-minute rotation."""
    cands = [
        (pid, st)
        for pid, st in pstates.items()
        if st.last_team_id == int(team_id)
        and st.games > 0
        and not math.isnan(st.ema_clean_min)
    ]
    if not cands:
        return {
            "rapm_rot": 0.0,
            "orapm_rot": 0.0,
            "drapm_rot": 0.0,
            "rapm_scoring_tilt": 0.0,
            "rapm_missing": 0.0,
            "clean_pm_rot": 0.0,
            "box_pm_rot": 0.0,
            "clean_vs_box_pm_rot": 0.0,
            "clean_missing_min": 0.0,
        }
    cands.sort(key=lambda x: -float(x[1].ema_clean_min))
    rot = cands[: int(top_n)]

    exp_mins: list[float] = []
    orapms: list[float] = []
    drapms: list[float] = []
    clean_pms: list[float] = []
    box_pms: list[float] = []
    missing_rapm = 0.0
    missing_min = 0.0

    for pid, st in rot:
        streak = int(st.miss_streak)
        p_play = float(play_rates.get(streak, 0.88 if streak == 0 else 0.35))
        p_play = float(np.clip(p_play, 0.0, 1.0))
        c_min = max(0.0, float(st.ema_clean_min))
        o_r = float(orapm_map.get(pid, PRIOR_ORAPM_REPLACEMENT))
        d_r = float(drapm_map.get(pid, PRIOR_DRAPM_REPLACEMENT))
        net_r = o_r + d_r
        c_pm = 0.0 if math.isnan(st.ema_clean_pm_per_min) else float(st.ema_clean_pm_per_min)
        b_pm = 0.0 if math.isnan(st.ema_box_pm_per_min) else float(st.ema_box_pm_per_min)

        exp_mins.append(p_play * c_min)
        orapms.append(o_r)
        drapms.append(d_r)
        clean_pms.append(c_pm)
        box_pms.append(b_pm)

        if streak >= 1:
            miss_m = (1.0 - p_play) * c_min
            missing_min += miss_m
            # Relative to replacement fill
            missing_rapm += 5.0 * (miss_m / 240.0) * max(
                0.0, net_r - (REPLACEMENT_FILL_ORAPM + REPLACEMENT_FILL_DRAPM)
            )

    sum_m = float(sum(exp_mins))
    if sum_m >= 240.0 and sum_m > 0.0:
        weights = [m / sum_m for m in exp_mins]
        fill_frac = 0.0
    else:
        weights = [m / 240.0 for m in exp_mins]
        fill_frac = max(0.0, (240.0 - sum_m) / 240.0)

    team_orapm = 5.0 * (
        sum(w * o for w, o in zip(weights, orapms)) + fill_frac * REPLACEMENT_FILL_ORAPM
    )
    team_drapm = 5.0 * (
        sum(w * d for w, d in zip(weights, drapms)) + fill_frac * REPLACEMENT_FILL_DRAPM
    )
    team_rapm = team_orapm + team_drapm
    scoring_tilt = team_orapm - team_drapm

    clean_pm_rot = sum(m * c for m, c in zip(exp_mins, clean_pms))
    box_pm_rot = sum(m * b for m, b in zip(exp_mins, box_pms))

    return {
        "rapm_rot": float(team_rapm),
        "orapm_rot": float(team_orapm),
        "drapm_rot": float(team_drapm),
        "rapm_scoring_tilt": float(scoring_tilt),
        "rapm_missing": float(missing_rapm),
        "clean_pm_rot": float(clean_pm_rot),
        "box_pm_rot": float(box_pm_rot),
        "clean_vs_box_pm_rot": float(clean_pm_rot - box_pm_rot),
        "clean_missing_min": float(missing_min),
    }


def build_leverage_and_rapm_features(
    markets_df: pd.DataFrame,
    summary_games_df: pd.DataFrame,
    summary_players_df: pd.DataFrame,
    summary_stints_df: pd.DataFrame,
    *,
    team_half_life: float = DEFAULT_TEAM_HL,
    player_half_life: float = DEFAULT_PLAYER_HL,
) -> pd.DataFrame:
    """Compute strictly pregame leverage/garbage-filtered & Stint RAPM features."""
    base = markets_df[
        [
            "season",
            "season_type",
            "game_id",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_tricode",
            "away_tricode",
            "home_score",
            "away_score",
            "is_neutral",
        ]
    ].copy()
    base["game_id"] = base["game_id"].astype(str)
    base["game_date_ts"] = pd.to_datetime(base["game_date"]).dt.normalize()

    sg = summary_games_df.copy()
    sg["game_id"] = sg["game_id"].astype(str)
    merged = base.merge(sg, on="game_id", how="inner").sort_values(
        ["game_date_ts", "game_id"]
    ).reset_index(drop=True)

    sp = summary_players_df.copy()
    sp["game_id"] = sp["game_id"].astype(str)
    players_by_game = {gid: grp for gid, grp in sp.groupby("game_id", sort=False)}

    ss = summary_stints_df.copy()
    ss["game_id"] = ss["game_id"].astype(str)
    stints_by_game = {gid: grp for gid, grp in ss.groupby("game_id", sort=False)}

    t_alpha = _ema_alpha(team_half_life)
    p_alpha = _ema_alpha(player_half_life)
    lg_alpha = _ema_alpha(80.0)

    tstates: dict[int, TeamCleanState] = {}
    pstates: dict[str, CleanPlayerState] = {}
    play_hist: list[tuple[int, int]] = []
    dates_hist: dict[int, list[pd.Timestamp]] = {}
    rapm_engine = ExpandingTwoWayRapm()

    lg_raw_ortg = 114.5
    lg_clean_ortg = 114.5
    lg_pace = 99.0
    lg_reg_total = 226.5
    lg_games = 0

    current_season: str | None = None
    out_rows: list[dict[str, Any]] = []

    for gdate_ts, day_df in merged.groupby("game_date_ts", sort=True):
        day_season = str(day_df.iloc[0]["season"])
        if current_season is not None and day_season != current_season:
            # Offseason regression-to-mean & RAPM decay; reset player miss streaks
            rapm_engine.decay(DEFAULT_OFFSEASON_RAPM_DECAY)
            carry = DEFAULT_OFFSEASON_TEAM_CARRY
            for tid, st in tstates.items():
                st.season_games = 0
                st.raw_ortg = carry * st.raw_ortg + (1.0 - carry) * lg_raw_ortg
                st.raw_drtg = carry * st.raw_drtg + (1.0 - carry) * lg_raw_ortg
                st.adj_raw_ortg = carry * st.adj_raw_ortg + (1.0 - carry) * lg_raw_ortg
                st.adj_raw_drtg = carry * st.adj_raw_drtg + (1.0 - carry) * lg_raw_ortg
                st.clean_ortg = carry * st.clean_ortg + (1.0 - carry) * lg_clean_ortg
                st.clean_drtg = carry * st.clean_drtg + (1.0 - carry) * lg_clean_ortg
                st.adj_clean_ortg = carry * st.adj_clean_ortg + (1.0 - carry) * lg_clean_ortg
                st.adj_clean_drtg = carry * st.adj_clean_drtg + (1.0 - carry) * lg_clean_ortg
                st.lev_ortg = carry * st.lev_ortg + (1.0 - carry) * lg_clean_ortg
                st.lev_drtg = carry * st.lev_drtg + (1.0 - carry) * lg_clean_ortg
                st.adj_lev_ortg = carry * st.adj_lev_ortg + (1.0 - carry) * lg_clean_ortg
                st.adj_lev_drtg = carry * st.adj_lev_drtg + (1.0 - carry) * lg_clean_ortg
                st.raw_pace = carry * st.raw_pace + (1.0 - carry) * lg_pace
                st.clean_pace = carry * st.clean_pace + (1.0 - carry) * lg_pace
                st.margin_3q_ema *= carry
                st.final_margin_ema *= carry
                st.q4_giveback_ema *= carry
            for st_p in pstates.values():
                st_p.miss_streak = 0
        current_season = day_season

        orapm_map, drapm_map = rapm_engine.current_ratings()
        play_rates = learn_play_rate(play_hist)

        # PHASE 1: Emit strictly pregame features for all games on gdate_ts
        for g in day_df.itertuples(index=False):
            hid = int(g.home_team_id)
            aid = int(g.away_team_id)
            hst = tstates.get(hid) or TeamCleanState(
                raw_ortg=lg_raw_ortg,
                raw_drtg=lg_raw_ortg,
                adj_raw_ortg=lg_raw_ortg,
                adj_raw_drtg=lg_raw_ortg,
                clean_ortg=lg_clean_ortg,
                clean_drtg=lg_clean_ortg,
                adj_clean_ortg=lg_clean_ortg,
                adj_clean_drtg=lg_clean_ortg,
                lev_ortg=lg_clean_ortg,
                lev_drtg=lg_clean_ortg,
                adj_lev_ortg=lg_clean_ortg,
                adj_lev_drtg=lg_clean_ortg,
                raw_pace=lg_pace,
                clean_pace=lg_pace,
                raw_total_ema=lg_reg_total + 1.5,
                reg_total_ema=lg_reg_total,
            )
            ast = tstates.get(aid) or TeamCleanState(
                raw_ortg=lg_raw_ortg,
                raw_drtg=lg_raw_ortg,
                adj_raw_ortg=lg_raw_ortg,
                adj_raw_drtg=lg_raw_ortg,
                clean_ortg=lg_clean_ortg,
                clean_drtg=lg_clean_ortg,
                adj_clean_ortg=lg_clean_ortg,
                adj_clean_drtg=lg_clean_ortg,
                lev_ortg=lg_clean_ortg,
                lev_drtg=lg_clean_ortg,
                adj_lev_ortg=lg_clean_ortg,
                adj_lev_drtg=lg_clean_ortg,
                raw_pace=lg_pace,
                clean_pace=lg_pace,
                raw_total_ema=lg_reg_total + 1.5,
                reg_total_ema=lg_reg_total,
            )

            h_rapm = _project_team_rotation_rapm(
                pstates,
                hid,
                orapm_map=orapm_map,
                drapm_map=drapm_map,
                play_rates=play_rates,
            )
            a_rapm = _project_team_rotation_rapm(
                pstates,
                aid,
                orapm_map=orapm_map,
                drapm_map=drapm_map,
                play_rates=play_rates,
            )

            h_raw_net = hst.adj_raw_ortg - hst.adj_raw_drtg
            a_raw_net = ast.adj_raw_ortg - ast.adj_raw_drtg
            h_clean_net = hst.adj_clean_ortg - hst.adj_clean_drtg
            a_clean_net = ast.adj_clean_ortg - ast.adj_clean_drtg
            h_lev_net = hst.adj_lev_ortg - hst.adj_lev_drtg
            a_lev_net = ast.adj_lev_ortg - ast.adj_lev_drtg

            raw_net_diff = h_raw_net - a_raw_net
            clean_net_diff = h_clean_net - a_clean_net
            lev_net_diff = h_lev_net - a_lev_net
            garb_distort_diff = clean_net_diff - raw_net_diff
            lev_distort_diff = lev_net_diff - raw_net_diff

            exp_clean_pace = 0.5 * (hst.clean_pace + ast.clean_pace)
            exp_raw_pace = 0.5 * (hst.raw_pace + ast.raw_pace)
            exp_home_clean_off = 0.5 * (hst.adj_clean_ortg + ast.adj_clean_drtg)
            exp_away_clean_off = 0.5 * (ast.adj_clean_ortg + hst.adj_clean_drtg)
            clean_expected_total = exp_clean_pace * (exp_home_clean_off + exp_away_clean_off) / 100.0
            raw_expected_total = exp_raw_pace * (
                0.5 * (hst.adj_raw_ortg + ast.adj_raw_drtg)
                + 0.5 * (ast.adj_raw_ortg + hst.adj_raw_drtg)
            ) / 100.0

            # Rest / schedule context strictly as-of gdate_ts
            h_dates = dates_hist.get(hid, [])
            a_dates = dates_hist.get(aid, [])
            h_rest = float(min((gdate_ts - h_dates[-1]).days, 7)) if h_dates else 4.0
            a_rest = float(min((gdate_ts - a_dates[-1]).days, 7)) if a_dates else 4.0
            h_sched = schedule_density_asof(pd.Series(h_dates, dtype="datetime64[ns]"), gdate_ts)
            a_sched = schedule_density_asof(pd.Series(a_dates, dtype="datetime64[ns]"), gdate_ts)

            out_rows.append(
                {
                    "season": str(g.season),
                    "game_id": str(g.game_id),
                    "home_prior_games": int(hst.games),
                    "away_prior_games": int(ast.games),
                    "home_season_games": int(hst.season_games),
                    "away_season_games": int(ast.season_games),
                    # Team Clean / Leverage / Raw Efficiency
                    "home_adj_raw_net": float(h_raw_net),
                    "away_adj_raw_net": float(a_raw_net),
                    "raw_net_diff": float(raw_net_diff),
                    "home_adj_clean_ortg": float(hst.adj_clean_ortg),
                    "away_adj_clean_ortg": float(ast.adj_clean_ortg),
                    "home_adj_clean_drtg": float(hst.adj_clean_drtg),
                    "away_adj_clean_drtg": float(ast.adj_clean_drtg),
                    "home_adj_clean_net": float(h_clean_net),
                    "away_adj_clean_net": float(a_clean_net),
                    "clean_net_diff": float(clean_net_diff),
                    "lev_net_diff": float(lev_net_diff),
                    "garbage_distortion_diff": float(garb_distort_diff),
                    "leverage_distortion_diff": float(lev_distort_diff),
                    # 3Q dominance & 4Q blowout giveback
                    "margin_3q_diff": float(hst.margin_3q_ema - ast.margin_3q_ema),
                    "q4_giveback_diff": float(hst.q4_giveback_ema - ast.q4_giveback_ema),
                    # Clean Pace & Total Inflation Signals
                    "home_clean_pace": float(hst.clean_pace),
                    "away_clean_pace": float(ast.clean_pace),
                    "exp_clean_pace": float(exp_clean_pace),
                    "exp_raw_pace": float(exp_raw_pace),
                    "clean_vs_raw_pace_diff": float(exp_clean_pace - exp_raw_pace),
                    "clean_expected_total": float(clean_expected_total),
                    "raw_expected_total": float(raw_expected_total),
                    "clean_vs_raw_total_gap": float(clean_expected_total - raw_expected_total),
                    "recent_ot_inflation": float(
                        (hst.raw_total_ema - hst.reg_total_ema)
                        + (ast.raw_total_ema - ast.reg_total_ema)
                    ),
                    "recent_endgame_fta_sum": float(hst.endgame_fta_ema + ast.endgame_fta_ema),
                    # Stint RAPM (2-way + net + missing)
                    "home_rapm_rot": h_rapm["rapm_rot"],
                    "away_rapm_rot": a_rapm["rapm_rot"],
                    "rapm_rot_diff": float(h_rapm["rapm_rot"] - a_rapm["rapm_rot"]),
                    "home_orapm_rot": h_rapm["orapm_rot"],
                    "away_orapm_rot": a_rapm["orapm_rot"],
                    "home_drapm_rot": h_rapm["drapm_rot"],
                    "away_drapm_rot": a_rapm["drapm_rot"],
                    "orapm_sum": float(h_rapm["orapm_rot"] + a_rapm["orapm_rot"]),
                    "drapm_sum": float(h_rapm["drapm_rot"] + a_rapm["drapm_rot"]),
                    "rapm_scoring_tilt_sum": float(
                        h_rapm["rapm_scoring_tilt"] + a_rapm["rapm_scoring_tilt"]
                    ),
                    "rapm_missing_diff": float(h_rapm["rapm_missing"] - a_rapm["rapm_missing"]),
                    "clean_pm_rot_diff": float(h_rapm["clean_pm_rot"] - a_rapm["clean_pm_rot"]),
                    "clean_vs_box_pm_rot_diff": float(
                        h_rapm["clean_vs_box_pm_rot"] - a_rapm["clean_vs_box_pm_rot"]
                    ),
                    "clean_missing_min_diff": float(
                        h_rapm["clean_missing_min"] - a_rapm["clean_missing_min"]
                    ),
                    # Team Foul / FTR / Style Profiles (used directly & for Referee Interactions)
                    "home_off_ftr": float(hst.off_ftr_ema),
                    "away_off_ftr": float(ast.off_ftr_ema),
                    "home_def_ftr": float(hst.def_ftr_ema),
                    "away_def_ftr": float(ast.def_ftr_ema),
                    "team_ftr_sum": float(
                        0.5 * (hst.off_ftr_ema + ast.def_ftr_ema)
                        + 0.5 * (ast.off_ftr_ema + hst.def_ftr_ema)
                    ),
                    "team_net_ftr_diff": float(
                        (hst.off_ftr_ema - hst.def_ftr_ema) - (ast.off_ftr_ema - ast.def_ftr_ema)
                    ),
                    "home_pf_rate": float(hst.pf_rate_ema),
                    "away_pf_rate": float(ast.pf_rate_ema),
                    "away_minus_home_pf_rate": float(ast.pf_rate_ema - hst.pf_rate_ema),
                    "team_pf_rate_sum": float(hst.pf_rate_ema + ast.pf_rate_ema),
                    "lg_clean_pace_asof": float(lg_pace),
                    # Rest / schedule
                    "rest_diff": float(h_rest - a_rest),
                    "b2b_diff": float((h_rest == 1.0) - (a_rest == 1.0)),
                    "b2b_sum": float((h_rest == 1.0) + (a_rest == 1.0)),
                    "three_in_four_diff": float(
                        h_sched["three_in_four"] - a_sched["three_in_four"]
                    ),
                    "games_last7_diff": float(h_sched["games_last_n"] - a_sched["games_last_n"]),
                }
            )

        # PHASE 2: Update states with completed games on gdate_ts
        rapm_engine.decay(DEFAULT_DAILY_RAPM_DECAY)
        for g in day_df.itertuples(index=False):
            gid = str(g.game_id)
            hid = int(g.home_team_id)
            aid = int(g.away_team_id)

            # Update Stint RAPM normal equations
            gstints = stints_by_game.get(gid)
            if gstints is not None and not gstints.empty:
                rapm_engine.add_game_stints(
                    gstints,
                    league_ortg=lg_clean_ortg,
                    is_neutral=bool(g.is_neutral),
                )

            # Update Player Clean States + miss-streak play history
            gplayers = players_by_game.get(gid)
            if gplayers is not None and not gplayers.empty:
                seen_pids: set[str] = set()
                played_pids: set[str] = set()
                for pr in gplayers.itertuples(index=False):
                    pid = str(pr.player_id)
                    if not pid:
                        continue
                    if not bool(pr.dnp) and float(pr.box_minutes) > 0.0:
                        played_pids.add(pid)
                for tid in (hid, aid):
                    for pid, pst in list(pstates.items()):
                        if pst.last_team_id == tid and pst.games > 0:
                            play_hist.append((int(pst.miss_streak), 1 if pid in played_pids else 0))
                for pr in gplayers.itertuples(index=False):
                    pid = str(pr.player_id)
                    if not pid:
                        continue
                    seen_pids.add(pid)
                    pst = pstates.get(pid) or CleanPlayerState()
                    # Map ESPN team_id to markets team_id (identical ESPN IDs)
                    pst.last_team_id = int(pr.team_id)
                    bmin = float(pr.box_minutes)
                    cmin = float(pr.clean_minutes)
                    if bool(pr.dnp) or bmin <= 0.0:
                        pst.miss_streak += 1
                    else:
                        c_pm_rate = float(pr.clean_pm) / max(cmin, 1.0) if cmin >= 0.5 else 0.0
                        b_pm_rate = float(pr.box_pm) / max(bmin, 1.0)
                        pst.ema_clean_min = _ema_step(pst.ema_clean_min, cmin, pst.games, p_alpha)
                        pst.ema_box_min = _ema_step(pst.ema_box_min, bmin, pst.games, p_alpha)
                        pst.ema_clean_pm_per_min = _ema_step(
                            pst.ema_clean_pm_per_min, c_pm_rate, pst.games, p_alpha
                        )
                        pst.ema_box_pm_per_min = _ema_step(
                            pst.ema_box_pm_per_min, b_pm_rate, pst.games, p_alpha
                        )
                        pst.games += 1
                        pst.miss_streak = 0
                    pstates[pid] = pst
                for pid, pst in list(pstates.items()):
                    if pid not in seen_pids and pst.last_team_id in (hid, aid) and pst.games > 0:
                        pst.miss_streak += 1

            # Update Team Clean States (simultaneous pre-game opponent ratings)
            hst = tstates.get(hid) or TeamCleanState()
            ast = tstates.get(aid) or TeamCleanState()
            prev_h_raw_off, prev_h_raw_def = hst.adj_raw_ortg, hst.adj_raw_drtg
            prev_a_raw_off, prev_a_raw_def = ast.adj_raw_ortg, ast.adj_raw_drtg
            prev_h_cln_off, prev_h_cln_def = hst.adj_clean_ortg, hst.adj_clean_drtg
            prev_a_cln_off, prev_a_cln_def = ast.adj_clean_ortg, ast.adj_clean_drtg
            prev_h_lev_off, prev_h_lev_def = hst.adj_lev_ortg, hst.adj_lev_drtg
            prev_a_lev_off, prev_a_lev_def = ast.adj_lev_ortg, ast.adj_lev_drtg

            h_r_ortg = float(g.home_raw_ortg)
            a_r_ortg = float(g.away_raw_ortg)
            h_c_ortg = float(g.home_clean_ortg)
            a_c_ortg = float(g.away_clean_ortg)
            h_l_ortg = float(g.home_lev_ortg)
            a_l_ortg = float(g.away_lev_ortg)

            # Home update
            nh = hst.games
            hst.raw_ortg = _ema_step(hst.raw_ortg, h_r_ortg, nh, t_alpha)
            hst.raw_drtg = _ema_step(hst.raw_drtg, a_r_ortg, nh, t_alpha)
            hst.adj_raw_ortg = _ema_step(
                hst.adj_raw_ortg, h_r_ortg + (lg_raw_ortg - prev_a_raw_def), nh, t_alpha
            )
            hst.adj_raw_drtg = _ema_step(
                hst.adj_raw_drtg, a_r_ortg + (prev_a_raw_off - lg_raw_ortg), nh, t_alpha
            )
            hst.clean_ortg = _ema_step(hst.clean_ortg, h_c_ortg, nh, t_alpha)
            hst.clean_drtg = _ema_step(hst.clean_drtg, a_c_ortg, nh, t_alpha)
            hst.adj_clean_ortg = _ema_step(
                hst.adj_clean_ortg, h_c_ortg + (lg_clean_ortg - prev_a_cln_def), nh, t_alpha
            )
            hst.adj_clean_drtg = _ema_step(
                hst.adj_clean_drtg, a_c_ortg + (prev_a_cln_off - lg_clean_ortg), nh, t_alpha
            )
            hst.lev_ortg = _ema_step(hst.lev_ortg, h_l_ortg, nh, t_alpha)
            hst.lev_drtg = _ema_step(hst.lev_drtg, a_l_ortg, nh, t_alpha)
            hst.adj_lev_ortg = _ema_step(
                hst.adj_lev_ortg, h_l_ortg + (lg_clean_ortg - prev_a_lev_def), nh, t_alpha
            )
            hst.adj_lev_drtg = _ema_step(
                hst.adj_lev_drtg, a_l_ortg + (prev_a_lev_off - lg_clean_ortg), nh, t_alpha
            )
            hst.raw_pace = _ema_step(hst.raw_pace, float(g.raw_pace), nh, t_alpha)
            hst.clean_pace = _ema_step(hst.clean_pace, float(g.clean_pace), nh, t_alpha)
            f_marg = float(g.home_score - g.away_score)
            hst.margin_3q_ema = _ema_step(hst.margin_3q_ema, float(g.margin_3q), nh, t_alpha)
            hst.final_margin_ema = _ema_step(hst.final_margin_ema, f_marg, nh, t_alpha)
            hst.q4_giveback_ema = _ema_step(
                hst.q4_giveback_ema, float(g.q4_giveback_home), nh, t_alpha
            )
            tot_pts = float(g.home_score + g.away_score)
            hst.raw_total_ema = _ema_step(hst.raw_total_ema, tot_pts, nh, t_alpha)
            hst.reg_total_ema = _ema_step(hst.reg_total_ema, float(g.reg_total_pts), nh, t_alpha)
            hst.ot_rate_ema = _ema_step(hst.ot_rate_ema, float(g.n_ot_periods > 0), nh, t_alpha)
            hst.endgame_fta_ema = _ema_step(
                hst.endgame_fta_ema, float(g.home_endgame_fta + g.away_endgame_fta), nh, t_alpha
            )
            h_ftr = float(g.home_fta) / max(float(g.home_fga), 1.0)
            a_ftr = float(g.away_fta) / max(float(g.away_fga), 1.0)
            hst.off_ftr_ema = _ema_step(hst.off_ftr_ema, h_ftr, nh, t_alpha)
            hst.def_ftr_ema = _ema_step(hst.def_ftr_ema, a_ftr, nh, t_alpha)
            hst.pf_rate_ema = _ema_step(hst.pf_rate_ema, float(g.home_pf), nh, t_alpha)
            hst.opp_pf_rate_ema = _ema_step(hst.opp_pf_rate_ema, float(g.away_pf), nh, t_alpha)
            hst.fg3_rate_ema = _ema_step(
                hst.fg3_rate_ema, float(g.home_fg3a) / max(float(g.home_fga), 1.0), nh, t_alpha
            )
            hst.games += 1
            hst.season_games += 1
            tstates[hid] = hst
            dates_hist.setdefault(hid, []).append(gdate_ts)

            # Away update
            na = ast.games
            ast.raw_ortg = _ema_step(ast.raw_ortg, a_r_ortg, na, t_alpha)
            ast.raw_drtg = _ema_step(ast.raw_drtg, h_r_ortg, na, t_alpha)
            ast.adj_raw_ortg = _ema_step(
                ast.adj_raw_ortg, a_r_ortg + (lg_raw_ortg - prev_h_raw_def), na, t_alpha
            )
            ast.adj_raw_drtg = _ema_step(
                ast.adj_raw_drtg, h_r_ortg + (prev_h_raw_off - lg_raw_ortg), na, t_alpha
            )
            ast.clean_ortg = _ema_step(ast.clean_ortg, a_c_ortg, na, t_alpha)
            ast.clean_drtg = _ema_step(ast.clean_drtg, h_c_ortg, na, t_alpha)
            ast.adj_clean_ortg = _ema_step(
                ast.adj_clean_ortg, a_c_ortg + (lg_clean_ortg - prev_h_cln_def), na, t_alpha
            )
            ast.adj_clean_drtg = _ema_step(
                ast.adj_clean_drtg, h_c_ortg + (prev_h_cln_off - lg_clean_ortg), na, t_alpha
            )
            ast.lev_ortg = _ema_step(ast.lev_ortg, a_l_ortg, na, t_alpha)
            ast.lev_drtg = _ema_step(ast.lev_drtg, h_l_ortg, na, t_alpha)
            ast.adj_lev_ortg = _ema_step(
                ast.adj_lev_ortg, a_l_ortg + (lg_clean_ortg - prev_h_lev_def), na, t_alpha
            )
            ast.adj_lev_drtg = _ema_step(
                ast.adj_lev_drtg, h_l_ortg + (prev_h_lev_off - lg_clean_ortg), na, t_alpha
            )
            ast.raw_pace = _ema_step(ast.raw_pace, float(g.raw_pace), na, t_alpha)
            ast.clean_pace = _ema_step(ast.clean_pace, float(g.clean_pace), na, t_alpha)
            ast.margin_3q_ema = _ema_step(ast.margin_3q_ema, -float(g.margin_3q), na, t_alpha)
            ast.final_margin_ema = _ema_step(ast.final_margin_ema, -f_marg, na, t_alpha)
            ast.q4_giveback_ema = _ema_step(
                ast.q4_giveback_ema, -float(g.q4_giveback_home), na, t_alpha
            )
            ast.raw_total_ema = _ema_step(ast.raw_total_ema, tot_pts, na, t_alpha)
            ast.reg_total_ema = _ema_step(ast.reg_total_ema, float(g.reg_total_pts), na, t_alpha)
            ast.ot_rate_ema = _ema_step(ast.ot_rate_ema, float(g.n_ot_periods > 0), na, t_alpha)
            ast.endgame_fta_ema = _ema_step(
                ast.endgame_fta_ema, float(g.home_endgame_fta + g.away_endgame_fta), na, t_alpha
            )
            ast.off_ftr_ema = _ema_step(ast.off_ftr_ema, a_ftr, na, t_alpha)
            ast.def_ftr_ema = _ema_step(ast.def_ftr_ema, h_ftr, na, t_alpha)
            ast.pf_rate_ema = _ema_step(ast.pf_rate_ema, float(g.away_pf), na, t_alpha)
            ast.opp_pf_rate_ema = _ema_step(ast.opp_pf_rate_ema, float(g.home_pf), na, t_alpha)
            ast.fg3_rate_ema = _ema_step(
                ast.fg3_rate_ema, float(g.away_fg3a) / max(float(g.away_fga), 1.0), na, t_alpha
            )
            ast.games += 1
            ast.season_games += 1
            tstates[aid] = ast
            dates_hist.setdefault(aid, []).append(gdate_ts)

            # Update trailing league baselines
            lg_raw_ortg = _ema_step(lg_raw_ortg, 0.5 * (h_r_ortg + a_r_ortg), lg_games, lg_alpha)
            lg_clean_ortg = _ema_step(
                lg_clean_ortg, 0.5 * (h_c_ortg + a_c_ortg), lg_games, lg_alpha
            )
            lg_pace = _ema_step(lg_pace, float(g.clean_pace), lg_games, lg_alpha)
            lg_reg_total = _ema_step(lg_reg_total, float(g.reg_total_pts), lg_games, lg_alpha)
            lg_games += 1

    return pd.DataFrame(out_rows)
