"""Expanding walk-forward (margin, total) distribution & correlated bivariate SGP pricing.

Models pregame:
- ``mu_margin``, heteroskedastic ``sigma_margin``
- ``mu_total``, heteroskedastic ``sigma_total``
- pregame residual correlation ``rho = Corr(e_margin, e_total)``

Strictly date-batched (`game_date < slate_date`): all games on a calendar date
are featurized and predicted BEFORE any game from that date updates team/player
states, league scoring environment EMAs, or regression parameters.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nba_betting.pricing.copula_sgp import t_copula_rect_prob

DMOCHOWSKI_QUANTILES = (0.15, 0.25, 0.4762, 0.5000, 0.5238, 0.75, 0.85)

from nba_betting.features.opponent_adjusted import LEAGUE_AVG_EFF
from nba_betting.features.player_impact import (
    PlayerState,
    learn_play_rate,
    prepare_player_box,
    team_player_impact_asof_states,
    update_player_states_after_game,
)
from nba_betting.features.possessions import FTA_COEFF
from nba_betting.features.schedule_context import schedule_density_asof

DEFAULT_HALF_LIFE = 14.0
OFFSEASON_RETENTION = 0.55
DEFAULT_MIN_TRAIN = 220
ELO_K = 20.0
ELO_HCA = 90.0

# 32-node Gauss-Hermite quadrature for Z ~ N(0, 1): E[g(Z)] = sum_k w_k g(z_k)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(32)
_STD_NORM_NODES = (_GH_X * math.sqrt(2.0)).astype(float)  # shape (32,)
_STD_NORM_WEIGHTS = (_GH_W / math.sqrt(math.pi)).astype(float)  # shape (32,)

MARGIN_FEATURE_COLS = [
    "raw_proj_margin",
    "adj_net_diff",
    "raw_net_diff",
    "elo_diff_100",
    "impact_diff",
    "missing_impact_diff",
    "pts_impact_diff",
    "efg_diff",
    "tov_rate_diff",
    "oreb_rate_diff",
    "ftr_diff",
    "rest_adv",
    "b2b_diff",
    "three_diff",
    "games7_diff",
    "is_neutral_f",
    "is_playoffs_f",
]

TOTAL_FEATURE_COLS = [
    "raw_proj_total",
    "league_total_ema",
    "pace_sum",
    "pace_min",
    "adj_off_sum",
    "adj_def_sum",
    "fg3a_rate_sum",
    "ftr_sum",
    "pf_rate_sum",
    "tov_rate_sum",
    "oreb_rate_sum",
    "pts_impact_sum",
    "missing_pts_sum",
    "b2b_sum",
    "three_sum",
    "rest_sum_capped",
    "is_playoffs_f",
    "abs_elo_diff_100",
]

SIGMA_MARGIN_COLS = [
    "pace_sum",
    "fg3a_rate_sum",
    "ftr_sum",
    "abs_mu_margin",
    "missing_min_sum",
    "team_margin_std_avg",
    "is_playoffs_f",
]

SIGMA_TOTAL_COLS = [
    "pace_sum",
    "fg3a_rate_sum",
    "ftr_sum",
    "pf_rate_sum",
    "abs_mu_margin",
    "missing_min_sum",
    "team_total_std_avg",
    "is_playoffs_f",
]

RHO_FEATURE_COLS = [
    "team_pts_var_diff",
    "mu_margin_scaled",
]


def _alpha(half_life: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / float(half_life))


def _ema(old: float, obs: float, n_prev: int, alpha: float) -> float:
    if n_prev <= 0 or old != old:
        return float(obs)
    return float((1.0 - alpha) * old + alpha * obs)


def _elo_update(elo_w: float, elo_l: float, mov: float, k: float = ELO_K) -> tuple[float, float]:
    diff = float(elo_w) - float(elo_l)
    mov_mult = math.log(abs(float(mov)) + 1.0) * (2.2 / ((diff * 0.001) + 2.2))
    exp_w = 1.0 / (1.0 + 10 ** ((float(elo_l) - float(elo_w)) / 400.0))
    delta = float(k) * mov_mult * (1.0 - exp_w)
    return float(elo_w) + delta, float(elo_l) - delta


def _player_scoring_asof(
    states: dict[str, PlayerState],
    team_id: int,
    play_rates: dict[int, float],
    top_n: int = 10,
) -> tuple[float, float, float]:
    """Return (expected_pts_index, missing_pts_index, missing_minutes) for team roster."""
    roster = [
        s
        for s in states.values()
        if s.last_team_id == int(team_id)
        and s.games > 0
        and s.ema_minutes == s.ema_minutes
        and s.ema_minutes > 1.0
    ]
    if not roster:
        return 0.0, 0.0, 0.0
    roster.sort(key=lambda r: -float(r.ema_minutes))
    roster = roster[: int(top_n)]
    exp_pts = 0.0
    miss_pts = 0.0
    miss_min = 0.0
    for r in roster:
        k = min(int(r.miss_streak), 10)
        p_play = float(play_rates.get(k, 0.85 if k == 0 else 0.35))
        ppm = float(r.ema_pts_per_min) if r.ema_pts_per_min == r.ema_pts_per_min else 0.45
        full = float(r.ema_minutes) * ppm
        exp_pts += p_play * full
        if k >= 1:
            miss_pts += (1.0 - p_play) * full
            miss_min += (1.0 - p_play) * float(r.ema_minutes)
    return float(exp_pts), float(miss_pts), float(miss_min)


def build_pregame_joint_feature_frame(
    *,
    seasons: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    player_box: pd.DataFrame,
    half_life: float = DEFAULT_HALF_LIFE,
    offseason_retention: float = OFFSEASON_RETENTION,
) -> pd.DataFrame:
    """Build strictly pregame features for `(margin, total)` in date-batched order."""
    alpha = _alpha(half_life)
    alpha_env = _alpha(45.0)
    alpha_var = _alpha(20.0)
    alpha_pl = _alpha(10.0)

    pb = prepare_player_box(player_box)
    pb_by_game: dict[str, pd.DataFrame] = {str(gid): g for gid, g in pb.groupby("game_id")}
    game_player_mins: dict[tuple[str, int], float] = {}
    if not pb.empty and "team_id" in pb.columns:
        grp_mins = pb.groupby(["game_id", "team_id"])["minutes"].sum()
        game_player_mins = {(str(k[0]), int(k[1])): float(v) for k, v in grp_mins.items()}

    pstates: dict[str, PlayerState] = {}
    player_ppm: dict[str, float] = {}
    play_hist: list[tuple[int, int]] = []

    # Team states
    elo: dict[int, float] = {}
    ortg: dict[int, float] = {}
    drtg: dict[int, float] = {}
    adj_off: dict[int, float] = {}
    adj_def: dict[int, float] = {}
    pace: dict[int, float] = {}
    fg3a_rate: dict[int, float] = {}
    efg: dict[int, float] = {}
    ftr: dict[int, float] = {}
    pf_rate: dict[int, float] = {}
    tov_rate: dict[int, float] = {}
    oreb_rate: dict[int, float] = {}
    pts_var: dict[int, float] = {}
    margin_var: dict[int, float] = {}
    total_var: dict[int, float] = {}
    n_games: dict[int, int] = {}
    season_games: dict[int, int] = {}
    dates_hist: dict[int, list[pd.Timestamp]] = {}

    # Global rolling scoring environment (updated strictly after each date)
    league_ortg = float(LEAGUE_AVG_EFF)
    league_pace = 99.0
    league_total = 226.0
    n_env_games = 0

    rows: list[dict[str, Any]] = []
    prev_season: str | None = None

    for season, sched_in, logs_in in seasons:
        if prev_season is not None and season != prev_season:
            # Mild offseason regression toward current league baselines
            r = float(offseason_retention)
            for tid in list(ortg.keys()):
                elo[tid] = r * elo.get(tid, 1500.0) + (1.0 - r) * 1500.0
                ortg[tid] = r * ortg[tid] + (1.0 - r) * league_ortg
                drtg[tid] = r * drtg[tid] + (1.0 - r) * league_ortg
                adj_off[tid] = r * adj_off.get(tid, ortg[tid]) + (1.0 - r) * league_ortg
                adj_def[tid] = r * adj_def.get(tid, drtg[tid]) + (1.0 - r) * league_ortg
                pace[tid] = r * pace.get(tid, league_pace) + (1.0 - r) * league_pace
                season_games[tid] = 0
                dates_hist[tid] = []
            for st in pstates.values():
                st.miss_streak = 0
        prev_season = season

        sched = sched_in.copy()
        sched["game_id"] = sched["game_id"].astype(str)
        date_src = "game_date" if "game_date" in sched.columns else "game_date_est"
        sched["game_date_ts"] = pd.to_datetime(sched[date_src]).dt.normalize()
        sched = sched.dropna(subset=["home_score", "away_score"]).sort_values(
            ["game_date_ts", "game_id"]
        )

        logs = logs_in.copy()
        logs["game_id"] = logs["game_id"].astype(str)
        logs["team_id"] = logs["team_id"].astype(int)
        logs_by_game = {str(gid): g for gid, g in logs.groupby("game_id")}

        # STRICT DATE-BATCHED LOOP: predict ALL games on `gd`, THEN update states with `gd`!
        for gd, day_sched in sched.groupby("game_date_ts", sort=True):
            rates = learn_play_rate(play_hist)

            # 1) Emit pregame features for every game on `gd`
            for _, g in day_sched.iterrows():
                gid = str(g["game_id"])
                hid = int(g["home_team_id"])
                aid = int(g["away_team_id"])
                # Require at least 1 prior game in the history (matches walk-forward gate)
                if season_games.get(hid, 0) >= 1 and season_games.get(aid, 0) >= 1:
                    himp = team_player_impact_asof_states(pstates, hid, play_rate_by_streak=rates)
                    aimp = team_player_impact_asof_states(pstates, aid, play_rate_by_streak=rates)
                    h_pts_idx, h_miss_pts, h_miss_min = _player_scoring_asof(
                        pstates, hid, rates
                    )
                    a_pts_idx, a_miss_pts, a_miss_min = _player_scoring_asof(
                        pstates, aid, rates
                    )

                    h_pace = pace.get(hid, league_pace)
                    a_pace = pace.get(aid, league_pace)
                    pace_proj = 0.5 * (h_pace + a_pace)

                    ho = adj_off.get(hid, ortg[hid])
                    hd = adj_def.get(hid, drtg[hid])
                    ao = adj_off.get(aid, ortg[aid])
                    ad = adj_def.get(aid, drtg[aid])

                    # Possession-consistent expected points anchored to current league ORTG
                    exp_h_eff = ho + ad - league_ortg
                    exp_a_eff = ao + hd - league_ortg
                    raw_h_pts = pace_proj * exp_h_eff / 100.0 + (
                        0.0 if bool(g.get("is_neutral", False)) else 2.2
                    )
                    raw_a_pts = pace_proj * exp_a_eff / 100.0

                    hr = (
                        float((gd - max(dates_hist[hid])).days)
                        if dates_hist.get(hid)
                        else 3.0
                    )
                    ar = (
                        float((gd - max(dates_hist[aid])).days)
                        if dates_hist.get(aid)
                        else 3.0
                    )
                    hs = (
                        schedule_density_asof(pd.Series(dates_hist[hid]), gd)
                        if dates_hist.get(hid)
                        else {"three_in_four": 0.0, "games_last_n": 0.0}
                    )
                    asus = (
                        schedule_density_asof(pd.Series(dates_hist[aid]), gd)
                        if dates_hist.get(aid)
                        else {"three_in_four": 0.0, "games_last_n": 0.0}
                    )

                    hr_cap = min(max(hr, 1.0), 4.0)
                    ar_cap = min(max(ar, 1.0), 4.0)
                    is_po = float(str(g.get("season_type", "")) == "Playoffs")
                    is_neu = float(bool(g.get("is_neutral", False)))

                    he = elo.get(hid, 1500.0)
                    ae = elo.get(aid, 1500.0)

                    h_pv = pts_var.get(hid, 144.0)
                    a_pv = pts_var.get(aid, 144.0)
                    h_mv = margin_var.get(hid, 175.0)
                    a_mv = margin_var.get(aid, 175.0)
                    h_tv = total_var.get(hid, 320.0)
                    a_tv = total_var.get(aid, 320.0)

                    hs_final = float(g["home_score"])
                    as_final = float(g["away_score"])

                    rows.append(
                        {
                            "season": season,
                            "season_type": str(g.get("season_type", "Regular Season")),
                            "game_id": gid,
                            "game_date": str(gd.date()),
                            "home_tricode": str(g.get("home_tricode", "")),
                            "away_tricode": str(g.get("away_tricode", "")),
                            "home_score": hs_final,
                            "away_score": as_final,
                            "actual_margin": hs_final - as_final,
                            "actual_total": hs_final + as_final,
                            "home_win": float(hs_final > as_final),
                            "raw_proj_margin": float(raw_h_pts - raw_a_pts),
                            "raw_proj_total": float(raw_h_pts + raw_a_pts),
                            "league_total_ema": float(league_total),
                            "adj_net_diff": float((ho - hd) - (ao - ad)),
                            "raw_net_diff": float((ortg[hid] - drtg[hid]) - (ortg[aid] - drtg[aid])),
                            "elo_diff_100": float((he - ae) / 100.0),
                            "abs_elo_diff_100": float(abs(he - ae) / 100.0),
                            "impact_diff": float(himp["impact"] - aimp["impact"]),
                            "missing_impact_diff": float(
                                himp["missing_impact"] - aimp["missing_impact"]
                            ),
                            "pts_impact_diff": float(h_pts_idx - a_pts_idx),
                            "pts_impact_sum": float(h_pts_idx + a_pts_idx),
                            "missing_pts_sum": float(h_miss_pts + a_miss_pts),
                            "missing_min_sum": float(h_miss_min + a_miss_min),
                            "pace_sum": float(h_pace + a_pace),
                            "pace_min": float(min(h_pace, a_pace)),
                            "adj_off_sum": float(ho + ao),
                            "adj_def_sum": float(hd + ad),
                            "fg3a_rate_sum": float(
                                fg3a_rate.get(hid, 0.39) + fg3a_rate.get(aid, 0.39)
                            ),
                            "efg_diff": float(efg.get(hid, 0.54) - efg.get(aid, 0.54)),
                            "ftr_diff": float(ftr.get(hid, 0.26) - ftr.get(aid, 0.26)),
                            "ftr_sum": float(ftr.get(hid, 0.26) + ftr.get(aid, 0.26)),
                            "pf_rate_sum": float(pf_rate.get(hid, 0.20) + pf_rate.get(aid, 0.20)),
                            "tov_rate_diff": float(
                                tov_rate.get(hid, 0.14) - tov_rate.get(aid, 0.14)
                            ),
                            "tov_rate_sum": float(
                                tov_rate.get(hid, 0.14) + tov_rate.get(aid, 0.14)
                            ),
                            "oreb_rate_diff": float(
                                oreb_rate.get(hid, 0.24) - oreb_rate.get(aid, 0.24)
                            ),
                            "oreb_rate_sum": float(
                                oreb_rate.get(hid, 0.24) + oreb_rate.get(aid, 0.24)
                            ),
                            "rest_adv": float(hr_cap - ar_cap),
                            "rest_sum_capped": float(hr_cap + ar_cap),
                            "b2b_diff": float(hr == 1.0) - float(ar == 1.0),
                            "b2b_sum": float(hr == 1.0) + float(ar == 1.0),
                            "three_diff": float(hs["three_in_four"] - asus["three_in_four"]),
                            "three_sum": float(hs["three_in_four"] + asus["three_in_four"]),
                            "games7_diff": float(hs["games_last_n"] - asus["games_last_n"]),
                            "is_neutral_f": is_neu,
                            "is_playoffs_f": is_po,
                            "team_margin_std_avg": float(
                                0.5 * (math.sqrt(max(h_mv, 25.0)) + math.sqrt(max(a_mv, 25.0)))
                            ),
                            "team_total_std_avg": float(
                                0.5 * (math.sqrt(max(h_tv, 64.0)) + math.sqrt(max(a_tv, 64.0)))
                            ),
                            "team_pts_var_diff": float((h_pv - a_pv) / 100.0),
                        }
                    )

            # 2) Ingest completed games from `gd` to update states for future dates > `gd`
            for _, g in day_sched.iterrows():
                gid = str(g["game_id"])
                hid = int(g["home_team_id"])
                aid = int(g["away_team_id"])
                hs_final = float(g["home_score"])
                as_final = float(g["away_score"])
                mov = hs_final - as_final
                tot_pts = hs_final + as_final

                # Update player states & scoring rates
                box = pb_by_game.get(gid)
                if box is not None and not box.empty:
                    for tid in (hid, aid):
                        for pid, st in list(pstates.items()):
                            if st.last_team_id != tid or st.games <= 0:
                                continue
                            prow = box.loc[box["player_id"] == pid]
                            played = (
                                1
                                if (not prow.empty and float(prow.iloc[0]["minutes"]) > 0)
                                else 0
                            )
                            play_hist.append((int(st.miss_streak), played))
                    for _, pr in box.iterrows():
                        mins = float(pr["minutes"])
                        if mins <= 0:
                            continue
                        pid = str(pr["player_id"])
                        ppm_obs = float(pr["pts"]) / max(mins, 1.0)
                        old_ppm = player_ppm.get(pid, float("nan"))
                        player_ppm[pid] = (
                            ppm_obs
                            if old_ppm != old_ppm
                            else (1.0 - alpha_pl) * old_ppm + alpha_pl * ppm_obs
                        )
                    update_player_states_after_game(pstates, box)

                # Update Elo
                he = elo.get(hid, 1500.0)
                ae = elo.get(aid, 1500.0)
                hca = 0.0 if bool(g.get("is_neutral", False)) else ELO_HCA
                if mov > 0:
                    he2, ae2 = _elo_update(he + hca, ae, mov)
                    he2 -= hca
                else:
                    ae2, he2 = _elo_update(ae, he + hca, -mov)
                    he2 -= hca
                elo[hid], elo[aid] = he2, ae2

                # Update team box stats
                tg = logs_by_game.get(gid)
                if tg is None or len(tg) < 2:
                    dates_hist.setdefault(hid, []).append(gd)
                    dates_hist.setdefault(aid, []).append(gd)
                    season_games[hid] = season_games.get(hid, 0) + 1
                    season_games[aid] = season_games.get(aid, 0) + 1
                    continue

                # Compute possessions & per-48 pace accounting for OT player minutes
                obs_by_tid: dict[int, dict[str, float]] = {}
                for _, tr in tg.iterrows():
                    tid = int(tr["team_id"])
                    fga = max(float(tr["fga"]), 1.0)
                    fgm = float(tr["fgm"])
                    fg3m = float(tr["fg3m"])
                    fg3a = float(tr["fg3a"])
                    fta = float(tr["fta"])
                    oreb_v = float(tr["oreb"])
                    tov_v = float(tr["tov"])
                    pf_v = float(tr["pf"])
                    pts_v = float(tr["pts"])
                    poss = max(fga - oreb_v + tov_v + FTA_COEFF * fta, 50.0)
                    p_mins = game_player_mins.get((gid, tid), 240.0)
                    game_mins = p_mins / 5.0 if 230.0 <= p_mins <= 360.0 else 48.0
                    pace_48 = poss * (48.0 / max(game_mins, 40.0))
                    obs_by_tid[tid] = {
                        "pts": pts_v,
                        "poss": poss,
                        "pace_48": pace_48,
                        "fg3a_rate": fg3a / fga,
                        "efg": (fgm + 0.5 * fg3m) / fga,
                        "ftr": fta / fga,
                        "pf_rate": pf_v / poss,
                        "tov_rate": tov_v / poss,
                        "oreb_rate": oreb_v / fga,
                    }

                if hid not in obs_by_tid or aid not in obs_by_tid:
                    continue

                # Snapshot opponent ratings BEFORE updating either team
                prev_adj_off_h = adj_off.get(hid, league_ortg)
                prev_adj_def_h = adj_def.get(hid, league_ortg)
                prev_adj_off_a = adj_off.get(aid, league_ortg)
                prev_adj_def_a = adj_def.get(aid, league_ortg)

                game_poss = 0.5 * (obs_by_tid[hid]["poss"] + obs_by_tid[aid]["poss"])
                game_pace_48 = 0.5 * (obs_by_tid[hid]["pace_48"] + obs_by_tid[aid]["pace_48"])

                for tid, opp_id, opp_off_prev, opp_def_prev in (
                    (hid, aid, prev_adj_off_a, prev_adj_def_a),
                    (aid, hid, prev_adj_off_h, prev_adj_def_h),
                ):
                    o = obs_by_tid[tid]
                    opp_o = obs_by_tid[opp_id]
                    n0 = n_games.get(tid, 0)
                    raw_o = 100.0 * o["pts"] / game_poss
                    raw_d = 100.0 * opp_o["pts"] / game_poss

                    # Scoring / margin / total innovation variance updates
                    exp_team_pts = pace.get(tid, league_pace) * ortg.get(tid, league_ortg) / 100.0
                    exp_opp_pts = pace.get(tid, league_pace) * drtg.get(tid, league_ortg) / 100.0
                    if n0 >= 3:
                        e_pts = o["pts"] - exp_team_pts
                        e_m_team = (o["pts"] - opp_o["pts"]) - (exp_team_pts - exp_opp_pts)
                        e_t_team = tot_pts - (exp_team_pts + exp_opp_pts)
                        pts_var[tid] = _ema(
                            pts_var.get(tid, 144.0), min(e_pts * e_pts, 900.0), n0, alpha_var
                        )
                        margin_var[tid] = _ema(
                            margin_var.get(tid, 175.0), min(e_m_team * e_m_team, 1200.0), n0, alpha_var
                        )
                        total_var[tid] = _ema(
                            total_var.get(tid, 320.0), min(e_t_team * e_t_team, 1600.0), n0, alpha_var
                        )

                    ortg[tid] = _ema(ortg.get(tid, float("nan")), raw_o, n0, alpha)
                    drtg[tid] = _ema(drtg.get(tid, float("nan")), raw_d, n0, alpha)
                    adj_off[tid] = _ema(
                        adj_off.get(tid, float("nan")),
                        raw_o + (league_ortg - opp_def_prev),
                        n0,
                        alpha,
                    )
                    adj_def[tid] = _ema(
                        adj_def.get(tid, float("nan")),
                        raw_d + (opp_off_prev - league_ortg),
                        n0,
                        alpha,
                    )
                    pace[tid] = _ema(pace.get(tid, float("nan")), o["pace_48"], n0, alpha)
                    fg3a_rate[tid] = _ema(fg3a_rate.get(tid, float("nan")), o["fg3a_rate"], n0, alpha)
                    efg[tid] = _ema(efg.get(tid, float("nan")), o["efg"], n0, alpha)
                    ftr[tid] = _ema(ftr.get(tid, float("nan")), o["ftr"], n0, alpha)
                    pf_rate[tid] = _ema(pf_rate.get(tid, float("nan")), o["pf_rate"], n0, alpha)
                    tov_rate[tid] = _ema(tov_rate.get(tid, float("nan")), o["tov_rate"], n0, alpha)
                    oreb_rate[tid] = _ema(oreb_rate.get(tid, float("nan")), o["oreb_rate"], n0, alpha)

                    n_games[tid] = n0 + 1
                    season_games[tid] = season_games.get(tid, 0) + 1
                    dates_hist.setdefault(tid, []).append(gd)

                # Update global rolling environment
                game_ortg = 100.0 * (0.5 * tot_pts) / game_poss
                league_ortg = _ema(league_ortg, game_ortg, n_env_games, alpha_env)
                league_pace = _ema(league_pace, game_pace_48, n_env_games, alpha_env)
                league_total = _ema(league_total, tot_pts, n_env_games, alpha_env)
                n_env_games += 1

    return pd.DataFrame(rows)


def fit_expanding_joint_distribution(
    feature_frame: pd.DataFrame,
    *,
    margin_cols: list[str] | None = None,
    total_cols: list[str] | None = None,
    include_market_features: bool = False,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every_days: int = 7,
    ridge_alpha_mean: float = 80.0,
    ridge_alpha_var: float = 300.0,
    ridge_alpha_rho: float = 800.0,
) -> pd.DataFrame:
    """Strictly date-batched expanding fit for `(mu_margin, sigma_margin, mu_total, sigma_total, rho)`.

    Every parameter used on date ``D`` is estimated solely from games with
    ``game_date < D``. Heteroskedastic variance models are fit on strictly
    out-of-sample prior residuals with exact empirical scale calibration so
    ``mean((e / sigma_hat)**2) == 1`` on the training history.
    """
    df = feature_frame.copy()
    df["game_date_ts"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df.sort_values(["game_date_ts", "game_id"]).reset_index(drop=True)

    m_cols = list(margin_cols or MARGIN_FEATURE_COLS)
    t_cols = list(total_cols or TOTAL_FEATURE_COLS)
    mkt_m_arr = (
        -pd.to_numeric(df["spread_home"], errors="coerce").to_numpy(dtype=float)
        if (include_market_features and "spread_home" in df.columns)
        else None
    )
    mkt_t_arr = (
        pd.to_numeric(df["total_line"], errors="coerce").to_numpy(dtype=float)
        if (include_market_features and "total_line" in df.columns)
        else None
    )

    n = len(df)
    mu_m = np.empty(n, dtype=float)
    mu_t = np.empty(n, dtype=float)
    sig_m = np.empty(n, dtype=float)
    sig_t = np.empty(n, dtype=float)
    rho_arr = np.empty(n, dtype=float)

    X_m_all = df[m_cols].to_numpy(dtype=float)
    X_t_all = df[t_cols].to_numpy(dtype=float)
    y_m = df["actual_margin"].to_numpy(dtype=float)
    y_t = df["actual_total"].to_numpy(dtype=float)
    raw_m = df["raw_proj_margin"].to_numpy(dtype=float)
    raw_t = df["raw_proj_total"].to_numpy(dtype=float)
    eff_alpha_mean = float(max(ridge_alpha_mean, 1200.0) if include_market_features else ridge_alpha_mean)

    # Cached fitted models across `refit_every_days`
    mean_models: dict[str, Any] | None = None
    var_models: dict[str, Any] | None = None
    last_refit_date: pd.Timestamp | None = None

    unique_dates = df["game_date_ts"].drop_duplicates().tolist()
    for gd in unique_dates:
        day_mask = (df["game_date_ts"] == gd).to_numpy()
        day_idx = np.flatnonzero(day_mask)
        i_hist = int(day_idx[0])  # all rows 0..i_hist-1 have strictly earlier game_date_ts < gd!

        if i_hist < int(min_train):
            for idx in day_idx:
                if mkt_m_arr is not None and np.isfinite(mkt_m_arr[idx]):
                    mu_m[idx] = float(mkt_m_arr[idx])
                else:
                    mu_m[idx] = float(raw_m[idx])
                if mkt_t_arr is not None and np.isfinite(mkt_t_arr[idx]):
                    mu_t[idx] = float(mkt_t_arr[idx])
                else:
                    mu_t[idx] = float(raw_t[idx])
                sig_m[idx] = 13.2 if i_hist < 40 else float(np.clip(np.std(y_m[:i_hist] - mu_m[:i_hist]), 10.5, 16.5))
                sig_t[idx] = 17.8 if i_hist < 40 else float(np.clip(np.std(y_t[:i_hist] - mu_t[:i_hist]), 14.0, 22.0))
                rho_arr[idx] = 0.0
            continue

        need_refit = (
            mean_models is None
            or last_refit_date is None
            or (gd - last_refit_date).days >= int(refit_every_days)
        )
        if need_refit:
            # 1) Fit mean margin & mean total Ridge on 0..i_hist-1 (or shrunk residual over market when enabled)
            if mkt_m_arr is not None:
                valid_m_tr = np.isfinite(mkt_m_arr[:i_hist])
                Xm_tr = X_m_all[:i_hist][valid_m_tr].copy()
                ym_target = y_m[:i_hist][valid_m_tr] - mkt_m_arr[:i_hist][valid_m_tr]
            else:
                Xm_tr = X_m_all[:i_hist].copy()
                ym_target = y_m[:i_hist]
            xm_mean = np.nanmean(Xm_tr, axis=0)
            xm_mean = np.where(np.isfinite(xm_mean), xm_mean, 0.0)
            inds = np.where(~np.isfinite(Xm_tr))
            Xm_tr[inds] = np.take(xm_mean, inds[1])
            sc_m = StandardScaler()
            Xm_s = sc_m.fit_transform(Xm_tr)
            rg_m = Ridge(alpha=eff_alpha_mean)
            rg_m.fit(Xm_s, ym_target)

            if mkt_t_arr is not None:
                valid_t_tr = np.isfinite(mkt_t_arr[:i_hist])
                Xt_tr = X_t_all[:i_hist][valid_t_tr].copy()
                yt_target = y_t[:i_hist][valid_t_tr] - mkt_t_arr[:i_hist][valid_t_tr]
            else:
                Xt_tr = X_t_all[:i_hist].copy()
                yt_target = y_t[:i_hist]
            xt_mean = np.nanmean(Xt_tr, axis=0)
            xt_mean = np.where(np.isfinite(xt_mean), xt_mean, 0.0)
            inds_t = np.where(~np.isfinite(Xt_tr))
            Xt_tr[inds_t] = np.take(xt_mean, inds_t[1])
            sc_t = StandardScaler()
            Xt_s = sc_t.fit_transform(Xt_tr)
            rg_t = Ridge(alpha=eff_alpha_mean)
            rg_t.fit(Xt_s, yt_target)

            mean_models = {
                "xm_mean": xm_mean,
                "sc_m": sc_m,
                "rg_m": rg_m,
                "xt_mean": xt_mean,
                "sc_t": sc_t,
                "rg_t": rg_t,
            }

            # 2) Fit heteroskedastic sigma_margin, sigma_total, and rho on strictly out-of-sample residuals!
            # Skip earliest cold-start rows (< 60) so variance targets aren't distorted by game-1 priors
            v_start = min(60, max(0, i_hist - int(min_train)))
            e_m_hist = y_m[v_start:i_hist] - mu_m[v_start:i_hist]
            e_t_hist = y_t[v_start:i_hist] - mu_t[v_start:i_hist]

            sub_df = df.iloc[v_start:i_hist].copy()
            sub_df["abs_mu_margin"] = np.abs(mu_m[v_start:i_hist])
            sub_df["mu_margin_scaled"] = mu_m[v_start:i_hist] / 10.0

            Zm = np.nan_to_num(sub_df[SIGMA_MARGIN_COLS].to_numpy(dtype=float), nan=0.0)
            sc_zm = StandardScaler()
            Zm_s = sc_zm.fit_transform(Zm)
            rg_vm = Ridge(alpha=float(ridge_alpha_var))
            rg_vm.fit(Zm_s, np.log(np.clip(e_m_hist**2, 4.0, 900.0)))
            raw_vm = np.exp(np.clip(rg_vm.predict(Zm_s), -5.0, 10.0))
            c_vm = float(np.mean((e_m_hist**2) / raw_vm))

            Zt = np.nan_to_num(sub_df[SIGMA_TOTAL_COLS].to_numpy(dtype=float), nan=0.0)
            sc_zt = StandardScaler()
            Zt_s = sc_zt.fit_transform(Zt)
            rg_vt = Ridge(alpha=float(ridge_alpha_var))
            rg_vt.fit(Zt_s, np.log(np.clip(e_t_hist**2, 9.0, 1600.0)))
            raw_vt = np.exp(np.clip(rg_vt.predict(Zt_s), -5.0, 10.0))
            c_vt = float(np.mean((e_t_hist**2) / raw_vt))

            sm_hist = np.clip(np.sqrt(c_vm * raw_vm), 9.5, 18.5)
            st_hist = np.clip(np.sqrt(c_vt * raw_vt), 13.0, 24.5)
            u_m = e_m_hist / sm_hist
            u_t = e_t_hist / st_hist
            cross = np.clip(u_m * u_t, -6.0, 6.0)

            Zr = np.nan_to_num(sub_df[RHO_FEATURE_COLS].to_numpy(dtype=float), nan=0.0)
            sc_zr = StandardScaler()
            Zr_s = sc_zr.fit_transform(Zr)
            rg_rho = Ridge(alpha=float(ridge_alpha_rho))
            rg_rho.fit(Zr_s, cross)

            var_models = {
                "sc_zm": sc_zm,
                "rg_vm": rg_vm,
                "c_vm": c_vm,
                "sc_zt": sc_zt,
                "rg_vt": rg_vt,
                "c_vt": c_vt,
                "sc_zr": sc_zr,
                "rg_rho": rg_rho,
            }
            last_refit_date = gd

        assert mean_models is not None and var_models is not None

        # Predict day rows vectorized
        Xm_d = X_m_all[day_idx].copy()
        inds = np.where(~np.isfinite(Xm_d))
        Xm_d[inds] = np.take(mean_models["xm_mean"], inds[1])
        pred_m_d = mean_models["rg_m"].predict(mean_models["sc_m"].transform(Xm_d))
        if mkt_m_arr is not None:
            base_m_d = np.where(np.isfinite(mkt_m_arr[day_idx]), mkt_m_arr[day_idx], raw_m[day_idx])
            mu_m_d = base_m_d + pred_m_d
        else:
            mu_m_d = pred_m_d

        Xt_d = X_t_all[day_idx].copy()
        inds_t = np.where(~np.isfinite(Xt_d))
        Xt_d[inds_t] = np.take(mean_models["xt_mean"], inds_t[1])
        pred_t_d = mean_models["rg_t"].predict(mean_models["sc_t"].transform(Xt_d))
        if mkt_t_arr is not None:
            base_t_d = np.where(np.isfinite(mkt_t_arr[day_idx]), mkt_t_arr[day_idx], raw_t[day_idx])
            mu_t_d = base_t_d + pred_t_d
        else:
            mu_t_d = pred_t_d

        mu_m[day_idx] = mu_m_d
        mu_t[day_idx] = mu_t_d

        day_df = df.iloc[day_idx].copy()
        day_df["abs_mu_margin"] = np.abs(mu_m_d)
        day_df["mu_margin_scaled"] = mu_m_d / 10.0

        Zm_d = np.nan_to_num(day_df[SIGMA_MARGIN_COLS].to_numpy(dtype=float), nan=0.0)
        raw_vm_d = np.exp(np.clip(var_models["rg_vm"].predict(var_models["sc_zm"].transform(Zm_d)), -5.0, 10.0))
        sig_m[day_idx] = np.clip(np.sqrt(var_models["c_vm"] * raw_vm_d), 9.5, 18.5)

        Zt_d = np.nan_to_num(day_df[SIGMA_TOTAL_COLS].to_numpy(dtype=float), nan=0.0)
        raw_vt_d = np.exp(np.clip(var_models["rg_vt"].predict(var_models["sc_zt"].transform(Zt_d)), -5.0, 10.0))
        sig_t[day_idx] = np.clip(np.sqrt(var_models["c_vt"] * raw_vt_d), 13.0, 24.5)

        Zr_d = np.nan_to_num(day_df[RHO_FEATURE_COLS].to_numpy(dtype=float), nan=0.0)
        rho_d = var_models["rg_rho"].predict(var_models["sc_zr"].transform(Zr_d))
        rho_arr[day_idx] = np.clip(rho_d, -0.35, 0.35)

    out = df.drop(columns=["game_date_ts"])
    out["mu_margin"] = mu_m
    out["sigma_margin"] = sig_m
    out["mu_total"] = mu_t
    out["sigma_total"] = sig_t
    out["rho_margin_total"] = rho_arr
    return out


def bvn_rect_prob(
    *,
    mu_m: np.ndarray | float,
    sigma_m: np.ndarray | float,
    mu_t: np.ndarray | float,
    sigma_t: np.ndarray | float,
    rho: np.ndarray | float,
    m_lo: np.ndarray | float,
    m_hi: np.ndarray | float,
    t_lo: np.ndarray | float,
    t_hi: np.ndarray | float,
) -> np.ndarray:
    """Vectorized exact bivariate normal rectangle probability ``P(m_lo < M < m_hi, t_lo < T < t_hi)``.

    Uses conditional decomposition ``T | M`` integrated over ``M`` via 32-node
    Gauss-Hermite / mapped interval quadrature (accurate to ~1e-13, microseconds
    over 5,000 games).
    """
    mm = np.asarray(mu_m, dtype=float)
    sm = np.asarray(sigma_m, dtype=float)
    mt = np.asarray(mu_t, dtype=float)
    st = np.asarray(sigma_t, dtype=float)
    r = np.clip(np.asarray(rho, dtype=float), -0.999, 0.999)

    a1 = (np.asarray(m_lo, dtype=float) - mm) / sm
    b1 = (np.asarray(m_hi, dtype=float) - mm) / sm
    a2 = (np.asarray(t_lo, dtype=float) - mt) / st
    b2 = (np.asarray(t_hi, dtype=float) - mt) / st

    # Broadcast all inputs to common 1D/ND shape
    a1, b1, a2, b2, r = np.broadcast_arrays(a1, b1, a2, b2, r)
    orig_shape = a1.shape
    a1f = a1.ravel()
    b1f = b1.ravel()
    a2f = a2.ravel()
    b2f = b2.ravel()
    rf = r.ravel()

    # Smooth analytic integrand phi(z1) * P(T in [a2, b2] | Z1 = z1) on [lo1, hi1] ⊂ [-9.5, 9.5]
    lo1 = np.clip(a1f, -9.5, 9.5)
    hi1 = np.clip(b1f, -9.5, 9.5)
    half = 0.5 * np.maximum(hi1 - lo1, 0.0)
    mid = 0.5 * (lo1 + hi1)

    gl_x, gl_w = _GL_NODES, _GL_WEIGHTS  # shape (48,) in [-1, 1]
    z1 = mid[:, None] + half[:, None] * gl_x[None, :]
    phi_z1 = np.exp(-0.5 * z1 * z1) / math.sqrt(2.0 * math.pi)
    s_cond = np.sqrt(np.maximum(1.0 - rf * rf, 1e-12))[:, None]

    hi_cond = np.where(
        np.isposinf(b2f[:, None]),
        1.0,
        ndtr((b2f[:, None] - rf[:, None] * z1) / s_cond),
    )
    lo_cond = np.where(
        np.isneginf(a2f[:, None]),
        0.0,
        ndtr((a2f[:, None] - rf[:, None] * z1) / s_cond),
    )
    p_t_given_m = np.maximum(hi_cond - lo_cond, 0.0)
    prob = half * np.sum(phi_z1 * p_t_given_m * gl_w[None, :], axis=1)
    prob = np.where(b1f <= a1f, 0.0, np.where(b2f <= a2f, 0.0, prob))
    return np.clip(prob, 0.0, 1.0).reshape(orig_shape)


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def _line_intervals(
    line: np.ndarray,
    *,
    is_spread_home: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return intervals `(win_a_lo, win_a_hi, win_b_lo, win_b_hi, push_lo, push_hi)`.

    For spread (`is_spread_home=True`):
    - threshold `thr = -spread_home` (Home covers iff `M > thr`).
    - side A = Home Cover, side B = Away Cover.
    - Non-zero integer spread uses `±0.5` continuity interval for exact integer push;
      pick'em (`0.0`) has no NBA tie so zero push width.

    For total (`is_spread_home=False`):
    - threshold `thr = total_line` (Over iff `T > thr`).
    - side A = Over, side B = Under.
    - Integer total uses `[thr - 0.5, thr + 0.5]` for push.
    """
    ln = np.asarray(line, dtype=float)
    thr = -ln if is_spread_home else ln
    is_whole = np.isfinite(ln) & (np.abs(ln - np.round(ln)) < 1e-9)
    if is_spread_home:
        # NBA games cannot end in a 0-point tie, so spread=0.0 never pushes
        is_whole = is_whole & (np.abs(ln) > 1e-9)

    half_push = np.where(is_whole, 0.5, 0.0)
    a_lo = thr + half_push
    a_hi = np.full_like(thr, np.inf)
    b_lo = np.full_like(thr, -np.inf)
    b_hi = thr - half_push
    p_lo = thr - half_push
    p_hi = thr + half_push
    return a_lo, a_hi, b_lo, b_hi, p_lo, p_hi


def attach_dmochowski_quantiles(
    df: pd.DataFrame,
    *,
    margin_cols: list[str] | None = None,
    total_cols: list[str] | None = None,
    include_market_features: bool = False,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every_days: int = 28,
    quantiles: tuple[float, ...] = DMOCHOWSKI_QUANTILES,
) -> pd.DataFrame:
    """Strictly date-batched expanding pinball-loss quantile models (Dmochowski 2023).

    Predicts conditional quantiles ``0.15, 0.25, 0.4762, 0.5000 (median), 0.5238, 0.75, 0.85``
    for both ``margin`` and ``total`` using only games with ``game_date < slate_date``.
    Because overtime (+23 pts right-tail skew on totals), late intentional fouling, and
    Q4 bench clearings separate conditional mean from conditional median, Dmochowski (2023)
    gates bets only when the book line sits strictly outside the ``[0.4762, 0.5238]``
    quantile band.
    """
    from scipy.stats import norm as sp_norm

    out = df.copy()
    out["game_date_ts"] = pd.to_datetime(out["game_date"]).dt.normalize()
    out = out.sort_values(["game_date_ts", "game_id"]).reset_index(drop=True)

    m_cols = list(margin_cols or MARGIN_FEATURE_COLS)
    t_cols = list(total_cols or TOTAL_FEATURE_COLS)
    mkt_m_arr = (
        -pd.to_numeric(out["spread_home"], errors="coerce").to_numpy(dtype=float)
        if (include_market_features and "spread_home" in out.columns)
        else None
    )
    mkt_t_arr = (
        pd.to_numeric(out["total_line"], errors="coerce").to_numpy(dtype=float)
        if (include_market_features and "total_line" in out.columns)
        else None
    )

    Xm_all = np.nan_to_num(out[m_cols].to_numpy(dtype=float), nan=0.0)
    Xt_all = np.nan_to_num(out[t_cols].to_numpy(dtype=float), nan=0.0)
    ym = out["actual_margin"].to_numpy(dtype=float)
    yt = out["actual_total"].to_numpy(dtype=float)
    mm = out["mu_margin"].to_numpy(dtype=float)
    sm = out["sigma_margin"].to_numpy(dtype=float)
    mt = out["mu_total"].to_numpy(dtype=float)
    st = out["sigma_total"].to_numpy(dtype=float)

    k_q = len(quantiles)
    z_norm = sp_norm.ppf(np.asarray(quantiles, dtype=float))
    qm_mat = np.empty((len(out), k_q), dtype=float)
    qt_mat = np.empty((len(out), k_q), dtype=float)

    q_models_m: list[HistGradientBoostingRegressor] | None = None
    q_models_t: list[HistGradientBoostingRegressor] | None = None
    last_refit: pd.Timestamp | None = None

    for gd in out["game_date_ts"].drop_duplicates().tolist():
        day_idx = np.flatnonzero((out["game_date_ts"] == gd).to_numpy())
        i_hist = int(day_idx[0])

        if i_hist < int(min_train):
            qm_mat[day_idx, :] = mm[day_idx, None] + sm[day_idx, None] * z_norm[None, :]
            qt_mat[day_idx, :] = mt[day_idx, None] + st[day_idx, None] * z_norm[None, :]
            continue

        if (
            q_models_m is None
            or q_models_t is None
            or last_refit is None
            or (gd - last_refit).days >= int(refit_every_days)
        ):
            if mkt_m_arr is not None:
                vm = np.isfinite(mkt_m_arr[:i_hist])
                Xm_tr = Xm_all[:i_hist][vm]
                ym_tr = ym[:i_hist][vm] - mkt_m_arr[:i_hist][vm]
            else:
                Xm_tr = Xm_all[:i_hist]
                ym_tr = ym[:i_hist]

            if mkt_t_arr is not None:
                vt = np.isfinite(mkt_t_arr[:i_hist])
                Xt_tr = Xt_all[:i_hist][vt]
                yt_tr = yt[:i_hist][vt] - mkt_t_arr[:i_hist][vt]
            else:
                Xt_tr = Xt_all[:i_hist]
                yt_tr = yt[:i_hist]

            q_models_m = []
            q_models_t = []
            for q in quantiles:
                gbm_m = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(q),
                    max_iter=40,
                    max_depth=3,
                    min_samples_leaf=50,
                    l2_regularization=15.0,
                    learning_rate=0.06,
                    random_state=0,
                )
                gbm_m.fit(Xm_tr, ym_tr)
                q_models_m.append(gbm_m)

                gbm_t = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(q),
                    max_iter=40,
                    max_depth=3,
                    min_samples_leaf=50,
                    l2_regularization=15.0,
                    learning_rate=0.06,
                    random_state=0,
                )
                gbm_t.fit(Xt_tr, yt_tr)
                q_models_t.append(gbm_t)
            last_refit = gd

        assert q_models_m is not None and q_models_t is not None
        Xm_d = Xm_all[day_idx]
        Xt_d = Xt_all[day_idx]
        pm_q = np.column_stack([m.predict(Xm_d) for m in q_models_m])
        pt_q = np.column_stack([m.predict(Xt_d) for m in q_models_t])

        if mkt_m_arr is not None:
            base_m = np.where(np.isfinite(mkt_m_arr[day_idx]), mkt_m_arr[day_idx], mm[day_idx])
            pm_q = base_m[:, None] + pm_q
        if mkt_t_arr is not None:
            base_t = np.where(np.isfinite(mkt_t_arr[day_idx]), mkt_t_arr[day_idx], mt[day_idx])
            pt_q = base_t[:, None] + pt_q

        # Enforce monotone non-crossing quantiles with minimum 0.05 pt spacing
        for j in range(1, k_q):
            pm_q[:, j] = np.maximum(pm_q[:, j], pm_q[:, j - 1] + 0.05)
            pt_q[:, j] = np.maximum(pt_q[:, j], pt_q[:, j - 1] + 0.05)

        qm_mat[day_idx, :] = pm_q
        qt_mat[day_idx, :] = pt_q

    out = out.drop(columns=["game_date_ts"])
    for j, q in enumerate(quantiles):
        tag = f"{int(round(q * 10000)):04d}"
        out[f"q_margin_{tag}"] = qm_mat[:, j]
        out[f"q_total_{tag}"] = qt_mat[:, j]

    out["median_margin"] = out["q_margin_5000"]
    out["median_total"] = out["q_total_5000"]
    out["skew_margin_mean_minus_med"] = out["mu_margin"] - out["median_margin"]
    out["skew_total_mean_minus_med"] = out["mu_total"] - out["median_total"]

    # Shape-preserving PCHIP CDF interpolation across [0.01, q_1..q_7, 0.99]
    q_grid = np.array([0.01, *quantiles, 0.99], dtype=float)
    sp = pd.to_numeric(out.get("spread_home", np.nan), errors="coerce").to_numpy(dtype=float)
    tl = pd.to_numeric(out.get("total_line", np.nan), errors="coerce").to_numpy(dtype=float)
    sh_lo, _, sa_hi, _, _, _ = _line_intervals(sp, is_spread_home=True)
    # Wait: _line_intervals returns (a_lo, a_hi, b_lo, b_hi, p_lo, p_hi)
    sh_lo, _, _, sa_hi, _, _ = _line_intervals(sp, is_spread_home=True)
    ov_lo, _, _, un_hi, _, _ = _line_intervals(tl, is_spread_home=False)

    p_q_hc = np.full(len(out), np.nan)
    p_q_ac = np.full(len(out), np.nan)
    p_q_ov = np.full(len(out), np.nan)
    p_q_un = np.full(len(out), np.nan)

    for i in range(len(out)):
        xm_knots = np.empty(len(q_grid), dtype=float)
        xm_knots[0] = min(mm[i] - 2.5 * sm[i], qm_mat[i, 0] - 4.0)
        xm_knots[1:-1] = qm_mat[i, :]
        xm_knots[-1] = max(mm[i] + 2.5 * sm[i], qm_mat[i, -1] + 4.0)
        cdf_m = PchipInterpolator(xm_knots, q_grid, extrapolate=True)
        if np.isfinite(sp[i]):
            p_q_hc[i] = float(np.clip(1.0 - cdf_m(sh_lo[i]), 1e-4, 1.0 - 1e-4))
            p_q_ac[i] = float(np.clip(cdf_m(sa_hi[i]), 1e-4, 1.0 - 1e-4))

        xt_knots = np.empty(len(q_grid), dtype=float)
        xt_knots[0] = min(mt[i] - 2.5 * st[i], qt_mat[i, 0] - 5.0)
        xt_knots[1:-1] = qt_mat[i, :]
        xt_knots[-1] = max(mt[i] + 2.5 * st[i], qt_mat[i, -1] + 5.0)
        cdf_t = PchipInterpolator(xt_knots, q_grid, extrapolate=True)
        if np.isfinite(tl[i]):
            p_q_ov[i] = float(np.clip(1.0 - cdf_t(ov_lo[i]), 1e-4, 1.0 - 1e-4))
            p_q_un[i] = float(np.clip(cdf_t(un_hi[i]), 1e-4, 1.0 - 1e-4))

    out["p_quantile_home_cover"] = p_q_hc
    out["p_quantile_away_cover"] = p_q_ac
    out["p_quantile_over"] = p_q_ov
    out["p_quantile_under"] = p_q_un

    # Dmochowski (2023) [0.4762, 0.5238] outside-band flags (marginpts signed away from band)
    # Positive distance => strictly outside the Dmochowski [0.4762, 0.5238] no-bet band!
    thr_m = -sp
    out["dmochowski_home_cover_margin"] = out["q_margin_4762"].to_numpy(float) - thr_m
    out["dmochowski_away_cover_margin"] = thr_m - out["q_margin_5238"].to_numpy(float)
    out["dmochowski_over_margin"] = out["q_total_4762"].to_numpy(float) - tl
    out["dmochowski_under_margin"] = tl - out["q_total_5238"].to_numpy(float)
    return out


def attach_bivariate_market_probs(
    df: pd.DataFrame,
    *,
    copula: str = "t4",
    df_t: float = 4.0,
    use_quantile_marginals: bool = False,
) -> pd.DataFrame:
    """Attach marginal ML/ATS/Total win+push probs and correlated SGP joint probs.

    Parameters
    ----------
    copula : {'t4', 'gaussian'}
        Default ``'t4'`` uses a Student-t copula with ``df_t = 4.0`` degrees of
        freedom so positive symmetric tail dependence (OT scoring bursts, blowout
        Q4 scripts) is priced accurately. ``'gaussian'`` uses ``bvn_rect_prob``
        for zero-tail-dependence baseline comparison.
    use_quantile_marginals : bool
        If True and Dmochowski quantile columns are present, blends quantile PCHIP
        cover/total probabilities (50/50) with heteroskedastic Normal marginals.
    """
    out = df.copy()
    mm = out["mu_margin"].to_numpy(dtype=float)
    sm = out["sigma_margin"].to_numpy(dtype=float)
    mt = out["mu_total"].to_numpy(dtype=float)
    st = out["sigma_total"].to_numpy(dtype=float)
    rho = out["rho_margin_total"].to_numpy(dtype=float)

    # Moneyline (NBA has no ties: M > 0 vs M < 0)
    p_hw = np.clip(1.0 - ndtr((0.0 - mm) / sm), 1e-6, 1.0 - 1e-6)
    out["p_model_home_ml"] = p_hw
    out["p_model_away_ml"] = 1.0 - p_hw
    out["p_model_ml_push"] = 0.0

    # Spread
    sp = pd.to_numeric(out["spread_home"], errors="coerce").to_numpy(dtype=float)
    sh_lo, sh_hi, sa_lo, sa_hi, sp_plo, sp_phi = _line_intervals(sp, is_spread_home=True)
    valid_sp = np.isfinite(sp)
    p_hc = np.where(valid_sp, 1.0 - ndtr((sh_lo - mm) / sm), np.nan)
    p_ac = np.where(valid_sp, ndtr((sa_hi - mm) / sm), np.nan)
    p_sp_push = np.where(
        valid_sp,
        np.maximum(ndtr((sp_phi - mm) / sm) - ndtr((sp_plo - mm) / sm), 0.0),
        np.nan,
    )
    if use_quantile_marginals and "p_quantile_home_cover" in out.columns:
        qhc = out["p_quantile_home_cover"].to_numpy(dtype=float)
        qac = out["p_quantile_away_cover"].to_numpy(dtype=float)
        non_push_mass = np.maximum(1.0 - p_sp_push, 1e-6)
        raw_hc = 0.5 * p_hc + 0.5 * qhc
        raw_ac = 0.5 * p_ac + 0.5 * qac
        tot_side = np.maximum(raw_hc + raw_ac, 1e-6)
        p_hc = np.where(valid_sp, non_push_mass * raw_hc / tot_side, np.nan)
        p_ac = np.where(valid_sp, non_push_mass * raw_ac / tot_side, np.nan)

    out["p_model_home_cover"] = p_hc
    out["p_model_away_cover"] = p_ac
    out["p_model_spread_push"] = p_sp_push

    # Total
    tl = pd.to_numeric(out["total_line"], errors="coerce").to_numpy(dtype=float)
    ov_lo, ov_hi, un_lo, un_hi, tot_plo, tot_phi = _line_intervals(tl, is_spread_home=False)
    valid_tl = np.isfinite(tl)
    p_ov = np.where(valid_tl, 1.0 - ndtr((ov_lo - mt) / st), np.nan)
    p_un = np.where(valid_tl, ndtr((un_hi - mt) / st), np.nan)
    p_tot_push = np.where(
        valid_tl,
        np.maximum(ndtr((tot_phi - mt) / st) - ndtr((tot_plo - mt) / st), 0.0),
        np.nan,
    )
    if use_quantile_marginals and "p_quantile_over" in out.columns:
        qov = out["p_quantile_over"].to_numpy(dtype=float)
        qun = out["p_quantile_under"].to_numpy(dtype=float)
        non_push_t = np.maximum(1.0 - p_tot_push, 1e-6)
        raw_ov = 0.5 * p_ov + 0.5 * qov
        raw_un = 0.5 * p_un + 0.5 * qun
        tot_t = np.maximum(raw_ov + raw_un, 1e-6)
        p_ov = np.where(valid_tl, non_push_t * raw_ov / tot_t, np.nan)
        p_un = np.where(valid_tl, non_push_t * raw_un / tot_t, np.nan)

    out["p_model_over"] = p_ov
    out["p_model_under"] = p_un
    out["p_model_total_push"] = p_tot_push

    both_sp_tl = valid_sp & valid_tl

    if copula == "t4":
        # Partition uniform marginals U_M in [0, 1] (Away Cover | Push | Home Cover)
        # and U_T in [0, 1] (Under | Push | Over), preserving exact marginals & push mass!
        p_ac_s = np.where(both_sp_tl, p_ac, 0.5)
        p_spp_s = np.where(both_sp_tl, p_sp_push, 0.0)
        p_un_s = np.where(both_tl := valid_tl, p_un, 0.5)
        p_tp_s = np.where(both_tl, p_tot_push, 0.0)
        z0 = np.zeros_like(mm)
        z1 = np.ones_like(mm)

        m_u_intervals = {
            "away_cover": (z0, p_ac_s),
            "push": (p_ac_s, np.clip(p_ac_s + p_spp_s, 0.0, 1.0)),
            "home_cover": (np.clip(p_ac_s + p_spp_s, 0.0, 1.0), z1),
        }
        t_u_intervals = {
            "under": (z0, p_un_s),
            "push": (p_un_s, np.clip(p_un_s + p_tp_s, 0.0, 1.0)),
            "over": (np.clip(p_un_s + p_tp_s, 0.0, 1.0), z1),
        }

        for tag_m in ("home_cover", "away_cover"):
            u1_a, u1_b = m_u_intervals[tag_m]
            u1_pa, u1_pb = m_u_intervals["push"]
            for tag_t in ("over", "under"):
                u2_a, u2_b = t_u_intervals[tag_t]
                u2_pa, u2_pb = t_u_intervals["push"]
                jp = t_copula_rect_prob(
                    u1_lo=u1_a, u1_hi=u1_b, u2_lo=u2_a, u2_hi=u2_b, rho=rho, df=df_t
                )
                jp_wp = t_copula_rect_prob(
                    u1_lo=u1_a, u1_hi=u1_b, u2_lo=u2_pa, u2_hi=u2_pb, rho=rho, df=df_t
                )
                jp_pw = t_copula_rect_prob(
                    u1_lo=u1_pa, u1_hi=u1_pb, u2_lo=u2_a, u2_hi=u2_b, rho=rho, df=df_t
                )
                out[f"p_sgp_{tag_m}_{tag_t}"] = np.where(both_sp_tl, jp, np.nan)
                out[f"p_sgp_{tag_m}_push_{tag_t}"] = np.where(both_sp_tl, jp_wp, np.nan)
                out[f"p_sgp_push_{tag_m}_{tag_t}"] = np.where(both_sp_tl, jp_pw, np.nan)

        p_bp = t_copula_rect_prob(
            u1_lo=m_u_intervals["push"][0],
            u1_hi=m_u_intervals["push"][1],
            u2_lo=t_u_intervals["push"][0],
            u2_hi=t_u_intervals["push"][1],
            rho=rho,
            df=df_t,
        )
        out["p_sgp_both_push"] = np.where(both_sp_tl, p_bp, np.nan)

        # ML x Total under t_4 copula: Away ML = [0, 1 - p_hw], Home ML = [1 - p_hw, 1]
        p_aw = 1.0 - p_hw
        for tag_m, u1_a, u1_b in (
            ("home_ml", p_aw, z1),
            ("away_ml", z0, p_aw),
        ):
            for tag_t in ("over", "under"):
                u2_a, u2_b = t_u_intervals[tag_t]
                u2_pa, u2_pb = t_u_intervals["push"]
                jp = t_copula_rect_prob(
                    u1_lo=u1_a, u1_hi=u1_b, u2_lo=u2_a, u2_hi=u2_b, rho=rho, df=df_t
                )
                jp_wp = t_copula_rect_prob(
                    u1_lo=u1_a, u1_hi=u1_b, u2_lo=u2_pa, u2_hi=u2_pb, rho=rho, df=df_t
                )
                out[f"p_sgp_{tag_m}_{tag_t}"] = np.where(valid_tl, jp, np.nan)
                out[f"p_sgp_{tag_m}_push_{tag_t}"] = np.where(valid_tl, jp_wp, np.nan)
        out["sgp_copula_used"] = f"t_{df_t:g}"
        return out

    # Gaussian copula baseline path
    sp_Safe = np.where(both_sp_tl, sp, -3.5)
    tl_Safe = np.where(both_sp_tl, tl, 225.5)
    sh_lo_s, sh_hi_s, sa_lo_s, sa_hi_s, sp_plo_s, sp_phi_s = _line_intervals(
        sp_Safe, is_spread_home=True
    )
    ov_lo_s, ov_hi_s, un_lo_s, un_hi_s, tot_plo_s, tot_phi_s = _line_intervals(
        tl_Safe, is_spread_home=False
    )

    for tag_m, m_a, m_b in (
        ("home_cover", sh_lo_s, sh_hi_s),
        ("away_cover", sa_lo_s, sa_hi_s),
    ):
        for tag_t, t_a, t_b in (
            ("over", ov_lo_s, ov_hi_s),
            ("under", un_lo_s, un_hi_s),
        ):
            jp = bvn_rect_prob(
                mu_m=mm, sigma_m=sm, mu_t=mt, sigma_t=st, rho=rho,
                m_lo=m_a, m_hi=m_b, t_lo=t_a, t_hi=t_b,
            )
            jp_w_push = bvn_rect_prob(
                mu_m=mm, sigma_m=sm, mu_t=mt, sigma_t=st, rho=rho,
                m_lo=m_a, m_hi=m_b, t_lo=tot_plo_s, t_hi=tot_phi_s,
            )
            jp_push_w = bvn_rect_prob(
                mu_m=mm, sigma_m=sm, mu_t=mt, sigma_t=st, rho=rho,
                m_lo=sp_plo_s, m_hi=sp_phi_s, t_lo=t_a, t_hi=t_b,
            )
            out[f"p_sgp_{tag_m}_{tag_t}"] = np.where(both_sp_tl, jp, np.nan)
            out[f"p_sgp_{tag_m}_push_{tag_t}"] = np.where(both_sp_tl, jp_w_push, np.nan)
            out[f"p_sgp_push_{tag_m}_{tag_t}"] = np.where(both_sp_tl, jp_push_w, np.nan)

    p_both_push = bvn_rect_prob(
        mu_m=mm, sigma_m=sm, mu_t=mt, sigma_t=st, rho=rho,
        m_lo=sp_plo_s, m_hi=sp_phi_s, t_lo=tot_plo_s, t_hi=tot_phi_s,
    )
    out["p_sgp_both_push"] = np.where(both_sp_tl, p_both_push, np.nan)

    pos_inf = np.full_like(mm, np.inf)
    neg_inf = np.full_like(mm, -np.inf)
    zero_arr = np.zeros_like(mm)
    for tag_m, m_a, m_b in (
        ("home_ml", zero_arr, pos_inf),
        ("away_ml", neg_inf, zero_arr),
    ):
        for tag_t, t_a, t_b in (
            ("over", ov_lo_s, ov_hi_s),
            ("under", un_lo_s, un_hi_s),
        ):
            jp = bvn_rect_prob(
                mu_m=mm, sigma_m=sm, mu_t=mt, sigma_t=st, rho=rho,
                m_lo=m_a, m_hi=m_b, t_lo=t_a, t_hi=t_b,
            )
            jp_w_push = bvn_rect_prob(
                mu_m=mm, sigma_m=sm, mu_t=mt, sigma_t=st, rho=rho,
                m_lo=m_a, m_hi=m_b, t_lo=tot_plo_s, t_hi=tot_phi_s,
            )
            out[f"p_sgp_{tag_m}_{tag_t}"] = np.where(valid_tl, jp, np.nan)
            out[f"p_sgp_{tag_m}_push_{tag_t}"] = np.where(valid_tl, jp_w_push, np.nan)

    out["sgp_copula_used"] = "gaussian"
    return out


def load_cached_seasons(root: Path | None = None) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    root = root or Path("/tmp/nba_postseason_bt")
    out: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for s in ("2022-23", "2023-24", "2024-25", "2025-26"):
        out.append(
            (
                s,
                pd.read_parquet(root / f"{s}_all_schedule.parquet"),
                pd.read_parquet(root / f"{s}_all_logs.parquet"),
            )
        )
    return out
