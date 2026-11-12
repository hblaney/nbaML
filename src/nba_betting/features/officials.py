"""Strictly-expanding pregame NBA referee crew profiles & team interactions.

NBA official crew assignments are published at 9:00 AM ET on game day and stored
in ESPN summary ``gameInfo.officials`` (``order=1`` Crew Chief, ``order=2`` Referee,
``order=3`` Umpire; ``order=4`` playoff alternate excluded).

Zero-leakage guarantees:
- Profiles for games on ``game_date`` use ONLY completed games with
  ``prior_game_date < game_date``.
- Game residuals are centered against trailing contemporary league means (plus
  pregame team foul/FTA expectations) so league-wide officiating shifts (such as
  the Feb 2024 NBA scoring/whistle shift) never create spurious referee bias.
- Empirical-Bayes (James-Stein style) pseudocount shrinkage pulls low-sample /
  G-League call-up officials smoothly toward 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_REF_DAILY_DECAY = 0.9990
KAPPA_FOUL = 35.0
KAPPA_PACE = 50.0
KAPPA_BIAS = 60.0
CREW_WEIGHTS = (1.20, 0.90, 0.90)  # Sums to 3.0; 40% Crew Chief, 30% Referee, 30% Umpire


def _ema_alpha(half_life: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / float(half_life))


def _ema_step(old: float, obs: float, n_prev: int, alpha: float) -> float:
    if n_prev <= 0 or math.isnan(old):
        return float(obs)
    return float((1.0 - alpha) * old + alpha * obs)


@dataclass
class OfficialProfile:
    eff_games: float = 0.0
    total_games: int = 0
    chief_games: int = 0
    sum_pf_res: float = 0.0
    sum_adj_pf_res: float = 0.0
    sum_fta_res: float = 0.0
    sum_adj_fta_res: float = 0.0
    sum_pace_res: float = 0.0
    sum_reg_tot_res: float = 0.0
    sum_tot_close_res: float = 0.0
    sum_home_foul_res: float = 0.0
    sum_home_fta_res: float = 0.0
    sum_home_ats_res: float = 0.0

    def decay(self, factor: float) -> None:
        self.eff_games *= factor
        self.sum_pf_res *= factor
        self.sum_adj_pf_res *= factor
        self.sum_fta_res *= factor
        self.sum_adj_fta_res *= factor
        self.sum_pace_res *= factor
        self.sum_reg_tot_res *= factor
        self.sum_tot_close_res *= factor
        self.sum_home_foul_res *= factor
        self.sum_home_fta_res *= factor
        self.sum_home_ats_res *= factor

    def shrunk(self, attr: str, kappa: float) -> float:
        n = self.eff_games
        if n <= 1e-6:
            return 0.0
        mean_val = float(getattr(self, attr)) / n
        return float((n / (n + kappa)) * mean_val)


def build_officials_and_interaction_features(
    markets_df: pd.DataFrame,
    summary_games_df: pd.DataFrame,
    leverage_features_df: pd.DataFrame,
    *,
    daily_decay: float = DEFAULT_REF_DAILY_DECAY,
    kappa_foul: float = KAPPA_FOUL,
    kappa_pace: float = KAPPA_PACE,
    kappa_bias: float = KAPPA_BIAS,
) -> pd.DataFrame:
    """Build pregame referee crew tendency profiles + team style interactions."""
    mkt_cols = [
        "season",
        "game_id",
        "game_date",
        "home_score",
        "away_score",
        "spread_home",
        "total_line",
        "is_neutral",
    ]
    base = markets_df[mkt_cols].copy()
    base["game_id"] = base["game_id"].astype(str)
    base["game_date_ts"] = pd.to_datetime(base["game_date"]).dt.normalize()

    sg = summary_games_df[
        [
            "game_id",
            "n_officials",
            "ref_1",
            "ref_2",
            "ref_3",
            "ref_1_display",
            "ref_2_display",
            "ref_3_display",
            "home_pf",
            "away_pf",
            "home_fta",
            "away_fta",
            "home_fga",
            "away_fga",
            "clean_pace",
            "reg_total_pts",
        ]
    ].copy()
    sg["game_id"] = sg["game_id"].astype(str)

    lf = leverage_features_df[
        [
            "season",
            "game_id",
            "team_ftr_sum",
            "team_net_ftr_diff",
            "home_pf_rate",
            "away_pf_rate",
            "away_minus_home_pf_rate",
            "team_pf_rate_sum",
            "home_clean_pace",
            "away_clean_pace",
            "lg_clean_pace_asof",
        ]
    ].copy()
    lf["game_id"] = lf["game_id"].astype(str)

    merged = (
        base.merge(sg, on="game_id", how="inner")
        .merge(lf, on=["season", "game_id"], how="inner")
        .sort_values(["game_date_ts", "game_id"])
        .reset_index(drop=True)
    )

    lg_alpha = _ema_alpha(120.0)
    lg_pf_tot = 40.0
    lg_fta_tot = 45.0
    lg_ftr_sum = 0.52
    lg_pace = 99.0
    lg_reg_tot = 226.5
    lg_home_foul_adv = 0.5  # away_pf - home_pf baseline
    lg_home_fta_adv = 1.0  # home_fta - away_fta baseline
    lg_games = 0

    profiles: dict[str, OfficialProfile] = {}
    out_rows: list[dict[str, Any]] = []

    for gdate_ts, day_df in merged.groupby("game_date_ts", sort=True):
        # PHASE 1: Compute strictly pregame crew features for all games on gdate_ts
        for g in day_df.itertuples(index=False):
            refs = [r for r in (str(g.ref_1), str(g.ref_2), str(g.ref_3)) if r]
            has_offs = len(refs) > 0

            crew_pf = 0.0
            crew_adj_pf = 0.0
            crew_fta = 0.0
            crew_adj_fta = 0.0
            crew_pace = 0.0
            crew_reg_tot = 0.0
            crew_tot_close = 0.0
            crew_home_foul = 0.0
            crew_home_fta = 0.0
            crew_home_ats = 0.0
            chief_adj_fta = 0.0
            chief_home_foul = 0.0
            crew_exp_games = 0.0

            if has_offs:
                w_raw = CREW_WEIGHTS[: len(refs)]
                w_scale = 3.0 / sum(w_raw)
                weights = [w * w_scale for w in w_raw]

                for idx, (rname, w) in enumerate(zip(refs, weights)):
                    prof = profiles.get(rname)
                    if prof is None:
                        continue
                    pf_s = prof.shrunk("sum_pf_res", kappa_foul)
                    apf_s = prof.shrunk("sum_adj_pf_res", kappa_foul)
                    fta_s = prof.shrunk("sum_fta_res", kappa_foul)
                    afta_s = prof.shrunk("sum_adj_fta_res", kappa_foul)
                    pace_s = prof.shrunk("sum_pace_res", kappa_pace)
                    rtot_s = prof.shrunk("sum_reg_tot_res", kappa_pace)
                    tcl_s = prof.shrunk("sum_tot_close_res", kappa_pace)
                    hf_s = (
                        0.0
                        if bool(g.is_neutral)
                        else prof.shrunk("sum_home_foul_res", kappa_bias)
                    )
                    hfta_s = (
                        0.0
                        if bool(g.is_neutral)
                        else prof.shrunk("sum_home_fta_res", kappa_bias)
                    )
                    hats_s = (
                        0.0
                        if bool(g.is_neutral)
                        else prof.shrunk("sum_home_ats_res", kappa_bias)
                    )

                    crew_pf += w * pf_s
                    crew_adj_pf += w * apf_s
                    crew_fta += w * fta_s
                    crew_adj_fta += w * afta_s
                    crew_pace += w * pace_s
                    crew_reg_tot += w * rtot_s
                    crew_tot_close += w * tcl_s
                    crew_home_foul += w * hf_s
                    crew_home_fta += w * hfta_s
                    crew_home_ats += w * hats_s
                    crew_exp_games += float(prof.total_games) / len(refs)

                    if idx == 0:
                        chief_adj_fta = afta_s
                        chief_home_foul = hf_s

            # Centered team style factors for orthogonal interaction terms
            ftr_sum_c = float(g.team_ftr_sum) - lg_ftr_sum
            pf_sum_c = float(g.team_pf_rate_sum) - lg_pf_tot
            away_pf_c = float(g.away_pf_rate) - 0.5 * lg_pf_tot
            pace_sum_c = (float(g.home_clean_pace) + float(g.away_clean_pace)) - 2.0 * float(
                g.lg_clean_pace_asof
            )

            out_rows.append(
                {
                    "season": str(g.season),
                    "game_id": str(g.game_id),
                    "has_officials": float(has_offs),
                    "ref_1": str(g.ref_1),
                    "ref_2": str(g.ref_2),
                    "ref_3": str(g.ref_3),
                    "crew_exp_games": float(crew_exp_games),
                    # Main crew lifts (shrunk, contemporary-league centered)
                    "crew_foul_lift": float(crew_pf),
                    "crew_adj_foul_lift": float(crew_adj_pf),
                    "crew_fta_lift": float(crew_fta),
                    "crew_adj_fta_lift": float(crew_adj_fta),
                    "crew_pace_lift": float(crew_pace),
                    "crew_reg_total_lift": float(crew_reg_tot),
                    "crew_total_vs_close_lift": float(crew_tot_close),
                    "crew_home_foul_bias": float(crew_home_foul),
                    "crew_home_fta_bias": float(crew_home_fta),
                    "crew_home_ats_bias": float(crew_home_ats),
                    "chief_adj_fta_lift": float(chief_adj_fta),
                    "chief_home_foul_bias": float(chief_home_foul),
                    # Team x Crew Interactions
                    "ix_crew_fta_x_ftr_sum": float(crew_adj_fta * ftr_sum_c * 10.0),
                    "ix_crew_foul_x_pf_sum": float(crew_adj_pf * pf_sum_c * 0.1),
                    "ix_crew_fta_x_net_ftr_diff": float(
                        crew_adj_fta * float(g.team_net_ftr_diff) * 10.0
                    ),
                    "ix_crew_foul_x_pf_diff": float(
                        crew_adj_pf * float(g.away_minus_home_pf_rate) * 0.1
                    ),
                    "ix_crew_home_bias_x_away_pf": float(crew_home_foul * away_pf_c * 0.2),
                    "ix_crew_pace_x_team_pace": float(crew_pace * pace_sum_c * 0.1),
                }
            )

        # PHASE 2: Decay and update official profiles from completed games on gdate_ts
        for prof in profiles.values():
            prof.decay(daily_decay)

        for g in day_df.itertuples(index=False):
            pf_tot = float(g.home_pf + g.away_pf)
            fta_tot = float(g.home_fta + g.away_fta)
            pace_val = float(g.clean_pace)
            reg_tot = float(g.reg_total_pts)
            actual_tot = float(g.home_score + g.away_score)
            actual_marg = float(g.home_score - g.away_score)

            pf_res = pf_tot - lg_pf_tot
            # Subtract team pregame foul expectation (shrunk 50% toward league mean)
            exp_team_pf = 0.5 * float(g.team_pf_rate_sum) + 0.5 * lg_pf_tot
            adj_pf_res = pf_tot - exp_team_pf

            fta_res = fta_tot - lg_fta_tot
            fga_tot = max(float(g.home_fga + g.away_fga), 140.0)
            exp_team_fta = 0.5 * (0.5 * float(g.team_ftr_sum) + 0.5 * lg_ftr_sum) * fga_tot
            adj_fta_res = fta_tot - exp_team_fta

            pace_res = pace_val - lg_pace
            reg_tot_res = reg_tot - lg_reg_tot

            tline = float(g.total_line) if pd.notna(g.total_line) else actual_tot
            tot_close_res = float(np.clip(actual_tot - tline, -35.0, 35.0))

            home_foul_adv = float(g.away_pf - g.home_pf)
            home_fta_adv = float(g.home_fta - g.away_fta)
            hf_res = 0.0 if bool(g.is_neutral) else (home_foul_adv - lg_home_foul_adv)
            hfta_res = 0.0 if bool(g.is_neutral) else (home_fta_adv - lg_home_fta_adv)

            shome = float(g.spread_home) if pd.notna(g.spread_home) else -actual_marg
            hats_res = (
                0.0
                if bool(g.is_neutral)
                else float(np.clip(actual_marg + shome, -28.0, 28.0))
            )

            refs = [r for r in (str(g.ref_1), str(g.ref_2), str(g.ref_3)) if r]
            for idx, rname in enumerate(refs):
                prof = profiles.get(rname)
                if prof is None:
                    prof = OfficialProfile()
                    profiles[rname] = prof
                prof.eff_games += 1.0
                prof.total_games += 1
                if idx == 0:
                    prof.chief_games += 1
                prof.sum_pf_res += pf_res
                prof.sum_adj_pf_res += adj_pf_res
                prof.sum_fta_res += fta_res
                prof.sum_adj_fta_res += adj_fta_res
                prof.sum_pace_res += pace_res
                prof.sum_reg_tot_res += reg_tot_res
                prof.sum_tot_close_res += tot_close_res
                prof.sum_home_foul_res += hf_res
                prof.sum_home_fta_res += hfta_res
                prof.sum_home_ats_res += hats_res

            # Update trailing contemporary league baselines (30-game half-life tracks mid-season rule/whistle shifts)
            lg_fast_alpha = _ema_alpha(30.0)
            lg_pf_tot = _ema_step(lg_pf_tot, pf_tot, lg_games, lg_fast_alpha)
            lg_fta_tot = _ema_step(lg_fta_tot, fta_tot, lg_games, lg_fast_alpha)
            obs_ftr_sum = float(g.home_fta) / max(float(g.home_fga), 1.0) + float(
                g.away_fta
            ) / max(float(g.away_fga), 1.0)
            lg_ftr_sum = _ema_step(lg_ftr_sum, obs_ftr_sum, lg_games, lg_fast_alpha)
            lg_pace = _ema_step(lg_pace, pace_val, lg_games, lg_fast_alpha)
            lg_reg_tot = _ema_step(lg_reg_tot, reg_tot, lg_games, lg_fast_alpha)
            if not bool(g.is_neutral):
                lg_home_foul_adv = _ema_step(
                    lg_home_foul_adv, home_foul_adv, lg_games, lg_alpha
                )
                lg_home_fta_adv = _ema_step(
                    lg_home_fta_adv, home_fta_adv, lg_games, lg_alpha
                )
            lg_games += 1

    return pd.DataFrame(out_rows)
