"""Strict walk-forward NBA player prop backtest & calibration engine (2022-26).

Evaluates five zero-leakage forecasting systems on real historical two-way prop lines:
1. ``book_rolling10_vig115``: Standard naive rolling-10 expectation + Negative Binomial
   dispersion priced at a symmetric -115/-115 vig benchmark (and constant 50/50 at line).
2. ``v15_rolling10_poisson``: Legacy v1.5 trailing-10 Poisson CDF.
3. ``structural_bimodal_raw``: Non-linear teammate-absence redistribution + spread/total
   conditioned bimodal blowout/OT Negative Binomial mixture (no market odds input).
4. ``structural_bimodal_calibrated``: Expanding-window Platt calibration per stat on
   strictly prior games (pure model, no book price input).
5. ``market_structural_alpha``: Expanding-window logistic regression combining de-vigged
   pregame implied probability with the structural blowout mixture shift, starter
   blowout Over-trap, bench garbage-time boost, non-linear absence innovation, and
   spot-starter regression. Refits strictly on prior games (zero lookahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from nba_betting.backtest.metrics import brier_score, log_loss
from nba_betting.features.prop_redistribution import RedistributionTracker
from nba_betting.models.prop_distribution import (
    DEFAULT_NB_ALPHA,
    p_over_nbinom,
    project_distributional_props,
    unimodal_p_over_from_rates,
)
from nba_betting.models.props import p_over_poisson
from nba_betting.pricing.american import devig_two_way
from nba_betting.pricing.kelly import american_to_decimal
from nba_betting.pricing.prop_parlay import (
    build_eligible_prop_legs,
    enumerate_prop_parlays,
    select_nonoverlapping_prop_parlays,
    simulate_prop_parlay_kelly,
)

STAT_TO_BOX_COL = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "pra": "pra",
}


@dataclass(frozen=True)
class PropCalibrationSummary:
    n: int
    brier: float
    log_loss: float
    hit_rate: float
    ece_10bin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "hit_rate": self.hit_rate,
            "ece_10bin": self.ece_10bin,
        }


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    yt = np.asarray(y, dtype=float)
    if len(p) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ids = np.digitize(p, bins[1:-1], right=False)
    ece = 0.0
    for b in range(int(n_bins)):
        mask = ids == b
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / len(p)) * abs(float(np.mean(p[mask]) - np.mean(yt[mask])))
    return float(ece)


def summarize_prop_calibration(probs: np.ndarray, y: np.ndarray) -> PropCalibrationSummary:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    yt = np.asarray(y, dtype=float)
    return PropCalibrationSummary(
        n=int(len(yt)),
        brier=float(brier_score(p, yt)),
        log_loss=float(log_loss(p, yt)),
        hit_rate=float(((p >= 0.5) == (yt == 1.0)).mean()) if len(yt) else 0.0,
        ece_10bin=expected_calibration_error(p, yt, n_bins=10),
    )


def build_walkforward_prop_frame(
    *,
    player_boxscores: pd.DataFrame,
    game_markets: pd.DataFrame,
    historical_props: pd.DataFrame,
    min_prior_games: int = 5,
    line_col: str = "open_line",
    over_price_col: str = "open_over_price",
    under_price_col: str = "open_under_price",
) -> pd.DataFrame:
    """Replay boxscores chronologically and score pregame props with zero leakage."""
    box = player_boxscores.copy()
    mkt = game_markets.copy()
    props = historical_props.dropna(subset=[line_col, over_price_col, under_price_col]).copy()

    for d in (box, mkt, props):
        d["game_id"] = d["game_id"].astype(str)
    for d in (box, props):
        d["player_id"] = d["player_id"].astype(str)

    mkt_map = mkt.set_index("game_id")[["spread_home", "total_line"]].to_dict("index")
    props_by_game = {gid: g for gid, g in props.groupby("game_id")}

    box = box.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)
    tracker = RedistributionTracker()
    min_col = "minutes" if "minutes" in box.columns else "min"

    rows: list[dict[str, Any]] = []
    for (gd, gid), gdf in box.groupby(["game_date", "game_id"], sort=True):
        minfo = mkt_map.get(str(gid))
        sp_h = (
            float(minfo["spread_home"])
            if minfo is not None and pd.notna(minfo.get("spread_home"))
            else 0.0
        )
        tot_l = (
            float(minfo["total_line"])
            if minfo is not None and pd.notna(minfo.get("total_line"))
            else 226.5
        )
        gp = props_by_game.get(str(gid))

        if gp is not None and not gp.empty:
            for tid, tdf in gdf.groupby("team_id"):
                tid_i = int(tid)
                opp_tid = int(tdf["opp_team_id"].iloc[0]) if "opp_team_id" in tdf.columns else None
                is_home = bool(tdf["is_home"].iloc[0]) if "is_home" in tdf.columns else True
                tsp = sp_h if is_home else -sp_h

                team_rates = tracker.project_team_pregame(
                    tid_i,
                    opp_tid,
                    gd,
                    team_spread=tsp,
                    total_line=tot_l,
                    min_games=min_prior_games,
                )
                active_rows = {
                    str(r["player_id"]): r
                    for _, r in tdf.iterrows()
                    if float(r.get(min_col, 0.0) or 0.0) > 0.0
                }
                t_props = gp.loc[gp["player_id"].isin(active_rows.keys())]

                for _, pr in t_props.iterrows():
                    pid = str(pr["player_id"])
                    rates = team_rates.get(pid)
                    if rates is None or rates.games < int(min_prior_games):
                        continue
                    stat = str(pr["stat"])
                    if stat not in STAT_TO_BOX_COL:
                        continue
                    roll = tracker.rolling10_expectation(pid, stat, window=10)
                    if roll is None:
                        continue

                    line = float(pr[line_col])
                    am_over = float(pr[over_price_col])
                    am_under = float(pr[under_price_col])
                    if am_over == 0 or am_under == 0:
                        continue
                    dec_ov = american_to_decimal(am_over)
                    dec_un = american_to_decimal(am_under)
                    overround = (1.0 / dec_ov) + (1.0 / dec_un)
                    if not (1.00 <= overround <= 1.15):
                        continue
                    fair_ov, _ = devig_two_way(am_over, am_under)

                    act_row = active_rows[pid]
                    pts_a = float(act_row.get("pts", 0.0) or 0.0)
                    reb_a = float(act_row.get("reb", 0.0) or 0.0)
                    ast_a = float(act_row.get("ast", 0.0) or 0.0)
                    fg3_a = float(act_row.get("fg3m", 0.0) or 0.0)
                    actual_val = {
                        "points": pts_a,
                        "rebounds": reb_a,
                        "assists": ast_a,
                        "threes": fg3_a,
                        "pra": pts_a + reb_a + ast_a,
                    }[stat]
                    if actual_val == line:
                        continue  # grade non-push decisions cleanly

                    dist_proj = project_distributional_props(
                        rates,
                        game_id=str(gid),
                        game_date=str(gd),
                        team_spread=tsp,
                        total_line=tot_l,
                    )
                    _, mu_roll10 = roll
                    p_pois_v15 = p_over_poisson(mu_roll10, line)
                    p_roll10_nb = p_over_nbinom(mu_roll10, line, nb_alpha=DEFAULT_NB_ALPHA[stat])
                    p_bi_raw = dist_proj.p_over(stat, line)
                    rate_pm = {
                        "points": rates.pts_per_min,
                        "rebounds": rates.reb_per_min,
                        "assists": rates.ast_per_min,
                        "threes": rates.fg3m_per_min,
                        "pra": rates.pra_per_min,
                    }[stat]
                    p_uni_raw = unimodal_p_over_from_rates(rates.adj_minutes, rate_pm, stat, line)

                    rows.append(
                        {
                            "season": str(pr["season"]),
                            "game_date": str(gd),
                            "game_id": str(gid),
                            "player_id": pid,
                            "player_name": rates.player_name,
                            "team_abbreviation": rates.team_abbreviation,
                            "market": str(pr.get("market") or stat),
                            "stat": stat,
                            "book": str(pr.get("book") or ""),
                            "line": line,
                            "book_over": am_over,
                            "book_under": am_under,
                            "book_implied_over_fair": float(fair_ov),
                            "overround": float(overround),
                            "actual": actual_val,
                            "y_over": 1.0 if actual_val > line else 0.0,
                            "exp_roll10": float(mu_roll10),
                            "model_exp": dist_proj.expected(stat),
                            "base_minutes": rates.base_minutes,
                            "adj_minutes": rates.adj_minutes,
                            "exp_minutes": dist_proj.exp_minutes,
                            "team_spread": float(tsp),
                            "total_line": float(tot_l),
                            "delta_missing_min": rates.delta_missing_min,
                            "delta_missing_usg": rates.delta_missing_usg,
                            "spot_starter_flag": rates.spot_starter_flag,
                            "bench_demotion_flag": rates.bench_demotion_flag,
                            "is_b2b": rates.is_b2b,
                            "is_home": 1.0 if is_home else 0.0,
                            "p_v15_poisson": float(p_pois_v15),
                            "p_roll10_nb_vig115": float(p_roll10_nb),
                            "p_structural_unimodal": float(p_uni_raw),
                            "p_structural_bimodal_raw": float(p_bi_raw),
                        }
                    )

        # Strictly postgame updates
        for tid, tdf in gdf.groupby("team_id"):
            opp_tid = int(tdf["opp_team_id"].iloc[0]) if "opp_team_id" in tdf.columns else None
            tracker.update_team_postgame(tdf, game_date=gd, opp_team_id=opp_tid)

    return pd.DataFrame(rows)


def _alpha_design_matrix(df: pd.DataFrame) -> np.ndarray:
    l_mkt = logit(np.clip(df["book_implied_over_fair"].to_numpy(float), 1e-4, 1.0 - 1e-4))
    l_bi = logit(np.clip(df["p_structural_bimodal_raw"].to_numpy(float), 1e-4, 1.0 - 1e-4))
    l_uni = logit(np.clip(df["p_structural_unimodal"].to_numpy(float), 1e-4, 1.0 - 1e-4))
    diff_bi = np.clip(l_bi - l_mkt, -1.5, 1.5)
    blow_shift = l_bi - l_uni

    tsp = df["team_spread"].to_numpy(float)
    abs_sp = np.abs(tsp)
    bm = df["base_minutes"].to_numpy(float)
    st_fav_blow = ((bm >= 28.0).astype(float)) * np.maximum(0.0, -tsp - 6.5) / 10.0
    st_dog_blow = ((bm >= 28.0).astype(float)) * np.maximum(0.0, tsp - 6.5) / 10.0
    bn_blow = ((bm < 22.0).astype(float)) * np.maximum(0.0, abs_sp - 8.0) / 10.0

    d_min = np.clip(df["delta_missing_min"].to_numpy(float) / 20.0, -1.5, 1.5)
    promo = df["spot_starter_flag"].to_numpy(float)
    tot_z = (df["total_line"].to_numpy(float) - 226.5) / 12.0

    is_pts = (df["stat"] == "points").astype(float).to_numpy()
    is_reb = (df["stat"] == "rebounds").astype(float).to_numpy()
    is_ast = (df["stat"] == "assists").astype(float).to_numpy()
    is_3pm = (df["stat"] == "threes").astype(float).to_numpy()
    is_pra = (df["stat"] == "pra").astype(float).to_numpy()

    return np.column_stack(
        [
            l_mkt,
            diff_bi,
            blow_shift,
            st_fav_blow,
            st_dog_blow,
            bn_blow,
            d_min,
            promo,
            tot_z,
            is_pts,
            is_reb,
            is_ast,
            is_3pm,
            is_pra,
        ]
    )


def fit_expanding_prop_calibrators(
    frame: pd.DataFrame,
    *,
    min_train_days: int = 25,
    refit_every_days: int = 10,
    C_alpha: float = 0.5,
) -> pd.DataFrame:
    """Attach strictly out-of-sample expanding Platt & Market+Structural Alpha probs.

    Refits by chronological ``game_date`` blocks so zero same-day labels ever leak.
    """
    if frame.empty:
        return frame.copy()
    df = frame.sort_values(["game_date", "game_id", "player_id", "stat"]).reset_index(drop=True)
    X_all = _alpha_design_matrix(df)
    l_raw = logit(np.clip(df["p_structural_bimodal_raw"].to_numpy(float), 1e-4, 1.0 - 1e-4))
    y_all = df["y_over"].to_numpy(int)

    p_platt = np.copy(df["p_structural_bimodal_raw"].to_numpy(float))
    p_alpha = np.copy(df["book_implied_over_fair"].to_numpy(float))

    dates = sorted(df["game_date"].unique())
    alpha_clf: LogisticRegression | None = None
    stat_clfs: dict[str, LogisticRegression] = {}
    last_fit_day_idx = -999

    for d_idx, gd in enumerate(dates):
        cur_mask = (df["game_date"] == gd).to_numpy()
        if d_idx < int(min_train_days):
            continue

        if alpha_clf is None or (d_idx - last_fit_day_idx) >= int(refit_every_days):
            past_mask = (df["game_date"] < gd).to_numpy()
            if np.sum(past_mask) >= 500 and len(np.unique(y_all[past_mask])) == 2:
                clf = LogisticRegression(C=float(C_alpha), max_iter=500)
                clf.fit(X_all[past_mask], y_all[past_mask])
                alpha_clf = clf

            for st_name in df["stat"].unique():
                st_past = past_mask & (df["stat"] == st_name).to_numpy()
                if np.sum(st_past) >= 200 and len(np.unique(y_all[st_past])) == 2:
                    s_clf = LogisticRegression(C=2.0, max_iter=300)
                    s_clf.fit(l_raw[st_past].reshape(-1, 1), y_all[st_past])
                    stat_clfs[str(st_name)] = s_clf
            last_fit_day_idx = d_idx

        if alpha_clf is not None:
            p_alpha[cur_mask] = alpha_clf.predict_proba(X_all[cur_mask])[:, 1]
        for st_name, s_clf in stat_clfs.items():
            st_cur = cur_mask & (df["stat"] == st_name).to_numpy()
            if np.any(st_cur):
                p_platt[st_cur] = s_clf.predict_proba(l_raw[st_cur].reshape(-1, 1))[:, 1]

    df["p_structural_bimodal_cal"] = np.clip(p_platt, 1e-5, 1.0 - 1e-5)
    df["model_p_over"] = np.clip(p_alpha, 1e-5, 1.0 - 1e-5)
    df["model_p_under"] = 1.0 - df["model_p_over"]
    return df


def evaluate_prop_parlay_slices(
    scored_frame: pd.DataFrame,
    *,
    sizes: tuple[int, ...] = (2, 3),
    min_price_edge: float = 0.04,
    max_per_day: int = 1,
    starting_bankroll: float = 50.0,
    kelly_fraction: float = 0.15,
    max_bet_frac: float = 0.03,
) -> dict[str, Any]:
    """Run 2-leg and 3-leg cross-game prop parlays + $50 Kelly per split."""
    legs = build_eligible_prop_legs(
        scored_frame,
        p_over_col="model_p_over",
        fair_over_col="book_implied_over_fair",
        over_am_col="book_over",
        under_am_col="book_under",
        y_over_col="y_over",
        require_vs_vigged=True,
    )

    slices = {
        "train_2022_24": ["2022-23", "2023-24"],
        "confirm_2024_25": ["2024-25"],
        "holdout_2025_26": ["2025-26"],
    }
    results: dict[str, Any] = {"n_eligible_legs_total": int(len(legs)), "by_leg_size": {}}

    for n_legs in sizes:
        cands = enumerate_prop_parlays(
            legs,
            sizes=(int(n_legs),),
            min_price_edge=min_price_edge,
            allow_same_game=False,
        )
        picked = select_nonoverlapping_prop_parlays(cands, max_per_day=max_per_day)
        leg_res: dict[str, Any] = {}
        for split_name, season_list in slices.items():
            sub_t = (
                picked.loc[picked["season"].isin(season_list)].copy()
                if not picked.empty
                else picked
            )
            _, summary = simulate_prop_parlay_kelly(
                sub_t,
                starting_bankroll=starting_bankroll,
                kelly_fraction=kelly_fraction,
                max_bet_frac=max_bet_frac,
            )
            leg_res[split_name] = summary
        # Also per-season breakdown
        by_season: dict[str, Any] = {}
        for s in ("2022-23", "2023-24", "2024-25", "2025-26"):
            sub_s = picked.loc[picked["season"] == s].copy() if not picked.empty else picked
            _, sum_s = simulate_prop_parlay_kelly(
                sub_s,
                starting_bankroll=starting_bankroll,
                kelly_fraction=kelly_fraction,
                max_bet_frac=max_bet_frac,
            )
            by_season[s] = sum_s
        leg_res["by_season"] = by_season
        results["by_leg_size"][str(int(n_legs))] = leg_res

    return results
