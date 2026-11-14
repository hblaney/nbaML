"""HoopVantage-inspired expanding margin model (no odds).

Walk-forward Ridge predicts home margin from prior-box player impact,
miss-streak availability, opponent-adjusted team nets, 538-style Elo,
and schedule/rest features; win probability is ``Φ(margin / σ)``.

Confirm-gated on 2024-25 vs EMA-16+rest+Platt (0.20931); pooled beats 0.21447.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from nba_betting.features.opponent_adjusted import LEAGUE_AVG_EFF
from nba_betting.features.player_impact import (
    PlayerState,
    learn_play_rate,
    prepare_player_box,
    team_player_impact_asof_states,
    update_player_states_after_game,
)
from nba_betting.features.schedule_context import schedule_density_asof
from nba_betting.features.team_ratings import prepare_team_game_rows

DEFAULT_SIGMA = 11.8
DEFAULT_RIDGE_ALPHA = 2.0
DEFAULT_MIN_TRAIN = 250
DEFAULT_REFIT_EVERY = 50
DEFAULT_EMA_HL = 16.0
ELO_K = 20.0
ELO_HCA = 100.0
FEATURE_COLS = [
    "impact_diff",
    "missing_impact_diff",
    "exp_min_diff",
    "adj_net_diff",
    "net_diff",
    "elo_diff",
    "rest_diff",
    "b2b_diff",
    "three_diff",
    "games7_diff",
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


def build_player_impact_feature_frame(
    *,
    seasons: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    player_box: pd.DataFrame,
    half_life: float = DEFAULT_EMA_HL,
) -> pd.DataFrame:
    """Chronological as-of feature rows for (season, game_id). No future boxes."""
    alpha = _alpha(half_life)
    league = float(LEAGUE_AVG_EFF)
    pb = prepare_player_box(player_box)
    pb_by_game = {str(gid): g for gid, g in pb.groupby("game_id")}

    pstates: dict[str, PlayerState] = {}
    play_hist: list[tuple[int, int]] = []
    elo: dict[int, float] = {}
    ortg: dict[int, float] = {}
    drtg: dict[int, float] = {}
    off: dict[int, float] = {}
    deff: dict[int, float] = {}
    n_games: dict[int, int] = {}
    dates_hist: dict[int, list[pd.Timestamp]] = {}
    rows: list[dict[str, Any]] = []

    for season, sched_in, logs in seasons:
        sched = sched_in.copy()
        sched["game_id"] = sched["game_id"].astype(str)
        if "game_date" not in sched.columns:
            sched["game_date"] = pd.to_datetime(sched["game_date_est"]).dt.normalize()
        else:
            sched["game_date"] = pd.to_datetime(sched["game_date"]).dt.normalize()
        sched = sched.dropna(subset=["home_score", "away_score"]).sort_values(
            ["game_date", "game_id"]
        )
        prep = prepare_team_game_rows(logs)
        pairs = prep.groupby("game_id")["team_id"].agg(list)
        opp: dict[tuple[str, int], int] = {}
        for gid, tids in pairs.items():
            if len(tids) != 2:
                continue
            a, b = int(tids[0]), int(tids[1])
            opp[(str(gid), a)] = b
            opp[(str(gid), b)] = a
        prep = prep.copy()
        prep["game_id"] = prep["game_id"].astype(str)
        prep["opp_team_id"] = [
            opp.get((str(g), int(t)), np.nan) for g, t in zip(prep["game_id"], prep["team_id"])
        ]
        prep = prep.dropna(subset=["opp_team_id"])
        prep["opp_team_id"] = prep["opp_team_id"].astype(int)
        by_game = prep.groupby("game_id")

        for _, g in sched.iterrows():
            gid = str(g["game_id"])
            gd = pd.Timestamp(g["game_date"]).normalize()
            hid = int(g["home_team_id"])
            aid = int(g["away_team_id"])
            rates = learn_play_rate(play_hist)

            if n_games.get(hid, 0) >= 1 and n_games.get(aid, 0) >= 1:
                himp = team_player_impact_asof_states(pstates, hid, play_rate_by_streak=rates)
                aimp = team_player_impact_asof_states(pstates, aid, play_rate_by_streak=rates)
                h_net = ortg[hid] - drtg[hid]
                a_net = ortg[aid] - drtg[aid]
                h_adj = off.get(hid, ortg[hid]) - deff.get(hid, drtg[hid])
                a_adj = off.get(aid, ortg[aid]) - deff.get(aid, drtg[aid])
                he = elo.get(hid, 1500.0)
                ae = elo.get(aid, 1500.0)
                hr = float((gd - max(dates_hist[hid])).days)
                ar = float((gd - max(dates_hist[aid])).days)
                hs = schedule_density_asof(pd.Series(dates_hist[hid]), gd)
                asus = schedule_density_asof(pd.Series(dates_hist[aid]), gd)
                rows.append(
                    {
                        "season": season,
                        "game_id": gid,
                        "game_date": str(gd.date()),
                        "home_win": float(g["home_score"] > g["away_score"]),
                        "actual_margin": float(g["home_score"] - g["away_score"]),
                        "impact_diff": himp["impact"] - aimp["impact"],
                        "missing_impact_diff": himp["missing_impact"] - aimp["missing_impact"],
                        "exp_min_diff": himp["exp_minutes"] - aimp["exp_minutes"],
                        "net_diff": h_net - a_net,
                        "adj_net_diff": h_adj - a_adj,
                        "rest_diff": hr - ar,
                        "b2b_diff": float(hr == 1.0) - float(ar == 1.0),
                        "three_diff": hs["three_in_four"] - asus["three_in_four"],
                        "games7_diff": hs["games_last_n"] - asus["games_last_n"],
                        "elo_diff": he - ae,
                    }
                )

            box = pb_by_game.get(gid)
            if box is not None and not box.empty:
                for tid in (hid, aid):
                    for pid, st in list(pstates.items()):
                        if st.last_team_id != tid or st.games <= 0:
                            continue
                        played = 0
                        prow = box.loc[box["player_id"] == pid]
                        if not prow.empty and float(prow.iloc[0]["minutes"]) > 0:
                            played = 1
                        play_hist.append((int(st.miss_streak), played))
                update_player_states_after_game(pstates, box)

            try:
                tg = by_game.get_group(gid)
            except KeyError:
                continue

            mov = float(g["home_score"] - g["away_score"])
            he = elo.get(hid, 1500.0)
            ae = elo.get(aid, 1500.0)
            if mov > 0:
                he2, ae2 = _elo_update(he + ELO_HCA, ae, mov)
                he2 -= ELO_HCA
            else:
                ae2, he2 = _elo_update(ae, he + ELO_HCA, -mov)
                he2 -= ELO_HCA
            elo[hid], elo[aid] = he2, ae2

            for _, tr in tg.iterrows():
                tid = int(tr["team_id"])
                n0 = n_games.get(tid, 0)
                ortg[tid] = _ema(ortg.get(tid, float("nan")), float(tr["ortg"]), n0, alpha)
                drtg[tid] = _ema(drtg.get(tid, float("nan")), float(tr["drtg"]), n0, alpha)
                odef = deff.get(int(tr["opp_team_id"]), league)
                ooff = off.get(int(tr["opp_team_id"]), league)
                off[tid] = _ema(
                    off.get(tid, float("nan")), float(tr["ortg"]) + (league - odef), n0, alpha
                )
                deff[tid] = _ema(
                    deff.get(tid, float("nan")), float(tr["drtg"]) + (ooff - league), n0, alpha
                )
                n_games[tid] = n0 + 1
                dates_hist.setdefault(tid, []).append(gd)

    return pd.DataFrame(rows)


def expanding_ridge_normal_probs(
    feature_frame: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    sigma: float = DEFAULT_SIGMA,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every: int = DEFAULT_REFIT_EVERY,
) -> pd.DataFrame:
    """Fit expanding Ridge on actual margin; emit Φ(m̂/σ) win probs."""
    df = feature_frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    X = np.nan_to_num(df[feature_cols].to_numpy(dtype=float), nan=0.0)
    X = np.column_stack([np.ones(len(df)), X])  # intercept = HCA
    y = df["actual_margin"].to_numpy(dtype=float)
    p = np.empty(len(df), dtype=float)
    model: Any = None
    last = -1
    for i in range(len(df)):
        if i < min_train:
            # Stable home prior until enough history (intercept column is 1.0).
            p[i] = float(norm.cdf(1.0 / float(sigma)))
            continue
        if model is None or i == min_train or (i - last) >= refit_every:
            rg = Ridge(alpha=float(ridge_alpha))
            rg.fit(X[:i], y[:i])
            model = rg
            last = i
        mhat = float(model.predict(X[i : i + 1])[0])
        p[i] = float(np.clip(norm.cdf(mhat / float(sigma)), 1e-6, 1 - 1e-6))
    out = df.copy()
    out["p_home_win"] = p
    out["p_away_win"] = 1.0 - out["p_home_win"]
    pick_home = out["p_home_win"] >= 0.5
    out["ml_pick_home"] = pick_home
    out["p_pick"] = np.where(pick_home, out["p_home_win"], out["p_away_win"])
    out["ml_correct"] = pick_home == (out["home_win"] == 1.0)
    return out


def run_player_impact_walkforward(
    *,
    root: Path | None = None,
    player_box_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience runner over cached season schedule/logs + player boxes."""
    root = root or Path("/tmp/nba_postseason_bt")
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    season_frames: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for s in seasons:
        season_frames.append(
            (
                s,
                pd.read_parquet(root / f"{s}_all_schedule.parquet"),
                pd.read_parquet(root / f"{s}_all_logs.parquet"),
            )
        )
    box_path = player_box_path or Path("/tmp/nba_full_bt/espn_player_box.parquet")
    box = pd.read_parquet(box_path)
    feats = build_player_impact_feature_frame(seasons=season_frames, player_box=box)
    graded = expanding_ridge_normal_probs(feats)
    from nba_betting.backtest.metrics import brier_score, log_loss

    p = graded["p_home_win"].to_numpy(float)
    y = graded["home_win"].to_numpy(float)
    metrics = {
        "n": int(len(graded)),
        "hit": float(((p >= 0.5) == (y == 1)).mean()),
        "brier": float(brier_score(p, y)),
        "ll": float(log_loss(p, y)),
        "model": "player_impact_ridge_normal",
        "sigma": DEFAULT_SIGMA,
        "ridge_alpha": DEFAULT_RIDGE_ALPHA,
    }
    return graded, metrics
