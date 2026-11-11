"""As-of feature enrichment on top of raw no-odds moneyline probs.

Walk-forward logistic on ``logit(p_raw)`` plus schedule / prior-box availability /
opponent-adjusted net residuals. Fit only on earlier games (no leakage).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba_betting.features.opponent_adjusted import LEAGUE_AVG_EFF
from nba_betting.features.player_availability import availability_asof, prepare_player_minutes
from nba_betting.features.schedule_context import schedule_density_asof
from nba_betting.features.team_ratings import prepare_team_game_rows

# Confirm-gated on 2024-25 (beats EMA-16+rest+Platt confirm 0.20931 and pooled 0.21447).
DEFAULT_AVAIL_K = 0.05
DEFAULT_SCHED_K = 0.8
DEFAULT_ADJ_K = 0.15
DEFAULT_LOGIT_C = 5.0
DEFAULT_MIN_TRAIN = 200
DEFAULT_REFIT_EVERY = 50
DEFAULT_EMA_HALF_LIFE = 16.0


def _ema_alpha(half_life: float = DEFAULT_EMA_HALF_LIFE) -> float:
    return 1.0 - math.exp(math.log(0.5) / float(half_life))


def _ema_update(old: float, obs: float, n_prev: int, alpha: float) -> float:
    if n_prev <= 0 or old != old:
        return float(obs)
    return float((1.0 - alpha) * old + alpha * obs)


def _lookup_avail(pm: dict[int, pd.DataFrame], team_id: int, gd: pd.Timestamp):
    frame = pm.get(int(team_id))
    if frame is None or frame.empty:
        return availability_asof(
            pd.DataFrame(columns=["game_id", "game_date", "team_id", "player_id", "minutes"]),
            int(team_id),
            gd,
        )
    return availability_asof(frame, int(team_id), gd)


def build_asof_enrich_features(
    *,
    seasons: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    player_minutes: pd.DataFrame,
    half_life: float = DEFAULT_EMA_HALF_LIFE,
) -> pd.DataFrame:
    """Build as-of feature rows for (season, game_id).

    ``seasons`` entries are ``(season, schedule, team_logs)``.
    Player minutes / ratings use strictly prior tips only.
    """
    alpha = _ema_alpha(half_life)
    league = float(LEAGUE_AVG_EFF)
    pm_all = prepare_player_minutes(player_minutes)
    pm = {int(tid): g.reset_index(drop=True) for tid, g in pm_all.groupby("team_id")}
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

        ortg: dict[int, float] = {}
        drtg: dict[int, float] = {}
        off: dict[int, float] = {}
        deff: dict[int, float] = {}
        n_games: dict[int, int] = {}
        dates_hist: dict[int, list[pd.Timestamp]] = {}

        for _, g in sched.iterrows():
            gid = str(g["game_id"])
            gd = pd.Timestamp(g["game_date"]).normalize()
            hid = int(g["home_team_id"])
            aid = int(g["away_team_id"])
            if n_games.get(hid, 0) >= 1 and n_games.get(aid, 0) >= 1:
                h_net_raw = ortg[hid] - drtg[hid]
                a_net_raw = ortg[aid] - drtg[aid]
                h_net_adj = off.get(hid, ortg[hid]) - deff.get(hid, drtg[hid])
                a_net_adj = off.get(aid, ortg[aid]) - deff.get(aid, drtg[aid])
                hav = _lookup_avail(pm, hid, gd)
                aav = _lookup_avail(pm, aid, gd)
                hs = schedule_density_asof(pd.Series(dates_hist[hid]), gd)
                asus = schedule_density_asof(pd.Series(dates_hist[aid]), gd)
                miss_diff = float(hav.missing_minutes - aav.missing_minutes)
                three_diff = float(hs["three_in_four"] - asus["three_in_four"])
                games7_diff = float(hs["games_last_n"] - asus["games_last_n"])
                adj_residual = float((h_net_adj - a_net_adj) - (h_net_raw - a_net_raw))
                rows.append(
                    {
                        "season": season,
                        "game_id": gid,
                        "game_date": str(gd.date()),
                        "miss_diff": miss_diff,
                        "three_diff": three_diff,
                        "games7_diff": games7_diff,
                        "adj_residual": adj_residual,
                    }
                )

            try:
                tg = by_game.get_group(gid)
            except KeyError:
                continue
            for _, tr in tg.iterrows():
                tid = int(tr["team_id"])
                n0 = n_games.get(tid, 0)
                ortg[tid] = _ema_update(ortg.get(tid, float("nan")), float(tr["ortg"]), n0, alpha)
                drtg[tid] = _ema_update(drtg.get(tid, float("nan")), float(tr["drtg"]), n0, alpha)
                odef = deff.get(int(tr["opp_team_id"]), league)
                ooff = off.get(int(tr["opp_team_id"]), league)
                off[tid] = _ema_update(
                    off.get(tid, float("nan")), float(tr["ortg"]) + (league - odef), n0, alpha
                )
                deff[tid] = _ema_update(
                    deff.get(tid, float("nan")), float(tr["drtg"]) + (ooff - league), n0, alpha
                )
                n_games[tid] = n0 + 1
                dates_hist.setdefault(tid, []).append(gd)

    return pd.DataFrame(rows)


def margin_shift_from_features(
    feats: pd.DataFrame,
    *,
    avail_k: float = DEFAULT_AVAIL_K,
    sched_k: float = DEFAULT_SCHED_K,
    adj_k: float = DEFAULT_ADJ_K,
) -> np.ndarray:
    """Home-perspective latent margin shift from as-of features."""
    miss = feats["miss_diff"].to_numpy(dtype=float)
    three = feats["three_diff"].to_numpy(dtype=float)
    games7 = feats["games7_diff"].to_numpy(dtype=float)
    adj = feats["adj_residual"].to_numpy(dtype=float)
    return (
        -float(avail_k) * miss
        - float(sched_k) * three
        - 0.25 * float(sched_k) * np.maximum(0.0, games7)
        + float(adj_k) * adj
    )


def expanding_logit_enrich(
    p_raw: np.ndarray,
    y: np.ndarray,
    shift: np.ndarray,
    *,
    C: float = DEFAULT_LOGIT_C,
    min_train: int = DEFAULT_MIN_TRAIN,
    refit_every: int = DEFAULT_REFIT_EVERY,
) -> np.ndarray:
    """Expanding logistic on [logit(p_raw), shift]; earlier games only."""
    p_raw = np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=float)
    shift = np.nan_to_num(np.asarray(shift, dtype=float), nan=0.0)
    logit = np.log(p_raw / (1.0 - p_raw))
    X = np.column_stack([logit, shift])
    out = np.empty(len(y), dtype=float)
    model: Any = None
    last = -1
    for i in range(len(y)):
        if i < min_train:
            out[i] = float(p_raw[i])
            continue
        if model is None or i == min_train or (i - last) >= refit_every:
            lr = LogisticRegression(C=float(C), solver="lbfgs", max_iter=500)
            lr.fit(X[:i], y[:i])
            model = lr
            last = i
        out[i] = float(model.predict_proba(X[i : i + 1])[0, 1])
    return out


def enrich_graded_predictions(
    graded: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    avail_k: float = DEFAULT_AVAIL_K,
    sched_k: float = DEFAULT_SCHED_K,
    adj_k: float = DEFAULT_ADJ_K,
    C: float = DEFAULT_LOGIT_C,
    p_col: str = "p_home_win",
) -> pd.DataFrame:
    """Join as-of features and replace ML probs with enriched walk-forward probs."""
    df = graded.copy()
    df["game_id"] = df["game_id"].astype(str)
    feats = feature_rows.copy()
    feats["game_id"] = feats["game_id"].astype(str)
    keys = ["season", "game_id"] if "season" in df.columns and "season" in feats.columns else ["game_id"]
    merged = df.merge(feats, on=keys, how="inner", suffixes=("", "_feat"))
    merged = merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    shift = margin_shift_from_features(
        merged, avail_k=avail_k, sched_k=sched_k, adj_k=adj_k
    )
    if "p_home_win_raw" not in merged.columns:
        merged["p_home_win_raw"] = merged[p_col].astype(float)
    enriched = expanding_logit_enrich(
        merged["p_home_win_raw"].to_numpy(dtype=float),
        merged["home_win"].to_numpy(dtype=float),
        shift,
        C=C,
    )
    merged[p_col] = enriched
    merged["p_away_win"] = 1.0 - merged[p_col]
    pick_home = merged[p_col] >= 0.5
    merged["ml_pick_home"] = pick_home
    merged["p_pick"] = np.where(pick_home, merged[p_col], merged["p_away_win"])
    if "home_win" in merged.columns:
        merged["ml_correct"] = pick_home == (merged["home_win"] == 1.0)
    return merged
