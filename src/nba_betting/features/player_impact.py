"""HoopVantage-inspired player impact + missed-game availability (as-of only).

For each player we keep EMA minutes and plus-minus-per-minute from prior boxes.
Availability uses consecutive zero-minute / absent games (miss streak), with
play rates learned from earlier games only — the second leg of HoopVantage's
availability rule when an official injury PDF archive is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HALF_LIFE_GAMES = 20.0
REPLACEMENT_PM_PER_MIN = 0.0


@dataclass
class PlayerState:
    games: int = 0
    ema_minutes: float = float("nan")
    ema_pm_per_min: float = float("nan")
    ema_pts_per_min: float = float("nan")
    miss_streak: int = 0
    last_team_id: int | None = None


def _alpha(half_life: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / float(half_life))


def _ema(old: float, obs: float, n_prev: int, alpha: float) -> float:
    if n_prev <= 0 or old != old:
        return float(obs)
    return float((1.0 - alpha) * old + alpha * obs)


def prepare_player_box(player_box: pd.DataFrame) -> pd.DataFrame:
    df = player_box.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(str)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["plus_minus"] = pd.to_numeric(df.get("plus_minus", 0.0), errors="coerce").fillna(0.0)
    df["pts"] = pd.to_numeric(df.get("pts", 0.0), errors="coerce").fillna(0.0)
    if "dnp" in df.columns:
        df.loc[df["dnp"].astype(bool), ["minutes", "plus_minus", "pts"]] = 0.0
    return df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)


def team_player_impact_asof_states(
    states: dict[str, PlayerState],
    team_id: int,
    *,
    play_rate_by_streak: dict[int, float],
    default_play_rate: float = 0.85,
    top_n: int = 9,
    half_life: float = DEFAULT_HALF_LIFE_GAMES,
) -> dict[str, float]:
    """Sum expected on-court impact for a team's recent rotation (state already as-of)."""
    members = [
        (pid, st)
        for pid, st in states.items()
        if st.last_team_id == int(team_id) and st.games > 0 and st.ema_minutes == st.ema_minutes
    ]
    if not members:
        return {
            "impact": 0.0,
            "exp_minutes": 0.0,
            "missing_impact": 0.0,
            "n_rotation": 0.0,
        }
    members.sort(key=lambda x: -float(x[1].ema_minutes))
    members = members[: int(top_n)]
    impact = 0.0
    exp_min = 0.0
    missing_impact = 0.0
    for pid, st in members:
        streak = int(st.miss_streak)
        p_play = float(play_rate_by_streak.get(streak, default_play_rate if streak == 0 else 0.35))
        p_play = float(np.clip(p_play, 0.0, 1.0))
        minutes = float(st.ema_minutes) if st.ema_minutes == st.ema_minutes else 0.0
        pm = float(st.ema_pm_per_min) if st.ema_pm_per_min == st.ema_pm_per_min else REPLACEMENT_PM_PER_MIN
        contrib = p_play * minutes * pm
        full = minutes * pm
        impact += contrib
        exp_min += p_play * minutes
        if streak >= 1:
            missing_impact += full * (1.0 - p_play)
    return {
        "impact": float(impact),
        "exp_minutes": float(exp_min),
        "missing_impact": float(missing_impact),
        "n_rotation": float(len(members)),
    }


def update_player_states_after_game(
    states: dict[str, PlayerState],
    game_rows: pd.DataFrame,
    *,
    half_life: float = DEFAULT_HALF_LIFE_GAMES,
) -> None:
    """Update player EMAs / miss streaks after a game's box is known."""
    alpha = _alpha(half_life)
    seen: set[str] = set()
    for _, r in game_rows.iterrows():
        pid = str(r["player_id"])
        if not pid:
            continue
        seen.add(pid)
        st = states.get(pid) or PlayerState()
        minutes = float(r["minutes"])
        team_id = int(r["team_id"])
        st.last_team_id = team_id
        if minutes <= 0.0:
            st.miss_streak = int(st.miss_streak) + 1
        else:
            pm_per = float(r["plus_minus"]) / max(minutes, 1e-6)
            pts_per = float(r["pts"]) / max(minutes, 1e-6)
            st.ema_minutes = _ema(st.ema_minutes, minutes, st.games, alpha)
            st.ema_pm_per_min = _ema(st.ema_pm_per_min, pm_per, st.games, alpha)
            st.ema_pts_per_min = _ema(st.ema_pts_per_min, pts_per, st.games, alpha)
            st.games = int(st.games) + 1
            st.miss_streak = 0
        states[pid] = st

    # Players previously on these teams who did not appear at all: treat as miss
    # (approximate inactive). Only bump miss streak for players whose last team
    # is one of the two teams in this game and who had ema_minutes history.
    team_ids = set(int(t) for t in game_rows["team_id"].unique())
    for pid, st in list(states.items()):
        if pid in seen:
            continue
        if st.last_team_id in team_ids and st.games > 0:
            st.miss_streak = int(st.miss_streak) + 1
            states[pid] = st


def learn_play_rate(
    history: list[tuple[int, int]],
    *,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0,
) -> dict[int, float]:
    """Expanding Beta-smoothed P(played | miss_streak) from (streak, played) pairs."""
    counts: dict[int, list[float]] = {}
    for streak, played in history:
        s = int(min(max(streak, 0), 10))
        bucket = counts.setdefault(s, [prior_alpha, prior_beta])
        if played:
            bucket[0] += 1.0
        else:
            bucket[1] += 1.0
    return {s: a / (a + b) for s, (a, b) in counts.items()}
