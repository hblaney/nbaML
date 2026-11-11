"""Opponent-adjusted team offense/defense ratings (as-of, no leakage).

Method (iterative KenPom-style, season-to-date with EMA game weights)
----------------------------------------------------------------------
Let league average efficiency be ``L`` (points per 100 possessions).

For each prior game of team ``t`` vs opponent ``o``:

* observed offensive rating ``ORTG_t``
* observed defensive rating ``DRTG_t``

Update estimates until convergence (or a fixed iteration count):

* ``off[t] = wavg( ORTG_t + (def[o] - L) )``
* ``def[t] = wavg( DRTG_t + (off[o] - L) )``

So a team that scores 110 against an elite defense (def=105 when L=110)
is credited with adj offense ``110 + (105-110) = 105`` wait — actually
``110 + (105 - 110) = 105`` would *lower* credit; the standard form is:

* ``off[t] = wavg( ORTG_t - (def[o] - L) )``  # subtract opp defense overage
  which equals ``wavg( ORTG_t + (L - def[o]) )``

A 110 ORTG vs a 105 DRTG opponent (stingy) becomes
``110 + (110 - 105) = 115`` adj offense — harder schedule credit.

We verify on toy data that better opponents raise adj offense for the same
raw ORTG (see tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_betting.features.team_ratings import DEFAULT_EMA_HALF_LIFE, prepare_team_game_rows

LEAGUE_AVG_EFF = 112.0
DEFAULT_ITERS = 8


@dataclass(frozen=True)
class AdjTeamRating:
    team_id: int
    games: int
    off: float
    defense: float  # "def" is reserved in some contexts
    pace: float
    asof_date: str

    @property
    def net(self) -> float:
        return float(self.off - self.defense)


def _ema_weights(n: int, half_life: float) -> np.ndarray:
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (ages / float(half_life))
    return w / w.sum()


def opponent_adjusted_asof(
    prepared: pd.DataFrame,
    team_id: int,
    asof_date: str | pd.Timestamp,
    *,
    half_life: float = DEFAULT_EMA_HALF_LIFE,
    max_games: int = 82,
    n_iters: int = DEFAULT_ITERS,
    league_avg: float = LEAGUE_AVG_EFF,
) -> AdjTeamRating | None:
    """EMA-weighted iterative opp-adjusted ORTG/DRTG using games before ``asof_date``.

    Uses all teams' prior games in ``prepared`` (same season frame) so opponent
    defenses/offenses are estimated from the same as-of history.
    """
    asof = pd.Timestamp(asof_date).normalize()
    hist_all = prepared.loc[prepared["game_date"] < asof].copy()
    if hist_all.empty:
        return None

    # Need opponent team_id on each row.
    if "opp_team_id" not in hist_all.columns:
        hist_all = _attach_opp_team_id(hist_all)
    team_hist = hist_all.loc[hist_all["team_id"] == team_id].tail(int(max_games))
    if team_hist.empty:
        return None

    # Restrict opponent pool to same as-of window (all teams).
    # Build per-team game lists once.
    teams = hist_all["team_id"].unique()
    off = {int(t): float(league_avg) for t in teams}
    deff = {int(t): float(league_avg) for t in teams}

    # Cap each team's history for speed.
    by_team: dict[int, pd.DataFrame] = {}
    weights: dict[int, np.ndarray] = {}
    for t in teams:
        th = hist_all.loc[hist_all["team_id"] == t].tail(int(max_games))
        by_team[int(t)] = th
        weights[int(t)] = _ema_weights(len(th), half_life)

    for _ in range(int(n_iters)):
        new_off: dict[int, float] = {}
        new_def: dict[int, float] = {}
        for t, th in by_team.items():
            w = weights[t]
            ortg = th["ortg"].to_numpy(dtype=float)
            drtg = th["drtg"].to_numpy(dtype=float)
            opp = th["opp_team_id"].to_numpy(dtype=int)
            opp_def = np.array([deff[int(o)] for o in opp], dtype=float)
            opp_off = np.array([off[int(o)] for o in opp], dtype=float)
            # Credit: raw ORTG + (L - opp_def); raw DRTG + (opp_off - L)
            adj_o = ortg + (float(league_avg) - opp_def)
            adj_d = drtg + (opp_off - float(league_avg))
            new_off[t] = float(np.average(adj_o, weights=w))
            new_def[t] = float(np.average(adj_d, weights=w))
        off, deff = new_off, new_def

    tid = int(team_id)
    pace = float(
        np.average(
            team_hist["pace"].to_numpy(dtype=float),
            weights=_ema_weights(len(team_hist), half_life),
        )
    )
    return AdjTeamRating(
        team_id=tid,
        games=int(len(team_hist)),
        off=float(off[tid]),
        defense=float(deff[tid]),
        pace=pace,
        asof_date=str(asof.date()),
    )


def _attach_opp_team_id(prepared: pd.DataFrame) -> pd.DataFrame:
    """Pair each team-game row with the opponent's team_id via game_id."""
    df = prepared.copy()
    # Two rows per game_id; map other team.
    pairs = df.groupby("game_id")["team_id"].agg(list)
    opp_map: dict[tuple[str, int], int] = {}
    for gid, tids in pairs.items():
        if len(tids) != 2:
            continue
        a, b = int(tids[0]), int(tids[1])
        opp_map[(str(gid), a)] = b
        opp_map[(str(gid), b)] = a
    df["opp_team_id"] = [
        opp_map.get((str(g), int(t)), np.nan) for g, t in zip(df["game_id"], df["team_id"])
    ]
    df = df.dropna(subset=["opp_team_id"])
    df["opp_team_id"] = df["opp_team_id"].astype(int)
    return df


def adj_offense_observation(raw_ortg: float, opp_def: float, league_avg: float = LEAGUE_AVG_EFF) -> float:
    """Single-game adjusted offense: credit harder defenses."""
    return float(raw_ortg + (float(league_avg) - float(opp_def)))


def verify_harder_schedule_raises_adj_offense() -> bool:
    """Verify the adjustment identity and that iterative ratings separate soft/tough D."""
    # Algebraic property of the method (frozen opponent defenses).
    same_raw = 110.0
    vs_soft = adj_offense_observation(same_raw, opp_def=120.0)
    vs_tough = adj_offense_observation(same_raw, opp_def=100.0)
    if not (vs_tough > vs_soft):
        return False

    # Iterative estimator: Soft piles up high DRTG games, Tough low DRTG; then A
    # posts the same raw ORTG against each. A's adj off must be higher vs Tough.
    base = pd.Timestamp("2024-01-01")
    rows: list[dict] = []
    for i in range(6):
        d = base + pd.Timedelta(days=i)
        # Soft (2) vs filler: Soft DRTG=122
        rows += [
            dict(game_id=f"s{i}", game_date=d, team_id=2, opp_team_id=90 + i, ortg=100.0, drtg=122.0, pace=100.0, team_abbreviation="SFT"),
            dict(game_id=f"s{i}", game_date=d, team_id=90 + i, opp_team_id=2, ortg=122.0, drtg=100.0, pace=100.0, team_abbreviation="X"),
            # Tough (3) vs filler: Tough DRTG=98
            dict(game_id=f"t{i}", game_date=d, team_id=3, opp_team_id=80 + i, ortg=100.0, drtg=98.0, pace=100.0, team_abbreviation="TGH"),
            dict(game_id=f"t{i}", game_date=d, team_id=80 + i, opp_team_id=3, ortg=98.0, drtg=100.0, pace=100.0, team_abbreviation="Y"),
        ]
    # A vs Soft then A vs Tough with identical raw ORTG=110
    d_soft = base + pd.Timedelta(days=10)
    d_tough = base + pd.Timedelta(days=11)
    rows += [
        dict(game_id="as", game_date=d_soft, team_id=1, opp_team_id=2, ortg=110.0, drtg=105.0, pace=100.0, team_abbreviation="A"),
        dict(game_id="as", game_date=d_soft, team_id=2, opp_team_id=1, ortg=105.0, drtg=110.0, pace=100.0, team_abbreviation="SFT"),
        dict(game_id="at", game_date=d_tough, team_id=1, opp_team_id=3, ortg=110.0, drtg=105.0, pace=100.0, team_abbreviation="A"),
        dict(game_id="at", game_date=d_tough, team_id=3, opp_team_id=1, ortg=105.0, drtg=110.0, pace=100.0, team_abbreviation="TGH"),
    ]
    prepared = pd.DataFrame(rows)
    soft_only = prepared.loc[prepared["game_id"] != "at"].copy()
    tough_only = prepared.loc[prepared["game_id"] != "as"].copy()
    r_soft = opponent_adjusted_asof(
        soft_only, 1, d_soft + pd.Timedelta(days=1), n_iters=12, max_games=20, half_life=100.0
    )
    r_tough = opponent_adjusted_asof(
        tough_only, 1, d_tough + pd.Timedelta(days=1), n_iters=12, max_games=20, half_life=100.0
    )
    if r_soft is None or r_tough is None:
        return False
    return bool(r_tough.off > r_soft.off + 0.5)
