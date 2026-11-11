"""Same-day 2-/3-leg parlay selector using model edge vs book ML prices.

Product rule: no straight bets. A ticket is claimed +EV only when the
independent-model joint probability beats the **posted parlay price**
(product of vigged American→decimal odds), not merely when each leg beats
the de-vigged two-way fair.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nba_betting.pricing.kelly import american_to_decimal


def parlay_decimal(american_prices: Sequence[float]) -> float:
    """Sportsbook-style parlay decimal = product of leg decimals."""
    dec = 1.0
    for p in american_prices:
        dec *= american_to_decimal(float(p))
    return float(dec)


def parlay_ev_per_unit(model_probs: Sequence[float], american_prices: Sequence[float]) -> float:
    """Expected profit per 1-unit stake under independence.

    ``EV = (∏ p_i) * (∏ decimal_i) - 1``. Positive only if the model joint
    probability clears the vigged parlay price.
    """
    if len(model_probs) != len(american_prices) or not model_probs:
        raise ValueError("model_probs and american_prices must be non-empty and equal length")
    joint = 1.0
    for p in model_probs:
        pp = float(p)
        if not 0.0 < pp < 1.0:
            return float("-inf")
        joint *= pp
    return float(joint * parlay_decimal(american_prices) - 1.0)


def build_eligible_legs(
    games: pd.DataFrame,
    *,
    model_home_col: str = "p_model_home",
    novig_home_col: str = "p_home_novig",
    home_ml_col: str = "home_ml",
    away_ml_col: str = "away_ml",
    y_home_col: str = "home_win",
    min_fair_edge: float = 0.03,
    date_col: str = "game_date",
    game_id_col: str = "game_id",
) -> pd.DataFrame:
    """One row per eligible side where model beats de-vigged fair by ``min_fair_edge``.

    Required columns: model home win prob, de-vigged home fair, both MLs, home_win, date, game_id.
    Away model / novig are ``1 - home``.
    """
    need = [model_home_col, novig_home_col, home_ml_col, away_ml_col, y_home_col, date_col, game_id_col]
    missing = [c for c in need if c not in games.columns]
    if missing:
        raise ValueError(f"games missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, g in games.iterrows():
        p_h = float(g[model_home_col])
        fair_h = float(g[novig_home_col])
        p_a = 1.0 - p_h
        fair_a = 1.0 - fair_h
        home_won = float(g[y_home_col])
        base = {
            "season": g.get("season"),
            "game_date": pd.Timestamp(g[date_col]).normalize(),
            "game_id": str(g[game_id_col]),
            "provider": g.get("provider"),
        }
        for side, p_m, p_fair, price, won in (
            ("home", p_h, fair_h, float(g[home_ml_col]), home_won == 1.0),
            ("away", p_a, fair_a, float(g[away_ml_col]), home_won == 0.0),
        ):
            edge = p_m - p_fair
            if edge < float(min_fair_edge):
                continue
            if not np.isfinite(price) or price == 0:
                continue
            rows.append(
                {
                    **base,
                    "side": side,
                    "p_model": p_m,
                    "p_fair_novig": p_fair,
                    "fair_edge": edge,
                    "american": price,
                    "leg_won": bool(won),
                    "decimal": american_to_decimal(price),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "game_date",
                "game_id",
                "provider",
                "side",
                "p_model",
                "p_fair_novig",
                "fair_edge",
                "american",
                "leg_won",
                "decimal",
            ]
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["game_date", "game_id", "side"]).reset_index(drop=True)


def _candidate_from_legs(legs: Sequence[pd.Series], *, n_legs: int) -> dict[str, Any] | None:
    game_ids = [str(L["game_id"]) for L in legs]
    if len(set(game_ids)) != n_legs:
        return None
    probs = [float(L["p_model"]) for L in legs]
    prices = [float(L["american"]) for L in legs]
    ev = parlay_ev_per_unit(probs, prices)
    dec = parlay_decimal(prices)
    joint = float(np.prod(probs))
    book_implied = 1.0 / dec
    won = all(bool(L["leg_won"]) for L in legs)
    return {
        "game_date": legs[0]["game_date"],
        "season": legs[0].get("season"),
        "n_legs": int(n_legs),
        "game_ids": "|".join(game_ids),
        "sides": "|".join(str(L["side"]) for L in legs),
        "americans": "|".join(f"{float(L['american']):.0f}" for L in legs),
        "p_joint_model": joint,
        "parlay_decimal": dec,
        "book_implied": book_implied,
        "ev_per_unit": ev,
        "survives_parlay_price": bool(ev > 0.0),
        "ticket_won": bool(won),
        "min_fair_edge_leg": float(min(float(L["fair_edge"]) for L in legs)),
    }


def enumerate_same_day_parlays(
    legs: pd.DataFrame,
    *,
    sizes: Sequence[int] = (2, 3),
    min_ev: float = 0.0,
) -> pd.DataFrame:
    """All same-day combinations of distinct games that clear ``min_ev`` at book price."""
    if legs.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for day, day_legs in legs.groupby("game_date", sort=True):
        rows = [day_legs.iloc[i] for i in range(len(day_legs))]
        # At most one side per game in a ticket — handled by unique game_ids check.
        for n in sizes:
            if len(rows) < n:
                continue
            for combo in combinations(range(len(rows)), n):
                cand = _candidate_from_legs([rows[i] for i in combo], n_legs=n)
                if cand is None:
                    continue
                if cand["ev_per_unit"] < float(min_ev):
                    continue
                if not cand["survives_parlay_price"] and float(min_ev) >= 0.0:
                    # min_ev >= 0 already implies EV gate; keep explicit.
                    continue
                records.append(cand)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["game_date", "ev_per_unit"], ascending=[True, False]).reset_index(drop=True)


def select_nonoverlapping_tickets(
    candidates: pd.DataFrame,
    *,
    max_per_day: int = 1,
) -> pd.DataFrame:
    """Greedy per-day: highest EV first, skip tickets that reuse a game_id already taken."""
    if candidates.empty:
        return candidates.copy()
    picked: list[pd.Series] = []
    for day, day_df in candidates.groupby("game_date", sort=True):
        used: set[str] = set()
        n = 0
        for _, row in day_df.sort_values("ev_per_unit", ascending=False).iterrows():
            gids = set(str(row["game_ids"]).split("|"))
            if gids & used:
                continue
            picked.append(row)
            used |= gids
            n += 1
            if n >= int(max_per_day):
                break
    if not picked:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame(picked).sort_values(["game_date", "ev_per_unit"], ascending=[True, False]).reset_index(drop=True)


def grade_tickets(
    tickets: pd.DataFrame,
    *,
    stake: float = 1.0,
) -> pd.DataFrame:
    """Attach stake / payout / pnl. Payout includes stake returned on a win."""
    if tickets.empty:
        out = tickets.copy()
        for c in ("stake", "payout", "pnl", "cum_pnl", "bankroll"):
            out[c] = []
        return out
    df = tickets.copy()
    stake_f = float(stake)
    stakes: list[float] = []
    payouts: list[float] = []
    pnls: list[float] = []
    for _, row in df.iterrows():
        stakes.append(stake_f)
        if bool(row["ticket_won"]):
            pay = stake_f * float(row["parlay_decimal"])
            payouts.append(pay)
            pnls.append(pay - stake_f)
        else:
            payouts.append(0.0)
            pnls.append(-stake_f)
    df["stake"] = stakes
    df["payout"] = payouts
    df["pnl"] = pnls
    df["cum_pnl"] = np.cumsum(pnls)
    return df


def summarize_ticket_pnl(
    graded: pd.DataFrame,
    *,
    starting_bankroll: float | None = None,
    stake: float = 1.0,
) -> dict[str, Any]:
    """Historical profit summary: stake, payout, ROI, max drawdown, ticket counts."""
    n = int(len(graded))
    if n == 0:
        return {
            "n_tickets": 0,
            "n_wins": 0,
            "hit_rate": None,
            "total_stake": 0.0,
            "total_payout": 0.0,
            "total_pnl": 0.0,
            "roi": None,
            "max_drawdown": 0.0,
            "avg_ev_per_unit": None,
            "avg_parlay_decimal": None,
            "by_n_legs": {},
        }
    total_stake = float(graded["stake"].sum())
    total_payout = float(graded["payout"].sum())
    total_pnl = float(graded["pnl"].sum())
    cum = graded["pnl"].to_numpy(dtype=float).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = float(np.min(cum - peak)) if len(cum) else 0.0
    # Optional bankroll path if starting bankroll given
    if starting_bankroll is not None:
        br = float(starting_bankroll) + cum
        br_peak = np.maximum.accumulate(br)
        dd = float(np.min(br - br_peak))
    by_legs: dict[str, Any] = {}
    for n_legs, g in graded.groupby("n_legs"):
        st = float(g["stake"].sum())
        pn = float(g["pnl"].sum())
        by_legs[str(int(n_legs))] = {
            "n_tickets": int(len(g)),
            "n_wins": int(g["ticket_won"].sum()),
            "total_stake": st,
            "total_pnl": pn,
            "roi": (pn / st) if st > 0 else None,
        }
    return {
        "n_tickets": n,
        "n_wins": int(graded["ticket_won"].sum()),
        "hit_rate": float(graded["ticket_won"].mean()),
        "total_stake": total_stake,
        "total_payout": total_payout,
        "total_pnl": total_pnl,
        "roi": (total_pnl / total_stake) if total_stake > 0 else None,
        "max_drawdown": dd,
        "avg_ev_per_unit": float(graded["ev_per_unit"].mean()),
        "avg_parlay_decimal": float(graded["parlay_decimal"].mean()),
        "by_n_legs": by_legs,
        "stake_per_ticket": float(stake),
        "starting_bankroll": starting_bankroll,
    }


def run_parlay_selector(
    games: pd.DataFrame,
    *,
    model_home_col: str = "p_model_home",
    min_fair_edge: float = 0.03,
    min_ev: float = 0.02,
    sizes: Sequence[int] = (2, 3),
    max_per_day: int = 1,
    stake: float = 1.0,
    starting_bankroll: float | None = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """End-to-end: legs → +EV same-day parlays → non-overlapping pick → graded summary."""
    legs = build_eligible_legs(
        games,
        model_home_col=model_home_col,
        min_fair_edge=min_fair_edge,
    )
    cands = enumerate_same_day_parlays(legs, sizes=sizes, min_ev=min_ev)
    # Only tickets that survive parlay price (EV>0). min_ev already enforces if >=0.
    if not cands.empty:
        cands = cands[cands["survives_parlay_price"]].copy()
    picked = select_nonoverlapping_tickets(cands, max_per_day=max_per_day)
    graded = grade_tickets(picked, stake=stake)
    if starting_bankroll is not None and not graded.empty:
        graded["bankroll"] = float(starting_bankroll) + graded["pnl"].cumsum()
    summary = summarize_ticket_pnl(graded, starting_bankroll=starting_bankroll, stake=stake)
    summary["n_eligible_legs"] = int(len(legs))
    summary["n_candidate_tickets"] = int(len(cands))
    summary["params"] = {
        "min_fair_edge": float(min_fair_edge),
        "min_ev": float(min_ev),
        "sizes": list(sizes),
        "max_per_day": int(max_per_day),
        "stake": float(stake),
        "model_home_col": model_home_col,
    }
    return legs, graded, summary
