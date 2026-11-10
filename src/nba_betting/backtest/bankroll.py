"""Simple paper bankroll: flat stake and ¼ Kelly (assumed juice when no book)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nba_betting.pricing.kelly import (
    DEFAULT_AMERICAN,
    KELLY_FRACTION,
    MAX_BET_FRAC,
    STARTING_BANKROLL,
    american_to_decimal,
    stake_amount,
)


def _pnl(stake: float, won: bool, american: float, push: bool = False) -> float:
    if push or stake <= 0:
        return 0.0
    if won:
        return stake * (american_to_decimal(american) - 1.0)
    return -stake


def simulate_bankroll(
    graded: pd.DataFrame,
    *,
    starting_bankroll: float = STARTING_BANKROLL,
    flat_stake: float = 1.0,
    american: float = DEFAULT_AMERICAN,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_frac: float = MAX_BET_FRAC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply flat + Kelly bets on ML model favorites (Phase A paper).

    Uses ``ml_pick_home`` / ``ml_correct`` / ``p_pick`` columns when present.
    """
    df = graded.copy()
    if df.empty or "ml_correct" not in df.columns:
        meta = {
            "bankroll_start": starting_bankroll,
            "bankroll_end": starting_bankroll,
            "flat_roi": 0.0,
            "kelly_roi": 0.0,
        }
        return df, meta

    bankroll = float(starting_bankroll)
    flat_bankroll = float(starting_bankroll)
    kelly_ends: list[float] = []
    flat_ends: list[float] = []
    flat_stakes: list[float] = []
    kelly_stakes: list[float] = []
    flat_pnls: list[float] = []
    kelly_pnls: list[float] = []

    for _, row in df.iterrows():
        correct = row.get("ml_correct")
        if pd.isna(correct):
            kelly_ends.append(bankroll)
            flat_ends.append(flat_bankroll)
            flat_stakes.append(0.0)
            kelly_stakes.append(0.0)
            flat_pnls.append(0.0)
            kelly_pnls.append(0.0)
            continue

        p_pick = float(row.get("p_pick", row.get("p_home_win", 0.5)))
        won = bool(correct)

        f_stake = min(flat_stake, flat_bankroll)
        f_pnl = _pnl(f_stake, won, american)
        flat_bankroll += f_pnl

        k_stake = stake_amount(
            bankroll,
            p_pick,
            american,
            fraction=kelly_fraction,
            max_bet_frac=max_bet_frac,
        )
        k_pnl = _pnl(k_stake, won, american)
        bankroll += k_pnl

        flat_stakes.append(f_stake)
        kelly_stakes.append(k_stake)
        flat_pnls.append(f_pnl)
        kelly_pnls.append(k_pnl)
        flat_ends.append(flat_bankroll)
        kelly_ends.append(bankroll)

    df["flat_stake"] = flat_stakes
    df["flat_pnl"] = flat_pnls
    df["flat_bankroll"] = flat_ends
    df["kelly_stake"] = kelly_stakes
    df["kelly_pnl"] = kelly_pnls
    df["bankroll_end"] = kelly_ends

    flat_roi = (flat_bankroll - starting_bankroll) / starting_bankroll
    kelly_roi = (bankroll - starting_bankroll) / starting_bankroll
    meta = {
        "bankroll_start": starting_bankroll,
        "bankroll_end": bankroll,
        "flat_bankroll_end": flat_bankroll,
        "flat_roi": flat_roi,
        "kelly_roi": kelly_roi,
        "assumed_american": american,
        "kelly_fraction": kelly_fraction,
        "max_bet_frac": max_bet_frac,
        "note": "Phase A paper bankroll assumes constant American juice on model ML picks",
    }
    df.attrs["bankroll_start"] = starting_bankroll
    return df, meta
