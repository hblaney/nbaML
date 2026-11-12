"""Cross-game 2-leg & 3-leg player prop parlay EV selector + fractional Kelly sizing.

Enforces three strict anti-leakage / independence invariants:
1. Every leg must clear its own **vigged** straight implied probability
   (``p_model > 1 / decimal_leg``) or optional ``min_fair_edge``.
2. Multi-leg tickets default to **distinct games** (``allow_same_game=False``) so
   ``p_joint = prod(p_model_i)`` holds exactly without unmodeled same-game correlation
   or sportsbook SGP payout haircuts. Even when ``allow_same_game=True``, two props on
   the same player are strictly forbidden.
3. Combined probability must beat the actual parlay payout price:
   ``p_joint >= (1 + min_price_edge) / prod(decimal_i)``.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nba_betting.pricing.kelly import american_to_decimal
from nba_betting.pricing.parlay_ev import clears_parlay_price, parlay_decimal
from nba_betting.pricing.prop_edge import PropEdgeRow


def build_eligible_prop_legs(
    props: pd.DataFrame | Sequence[PropEdgeRow],
    *,
    p_over_col: str = "model_p_over",
    fair_over_col: str = "book_implied_over_fair",
    over_am_col: str = "book_over",
    under_am_col: str = "book_under",
    y_over_col: str | None = "y_over",
    min_fair_edge: float = 0.0,
    min_vig_edge: float = 0.0,
    require_vs_vigged: bool = True,
    allowed_sides: Sequence[str] = ("over", "under"),
) -> pd.DataFrame:
    """Build one row per eligible Over/Under player prop leg."""
    if isinstance(props, pd.DataFrame):
        df = props.copy()
    else:
        df = pd.DataFrame([r.to_dict() for r in props])

    cols = [
        "season",
        "game_date",
        "game_id",
        "player_id",
        "player_name",
        "team_abbreviation",
        "market",
        "stat",
        "line",
        "side",
        "p_model",
        "p_fair_novig",
        "p_vigged",
        "fair_edge",
        "vig_edge",
        "ev_leg",
        "american",
        "decimal",
        "leg_won",
        "book",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    side_set = {s.lower() for s in allowed_sides}
    has_y = y_over_col is not None and y_over_col in df.columns
    rows: list[dict[str, Any]] = []

    for _, r in df.iterrows():
        p_ov = float(r[p_over_col])
        p_un = 1.0 - p_ov
        fair_ov = float(r[fair_over_col])
        fair_un = 1.0 - fair_ov
        am_ov = r.get(over_am_col)
        am_un = r.get(under_am_col)
        if am_ov is None or am_un is None or pd.isna(am_ov) or pd.isna(am_un):
            continue
        am_ov_f = float(am_ov)
        am_un_f = float(am_un)
        if am_ov_f == 0 or am_un_f == 0:
            continue

        y_ov = float(r[y_over_col]) if has_y and pd.notna(r[y_over_col]) else None
        pid = str(r.get("player_id") or r.get("player_name") or "")
        pname = str(r.get("player_name") or pid)
        gid = str(r.get("game_id") or r.get("event_id") or "")

        for side, pm, pf, am, won in (
            ("over", p_ov, fair_ov, am_ov_f, (y_ov == 1.0) if y_ov is not None else None),
            ("under", p_un, fair_un, am_un_f, (y_ov == 0.0) if y_ov is not None else None),
        ):
            if side not in side_set:
                continue
            dec = american_to_decimal(am)
            pvig = 1.0 / dec
            f_edge = pm - pf
            v_edge = pm - pvig
            if f_edge < float(min_fair_edge):
                continue
            if require_vs_vigged and v_edge <= float(min_vig_edge):
                continue
            rows.append(
                {
                    "season": r.get("season"),
                    "game_date": pd.Timestamp(r["game_date"]).normalize(),
                    "game_id": gid,
                    "player_id": pid,
                    "player_name": pname,
                    "team_abbreviation": str(r.get("team_abbreviation") or ""),
                    "market": str(r.get("market") or r.get("stat") or ""),
                    "stat": str(r.get("stat") or ""),
                    "line": float(r["line"]),
                    "side": side,
                    "p_model": pm,
                    "p_fair_novig": pf,
                    "p_vigged": pvig,
                    "fair_edge": f_edge,
                    "vig_edge": v_edge,
                    "ev_leg": pm * dec - 1.0,
                    "american": am,
                    "decimal": dec,
                    "leg_won": won,
                    "book": str(r.get("book") or ""),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.sort_values(["game_date", "ev_leg"], ascending=[True, False]).reset_index(drop=True)


def enumerate_prop_parlays(
    legs: pd.DataFrame,
    *,
    sizes: Sequence[int] = (2,),
    min_price_edge: float = 0.04,
    max_legs_per_day_pool: int = 12,
    max_parlay_decimal: float | None = 10.0,
    allow_same_game: bool = False,
) -> pd.DataFrame:
    """Enumerate same-day 2/3-leg prop parlays clearing the combined payout price.

    When ``allow_same_game=False`` (default), keeps at most the top leg per game
    before forming combinations so legs are strictly cross-game independent.
    """
    if legs.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for gd, day_df in legs.groupby("game_date", sort=True):
        day_sorted = day_df.sort_values("ev_leg", ascending=False)
        if not allow_same_game:
            pool_df = day_sorted.drop_duplicates("game_id").head(int(max_legs_per_day_pool))
        else:
            pool_df = day_sorted.drop_duplicates("player_id").head(int(max_legs_per_day_pool))

        rows = [pool_df.iloc[i] for i in range(len(pool_df))]
        for n in sizes:
            if len(rows) < int(n):
                continue
            for combo in combinations(range(len(rows)), int(n)):
                picked = [rows[i] for i in combo]
                gids = [str(L["game_id"]) for L in picked]
                pids = [str(L["player_id"]) for L in picked]
                if len(set(pids)) != len(picked):
                    continue
                if not allow_same_game and len(set(gids)) != len(picked):
                    continue
                probs = [float(L["p_model"]) for L in picked]
                ams = [float(L["american"]) for L in picked]
                dec = parlay_decimal(ams)
                if max_parlay_decimal is not None and dec > float(max_parlay_decimal):
                    continue
                if not clears_parlay_price(probs, ams, min_price_edge=min_price_edge):
                    continue
                joint = float(np.prod(probs))
                book_imp = 1.0 / dec
                ev = float(joint * dec - 1.0)
                pe = float(joint / book_imp - 1.0)
                wons = [L["leg_won"] for L in picked]
                has_grade = all(w is not None and not pd.isna(w) for w in wons)
                ticket_won = all(bool(w) for w in wons) if has_grade else None

                records.append(
                    {
                        "game_date": gd,
                        "season": picked[0].get("season"),
                        "n_legs": int(n),
                        "same_game": len(set(gids)) < len(picked),
                        "game_ids": "|".join(gids),
                        "player_ids": "|".join(pids),
                        "players": "|".join(str(L["player_name"]) for L in picked),
                        "stats": "|".join(str(L["stat"]) for L in picked),
                        "sides": "|".join(str(L["side"]) for L in picked),
                        "lines": "|".join(f"{float(L['line']):g}" for L in picked),
                        "americans": "|".join(f"{float(L['american']):+.0f}" for L in picked),
                        "p_joint_model": joint,
                        "parlay_decimal": dec,
                        "book_implied": book_imp,
                        "price_edge": pe,
                        "ev_per_unit": ev,
                        "ticket_won": ticket_won,
                        "min_vig_edge_leg": float(min(float(L["vig_edge"]) for L in picked)),
                    }
                )

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["game_date", "price_edge"], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_nonoverlapping_prop_parlays(
    candidates: pd.DataFrame,
    *,
    max_per_day: int = 2,
    score_col: str = "price_edge",
) -> pd.DataFrame:
    """Select up to ``max_per_day`` player/game-disjoint tickets per slate date."""
    if candidates.empty:
        return candidates.copy()
    picked: list[pd.Series] = []
    for _gd, day_df in candidates.groupby("game_date", sort=True):
        day_sorted = day_df.sort_values(score_col, ascending=False)
        used_games: set[str] = set()
        used_players: set[str] = set()
        count = 0
        for _, row in day_sorted.iterrows():
            gids = set(str(row["game_ids"]).split("|"))
            pids = set(str(row["player_ids"]).split("|"))
            if (gids & used_games) or (pids & used_players):
                continue
            picked.append(row)
            used_games |= gids
            used_players |= pids
            count += 1
            if count >= int(max_per_day):
                break
    if not picked:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame(picked).reset_index(drop=True)


def parlay_kelly_fraction(
    p_joint: float,
    parlay_dec: float,
    *,
    fraction: float = 0.15,
    max_bet_frac: float = 0.03,
) -> float:
    """Fractional Kelly stake share for a parlay with decimal payout ``parlay_dec``."""
    p = float(p_joint)
    b = float(parlay_dec) - 1.0
    if not (0.0 < p < 1.0) or b <= 0.0:
        return 0.0
    full_kelly = (b * p - (1.0 - p)) / b
    if full_kelly <= 0.0:
        return 0.0
    return float(min(float(fraction) * full_kelly, float(max_bet_frac)))


def simulate_prop_parlay_kelly(
    tickets: pd.DataFrame,
    *,
    starting_bankroll: float = 50.0,
    kelly_fraction: float = 0.15,
    max_bet_frac: float = 0.03,
    min_stake: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate chronological fractional-Kelly bankroll from ``starting_bankroll`` ($50 default).

    Stakes are floored to cents (``math.floor(raw * 100) / 100``). Tickets placed on
    the same ``game_date`` allocate from the start-of-day bankroll so simultaneous tips
    never compound intraday outcomes.
    """
    if tickets.empty:
        return tickets.copy(), {
            "starting_bankroll": float(starting_bankroll),
            "ending_bankroll": float(starting_bankroll),
            "net_profit": 0.0,
            "return_pct": 0.0,
            "n_tickets": 0,
            "n_wins": 0,
            "hit_rate": None,
            "total_staked": 0.0,
            "roi_on_staked": None,
            "max_drawdown": 0.0,
        }

    df = tickets.sort_values(["game_date", "price_edge"], ascending=[True, False]).reset_index(
        drop=True
    )
    bankroll = float(starting_bankroll)
    peak = bankroll
    max_dd = 0.0

    stakes: list[float] = []
    pnls: list[float] = []
    balances: list[float] = []

    for _gd, day_idx in df.groupby("game_date", sort=True).groups.items():
        day_start_br = bankroll
        remaining_br = bankroll
        day_pnl = 0.0
        for idx in day_idx:
            row = df.loc[idx]
            frac = parlay_kelly_fraction(
                float(row["p_joint_model"]),
                float(row["parlay_decimal"]),
                fraction=kelly_fraction,
                max_bet_frac=max_bet_frac,
            )
            raw_stake = math.floor(day_start_br * frac * 100.0) / 100.0
            stake = min(raw_stake, remaining_br)
            if stake < float(min_stake):
                stakes.append(0.0)
                pnls.append(0.0)
                balances.append(bankroll + day_pnl)
                continue
            remaining_br = round(remaining_br - stake, 2)
            won = bool(row["ticket_won"])
            pnl = round(stake * (float(row["parlay_decimal"]) - 1.0), 2) if won else -stake
            day_pnl = round(day_pnl + pnl, 2)
            stakes.append(stake)
            pnls.append(pnl)
            balances.append(round(day_start_br + day_pnl, 2))

        bankroll = round(day_start_br + day_pnl, 2)
        peak = max(peak, bankroll)
        max_dd = min(max_dd, bankroll - peak)

    df["kelly_stake"] = stakes
    df["kelly_pnl"] = pnls
    df["kelly_bankroll"] = balances

    staked_mask = df["kelly_stake"] > 0
    n_staked = int(staked_mask.sum())
    n_wins = int(df.loc[staked_mask, "ticket_won"].astype(bool).sum()) if n_staked else 0
    tot_staked = float(df["kelly_stake"].sum())
    tot_pnl = float(bankroll - float(starting_bankroll))

    # Also compute flat 1u ROI for reference
    unit_pnls = np.where(
        df["ticket_won"].astype(bool),
        df["parlay_decimal"].to_numpy(float) - 1.0,
        -1.0,
    )
    unit_roi = float(np.mean(unit_pnls)) if len(df) else None

    summary = {
        "starting_bankroll": float(starting_bankroll),
        "ending_bankroll": float(bankroll),
        "net_profit": round(tot_pnl, 2),
        "return_pct": float(tot_pnl / float(starting_bankroll)),
        "n_tickets": n_staked,
        "n_wins": n_wins,
        "hit_rate": (n_wins / n_staked) if n_staked else None,
        "total_staked": round(tot_staked, 2),
        "roi_on_staked": (float(df["kelly_pnl"].sum()) / tot_staked) if tot_staked > 0 else None,
        "unit_roi": unit_roi,
        "max_drawdown": round(float(max_dd), 2),
        "kelly_fraction": float(kelly_fraction),
        "max_bet_frac": float(max_bet_frac),
    }
    return df, summary
