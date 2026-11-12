"""Multi-market cross-game parlay & correlated bivariate SGP selector + seasonal Kelly bankroll.

Supports:
- 3 full closing markets per game: Moneyline (`ml`), Point Spread (`spread`), Total (`total`)
- Exact push settlement (`push -> 1.0x` decimal leg multiplier; N-leg drops to N-1 legs)
- Exact bivariate normal `(margin, total)` SGP pricing with push cross-terms
- Strictly start-of-day fractional Kelly sizing from a seasonal bankroll ($50 default)
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nba_betting.pricing.kelly import american_to_decimal


def settle_parlay_legs(
    outcomes: Sequence[str],
    decimals: Sequence[float],
) -> tuple[str, float, int]:
    """Settle a multi-leg parlay with push leg-reduction.

    Each ``outcome`` is ``'win'``, ``'push'``, or ``'loss'``.
    - Any ``'loss'`` -> ``('loss', 0.0, 0)``.
    - Pushed legs carry multiplier ``1.0`` (an N-leg ticket with 1 push and N-1
      wins settles at the N-1 leg decimal product).
    - All legs ``'push'`` -> ``('push', 1.0, 0)`` (stake refunded, zero profit).

    Returns ``(ticket_status, effective_decimal_multiplier, n_winning_legs)``.
    """
    if len(outcomes) != len(decimals) or not outcomes:
        raise ValueError("outcomes and decimals must be non-empty and equal length")
    for o in outcomes:
        if o == "loss":
            return "loss", 0.0, 0
        if o not in {"win", "push"}:
            raise ValueError(f"invalid leg outcome: {o!r}")

    eff_dec = 1.0
    n_wins = 0
    for o, d in zip(outcomes, decimals):
        if o == "win":
            eff_dec *= float(d)
            n_wins += 1
        # 'push' multiplies by 1.0
    if n_wins == 0:
        return "push", 1.0, 0
    return "win", float(eff_dec), int(n_wins)


def build_multimarket_legs(
    games: pd.DataFrame,
    *,
    markets: Sequence[str] = ("ml", "spread", "total"),
    require_vs_vigged: bool = True,
    min_fair_edge: float = 0.0,
    min_vig_ev: float = 0.0,
    require_dmochowski_gate: bool = False,
    dmochowski_buffer: float = 0.0,
) -> pd.DataFrame:
    """Build candidate legs across ML, ATS spread, and Over/Under totals.

    When ``require_dmochowski_gate=True`` (Dmochowski 2023), spread and total
    legs are kept ONLY when the book line sits strictly outside the model's
    conditional ``[0.4762, 0.5238]`` quantile band by at least ``dmochowski_buffer``
    points (preventing OT / Q4 bench-clearing skewness from creating fake mean edges).
    """
    allowed = set(markets)
    rows: list[dict[str, Any]] = []

    for idx, g in games.iterrows():
        gid = str(g["game_id"])
        gd = pd.Timestamp(g["game_date"]).normalize()
        season = g.get("season")
        base = {
            "row_idx": int(idx),
            "season": season,
            "game_date": gd,
            "game_id": gid,
            "home_tricode": g.get("home_tricode", ""),
            "away_tricode": g.get("away_tricode", ""),
        }

        candidates: list[tuple[str, str, float | None, float | None, float | None, float, float | None, str, float | None]] = []

        if "ml" in allowed:
            hw = g.get("home_win")
            if hw is not None and np.isfinite(hw):
                # For ML, 0-margin vs Dmochowski quantile band when present
                dm_ml_h = float(g["q_margin_4762"]) if "q_margin_4762" in g and pd.notna(g["q_margin_4762"]) else None
                dm_ml_a = (-float(g["q_margin_5238"])) if "q_margin_5238" in g and pd.notna(g["q_margin_5238"]) else None
                candidates.append(
                    (
                        "ml",
                        "home",
                        0.0,
                        g.get("home_ml"),
                        g.get("p_model_home_ml"),
                        0.0,
                        g.get("p_home_shin") if pd.notna(g.get("p_home_shin")) else g.get("p_home_novig"),
                        "win" if float(hw) == 1.0 else "loss",
                        dm_ml_h,
                    )
                )
                candidates.append(
                    (
                        "ml",
                        "away",
                        0.0,
                        g.get("away_ml"),
                        g.get("p_model_away_ml"),
                        0.0,
                        g.get("p_away_shin") if pd.notna(g.get("p_away_shin")) else g.get("p_away_novig"),
                        "win" if float(hw) == 0.0 else "loss",
                        dm_ml_a,
                    )
                )

        if "spread" in allowed and g.get("spread_home") is not None and np.isfinite(g.get("spread_home")):
            ats_res = str(g.get("ats_result", "missing"))
            if ats_res in {"home", "away", "push"}:
                p_sp_push = float(g.get("p_model_spread_push") or 0.0)
                sp_h = float(g["spread_home"])
                dm_hc = float(g["dmochowski_home_cover_margin"]) if "dmochowski_home_cover_margin" in g and pd.notna(g["dmochowski_home_cover_margin"]) else None
                dm_ac = float(g["dmochowski_away_cover_margin"]) if "dmochowski_away_cover_margin" in g and pd.notna(g["dmochowski_away_cover_margin"]) else None
                candidates.append(
                    (
                        "spread",
                        "home",
                        sp_h,
                        g.get("spread_home_odds"),
                        g.get("p_model_home_cover"),
                        p_sp_push,
                        g.get("p_spread_home_shin") if pd.notna(g.get("p_spread_home_shin")) else g.get("p_spread_home_novig"),
                        "push" if ats_res == "push" else ("win" if ats_res == "home" else "loss"),
                        dm_hc,
                    )
                )
                candidates.append(
                    (
                        "spread",
                        "away",
                        -sp_h,
                        g.get("spread_away_odds"),
                        g.get("p_model_away_cover"),
                        p_sp_push,
                        g.get("p_spread_away_shin") if pd.notna(g.get("p_spread_away_shin")) else g.get("p_spread_away_novig"),
                        "push" if ats_res == "push" else ("win" if ats_res == "away" else "loss"),
                        dm_ac,
                    )
                )

        if "total" in allowed and g.get("total_line") is not None and np.isfinite(g.get("total_line")):
            tot_res = str(g.get("total_result", "missing"))
            if tot_res in {"over", "under", "push"}:
                p_tot_push = float(g.get("p_model_total_push") or 0.0)
                tl = float(g["total_line"])
                dm_ov = float(g["dmochowski_over_margin"]) if "dmochowski_over_margin" in g and pd.notna(g["dmochowski_over_margin"]) else None
                dm_un = float(g["dmochowski_under_margin"]) if "dmochowski_under_margin" in g and pd.notna(g["dmochowski_under_margin"]) else None
                candidates.append(
                    (
                        "total",
                        "over",
                        tl,
                        g.get("over_odds"),
                        g.get("p_model_over"),
                        p_tot_push,
                        g.get("p_over_shin") if pd.notna(g.get("p_over_shin")) else g.get("p_over_novig"),
                        "push" if tot_res == "push" else ("win" if tot_res == "over" else "loss"),
                        dm_ov,
                    )
                )
                candidates.append(
                    (
                        "total",
                        "under",
                        tl,
                        g.get("under_odds"),
                        g.get("p_model_under"),
                        p_tot_push,
                        g.get("p_under_shin") if pd.notna(g.get("p_under_shin")) else g.get("p_under_novig"),
                        "push" if tot_res == "push" else ("win" if tot_res == "under" else "loss"),
                        dm_un,
                    )
                )

        for mkt, side, line, price, p_win, p_push, p_fair, outcome, dm_margin in candidates:
            if price is None or p_win is None or p_fair is None:
                continue
            if require_dmochowski_gate and mkt in {"spread", "total"}:
                if dm_margin is None or float(dm_margin) < float(dmochowski_buffer):
                    continue
            pr = float(price)
            pw = float(p_win)
            pp = float(p_push)
            pf = float(p_fair)
            if not (np.isfinite(pr) and pr != 0.0 and np.isfinite(pw) and np.isfinite(pf)):
                continue
            if not (0.0 < pw < 1.0):
                continue
            dec = american_to_decimal(pr)
            pw_cond = pw / max(1.0 - pp, 1e-6)
            fair_edge = pw_cond - pf
            exp_return = pw * dec + pp * 1.0
            vig_ev = exp_return - 1.0
            if fair_edge < float(min_fair_edge):
                continue
            if require_vs_vigged and vig_ev <= float(min_vig_ev):
                continue
            rows.append(
                {
                    **base,
                    "market": mkt,
                    "side": side,
                    "line": float(line) if line is not None else 0.0,
                    "american": pr,
                    "decimal": dec,
                    "p_win": pw,
                    "p_push": pp,
                    "p_fair_novig": pf,
                    "fair_edge": fair_edge,
                    "vig_ev": vig_ev,
                    "exp_return_1u": exp_return,
                    "outcome": outcome,
                    "dmochowski_margin_pts": float(dm_margin) if dm_margin is not None else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["game_date", "game_id", "market", "side"]).reset_index(drop=True)


def _sgp_pair_expected_return(
    leg_a: dict[str, Any],
    leg_b: dict[str, Any],
    game_row: pd.Series,
    *,
    sgp_vig_haircut: float = 1.0,
) -> tuple[float, float] | None:
    """Return ``(expected_return_1u, p_both_win)`` for a valid 2-leg SGP on one game.

    Valid SGP pairs combine one margin-side leg (``spread`` or ``ml``) with one
    ``total`` leg (``over`` or ``under``). ``ml + spread`` or two legs of the
    same market on the same game are rejected (``None``).
    """
    ma, mb = str(leg_a["market"]), str(leg_b["market"])
    if {ma, mb} not in ({"spread", "total"}, {"ml", "total"}):
        return None
    m_leg, t_leg = (leg_a, leg_b) if mb == "total" else (leg_b, leg_a)
    m_mkt = str(m_leg["market"])
    m_side = str(m_leg["side"])
    t_side = str(t_leg["side"])

    tag_m = f"{m_side}_cover" if m_mkt == "spread" else f"{m_side}_ml"
    p_ww = float(game_row.get(f"p_sgp_{tag_m}_{t_side}", np.nan))
    if not np.isfinite(p_ww):
        return None
    p_wp = float(game_row.get(f"p_sgp_{tag_m}_push_{t_side}", 0.0) or 0.0)
    if m_mkt == "spread":
        p_pw = float(game_row.get(f"p_sgp_push_{tag_m}_{t_side}", 0.0) or 0.0)
        p_pp = float(game_row.get("p_sgp_both_push", 0.0) or 0.0)
    else:
        p_pw = 0.0
        p_pp = 0.0

    dm = float(m_leg["decimal"])
    dt = float(t_leg["decimal"])
    full_dec = dm * dt * float(sgp_vig_haircut)
    exp_ret = p_ww * full_dec + p_wp * dm + p_pw * dt + p_pp * 1.0
    return float(exp_ret), float(p_ww)


def enumerate_multimarket_parlays(
    legs: pd.DataFrame,
    games_indexed: pd.DataFrame,
    *,
    sizes: Sequence[int] = (2, 3),
    min_price_edge: float = 0.08,
    max_parlay_decimal: float | None = 10.0,
    allow_cross_game: bool = True,
    allow_same_game: bool = True,
    sgp_vig_haircut: float = 1.0,
    max_legs_per_day_pool: int = 18,
) -> pd.DataFrame:
    """Enumerate same-day multi-market cross-game parlays and correlated SGPs.

    ``price_edge`` is exact push-adjusted ``E[R_ticket] - 1.0`` (which equals
    ``p_joint * parlay_decimal - 1`` whenever ``p_push == 0``).
    """
    if legs.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for day, day_df in legs.groupby("game_date", sort=True):
        # Keep at most top `max_legs_per_day_pool` legs by vig_ev per day for fast combinatorial search
        if len(day_df) > int(max_legs_per_day_pool):
            day_df = day_df.sort_values("vig_ev", ascending=False).head(int(max_legs_per_day_pool))
        day_records = day_df.to_dict(orient="records")

        for k in sizes:
            if len(day_records) < k:
                continue
            for combo in combinations(day_records, k):
                gids = [str(x["game_id"]) for x in combo]
                unique_gids = set(gids)
                n_unique = len(unique_gids)

                # Case 1: All distinct games (cross-game parlay)
                if n_unique == k:
                    if not allow_cross_game:
                        continue
                    same_game = False
                    raw_dec = float(np.prod([float(x["decimal"]) for x in combo]))
                    quoted_dec = raw_dec
                    exp_ret = float(np.prod([float(x["exp_return_1u"]) for x in combo]))
                    p_joint_win = float(np.prod([float(x["p_win"]) for x in combo]))

                # Case 2: 2-leg SGP on a single game
                elif k == 2 and n_unique == 1:
                    if not allow_same_game:
                        continue
                    same_game = True
                    grow = games_indexed.loc[int(combo[0]["row_idx"])]
                    sgp_res = _sgp_pair_expected_return(
                        combo[0], combo[1], grow, sgp_vig_haircut=sgp_vig_haircut
                    )
                    if sgp_res is None:
                        continue
                    exp_ret, p_joint_win = sgp_res
                    raw_dec = float(combo[0]["decimal"]) * float(combo[1]["decimal"])
                    quoted_dec = raw_dec * float(sgp_vig_haircut)

                # Case 3: 3-leg hybrid (2-leg SGP on game A + 1 independent leg on game B)
                elif k == 3 and n_unique == 2:
                    if not (allow_same_game and allow_cross_game):
                        continue
                    same_game = True
                    counts: dict[str, list[dict[str, Any]]] = {}
                    for x in combo:
                        counts.setdefault(str(x["game_id"]), []).append(x)
                    sgp_legs = [v for v in counts.values() if len(v) == 2][0]
                    ind_leg = [v for v in counts.values() if len(v) == 1][0][0]
                    grow = games_indexed.loc[int(sgp_legs[0]["row_idx"])]
                    sgp_res = _sgp_pair_expected_return(
                        sgp_legs[0], sgp_legs[1], grow, sgp_vig_haircut=sgp_vig_haircut
                    )
                    if sgp_res is None:
                        continue
                    sgp_ret, sgp_p_win = sgp_res
                    exp_ret = float(sgp_ret) * float(ind_leg["exp_return_1u"])
                    p_joint_win = float(sgp_p_win) * float(ind_leg["p_win"])
                    raw_dec = (
                        float(sgp_legs[0]["decimal"])
                        * float(sgp_legs[1]["decimal"])
                        * float(ind_leg["decimal"])
                    )
                    quoted_dec = raw_dec * float(sgp_vig_haircut)
                else:
                    continue

                if max_parlay_decimal is not None and quoted_dec > float(max_parlay_decimal):
                    continue

                ev_unit = exp_ret - 1.0
                if ev_unit < float(min_price_edge):
                    continue

                outcomes = [str(x["outcome"]) for x in combo]
                leg_decs = [float(x["decimal"]) for x in combo]
                status, eff_dec, n_win_legs = settle_parlay_legs(outcomes, leg_decs)
                # Apply SGP haircut only when both SGP legs won together
                if same_game and status == "win" and n_win_legs == k:
                    eff_dec *= float(sgp_vig_haircut)

                records.append(
                    {
                        "game_date": day,
                        "season": combo[0]["season"],
                        "n_legs": int(k),
                        "same_game": bool(same_game),
                        "game_ids": "|".join(gids),
                        "markets": "|".join(str(x["market"]) for x in combo),
                        "sides": "|".join(str(x["side"]) for x in combo),
                        "lines": "|".join(f"{float(x['line']):g}" for x in combo),
                        "americans": "|".join(f"{float(x['american']):.0f}" for x in combo),
                        "outcomes": "|".join(outcomes),
                        "p_joint_win": p_joint_win,
                        "parlay_decimal": quoted_dec,
                        "book_implied": 1.0 / quoted_dec,
                        "ev_per_unit": ev_unit,
                        "price_edge": ev_unit,
                        "ticket_status": status,
                        "ticket_won": status == "win",
                        "ticket_pushed": status == "push",
                        "effective_decimal": eff_dec,
                        "n_winning_legs": n_win_legs,
                        "n_pushed_legs": int(sum(1 for o in outcomes if o == "push")),
                    }
                )

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["game_date", "ev_per_unit"], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_daily_multimarket_tickets(
    candidates: pd.DataFrame,
    *,
    max_per_day: int = 1,
) -> pd.DataFrame:
    """Greedy non-overlapping ticket selection per day by ``ev_per_unit``."""
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values(["game_date", "ev_per_unit"], ascending=[True, False])
    keep_idx: list[int] = []
    cur_day = None
    used_games: set[str] = set()
    n = 0
    for idx_val, day_val, gids_str in zip(
        ordered.index.to_numpy(),
        ordered["game_date"].to_numpy(),
        ordered["game_ids"].to_numpy(),
    ):
        if day_val != cur_day:
            cur_day = day_val
            used_games = set()
            n = 0
        if n >= int(max_per_day):
            continue
        gids = set(str(gids_str).split("|"))
        if gids & used_games:
            continue
        keep_idx.append(int(idx_val))
        used_games |= gids
        n += 1
    if not keep_idx:
        return candidates.iloc[0:0].copy()
    return ordered.loc[keep_idx].reset_index(drop=True)


def _floor_cent(x: float) -> float:
    return math.floor(float(x) * 100.0 + 1e-12) / 100.0


def kelly_parlay_stake_fraction(
    ev_per_unit: float,
    parlay_decimal: float,
    *,
    kelly_fraction: float = 0.25,
    max_bet_frac: float = 0.05,
) -> float:
    """Fractional Kelly stake share ``kelly_fraction * (EV / (decimal - 1))``, capped."""
    b = float(parlay_decimal) - 1.0
    if b <= 1e-6 or float(ev_per_unit) <= 0.0:
        return 0.0
    full_kelly = float(ev_per_unit) / b
    return float(min(max(full_kelly * float(kelly_fraction), 0.0), float(max_bet_frac)))


def simulate_seasonal_kelly_bankroll(
    tickets: pd.DataFrame,
    *,
    starting_bankroll: float = 50.0,
    kelly_fraction: float = 0.25,
    max_bet_frac: float = 0.05,
    max_daily_frac: float = 0.12,
    fixed_pct: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay tickets chronologically with start-of-day fractional Kelly sizing.

    If ``fixed_pct`` is provided, stakes ``fixed_pct * day_bankroll`` instead of Kelly.
    Each season in ``tickets['season']`` is replayed from ``starting_bankroll``
    independently (no cross-season compounding).
    """
    if tickets.empty:
        return tickets.copy(), {
            "starting_bankroll": float(starting_bankroll),
            "by_season": {},
            "n_tickets": 0,
        }

    all_rows: list[dict[str, Any]] = []
    by_season: dict[str, Any] = {}

    for season, s_df in tickets.groupby("season", sort=True):
        br = float(starting_bankroll)
        peak = br
        max_dd = 0.0
        broke = False
        total_staked = 0.0
        total_pnl = 0.0
        n_staked = 0
        n_wins = 0
        n_pushes = 0
        n_partial_push_wins = 0
        n_sgp = 0

        for gd, day_df in s_df.groupby("game_date", sort=True):
            day_start_br = br
            day_staked = 0.0
            day_cap = _floor_cent(day_start_br * float(max_daily_frac))
            day_pnl = 0.0

            for _, row in day_df.sort_values("ev_per_unit", ascending=False).iterrows():
                if day_start_br <= 0.0 or broke:
                    frac = 0.0
                    stake = 0.0
                else:
                    if fixed_pct is not None:
                        frac = float(fixed_pct)
                    else:
                        frac = kelly_parlay_stake_fraction(
                            float(row["ev_per_unit"]),
                            float(row["parlay_decimal"]),
                            kelly_fraction=kelly_fraction,
                            max_bet_frac=max_bet_frac,
                        )
                    raw_stake = _floor_cent(day_start_br * frac)
                    rem = max(0.0, round(min(day_cap, day_start_br) - day_staked, 2))
                    stake = min(raw_stake, rem)
                    if stake < 0.01:
                        stake = 0.0

                eff_dec = float(row["effective_decimal"])
                status = str(row["ticket_status"])
                if stake <= 0.0:
                    payout = 0.0
                    pnl = 0.0
                elif status == "loss":
                    payout = 0.0
                    pnl = -stake
                elif status == "push":
                    payout = stake
                    pnl = 0.0
                else:
                    payout = round(stake * eff_dec, 2)
                    pnl = round(payout - stake, 2)

                if stake > 0.0:
                    n_staked += 1
                    total_staked = round(total_staked + stake, 2)
                    day_staked = round(day_staked + stake, 2)
                    day_pnl = round(day_pnl + pnl, 2)
                    total_pnl = round(total_pnl + pnl, 2)
                    if status == "win":
                        n_wins += 1
                        if int(row.get("n_pushed_legs", 0)) > 0:
                            n_partial_push_wins += 1
                    elif status == "push":
                        n_pushes += 1
                    if bool(row.get("same_game", False)):
                        n_sgp += 1

                rec = row.to_dict()
                rec.update(
                    {
                        "kelly_frac_used": frac,
                        "stake": stake,
                        "payout": payout,
                        "pnl": pnl,
                        "bankroll_day_start": day_start_br,
                    }
                )
                all_rows.append(rec)

            br = round(day_start_br + day_pnl, 2)
            if br <= 0.0:
                br = 0.0
                broke = True
            peak = max(peak, br)
            max_dd = min(max_dd, round(br - peak, 2))
            for r in reversed(all_rows):
                if r["season"] == season and r["game_date"] == gd and "bankroll_end_of_day" not in r:
                    r["bankroll_end_of_day"] = br
                elif "bankroll_end_of_day" in r:
                    break

        by_season[str(season)] = {
            "start": float(starting_bankroll),
            "end": float(br),
            "net_profit": round(br - float(starting_bankroll), 2),
            "total_staked": float(total_staked),
            "roi_on_staked": (total_pnl / total_staked) if total_staked > 0 else None,
            "max_drawdown": float(max_dd),
            "broke": bool(broke),
            "n_tickets": int(n_staked),
            "n_wins": int(n_wins),
            "n_partial_push_wins": int(n_partial_push_wins),
            "n_all_pushes": int(n_pushes),
            "n_sgp_tickets": int(n_sgp),
        }

    graded_df = pd.DataFrame(all_rows)
    return graded_df, {
        "starting_bankroll": float(starting_bankroll),
        "kelly_fraction": float(kelly_fraction),
        "max_bet_frac": float(max_bet_frac),
        "fixed_pct": fixed_pct,
        "by_season": by_season,
        "n_tickets": int(sum(v["n_tickets"] for v in by_season.values())),
    }


def fast_seasonal_kelly_summary(
    tickets: pd.DataFrame,
    *,
    starting_bankroll: float = 50.0,
    kelly_fraction: float = 0.25,
    max_bet_frac: float = 0.05,
    max_daily_frac: float = 0.12,
    fixed_pct: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Ultra-fast array-level seasonal Kelly simulator (zero pandas iterrows)."""
    if tickets.empty:
        return {}
    df = tickets.sort_values(["season", "game_date", "ev_per_unit"], ascending=[True, True, False])
    seasons = df["season"].to_numpy()
    dates = df["game_date"].to_numpy()
    evs = df["ev_per_unit"].to_numpy(dtype=float)
    decs = df["parlay_decimal"].to_numpy(dtype=float)
    eff_decs = df["effective_decimal"].to_numpy(dtype=float)
    statuses = df["ticket_status"].to_numpy()
    n_push_arr = (
        df["n_pushed_legs"].to_numpy(dtype=int)
        if "n_pushed_legs" in df.columns
        else np.zeros(len(df), dtype=int)
    )
    sgp_arr = (
        df["same_game"].to_numpy(dtype=bool)
        if "same_game" in df.columns
        else np.zeros(len(df), dtype=bool)
    )

    by_season: dict[str, dict[str, Any]] = {}
    n_rows = len(df)
    i = 0
    while i < n_rows:
        ssn = str(seasons[i])
        br = float(starting_bankroll)
        peak = br
        max_dd = 0.0
        broke = False
        total_staked = 0.0
        total_pnl = 0.0
        n_staked = 0
        n_wins = 0
        n_pushes = 0
        n_partial_push_wins = 0
        n_sgp = 0

        while i < n_rows and str(seasons[i]) == ssn:
            cur_date = dates[i]
            day_start_br = br
            day_staked = 0.0
            day_cap = _floor_cent(day_start_br * float(max_daily_frac))
            day_pnl = 0.0

            while i < n_rows and str(seasons[i]) == ssn and dates[i] == cur_date:
                if day_start_br > 0.0 and not broke:
                    if fixed_pct is not None:
                        frac = float(fixed_pct)
                    else:
                        frac = kelly_parlay_stake_fraction(
                            float(evs[i]),
                            float(decs[i]),
                            kelly_fraction=kelly_fraction,
                            max_bet_frac=max_bet_frac,
                        )
                    raw_stake = _floor_cent(day_start_br * frac)
                    rem = max(0.0, round(min(day_cap, day_start_br) - day_staked, 2))
                    stake = min(raw_stake, rem)
                    if stake < 0.01:
                        stake = 0.0
                else:
                    stake = 0.0

                if stake > 0.0:
                    st_str = str(statuses[i])
                    if st_str == "loss":
                        pnl = -stake
                    elif st_str == "push":
                        pnl = 0.0
                        n_pushes += 1
                    else:
                        payout = round(stake * float(eff_decs[i]), 2)
                        pnl = round(payout - stake, 2)
                        n_wins += 1
                        if int(n_push_arr[i]) > 0:
                            n_partial_push_wins += 1
                    if bool(sgp_arr[i]):
                        n_sgp += 1
                    n_staked += 1
                    total_staked = round(total_staked + stake, 2)
                    day_staked = round(day_staked + stake, 2)
                    day_pnl = round(day_pnl + pnl, 2)
                    total_pnl = round(total_pnl + pnl, 2)
                i += 1

            br = round(day_start_br + day_pnl, 2)
            if br <= 0.0:
                br = 0.0
                broke = True
            peak = max(peak, br)
            max_dd = min(max_dd, round(br - peak, 2))

        by_season[ssn] = {
            "start": float(starting_bankroll),
            "end": float(br),
            "net_profit": round(br - float(starting_bankroll), 2),
            "total_staked": float(total_staked),
            "roi_on_staked": (total_pnl / total_staked) if total_staked > 0 else None,
            "max_drawdown": float(max_dd),
            "broke": bool(broke),
            "n_tickets": int(n_staked),
            "n_wins": int(n_wins),
            "n_partial_push_wins": int(n_partial_push_wins),
            "n_all_pushes": int(n_pushes),
            "n_sgp_tickets": int(n_sgp),
        }
    return by_season
