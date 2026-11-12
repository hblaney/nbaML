"""Same-day (and same-game when multi-market) parlay selector.

A ticket is kept only when the **combined** model probability beats the
**actual parlay payout price** (product of vigged American→decimal odds):

    p_joint = ∏ p_model_i
    book_implied = 1 / ∏ decimal_i
    keep iff p_joint >= book_implied * (1 + min_price_edge)

Per-leg edge vs a straight de-vig price is optional pre-filter only — it is
not sufficient to claim +EV. No straight bets as the product.

Same-game: only when legs share ``game_id`` but differ in ``market``
(e.g. ML + spread). Two moneylines from the same game are rejected.
With ESPN ML-only closes, same-game multi-market tickets are unavailable.
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


def clears_parlay_price(
    model_probs: Sequence[float],
    american_prices: Sequence[float],
    *,
    min_price_edge: float = 0.0,
) -> bool:
    """True iff combined model p beats book parlay implied by ``min_price_edge`` relative.

    ``min_price_edge=0.25`` means require ``p_joint >= 1.25 * book_implied``.
    """
    dec = parlay_decimal(american_prices)
    if dec <= 1.0:
        return False
    book_implied = 1.0 / dec
    joint = 1.0
    for p in model_probs:
        pp = float(p)
        if not 0.0 < pp < 1.0:
            return False
        joint *= pp
    return bool(joint >= book_implied * (1.0 + float(min_price_edge)))


def build_eligible_legs(
    games: pd.DataFrame,
    *,
    model_home_col: str = "p_model_home",
    novig_home_col: str = "p_home_novig",
    home_ml_col: str = "home_ml",
    away_ml_col: str = "away_ml",
    exec_home_ml_col: str | None = None,
    exec_away_ml_col: str | None = None,
    exec_home_book_col: str | None = None,
    exec_away_book_col: str | None = None,
    devig_method: str | None = None,
    y_home_col: str = "home_win",
    min_fair_edge: float = 0.0,
    require_vs_vigged: bool = True,
    date_col: str = "game_date",
    game_id_col: str = "game_id",
) -> pd.DataFrame:
    """One row per eligible ML side.

    Default: model must beat the **vigged** straight implied (harder than novig).
    ``min_fair_edge`` is an extra floor vs de-vig fair (0 disables).
    When ``devig_method`` is supplied (e.g. ``"shin"``, ``"power"``, ``"probit"``),
    recomputes ``(fair_h, fair_a)`` from ``(home_ml_col, away_ml_col)`` using that
    devigging method. When ``exec_home_ml_col`` / ``exec_away_ml_col`` are supplied,
    uses best executable cross-book prices for payout grading and vigged-price gating.
    """
    from nba_betting.pricing.american import devig_two_way

    need = [model_home_col, home_ml_col, away_ml_col, y_home_col, date_col, game_id_col]
    if devig_method is None:
        need.append(novig_home_col)
    missing = [c for c in need if c not in games.columns]
    if missing:
        raise ValueError(f"games missing columns: {missing}")

    eh_col = exec_home_ml_col or home_ml_col
    ea_col = exec_away_ml_col or away_ml_col

    rows: list[dict[str, Any]] = []
    for _, g in games.iterrows():
        p_h = float(g[model_home_col])
        ref_h_ml = float(g[home_ml_col])
        ref_a_ml = float(g[away_ml_col])
        if (
            devig_method is not None
            and np.isfinite(ref_h_ml)
            and np.isfinite(ref_a_ml)
            and ref_h_ml != 0
            and ref_a_ml != 0
        ):
            fair_h, fair_a = devig_two_way(ref_h_ml, ref_a_ml, method=devig_method)
        else:
            fair_h = float(g[novig_home_col])
            fair_a = 1.0 - fair_h
        p_a = 1.0 - p_h
        home_won = float(g[y_home_col])
        h_book = g.get(exec_home_book_col) if exec_home_book_col else g.get("provider")
        a_book = g.get(exec_away_book_col) if exec_away_book_col else g.get("provider")
        base = {
            "season": g.get("season"),
            "game_date": pd.Timestamp(g[date_col]).normalize(),
            "game_id": str(g[game_id_col]),
            "market": "ml",
        }
        for side, p_m, p_fair, price, book_used, won in (
            ("home", p_h, fair_h, float(g[eh_col]), h_book, home_won == 1.0),
            ("away", p_a, fair_a, float(g[ea_col]), a_book, home_won == 0.0),
        ):
            if not np.isfinite(price) or price == 0:
                continue
            vig_imp = float(american_to_decimal(price))
            # implied from american: 1/decimal
            p_vig = 1.0 / vig_imp
            fair_edge = p_m - p_fair
            vig_edge = p_m - p_vig
            if fair_edge < float(min_fair_edge):
                continue
            if require_vs_vigged and vig_edge <= 0.0:
                continue
            rows.append(
                {
                    **base,
                    "provider": book_used,
                    "side": side,
                    "p_model": p_m,
                    "p_fair_novig": p_fair,
                    "p_vigged": p_vig,
                    "fair_edge": fair_edge,
                    "vig_edge": vig_edge,
                    "american": price,
                    "leg_won": bool(won),
                    "decimal": american_to_decimal(price),
                }
            )
    cols = [
        "season",
        "game_date",
        "game_id",
        "provider",
        "market",
        "side",
        "p_model",
        "p_fair_novig",
        "p_vigged",
        "fair_edge",
        "vig_edge",
        "american",
        "leg_won",
        "decimal",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.sort_values(["game_date", "game_id", "side"]).reset_index(drop=True)


def _legs_compatible(legs: Sequence[pd.Series]) -> bool:
    """Distinct games, or same game with distinct markets (SGP). No home+away ML."""
    gids = [str(L["game_id"]) for L in legs]
    markets = [str(L.get("market", "ml")) for L in legs]
    sides = [str(L["side"]) for L in legs]
    if len(set(gids)) == len(legs):
        return True
    # same-game: all same game_id, markets unique, and not two ML sides
    if len(set(gids)) != 1:
        return False
    if len(set(markets)) != len(legs):
        return False
    if markets.count("ml") > 1:
        return False
    if len(set(sides)) != len(legs) and all(m == "ml" for m in markets):
        return False
    return True


def _candidate_from_legs(
    legs: Sequence[pd.Series],
    *,
    n_legs: int,
    min_price_edge: float,
    max_parlay_decimal: float | None,
) -> dict[str, Any] | None:
    if len(legs) != n_legs or not _legs_compatible(legs):
        return None
    probs = [float(L["p_model"]) for L in legs]
    prices = [float(L["american"]) for L in legs]
    dec = parlay_decimal(prices)
    if max_parlay_decimal is not None and dec > float(max_parlay_decimal):
        return None
    if not clears_parlay_price(probs, prices, min_price_edge=min_price_edge):
        return None
    joint = float(np.prod(probs))
    book_implied = 1.0 / dec
    ev = float(joint * dec - 1.0)
    price_edge = joint / book_implied - 1.0
    gids = [str(L["game_id"]) for L in legs]
    same_game = len(set(gids)) == 1
    won = all(bool(L["leg_won"]) for L in legs)
    return {
        "game_date": legs[0]["game_date"],
        "season": legs[0].get("season"),
        "n_legs": int(n_legs),
        "same_game": bool(same_game),
        "game_ids": "|".join(gids),
        "markets": "|".join(str(L.get("market", "ml")) for L in legs),
        "sides": "|".join(str(L["side"]) for L in legs),
        "americans": "|".join(f"{float(L['american']):.0f}" for L in legs),
        "p_joint_model": joint,
        "parlay_decimal": dec,
        "book_implied": book_implied,
        "price_edge": float(price_edge),
        "ev_per_unit": ev,
        "survives_parlay_price": True,
        "ticket_won": bool(won),
        "min_fair_edge_leg": float(min(float(L["fair_edge"]) for L in legs)),
        "min_vig_edge_leg": float(min(float(L.get("vig_edge", L["fair_edge"])) for L in legs)),
    }


def enumerate_parlays(
    legs: pd.DataFrame,
    *,
    sizes: Sequence[int] = (2, 3),
    min_price_edge: float = 0.25,
    max_parlay_decimal: float | None = 8.0,
    same_day: bool = True,
    allow_same_game: bool = True,
) -> pd.DataFrame:
    """Enumerate tickets that clear the combined-vs-parlay-price gate.

    Grouping: by ``game_date`` when ``same_day`` (default). Same-game multi-market
    combos are included when ``allow_same_game`` and compatible legs exist.
    """
    if legs.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    groups = legs.groupby("game_date", sort=True) if same_day else [("all", legs)]
    for _, day_legs in groups:
        rows = [day_legs.iloc[i] for i in range(len(day_legs))]
        for n in sizes:
            if len(rows) < n:
                continue
            for combo in combinations(range(len(rows)), n):
                picked = [rows[i] for i in combo]
                gids = {str(L["game_id"]) for L in picked}
                if len(gids) == 1 and not allow_same_game:
                    continue
                if len(gids) == 1 and all(str(L.get("market", "ml")) == "ml" for L in picked):
                    # mutually exclusive ML sides — never a valid ticket
                    continue
                cand = _candidate_from_legs(
                    picked,
                    n_legs=n,
                    min_price_edge=min_price_edge,
                    max_parlay_decimal=max_parlay_decimal,
                )
                if cand is None:
                    continue
                records.append(cand)
    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["game_date", "price_edge"], ascending=[True, False])
        .reset_index(drop=True)
    )


# Back-compat alias
def enumerate_same_day_parlays(
    legs: pd.DataFrame,
    *,
    sizes: Sequence[int] = (2, 3),
    min_ev: float = 0.0,
    min_price_edge: float | None = None,
    max_parlay_decimal: float | None = 8.0,
) -> pd.DataFrame:
    """Same-day enumeration. ``min_ev`` maps to price edge when ``min_price_edge`` omitted.

    ``min_ev`` is EV/stake; equivalent relative price edge is not identical, so when
    only ``min_ev`` is passed we filter by ``ev_per_unit`` after a 0 price-edge clear.
    """
    if min_price_edge is None:
        cands = enumerate_parlays(
            legs,
            sizes=sizes,
            min_price_edge=0.0,
            max_parlay_decimal=max_parlay_decimal,
            same_day=True,
            allow_same_game=True,
        )
        if cands.empty:
            return cands
        return cands[cands["ev_per_unit"] >= float(min_ev)].reset_index(drop=True)
    return enumerate_parlays(
        legs,
        sizes=sizes,
        min_price_edge=min_price_edge,
        max_parlay_decimal=max_parlay_decimal,
        same_day=True,
        allow_same_game=True,
    )


def select_top_edge_tickets(
    candidates: pd.DataFrame,
    *,
    max_per_day: int = 1,
    min_price_edge: float | None = None,
    walkforward_quantile: float | None = None,
    score_col: str = "price_edge",
) -> pd.DataFrame:
    """Fewer tickets: top edge per day, optional walk-forward quantile gate.

    If ``walkforward_quantile`` is set (e.g. 0.9), keep the day's best ticket only
    when its score ≥ the expanding quantile of **prior days' best scores** (no leak).
    """
    if candidates.empty:
        return candidates.copy()
    df = candidates.copy()
    if min_price_edge is not None and "price_edge" in df.columns:
        df = df[df["price_edge"] >= float(min_price_edge)]
    if df.empty:
        return df

    picked: list[pd.Series] = []
    prior_best: list[float] = []
    for day, day_df in df.groupby("game_date", sort=True):
        day_sorted = day_df.sort_values(score_col, ascending=False)
        used: set[str] = set()
        n = 0
        day_best_score: float | None = None
        for _, row in day_sorted.iterrows():
            gids = set(str(row["game_ids"]).split("|"))
            if gids & used:
                continue
            score = float(row[score_col])
            if walkforward_quantile is not None and prior_best:
                thr = float(np.quantile(prior_best, float(walkforward_quantile)))
                if score < thr:
                    break  # remaining are worse
            picked.append(row)
            used |= gids
            if day_best_score is None:
                day_best_score = score
            n += 1
            if n >= int(max_per_day):
                break
        if day_best_score is not None:
            prior_best.append(day_best_score)
    if not picked:
        return df.iloc[0:0].copy()
    return (
        pd.DataFrame(picked)
        .sort_values(["game_date", score_col], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_nonoverlapping_tickets(
    candidates: pd.DataFrame,
    *,
    max_per_day: int = 1,
) -> pd.DataFrame:
    """Greedy per-day by ``price_edge`` (fallback ``ev_per_unit``)."""
    score = (
        "price_edge"
        if (not candidates.empty and "price_edge" in candidates.columns)
        else "ev_per_unit"
    )
    return select_top_edge_tickets(candidates, max_per_day=max_per_day, score_col=score)


def grade_tickets(
    tickets: pd.DataFrame,
    *,
    stake: float = 1.0,
) -> pd.DataFrame:
    """Attach stake / payout / pnl. Payout includes stake returned on a win."""
    if tickets.empty:
        out = tickets.copy()
        for c in ("stake", "payout", "pnl", "cum_pnl"):
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
            "avg_price_edge": None,
            "avg_parlay_decimal": None,
            "by_n_legs": {},
        }
    total_stake = float(graded["stake"].sum())
    total_payout = float(graded["payout"].sum())
    total_pnl = float(graded["pnl"].sum())
    cum = graded["pnl"].to_numpy(dtype=float).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = float(np.min(cum - peak)) if len(cum) else 0.0
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
        "avg_price_edge": float(graded["price_edge"].mean())
        if "price_edge" in graded.columns
        else None,
        "avg_parlay_decimal": float(graded["parlay_decimal"].mean()),
        "by_n_legs": by_legs,
        "stake_per_ticket": float(stake),
        "starting_bankroll": starting_bankroll,
    }


def run_parlay_selector(
    games: pd.DataFrame,
    *,
    model_home_col: str = "p_model_home",
    min_fair_edge: float = 0.0,
    require_vs_vigged: bool = True,
    min_price_edge: float = 1.0,
    min_ev: float | None = None,
    max_parlay_decimal: float | None = None,
    sizes: Sequence[int] = (2,),
    max_per_day: int = 1,
    walkforward_quantile: float | None = None,
    allow_same_game: bool = True,
    stake: float = 1.0,
    starting_bankroll: float | None = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Legs → tickets that beat parlay payout price → top-edge pick → graded summary.

    Defaults = 2-leg freeze from the 2022-24 train sweep (best edge for
    2-leg: ``min_price_edge=1.0``). Only length positive on train, confirm,
    and holdout. Longer legs kept out of defaults — see ``nba-parlay-ev.md``.
    """
    legs = build_eligible_legs(
        games,
        model_home_col=model_home_col,
        min_fair_edge=min_fair_edge,
        require_vs_vigged=require_vs_vigged,
    )
    cands = enumerate_parlays(
        legs,
        sizes=sizes,
        min_price_edge=min_price_edge,
        max_parlay_decimal=max_parlay_decimal,
        same_day=True,
        allow_same_game=allow_same_game,
    )
    if min_ev is not None and not cands.empty:
        cands = cands[cands["ev_per_unit"] >= float(min_ev)].copy()
    picked = select_top_edge_tickets(
        cands,
        max_per_day=max_per_day,
        walkforward_quantile=walkforward_quantile,
        score_col="price_edge",
    )
    graded = grade_tickets(picked, stake=stake)
    if starting_bankroll is not None and not graded.empty:
        graded["bankroll"] = float(starting_bankroll) + graded["pnl"].cumsum()
    summary = summarize_ticket_pnl(graded, starting_bankroll=starting_bankroll, stake=stake)
    summary["n_eligible_legs"] = int(len(legs))
    summary["n_candidate_tickets"] = int(len(cands))
    summary["n_same_game"] = (
        int(graded["same_game"].sum())
        if (not graded.empty and "same_game" in graded.columns)
        else 0
    )
    summary["params"] = {
        "min_fair_edge": float(min_fair_edge),
        "require_vs_vigged": bool(require_vs_vigged),
        "min_price_edge": float(min_price_edge),
        "min_ev": min_ev,
        "max_parlay_decimal": max_parlay_decimal,
        "sizes": list(sizes),
        "max_per_day": int(max_per_day),
        "walkforward_quantile": walkforward_quantile,
        "allow_same_game": bool(allow_same_game),
        "stake": float(stake),
        "model_home_col": model_home_col,
    }
    return legs, graded, summary


def simulate_seasonal_parlay_bankroll(
    tickets: pd.DataFrame,
    *,
    starting_bankroll: float = 50.0,
    stake_pct: float = 0.01,
    kelly_frac: float | None = None,
    max_bet_frac: float = 0.05,
    min_stake: float = 0.01,
) -> dict[str, Any]:
    """Replay sequential seasonal bankrolls (resetting to ``starting_bankroll`` per season).

    Stake is floored to cents (``floor(pct * bankroll * 100) / 100``), matching the
    protocol in ``docs/nba-parlay-ev.md``. When ``kelly_frac`` is set, uses fractional
    Kelly on ``(p_joint_model, parlay_decimal)`` capped at ``max_bet_frac``.
    """
    import math

    by_season: dict[str, dict[str, Any]] = {}
    if tickets.empty:
        return {"starting_bankroll": float(starting_bankroll), "seasons": by_season}

    df = tickets.sort_values(["season", "game_date"]).reset_index(drop=True)
    for season, g in df.groupby("season", sort=True):
        br = float(starting_bankroll)
        peak = br
        max_dd = 0.0
        n_staked = 0
        n_wins = 0
        total_staked = 0.0

        for _, row in g.iterrows():
            dec = float(row["parlay_decimal"])
            if kelly_frac is not None:
                b = dec - 1.0
                p = float(row["p_joint_model"])
                full_k = (b * p - (1.0 - p)) / b if b > 0 else 0.0
                frac = min(max(0.0, full_k * float(kelly_frac)), float(max_bet_frac))
            else:
                frac = float(stake_pct)

            stake = math.floor(br * frac * 100.0) / 100.0
            if stake < float(min_stake) or stake > br:
                continue
            n_staked += 1
            total_staked += stake
            if bool(row["ticket_won"]):
                br = round(br + stake * (dec - 1.0), 6)
                n_wins += 1
            else:
                br = round(br - stake, 6)
            peak = max(peak, br)
            max_dd = min(max_dd, br - peak)

        by_season[str(season)] = {
            "n_tickets": n_staked,
            "n_wins": n_wins,
            "start_bankroll": float(starting_bankroll),
            "end_bankroll": round(br, 2),
            "net_profit": round(br - float(starting_bankroll), 2),
            "total_staked": round(total_staked, 2),
            "roi": ((br - float(starting_bankroll)) / total_staked) if total_staked > 0 else None,
            "max_drawdown": round(max_dd, 2),
            "clears_start": bool(br > float(starting_bankroll)),
        }

    return {
        "starting_bankroll": float(starting_bankroll),
        "stake_pct": stake_pct,
        "kelly_frac": kelly_frac,
        "seasons": by_season,
    }
