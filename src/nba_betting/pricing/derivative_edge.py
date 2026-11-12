"""Edge calculation, parlay leg building, and seasonal $50 Kelly bankroll compounding
for NBA 1st Quarter (1Q) and 1st Half (1H) derivative markets.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from nba_betting.data.derivative_odds import DerivativeEventOdds
from nba_betting.models.derivatives import (
    PeriodPrediction,
    moneyline_two_way_probs,
    spread_cover_probs,
    standard_bookmaker_derivative_lines,
    total_over_under_probs,
)
from nba_betting.pricing.american import (
    american_to_implied,
    devig_two_way,
    implied_to_american,
)
from nba_betting.pricing.kelly import american_to_decimal, kelly_fraction


@dataclass(frozen=True)
class DerivativeEdgeRow:
    game_id: str
    game_date: str
    home_tricode: str
    away_tricode: str
    market: str  # h2h_q1 | spreads_q1 | totals_q1 | h2h_h1 | spreads_h1 | totals_h1
    side: str  # home | away | over | under
    line: float | None
    model_prob: float
    push_prob: float
    fair_american: float
    book_american: float
    book_implied_raw: float
    book_implied_fair: float
    edge: float  # model_prob - book_implied_fair
    vig_edge: float  # model_prob - book_implied_raw
    book: str
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _match_derivative_event(
    pred: PeriodPrediction, events: Iterable[DerivativeEventOdds]
) -> DerivativeEventOdds | None:
    for ev in events:
        if ev.home_tricode == pred.home_tricode and ev.away_tricode == pred.away_tricode:
            return ev
    return None


def _safe_fair_american(prob: float) -> float:
    p = float(np.clip(prob, 1e-5, 1.0 - 1e-5))
    return float(implied_to_american(p))


def build_derivative_edge_rows(
    predictions: Sequence[PeriodPrediction],
    events: Sequence[DerivativeEventOdds],
) -> list[DerivativeEdgeRow]:
    """Join ``PeriodPrediction`` rows to live/cached ``DerivativeEventOdds`` across all 6 markets."""
    rows: list[DerivativeEdgeRow] = []
    for pred in predictions:
        ev = _match_derivative_event(pred, events)
        if ev is None:
            continue

        period_specs = (
            (
                "q1",
                pred.exp_q1_margin,
                pred.sigma_q1_margin,
                pred.exp_q1_total,
                pred.sigma_q1_total,
                ev.q1_moneyline_home,
                ev.q1_moneyline_away,
                ev.q1_spread_home_point,
                ev.q1_spread_home_price,
                ev.q1_spread_away_point,
                ev.q1_spread_away_price,
                ev.q1_total_point,
                ev.q1_total_over_price,
                ev.q1_total_under_price,
            ),
            (
                "h1",
                pred.exp_h1_margin,
                pred.sigma_h1_margin,
                pred.exp_h1_total,
                pred.sigma_h1_total,
                ev.h1_moneyline_home,
                ev.h1_moneyline_away,
                ev.h1_spread_home_point,
                ev.h1_spread_home_price,
                ev.h1_spread_away_point,
                ev.h1_spread_away_price,
                ev.h1_total_point,
                ev.h1_total_over_price,
                ev.h1_total_under_price,
            ),
        )

        for (
            tag,
            exp_m,
            sig_m,
            exp_t,
            sig_t,
            ml_h,
            ml_a,
            sp_h_pt,
            sp_h_pr,
            sp_a_pt,
            sp_a_pr,
            tot_pt,
            ov_pr,
            un_pr,
        ) in period_specs:
            # 1. Period Moneyline (h2h_q1 / h2h_h1)
            if ml_h is not None and ml_a is not None:
                p_h, p_a, p_push = moneyline_two_way_probs(exp_m, sig_m)
                fair_h, fair_a = devig_two_way(ml_h, ml_a)
                for side, mp, bp, bf in (
                    ("home", p_h, ml_h, fair_h),
                    ("away", p_a, ml_a, fair_a),
                ):
                    raw_imp = float(american_to_implied(bp))
                    rows.append(
                        DerivativeEdgeRow(
                            game_id=pred.game_id,
                            game_date=pred.game_date,
                            home_tricode=pred.home_tricode,
                            away_tricode=pred.away_tricode,
                            market=f"h2h_{tag}",
                            side=side,
                            line=None,
                            model_prob=mp,
                            push_prob=p_push,
                            fair_american=_safe_fair_american(mp),
                            book_american=float(bp),
                            book_implied_raw=raw_imp,
                            book_implied_fair=float(bf),
                            edge=float(mp - bf),
                            vig_edge=float(mp - raw_imp),
                            book=ev.book,
                            event_id=ev.event_id,
                        )
                    )

            # 2. Period Spread (spreads_q1 / spreads_h1)
            if sp_h_pt is not None and sp_h_pr is not None and sp_a_pr is not None:
                p_cov_h, p_cov_a, p_push = spread_cover_probs(exp_m, sig_m, sp_h_pt)
                fair_h, fair_a = devig_two_way(sp_h_pr, sp_a_pr)
                a_line = float(-sp_h_pt) if sp_a_pt is None else float(sp_a_pt)
                for side, mp, bp, bf, ln in (
                    ("home", p_cov_h, sp_h_pr, fair_h, float(sp_h_pt)),
                    ("away", p_cov_a, sp_a_pr, fair_a, a_line),
                ):
                    raw_imp = float(american_to_implied(bp))
                    rows.append(
                        DerivativeEdgeRow(
                            game_id=pred.game_id,
                            game_date=pred.game_date,
                            home_tricode=pred.home_tricode,
                            away_tricode=pred.away_tricode,
                            market=f"spreads_{tag}",
                            side=side,
                            line=ln,
                            model_prob=mp,
                            push_prob=p_push,
                            fair_american=_safe_fair_american(mp),
                            book_american=float(bp),
                            book_implied_raw=raw_imp,
                            book_implied_fair=float(bf),
                            edge=float(mp - bf),
                            vig_edge=float(mp - raw_imp),
                            book=ev.book,
                            event_id=ev.event_id,
                        )
                    )

            # 3. Period Totals (totals_q1 / totals_h1)
            if tot_pt is not None and ov_pr is not None and un_pr is not None:
                p_ov, p_un, p_push = total_over_under_probs(exp_t, sig_t, tot_pt)
                fair_ov, fair_un = devig_two_way(ov_pr, un_pr)
                for side, mp, bp, bf in (
                    ("over", p_ov, ov_pr, fair_ov),
                    ("under", p_un, un_pr, fair_un),
                ):
                    raw_imp = float(american_to_implied(bp))
                    rows.append(
                        DerivativeEdgeRow(
                            game_id=pred.game_id,
                            game_date=pred.game_date,
                            home_tricode=pred.home_tricode,
                            away_tricode=pred.away_tricode,
                            market=f"totals_{tag}",
                            side=side,
                            line=float(tot_pt),
                            model_prob=mp,
                            push_prob=p_push,
                            fair_american=_safe_fair_american(mp),
                            book_american=float(bp),
                            book_implied_raw=raw_imp,
                            book_implied_fair=float(bf),
                            edge=float(mp - bf),
                            vig_edge=float(mp - raw_imp),
                            book=ev.book,
                            event_id=ev.event_id,
                        )
                    )

    return rows


def build_historical_derivative_bets(
    scored_df: pd.DataFrame,
    *,
    pricing_mode: str = "archived_espn",
    default_vig_american: float = -110.0,
    allowed_markets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create candidate side-level derivative bets from walk-forward scored rows.

    ``pricing_mode``:
    - ``"archived_espn"``: grade strictly against real archived ESPN BET / DraftKings
      1Q & 1H lines where present (no synthetic lines).
    - ``"standard_sharp"``: grade against garbage-time-adjusted bookmaker derivative
      lines derived from closing full-game spread/total at ``-110/-110`` vig.
    - ``"mechanical_proportional"``: grade against literal ``25%`` (1Q) / ``50.5%`` (1H)
      half-point-snapped full-game lines at ``-110/-110`` vig.
    """
    market_filter = set(allowed_markets) if allowed_markets is not None else None
    vig = float(default_vig_american)
    records: list[dict[str, Any]] = []

    for _, r in scored_df.iterrows():
        gid = str(r["game_id"])
        gdate = pd.Timestamp(r["game_date"]).normalize()
        season = str(r["season"])
        h_tri = str(r["home_tricode"])
        a_tri = str(r["away_tricode"])

        act_q1_m = float(r["q1_margin"])
        act_h1_m = float(r["h1_margin"])
        act_q1_t = float(r["q1_total"])
        act_h1_t = float(r["h1_total"])

        exp_q1_m, sig_q1_m = float(r["exp_q1_margin"]), float(r["sigma_q1_margin"])
        exp_h1_m, sig_h1_m = float(r["exp_h1_margin"]), float(r["sigma_h1_margin"])
        exp_q1_t, sig_q1_t = float(r["exp_q1_total"]), float(r["sigma_q1_total"])
        exp_h1_t, sig_h1_t = float(r["exp_h1_total"]), float(r["sigma_h1_total"])

        if pricing_mode == "archived_espn":
            prov = str(r.get("deriv_odds_provider") or "")
            if not prov or prov == "None" or prov == "nan":
                continue
            q1_sp_h = float(r["book_q1_spread_home"]) if pd.notna(r.get("book_q1_spread_home")) else np.nan
            q1_sp_h_am = float(r["book_q1_spread_home_odds"]) if pd.notna(r.get("book_q1_spread_home_odds")) else vig
            q1_sp_a_am = float(r["book_q1_spread_away_odds"]) if pd.notna(r.get("book_q1_spread_away_odds")) else vig

            h1_sp_h = float(r["book_h1_spread_home"]) if pd.notna(r.get("book_h1_spread_home")) else np.nan
            h1_sp_h_am = float(r["book_h1_spread_home_odds"]) if pd.notna(r.get("book_h1_spread_home_odds")) else vig
            h1_sp_a_am = float(r["book_h1_spread_away_odds"]) if pd.notna(r.get("book_h1_spread_away_odds")) else vig

            q1_tot = float(r["book_q1_total_line"]) if pd.notna(r.get("book_q1_total_line")) else np.nan
            has_q1_tot_juice = pd.notna(r.get("book_q1_over_odds")) and pd.notna(r.get("book_q1_under_odds"))
            q1_ov_am = float(r["book_q1_over_odds"]) if has_q1_tot_juice else vig
            q1_un_am = float(r["book_q1_under_odds"]) if has_q1_tot_juice else vig

            h1_tot = float(r["book_h1_total_line"]) if pd.notna(r.get("book_h1_total_line")) else np.nan
            has_h1_tot_juice = pd.notna(r.get("book_h1_over_odds")) and pd.notna(r.get("book_h1_under_odds"))
            h1_ov_am = float(r["book_h1_over_odds"]) if has_h1_tot_juice else vig
            h1_un_am = float(r["book_h1_under_odds"]) if has_h1_tot_juice else vig

            q1_ml_h = float(r["book_q1_home_ml"]) if pd.notna(r.get("book_q1_home_ml")) else np.nan
            q1_ml_a = float(r["book_q1_away_ml"]) if pd.notna(r.get("book_q1_away_ml")) else np.nan
            h1_ml_h = float(r["book_h1_home_ml"]) if pd.notna(r.get("book_h1_home_ml")) else np.nan
            h1_ml_a = float(r["book_h1_away_ml"]) if pd.notna(r.get("book_h1_away_ml")) else np.nan
        elif pricing_mode in ("standard_sharp", "mechanical_proportional"):
            mode_key = "sharp" if pricing_mode == "standard_sharp" else "proportional"
            lines = standard_bookmaker_derivative_lines(
                float(r["fg_spread_home"]),
                float(r["fg_total_line"]),
                mode=mode_key,
            )
            prov = pricing_mode
            q1_sp_h, q1_sp_h_am, q1_sp_a_am = lines["q1_spread_home"], vig, vig
            h1_sp_h, h1_sp_h_am, h1_sp_a_am = lines["h1_spread_home"], vig, vig
            q1_tot, q1_ov_am, q1_un_am = lines["q1_total_line"], vig, vig
            h1_tot, h1_ov_am, h1_un_am = lines["h1_total_line"], vig, vig
            q1_ml_h = q1_ml_a = h1_ml_h = h1_ml_a = np.nan
        else:
            raise ValueError(f"unknown pricing_mode: {pricing_mode!r}")

        def _emit(
            mkt: str,
            side: str,
            line_val: float | None,
            p_mod: float,
            p_fair: float,
            am: float,
            won: bool,
            pushed: bool,
        ) -> None:
            if market_filter is not None and mkt not in market_filter:
                return
            if not np.isfinite(am) or am == 0.0:
                return
            dec = american_to_decimal(am)
            p_vig = 1.0 / dec
            records.append(
                {
                    "season": season,
                    "game_date": gdate,
                    "game_id": gid,
                    "home_tricode": h_tri,
                    "away_tricode": a_tri,
                    "provider": prov,
                    "market": mkt,
                    "side": side,
                    "line": line_val,
                    "p_model": float(p_mod),
                    "p_fair_novig": float(p_fair),
                    "p_vigged": float(p_vig),
                    "fair_edge": float(p_mod - p_fair),
                    "vig_edge": float(p_mod - p_vig),
                    "american": float(am),
                    "decimal": float(dec),
                    "ev_per_unit": float(p_mod * dec - 1.0),
                    "leg_won": bool(won),
                    "leg_push": bool(pushed),
                }
            )

        # Spreads Q1 / H1
        for mkt, sp_h, am_h, am_a, exp_m, sig_m, act_m in (
            ("spreads_q1", q1_sp_h, q1_sp_h_am, q1_sp_a_am, exp_q1_m, sig_q1_m, act_q1_m),
            ("spreads_h1", h1_sp_h, h1_sp_h_am, h1_sp_a_am, exp_h1_m, sig_h1_m, act_h1_m),
        ):
            if np.isfinite(sp_h) and np.isfinite(am_h) and np.isfinite(am_a) and am_h != 0 and am_a != 0:
                p_h, p_a, _ = spread_cover_probs(exp_m, sig_m, sp_h)
                f_h, f_a = devig_two_way(am_h, am_a)
                cov = act_m + sp_h
                _emit(mkt, "home", sp_h, p_h, f_h, am_h, cov > 0, cov == 0)
                _emit(mkt, "away", -sp_h, p_a, f_a, am_a, cov < 0, cov == 0)

        # Totals Q1 / H1
        for mkt, tot_l, am_o, am_u, exp_t, sig_t, act_t in (
            ("totals_q1", q1_tot, q1_ov_am, q1_un_am, exp_q1_t, sig_q1_t, act_q1_t),
            ("totals_h1", h1_tot, h1_ov_am, h1_un_am, exp_h1_t, sig_h1_t, act_h1_t),
        ):
            if np.isfinite(tot_l) and np.isfinite(am_o) and np.isfinite(am_u) and am_o != 0 and am_u != 0:
                p_o, p_u, _ = total_over_under_probs(exp_t, sig_t, tot_l)
                f_o, f_u = devig_two_way(am_o, am_u)
                diff = act_t - tot_l
                _emit(mkt, "over", tot_l, p_o, f_o, am_o, diff > 0, diff == 0)
                _emit(mkt, "under", tot_l, p_u, f_u, am_u, diff < 0, diff == 0)

        # Moneylines Q1 / H1 (archived book lines)
        for mkt, ml_h, ml_a, exp_m, sig_m, act_m in (
            ("h2h_q1", q1_ml_h, q1_ml_a, exp_q1_m, sig_q1_m, act_q1_m),
            ("h2h_h1", h1_ml_h, h1_ml_a, exp_h1_m, sig_h1_m, act_h1_m),
        ):
            if np.isfinite(ml_h) and np.isfinite(ml_a) and ml_h != 0 and ml_a != 0:
                p_h, p_a, _ = moneyline_two_way_probs(exp_m, sig_m)
                f_h, f_a = devig_two_way(ml_h, ml_a)
                _emit(mkt, "home", None, p_h, f_h, ml_h, act_m > 0, act_m == 0)
                _emit(mkt, "away", None, p_a, f_a, ml_a, act_m < 0, act_m == 0)

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["game_date", "game_id", "market", "side"])
        .reset_index(drop=True)
    )


def enumerate_derivative_parlays(
    all_legs: pd.DataFrame,
    *,
    n_legs: int = 2,
    min_vig_edge_leg: float = 0.015,
    min_price_edge: float = 0.05,
    max_per_day: int = 1,
    max_legs_pool_per_day: int = 12,
) -> pd.DataFrame:
    """Build distinct-game same-day N-leg derivative parlays that clear combined parlay vig."""
    if all_legs.empty:
        return pd.DataFrame()

    # Keep at most the strongest edge leg per game so cross-game parlays are strictly independent
    elig = all_legs.loc[all_legs["vig_edge"] >= float(min_vig_edge_leg)].copy()
    if elig.empty:
        return pd.DataFrame()
    elig = (
        elig.sort_values(["game_date", "vig_edge"], ascending=[True, False])
        .drop_duplicates(subset=["game_date", "game_id"], keep="first")
        .reset_index(drop=True)
    )

    picked_rows: list[dict[str, Any]] = []
    for gdate, day_df in elig.groupby("game_date", sort=True):
        pool = day_df.head(int(max_legs_pool_per_day))
        if len(pool) < n_legs:
            continue
        rows = [pool.iloc[i] for i in range(len(pool))]
        day_cands: list[dict[str, Any]] = []
        for comb in combinations(range(len(rows)), n_legs):
            legs = [rows[i] for i in comb]
            probs = [float(L["p_model"]) for L in legs]
            decs = [float(L["decimal"]) for L in legs]
            parlay_dec = float(np.prod(decs))
            book_imp = 1.0 / parlay_dec
            p_joint = float(np.prod(probs))
            price_edge = p_joint / book_imp - 1.0
            if price_edge < float(min_price_edge):
                continue

            # Sportsbook push-void settlement: pushed leg reduces parlay decimal to 1.0; any loss -> 0 payout
            any_loss = any((not bool(L["leg_won"])) and (not bool(L["leg_push"])) for L in legs)
            all_push = all(bool(L["leg_push"]) for L in legs)
            if any_loss:
                settled_dec = 0.0
                won_flag = False
                push_flag = False
            elif all_push:
                settled_dec = 1.0
                won_flag = False
                push_flag = True
            else:
                eff_dec = 1.0
                for L in legs:
                    eff_dec *= 1.0 if bool(L["leg_push"]) else float(L["decimal"])
                settled_dec = float(eff_dec)
                won_flag = True
                push_flag = False

            parlay_am = float(implied_to_american(book_imp))
            day_cands.append(
                {
                    "season": str(legs[0]["season"]),
                    "game_date": gdate,
                    "n_legs": int(n_legs),
                    "game_ids": "|".join(str(L["game_id"]) for L in legs),
                    "markets": "|".join(f"{L['market']}:{L['side']}" for L in legs),
                    "p_model": p_joint,
                    "american": parlay_am,
                    "decimal": parlay_dec,
                    "settled_decimal": settled_dec,
                    "book_implied": book_imp,
                    "price_edge": float(price_edge),
                    "ev_per_unit": float(p_joint * parlay_dec - 1.0),
                    "leg_won": won_flag,
                    "leg_push": push_flag,
                }
            )
        if not day_cands:
            continue
        day_cands.sort(key=lambda x: -x["price_edge"])
        used_gids: set[str] = set()
        kept = 0
        for cand in day_cands:
            cg = set(cand["game_ids"].split("|"))
            if cg & used_gids:
                continue
            picked_rows.append(cand)
            used_gids |= cg
            kept += 1
            if kept >= int(max_per_day):
                break

    if not picked_rows:
        return pd.DataFrame()
    return pd.DataFrame(picked_rows).sort_values(["game_date", "price_edge"], ascending=[True, False]).reset_index(drop=True)


def simulate_seasonal_kelly_bankroll(
    bets_df: pd.DataFrame,
    *,
    starting_bankroll: float = 50.0,
    kelly_mult: float = 0.25,
    max_bet_frac: float = 0.05,
    max_daily_frac: float = 0.20,
    min_stake: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate fractional Kelly compounding resetting to ``$50`` each NBA season.

    Simultaneous bets on the same calendar date size off the start-of-day bankroll
    and scale down proportionally if total daily allocation exceeds ``max_daily_frac``.
    """
    if bets_df.empty:
        return bets_df.copy(), {"seasons": {}, "pooled_flat_roi": None}

    df = bets_df.sort_values(["season", "game_date"]).copy()
    season_summaries: dict[str, Any] = {}
    out_chunks: list[pd.DataFrame] = []

    for season, s_df in df.groupby("season", sort=True):
        bankroll = float(starting_bankroll)
        peak = bankroll
        max_dd = 0.0
        stakes_col: list[float] = []
        pnls_col: list[float] = []
        br_after_col: list[float] = []
        flat_pnls: list[float] = []

        for _, day_df in s_df.groupby("game_date", sort=True):
            sod_bankroll = bankroll
            raw_fracs: list[float] = []
            for _, row in day_df.iterrows():
                f = kelly_fraction(
                    float(row["p_model"]),
                    float(row["american"]),
                    fraction=float(kelly_mult),
                    max_bet_frac=float(max_bet_frac),
                )
                raw_fracs.append(f)
            tot_frac = sum(raw_fracs)
            scale = (float(max_daily_frac) / tot_frac) if tot_frac > float(max_daily_frac) and tot_frac > 0 else 1.0

            day_pnl = 0.0
            for idx_in_day, (_, row) in enumerate(day_df.iterrows()):
                frac = raw_fracs[idx_in_day] * scale
                stake = math.floor(sod_bankroll * frac * 100.0) / 100.0
                if stake < float(min_stake) or sod_bankroll < float(min_stake):
                    stake = 0.0

                won = bool(row["leg_won"])
                pushed = bool(row.get("leg_push", False))
                settled_dec = row.get("settled_decimal", None)
                dec = float(row["decimal"])

                if pushed:
                    pnl = 0.0
                    f_pnl = 0.0
                elif won:
                    mult = (float(settled_dec) - 1.0) if settled_dec is not None else (dec - 1.0)
                    pnl = round(stake * mult, 2)
                    f_pnl = mult
                else:
                    pnl = -stake
                    f_pnl = -1.0

                day_pnl = round(day_pnl + pnl, 2)
                stakes_col.append(stake)
                pnls_col.append(pnl)
                flat_pnls.append(f_pnl)
                br_after_col.append(round(sod_bankroll + day_pnl, 2))

            bankroll = round(max(0.0, sod_bankroll + day_pnl), 2)
            peak = max(peak, bankroll)
            max_dd = min(max_dd, bankroll - peak)

        s_out = s_df.copy()
        s_out["kelly_stake"] = stakes_col
        s_out["kelly_pnl"] = pnls_col
        s_out["flat_unit_pnl"] = flat_pnls
        s_out["bankroll_after"] = br_after_col
        out_chunks.append(s_out)

        active = s_out.loc[s_out["kelly_stake"] > 0]
        non_push = s_out.loc[~s_out["leg_push"].astype(bool)]
        tot_staked = float(active["kelly_stake"].sum())
        tot_kelly_pnl = float(active["kelly_pnl"].sum())
        season_summaries[str(season)] = {
            "n_signals": int(len(s_out)),
            "n_staked": int(len(active)),
            "n_wins": int(non_push["leg_won"].sum()),
            "n_pushes": int(s_out["leg_push"].astype(bool).sum()),
            "hit_rate": float(non_push["leg_won"].mean()) if len(non_push) > 0 else None,
            "flat_unit_pnl": float(sum(flat_pnls)),
            "flat_roi": float(sum(flat_pnls) / len(non_push)) if len(non_push) > 0 else None,
            "starting_bankroll": float(starting_bankroll),
            "ending_bankroll": float(bankroll),
            "bankroll_return_pct": float((bankroll - starting_bankroll) / starting_bankroll),
            "total_kelly_staked": round(tot_staked, 2),
            "kelly_turnover_roi": float(tot_kelly_pnl / tot_staked) if tot_staked > 0 else None,
            "max_drawdown": round(float(max_dd), 2),
        }

    all_out = pd.concat(out_chunks, ignore_index=True)
    non_push_all = all_out.loc[~all_out["leg_push"].astype(bool)]
    pooled_flat_roi = (
        float(all_out["flat_unit_pnl"].sum() / len(non_push_all))
        if len(non_push_all) > 0
        else None
    )
    meta = {
        "seasons": season_summaries,
        "n_total_signals": int(len(all_out)),
        "pooled_hit_rate": float(non_push_all["leg_won"].mean()) if len(non_push_all) > 0 else None,
        "pooled_flat_unit_pnl": float(all_out["flat_unit_pnl"].sum()),
        "pooled_flat_roi": pooled_flat_roi,
    }
    return all_out, meta
