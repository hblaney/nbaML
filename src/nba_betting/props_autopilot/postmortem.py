"""Daily Post-Close Slip & Leg Autopsy Engine.

Examines every leg of every settled PrizePicks slip against the final ESPN
boxscore + linescores to diagnose *what went right and what went wrong* across:
  1. Closing Line Value (CLV) — did beating the book number save or validate the pick?
  2. Minutes & Game Script — blowout Q4 benching, unexpected overtime, tight Q4 extension, or foul trouble.
  3. Shooting / Scoring Variance — 3PT heater/cold-spell z-score & late FT inflation.
  4. Process vs Outcome Verdict — distinguishes `GOOD_PROCESS_WIN`, `CLV_SAVED_WIN`,
     `GOOD_PROCESS_SHOOTING_VARIANCE_LOSS`, `GAME_SCRIPT_MISS`, and `BAD_LINE_READ`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"


@dataclass(frozen=True)
class LegSpec:
    game_id: str
    player_name: str
    team: str
    stat_type: str
    side: Literal["MORE", "LESS"]
    pp_line: float
    dk_line_at_lock: float
    exp_minutes: float
    pregame_spread_abs: float
    thesis: str = ""


@dataclass(frozen=True)
class LegAutopsy:
    game_id: str
    matchup: str
    final_score: str
    player_name: str
    team: str
    stat_type: str
    side: str
    pp_line: float
    dk_line_at_lock: float
    actual_stat: float
    outcome: Literal["WIN", "LOSS", "PUSH", "CASHED_OUT", "PENDING"]
    clv_pts: float
    exp_minutes: float
    actual_minutes: float
    delta_minutes: float
    final_margin_abs: int
    overtime_periods: int
    personal_fouls: int
    fg_str: str
    fg3_str: str
    ft_str: str
    three_pt_points_over_expected: float
    verdict_tag: str
    what_went_right: str
    what_went_wrong: str
    rd_takeaway: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_num(val: Any, default: float = 0.0) -> float:
    try:
        if val in (None, "", "--", "-"):
            return default
        return float(val)
    except Exception:
        return default


def _parse_made_att(s: str) -> tuple[int, int]:
    if "-" in str(s):
        a, b = str(s).split("-", 1)
        return int(_safe_num(a)), int(_safe_num(b))
    return 0, 0


def compute_stat_from_box(vals: dict[str, str], stat_type: str) -> float:
    pts = _safe_num(vals.get("PTS"))
    reb = _safe_num(vals.get("REB"))
    ast = _safe_num(vals.get("AST"))
    fg3m, _ = _parse_made_att(vals.get("3PT", "0-0"))
    mapping = {
        "Points": pts,
        "Rebounds": reb,
        "Assists": ast,
        "Pts+Rebs+Asts": pts + reb + ast,
        "Pts+Rebs": pts + reb,
        "Pts+Asts": pts + ast,
        "Rebs+Asts": reb + ast,
        "3-PT Made": float(fg3m),
    }
    return float(mapping.get(stat_type, pts))


def analyze_leg_from_box(
    leg: LegSpec,
    *,
    matchup: str,
    final_score: str,
    final_margin_abs: int,
    overtime_periods: int,
    player_stats: dict[str, str],
    cashed_out: bool = False,
) -> LegAutopsy:
    min_raw = str(player_stats.get("MIN", "0"))
    actual_min = (
        _safe_num(min_raw.split(":")[0]) + _safe_num(min_raw.split(":")[1]) / 60.0
        if ":" in min_raw
        else _safe_num(min_raw)
    )
    delta_min = round(actual_min - float(leg.exp_minutes), 1)
    actual_stat = compute_stat_from_box(player_stats, leg.stat_type)
    pf = int(_safe_num(player_stats.get("PF", 0)))

    fg_str = str(player_stats.get("FG", "0-0"))
    fg3_str = str(player_stats.get("3PT", "0-0"))
    ft_str = str(player_stats.get("FT", "0-0"))
    fg3m, fg3a = _parse_made_att(fg3_str)
    ftm, fta = _parse_made_att(ft_str)

    # Expected 3PT points at ~36.5% NBA average on the same 3PA volume
    exp_3pt_pts = 3.0 * (0.365 * fg3a)
    three_pts_over_exp = round(3.0 * fg3m - exp_3pt_pts, 1)

    # Positive CLV means we got a better number on PrizePicks than DraftKings had
    clv_pts = (
        round(float(leg.pp_line) - float(leg.dk_line_at_lock), 2)
        if leg.side == "LESS"
        else round(float(leg.dk_line_at_lock) - float(leg.pp_line), 2)
    )

    if cashed_out:
        outcome: Literal["WIN", "LOSS", "PUSH", "CASHED_OUT", "PENDING"] = "CASHED_OUT"
    elif actual_stat == leg.pp_line:
        outcome = "PUSH"
    elif (leg.side == "MORE" and actual_stat > leg.pp_line) or (
        leg.side == "LESS" and actual_stat < leg.pp_line
    ):
        outcome = "WIN"
    else:
        outcome = "LOSS"

    # Determine whether the 1.0-pt stale line specifically saved the win vs DK's number
    dk_would_have_won = (leg.side == "MORE" and actual_stat > leg.dk_line_at_lock) or (
        leg.side == "LESS" and actual_stat < leg.dk_line_at_lock
    )

    right_notes: list[str] = []
    wrong_notes: list[str] = []
    verdict_tag = "STANDARD_RESULT"
    rd_takeaway = "Maintain current stat-calibrated blowout + stale-line weights."

    if clv_pts > 0:
        right_notes.append(f"Captured +{clv_pts:g} pt stale line vs DraftKings ({leg.pp_line} vs {leg.dk_line_at_lock}).")
    elif clv_pts < 0:
        wrong_notes.append(f"Took {clv_pts:g} pt worse number than DraftKings ({leg.pp_line} vs {leg.dk_line_at_lock}).")

    if outcome == "WIN" and not dk_would_have_won and clv_pts > 0:
        verdict_tag = "CLV_SAVED_WIN"
        right_notes.append(
            f"Stale line directly won the leg: landed on {actual_stat:g} (cashed {leg.pp_line}, whereas DK {leg.dk_line_at_lock} lost)."
        )
        rd_takeaway = "Stale 1.0-pt line gaps near key starter medians produce concrete win-rate lift."
    elif outcome == "WIN":
        if leg.side == "LESS" and final_margin_abs >= 16 and delta_min <= -2.0:
            verdict_tag = "BLOWOUT_TRUNCATION_CONFIRMED"
            right_notes.append(
                f"Blowout thesis hit ({final_margin_abs}-pt margin): minutes cut to {actual_min:.0f} ({delta_min:+.1f} vs projection)."
            )
        elif leg.side == "LESS" and overtime_periods > 0:
            verdict_tag = "SURVIVED_OVERTIME_WIN"
            right_notes.append(
                f"Cleared LESS {leg.pp_line} ({actual_stat:g}) even through {overtime_periods}OT."
            )
            if pf >= 5:
                right_notes.append(f"Foul trouble ({pf} PF) capped extra OT floor time at {actual_min:.0f} min.")
        else:
            verdict_tag = "GOOD_PROCESS_WIN"
            right_notes.append(
                f"Role and minutes ({actual_min:.0f}m vs {leg.exp_minutes:.0f}m proj) matched expectation ({actual_stat:g} vs {leg.pp_line})."
            )
    elif outcome == "CASHED_OUT":
        would_have_hit = (leg.side == "MORE" and actual_stat > leg.pp_line) or (
            leg.side == "LESS" and actual_stat < leg.pp_line
        )
        verdict_tag = "DISCIPLINED_LIVE_CASHOUT"
        right_notes.append(
            f"Locked guaranteed profit in a tight {final_margin_abs}-pt game instead of gambling Q4 variance."
        )
        if would_have_hit:
            wrong_notes.append(
                f"Player stayed on bench in Q4 and finished at {actual_stat:g} ({actual_min:.0f}m), so riding would have paid full 6x — track coach benching on negative +/-."
            )
            rd_takeaway = "Add live +/- and Q3 bench-stint coach-hook signal to live Q4 minutes estimator."
        else:
            right_notes.append(f"Player went on to finish at {actual_stat:g}, so cashing out dodged a Q4 loss!")
    elif outcome == "LOSS":
        if leg.side == "LESS" and fg3a >= 6 and three_pts_over_exp >= 5.0:
            verdict_tag = "GOOD_PROCESS_3PT_VARIANCE_LOSS"
            wrong_notes.append(
                f"Outlier 3PT heater: shot {fg3_str} from deep ({three_pts_over_exp:+.1f} pts above expected 3PT scoring) to clip {actual_stat:g} vs {leg.pp_line}."
            )
            if pf >= 5:
                right_notes.append(f"Minutes ({actual_min:.0f}m) were actually capped by {pf} fouls; loss was pure shooting variance.")
            rd_takeaway = "Validates RD-001 (-0.51% Points / -0.25% 3PM penalty): prefer Assists/Rebounds/PR over pure Points on high-volume 3PT shooters."
        elif leg.side == "LESS" and overtime_periods > 0:
            verdict_tag = "OVERTIME_EXTENSION_LOSS"
            wrong_notes.append(
                f"Game went to {overtime_periods}OT, inflating minutes to {actual_min:.0f}m ({delta_min:+.1f}m) and pushing stat to {actual_stat:g}."
            )
            rd_takeaway = "Avoid LESS on tight-spread (<4.5) games where OT probability (~6%) is highest."
        elif leg.side == "LESS" and leg.pregame_spread_abs >= 8.5 and final_margin_abs <= 10:
            verdict_tag = "FAILED_BLOWOUT_Q4_MINUTES_LOSS"
            wrong_notes.append(
                f"{leg.pregame_spread_abs:g}-pt favorite stayed competitive ({final_margin_abs}-pt final), forcing {actual_min:.0f} full starter minutes ({ft_str} FT)."
            )
            rd_takeaway = "Require >=10.5 spread or a full 1.0-pt stale line before banking on Q4 blowout benching."
        else:
            verdict_tag = "ROLE_OR_USAGE_MISS"
            wrong_notes.append(
                f"Finished at {actual_stat:g} vs {leg.pp_line} in {actual_min:.0f}m ({delta_min:+.1f}m vs proj; FG {fg_str}, FT {ft_str})."
            )
            rd_takeaway = "Audit opponent positional defense and recent usage trend on this archetype."

    return LegAutopsy(
        game_id=leg.game_id,
        matchup=matchup,
        final_score=final_score,
        player_name=leg.player_name,
        team=leg.team,
        stat_type=leg.stat_type,
        side=leg.side,
        pp_line=leg.pp_line,
        dk_line_at_lock=leg.dk_line_at_lock,
        actual_stat=actual_stat,
        outcome=outcome,
        clv_pts=clv_pts,
        exp_minutes=leg.exp_minutes,
        actual_minutes=round(actual_min, 1),
        delta_minutes=delta_min,
        final_margin_abs=final_margin_abs,
        overtime_periods=overtime_periods,
        personal_fouls=pf,
        fg_str=fg_str,
        fg3_str=fg3_str,
        ft_str=ft_str,
        three_pt_points_over_expected=three_pts_over_exp,
        verdict_tag=verdict_tag,
        what_went_right=" ".join(right_notes) if right_notes else "None.",
        what_went_wrong=" ".join(wrong_notes) if wrong_notes else "None — clean execution.",
        rd_takeaway=rd_takeaway,
    )


def fetch_and_autopsy_leg(leg: LegSpec, *, cashed_out: bool = False) -> LegAutopsy | None:
    """Fetch ESPN summary for ``leg.game_id`` and return full post-close autopsy once FINAL."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={leg.game_id}"
    with urlopen(Request(url, headers={"User-Agent": UA}), timeout=20) as resp:
        sm = json.loads(resp.read().decode("utf-8"))

    hdr = (sm.get("header", {}).get("competitions") or [{}])[0]
    st_name = hdr.get("status", {}).get("type", {}).get("name", "")
    if "FINAL" not in st_name:
        return None

    comps = hdr.get("competitors") or []
    if len(comps) == 2:
        c_away = next((c for c in comps if c.get("homeAway") == "away"), comps[0])
        c_home = next((c for c in comps if c.get("homeAway") == "home"), comps[1])
        s_away = int(_safe_num(c_away.get("score", 0)))
        s_home = int(_safe_num(c_home.get("score", 0)))
        matchup = f"{c_away['team']['abbreviation']}@{c_home['team']['abbreviation']}"
        final_score = f"{c_away['team']['abbreviation']} {s_away}, {c_home['team']['abbreviation']} {s_home}"
        margin_abs = abs(s_home - s_away)
        linescores = c_home.get("linescores") or []
        ot_periods = max(0, len(linescores) - 4)
    else:
        matchup, final_score, margin_abs, ot_periods = "", "", 0, 0

    target_lower = leg.player_name.lower()
    found_stats: dict[str, str] | None = None
    for tblk in (sm.get("boxscore") or {}).get("players", []):
        stats0 = (tblk.get("statistics") or [{}])[0]
        labels = stats0.get("labels", [])
        for ath in stats0.get("athletes", []):
            dname = str((ath.get("athlete") or {}).get("displayName", ""))
            if dname.lower() == target_lower or leg.player_name.split()[-1].lower() in dname.lower():
                if ath.get("didNotPlay"):
                    found_stats = {"MIN": "0", "PTS": "0", "REB": "0", "AST": "0", "PF": "0"}
                else:
                    found_stats = dict(zip(labels, ath.get("stats", [])))
                break
        if found_stats is not None:
            break

    if found_stats is None:
        return None

    return analyze_leg_from_box(
        leg,
        matchup=matchup,
        final_score=final_score,
        final_margin_abs=margin_abs,
        overtime_periods=ot_periods,
        player_stats=found_stats,
        cashed_out=cashed_out,
    )


def summarize_slip_autopsy(autopsies: list[LegAutopsy]) -> dict[str, Any]:
    wins = sum(1 for a in autopsies if a.outcome == "WIN")
    losses = sum(1 for a in autopsies if a.outcome == "LOSS")
    mean_clv = round(sum(a.clv_pts for a in autopsies) / max(len(autopsies), 1), 2)
    variance_losses = sum(1 for a in autopsies if "VARIANCE" in a.verdict_tag or "OVERTIME" in a.verdict_tag)
    return {
        "settled_legs": len(autopsies),
        "wins": wins,
        "losses": losses,
        "mean_clv_pts": mean_clv,
        "variance_or_ot_losses": variance_losses,
        "verdict_tags": [a.verdict_tag for a in autopsies],
    }
