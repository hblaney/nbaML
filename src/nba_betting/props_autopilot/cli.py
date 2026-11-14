"""CLI for running the 24/7 NBA PrizePicks & Player Props Autopilot daemon or single cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from nba_betting.props_autopilot.engine import run_autopilot_cycle
from nba_betting.props_autopilot.prizepicks_math import evaluate_live_cashout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="24/7 NBA PrizePicks Scanner, Grader & R&D Autopilot")
    parser.add_argument(
        "--state-dir",
        default="/workspace/data/processed/prizepicks_autopilot",
        help="Directory for live status JSON and archived snapshots",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("once", help="Run one complete scan + grade + slip optimization cycle")

    p_daemon = sub.add_parser("daemon", help="Run continuous 24/7 autopilot loop")
    p_daemon.add_argument("--interval-seconds", type=int, default=900, help="Seconds between live scans (default 900 = 15m)")

    p_cash = sub.add_parser("cashout", help="Evaluate a live PrizePicks Early Payout cashout offer")
    p_cash.add_argument("--side", choices=["MORE", "LESS"], required=True)
    p_cash.add_argument("--line", type=float, required=True)
    p_cash.add_argument("--current", type=float, required=True)
    p_cash.add_argument("--rem-min", type=float, required=True)
    p_cash.add_argument("--rate", type=float, required=True)
    p_cash.add_argument("--full-payout", type=float, required=True)
    p_cash.add_argument("--offer", type=float, required=True)

    args = parser.parse_args(argv)
    state_dir = Path(args.state_dir)

    if args.cmd == "once":
        payload = run_autopilot_cycle(state_dir)
        print(
            json.dumps(
                {
                    "updated_at_cst": payload["updated_at_cst"],
                    "prizepicks_standard_props_live": payload["prizepicks_standard_props_live"],
                    "upcoming_games_scanned": payload["upcoming_games_scanned"],
                    "matched_two_way_props": payload["matched_two_way_props"],
                    "recommended_slip_type": (payload["recommended_slip"] or {}).get("slip_type"),
                    "recommended_slip_ev": (payload["recommended_slip"] or {}).get("ev"),
                    "n_top_edges": len(payload["top_edges"]),
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "daemon":
        print(f"[props-autopilot] Starting 24/7 loop (interval={args.interval_seconds}s) -> {state_dir}", flush=True)
        while True:
            try:
                res = run_autopilot_cycle(state_dir)
                rec = res.get("recommended_slip") or {}
                print(
                    f"[{res['updated_at_cst']}] live_pp={res['prizepicks_standard_props_live']} "
                    f"matched={res['matched_two_way_props']} best_slip={rec.get('slip_type')} ev={rec.get('ev')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[props-autopilot] cycle error: {exc}", flush=True)
            time.sleep(max(int(args.interval_seconds), 60))

    if args.cmd == "cashout":
        res = evaluate_live_cashout(
            side=args.side,
            line=args.line,
            current_stat=args.current,
            expected_remaining_minutes=args.rem_min,
            per_minute_rate=args.rate,
            full_payout_dollars=args.full_payout,
            cashout_offer_dollars=args.offer,
        )
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
