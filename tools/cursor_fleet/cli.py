"""CLI entry point for managing Cursor tmux sessions, running the push bridge, or serving MCP."""

from __future__ import annotations

import argparse
import json

from tools.cursor_fleet.hub_client import FleetHubClient
from tools.cursor_fleet.mcp_server import serve_stdio
from tools.cursor_fleet.push_daemon import TmuxPushBridge
from tools.cursor_fleet.tmux_manager import (
    capture_pane,
    kill_tmux_session,
    list_tmux_sessions,
    push_text_to_session,
    spawn_session,
)


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hub-url", default=None, help="Fleet Hub URL (default http://127.0.0.1:3000)")

    parser = argparse.ArgumentParser(
        description="Cursor Multi-Session tmux & Postgres Fleet CLI",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_spawn = sub.add_parser("spawn", parents=[common], help="Start a detached tmux session for Cursor CLI")
    p_spawn.add_argument("slug", help="Session name")
    p_spawn.add_argument("--workdir", default="/workspace")
    p_spawn.add_argument("--prompt", default=None, help="Opening prompt pushed immediately")

    sub.add_parser("ls", parents=[common], help="List active tmux sessions")

    p_push = sub.add_parser("push", parents=[common], help="Push a message straight into a tmux session pane")
    p_push.add_argument("slug")
    p_push.add_argument("text")

    p_cap = sub.add_parser("capture", parents=[common], help="Capture live output from a tmux session pane")
    p_cap.add_argument("slug")
    p_cap.add_argument("--lines", type=int, default=80)

    p_kill = sub.add_parser("kill", parents=[common], help="Stop a tmux session")
    p_kill.add_argument("slug")

    p_search = sub.add_parser("search", parents=[common], help="Search permanent Postgres message history")
    p_search.add_argument("query")
    p_search.add_argument("--scope", choices=["all", "inter_session", "user"], default="all")

    p_daemon = sub.add_parser("daemon", parents=[common], help="Run zero-polling outbox -> tmux push bridge")
    p_daemon.add_argument("--once", action="store_true", help="Run one sync cycle and exit")
    p_daemon.add_argument("--interval", type=float, default=0.8)

    sub.add_parser("mcp", parents=[common], help="Run stdio MCP server for Cursor CLI / IDE")

    args = parser.parse_args(argv)
    client = FleetHubClient(hub_url=args.hub_url)

    if args.cmd == "spawn":
        info = spawn_session(args.slug, workdir=args.workdir, initial_prompt=args.prompt)
        print(json.dumps(info, indent=2))
        return 0

    if args.cmd == "ls":
        print(json.dumps(list_tmux_sessions(), indent=2))
        return 0

    if args.cmd == "push":
        ok = push_text_to_session(args.slug, args.text)
        print(json.dumps({"ok": ok, "slug": args.slug}))
        return 0 if ok else 1

    if args.cmd == "capture":
        print(capture_pane(args.slug, lines=args.lines))
        return 0

    if args.cmd == "kill":
        ok = kill_tmux_session(args.slug)
        print(json.dumps({"killed": ok, "slug": args.slug}))
        return 0 if ok else 1

    if args.cmd == "search":
        res = client.search_history(args.query, scope=args.scope)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "daemon":
        bridge = TmuxPushBridge(client)
        if args.once:
            print(json.dumps(bridge.step(force_sync=True), indent=2))
            return 0
        print(f"[fleet-bridge] Watching {client.hub_url} -> local tmux...", flush=True)
        bridge.run_forever(interval_s=args.interval)
        return 0

    if args.cmd == "mcp":
        serve_stdio()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
