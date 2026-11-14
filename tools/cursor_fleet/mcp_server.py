"""MCP stdio server giving any Cursor CLI / IDE session tools to control tmux sessions & search Postgres."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from tools.cursor_fleet.hub_client import FleetHubClient
from tools.cursor_fleet.tmux_manager import (
    capture_pane,
    list_tmux_sessions,
    push_text_to_session,
    spawn_session,
)

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "fleet_spawn_tmux_session",
        "description": "Start a new Cursor CLI worker session inside tmux, register it in Postgres, and push its opening prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "Short kebab-case session name (e.g. props-worker)"},
                "title": {"type": "string", "description": "Human-readable session title"},
                "workdir": {"type": "string", "description": "Working directory path (default /workspace)"},
                "initial_prompt": {"type": "string", "description": "Opening prompt pushed straight into the session"},
            },
            "required": ["slug"],
        },
    },
    {
        "name": "fleet_list_sessions",
        "description": "List all local tmux sessions and Postgres-registered fleet sessions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fleet_message_session",
        "description": "Send a direct session-to-session message to another tmux or Cloud session (saved permanently in Postgres and pushed directly into the target pane without polling).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to_session": {"type": "string", "description": "Target session slug (e.g. coordinator or props-worker)"},
                "body": {"type": "string", "description": "Message text to push and archive"},
                "from_session": {"type": "string", "description": "Sender session slug (defaults to CURSOR_FLEET_SESSION)"},
            },
            "required": ["to_session", "body"],
        },
    },
    {
        "name": "fleet_reply_to_user",
        "description": "Post a clean message from this session to the Web/Phone UI chat stream in Postgres.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "Summary or update for the user on Web/Phone UI"},
                "from_session": {"type": "string", "description": "Sender session slug"},
            },
            "required": ["body"],
        },
    },
    {
        "name": "fleet_read_tmux_pane",
        "description": "Read the live terminal output of any local tmux session pane.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "Session slug to inspect"},
                "lines": {"type": "integer", "description": "Number of scrollback lines (default 100)"},
            },
            "required": ["slug"],
        },
    },
    {
        "name": "fleet_search_history",
        "description": "Full-text search across the permanent Postgres message archive (all past user<->session and session<->session conversations) when a session needs to recall past context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms or phrase"},
                "scope": {
                    "type": "string",
                    "enum": ["all", "inter_session", "user"],
                    "description": "Filter by message scope (default all)",
                },
                "session": {"type": "string", "description": "Optional session slug filter"},
            },
            "required": ["query"],
        },
    },
]


def handle_tool_call(name: str, args: dict[str, Any], client: FleetHubClient | None = None) -> dict[str, Any]:
    hub = client or FleetHubClient()
    default_sender = os.environ.get("CURSOR_FLEET_SESSION") or "coordinator"

    if name == "fleet_spawn_tmux_session":
        slug = str(args.get("slug") or "worker")
        title = str(args.get("title") or slug)
        workdir = str(args.get("workdir") or "/workspace")
        prompt = str(args.get("initial_prompt") or "")
        info = spawn_session(slug, workdir=workdir, initial_prompt=prompt or None)
        try:
            hub.sync_tmux_panes(
                [
                    {
                        "slug": info["slug"],
                        "title": title,
                        "workdir": info["workdir"],
                        "status": "active",
                        "paneSnapshot": capture_pane(info["slug"], lines=80),
                    }
                ]
            )
            if prompt:
                hub.log_session_message(
                    from_session=default_sender,
                    to_session=info["slug"],
                    body=prompt,
                    skip_push=True,
                )
        except Exception:  # noqa: BLE001
            pass
        return info

    if name == "fleet_list_sessions":
        local_tmux = list_tmux_sessions()
        hub_sessions: list[dict[str, Any]] = []
        try:
            hub_sessions = hub.get_json("/api/sessions").get("sessions") or []
        except Exception:  # noqa: BLE001
            pass
        return {"local_tmux": local_tmux, "hub_sessions": hub_sessions}

    if name == "fleet_message_session":
        to_sess = str(args.get("to_session") or "")
        body = str(args.get("body") or "")
        sender = str(args.get("from_session") or default_sender)
        pushed_local = push_text_to_session(to_sess, f"[From @{sender}]: {body}")
        logged = None
        try:
            logged = hub.log_session_message(
                from_session=sender,
                to_session=to_sess,
                body=body,
                skip_push=pushed_local,
            )
        except Exception:  # noqa: BLE001
            pass
        return {"pushed_to_tmux": pushed_local, "logged": bool(logged)}

    if name == "fleet_reply_to_user":
        body = str(args.get("body") or "")
        sender = str(args.get("from_session") or default_sender)
        res = hub.log_session_message(
            from_session=sender,
            to_session=None,
            body=body,
            skip_push=True,
        )
        return {"ok": True, "message": res.get("message")}

    if name == "fleet_read_tmux_pane":
        slug = str(args.get("slug") or "")
        lines = int(args.get("lines") or 100)
        return {"slug": slug, "output": capture_pane(slug, lines=lines)}

    if name == "fleet_search_history":
        q = str(args.get("query") or "")
        scope = str(args.get("scope") or "all")
        sess = args.get("session")
        results = hub.search_history(q, scope=scope, session=str(sess) if sess else None)
        return {"query": q, "count": len(results), "results": results}

    raise ValueError(f"Unknown MCP tool: {name}")


def handle_rpc_request(msg: dict[str, Any], client: FleetHubClient | None = None) -> dict[str, Any] | None:
    method = msg.get("method")
    req_id = msg.get("id")

    if method == "notifications/initialized" or req_id is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "cursor-fleet-mcp", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}

    if method == "tools/call":
        params = msg.get("params") or {}
        tname = str(params.get("name") or "")
        targs = params.get("arguments") or {}
        try:
            payload = handle_tool_call(tname, targs, client=client)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio() -> None:
    client = FleetHubClient()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_rpc_request(msg, client=client)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
