"""Tests for Cursor Fleet tmux session orchestration and MCP JSON-RPC server."""

from __future__ import annotations

import json
import time

from tools.cursor_fleet.mcp_server import handle_rpc_request
from tools.cursor_fleet.tmux_manager import (
    capture_pane,
    has_tmux_session,
    kill_tmux_session,
    push_text_to_session,
    sanitize_session_name,
    spawn_session,
)


def test_sanitize_session_name() -> None:
    assert sanitize_session_name("Props Scanner #1!") == "props-scanner-1"
    assert sanitize_session_name("   ") == "cursor-worker"


def test_tmux_spawn_push_capture_and_kill() -> None:
    slug = "fleet-unit-test-worker"
    kill_tmux_session(slug)
    try:
        info = spawn_session(slug, workdir="/workspace", command=["/bin/sh"])
        assert info["slug"] == slug
        assert has_tmux_session(slug)

        pushed = push_text_to_session(slug, "echo FLEET_PUSH_OK_42")
        assert pushed
        time.sleep(0.25)

        snap = capture_pane(slug, lines=40)
        assert "FLEET_PUSH_OK_42" in snap
    finally:
        kill_tmux_session(slug)
        assert not has_tmux_session(slug)


def test_mcp_rpc_initialize_and_list_tools() -> None:
    init_resp = handle_rpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "cursor-fleet-mcp"

    tools_resp = handle_rpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
    assert {
        "fleet_spawn_tmux_session",
        "fleet_list_sessions",
        "fleet_message_session",
        "fleet_read_tmux_pane",
        "fleet_search_history",
    }.issubset(tool_names)

    # Read pane via MCP tool call
    call_resp = handle_rpc_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "fleet_read_tmux_pane",
                "arguments": {"slug": "nonexistent-pane", "lines": 20},
            },
        }
    )
    assert call_resp is not None
    assert call_resp["result"]["isError"] is False
    parsed = json.loads(call_resp["result"]["content"][0]["text"])
    assert parsed["output"] == ""
