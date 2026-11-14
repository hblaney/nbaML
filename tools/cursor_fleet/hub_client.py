"""HTTP client connecting local tmux/MCP tools to the Postgres-backed Cursor Fleet Hub."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FleetHubClient:
    def __init__(
        self,
        hub_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        base = (hub_url or os.environ.get("CURSOR_FLEET_HUB_URL") or "http://127.0.0.1:3000").rstrip("/")
        self.hub_url = base
        self.api_key = (api_key or os.environ.get("CURSOR_API_KEY") or "").strip()
        self.timeout_s = timeout_s

    def _headers(self, is_json: bool = False) -> dict[str, str]:
        hdrs = {"Accept": "application/json"}
        if is_json:
            hdrs["Content-Type"] = "application/json"
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    def get_json(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        url = f"{self.hub_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        req = Request(url, headers=self._headers(), method="GET")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.hub_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=self._headers(is_json=True), method="POST")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def poll_outbox(self) -> list[dict[str, Any]]:
        res = self.get_json("/api/bridge")
        return list(res.get("outbox") or [])

    def ack_outbox(self, ids: list[str]) -> dict[str, Any]:
        if not ids:
            return {"ok": True, "acked": 0}
        return self.post_json("/api/bridge", {"op": "ack_outbox", "ids": ids})

    def sync_tmux_panes(self, panes: list[dict[str, Any]]) -> dict[str, Any]:
        return self.post_json("/api/bridge", {"op": "sync_tmux_panes", "panes": panes})

    def log_session_message(
        self,
        *,
        from_session: str,
        to_session: str | None,
        body: str,
        skip_push: bool = False,
    ) -> dict[str, Any]:
        return self.post_json(
            "/api/bridge",
            {
                "op": "session_message",
                "fromSession": from_session,
                "toSession": to_session,
                "body": body,
                "skipPush": skip_push,
            },
        )

    def register_session(
        self,
        *,
        title: str,
        slug: str,
        workdir: str = "/workspace",
        initial_prompt: str = "",
    ) -> dict[str, Any]:
        return self.post_json(
            "/api/sessions",
            {
                "title": title,
                "slug": slug,
                "kind": "tmux",
                "workdir": workdir,
                "initialPrompt": initial_prompt,
            },
        )

    def search_history(
        self,
        query: str,
        *,
        scope: str = "all",
        session: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {"q": query, "scope": scope}
        if session:
            params["session"] = session
        res = self.get_json("/api/search", params)
        return list(res.get("results") or [])
