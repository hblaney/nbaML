"""Zero-polling tmux push bridge: drains Postgres outbox -> tmux send-keys, and syncs live panes."""

from __future__ import annotations

import time
from typing import Any

from tools.cursor_fleet.hub_client import FleetHubClient
from tools.cursor_fleet.tmux_manager import (
    capture_pane,
    list_tmux_sessions,
    push_text_to_session,
    spawn_session,
)


class TmuxPushBridge:
    def __init__(self, client: FleetHubClient) -> None:
        self.client = client
        self._last_snapshots: dict[str, str] = {}
        self._last_sync_ts: float = 0.0

    def step(self, *, force_sync: bool = False) -> dict[str, Any]:
        delivered_ids: list[str] = []
        outbox = self.client.poll_outbox()

        for item in outbox:
            oid = str(item.get("id") or "")
            action = str(item.get("action") or "")
            target = str(item.get("tmux_target") or "")
            payload = str(item.get("payload_text") or "")
            workdir = str(item.get("workdir") or "/workspace")
            slug = target.split(":")[0] if target else "worker"

            if action == "spawn_tmux":
                spawn_session(slug, workdir=workdir, initial_prompt=payload or None)
                delivered_ids.append(oid)
            elif action == "push_keys":
                if not push_text_to_session(slug, payload):
                    spawn_session(slug, workdir=workdir, initial_prompt=payload)
                delivered_ids.append(oid)

        if delivered_ids:
            self.client.ack_outbox(delivered_ids)

        now = time.monotonic()
        sessions = list_tmux_sessions()
        changed_panes: list[dict[str, Any]] = []
        internal_daemons = {"fleet-hub-web", "fleet-bridge"}
        for s in sessions:
            slug = s["slug"]
            if slug in internal_daemons:
                continue
            snap = capture_pane(slug, lines=100)
            if force_sync or (now - self._last_sync_ts >= 6.0) or self._last_snapshots.get(slug) != snap:
                self._last_snapshots[slug] = snap
                changed_panes.append(
                    {
                        "slug": slug,
                        "title": slug.replace("-", " ").title(),
                        "workdir": s.get("workdir") or "/workspace",
                        "status": "active",
                        "paneSnapshot": snap,
                    }
                )

        if changed_panes:
            self.client.sync_tmux_panes(changed_panes)
            self._last_sync_ts = now

        return {
            "delivered": len(delivered_ids),
            "synced_panes": len(changed_panes),
        }

    def run_forever(self, *, interval_s: float = 0.8) -> None:
        self.step(force_sync=True)
        while True:
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                print(f"[fleet-bridge] transient error: {exc}", flush=True)
            time.sleep(interval_s)
