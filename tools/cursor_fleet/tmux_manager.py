"""Safe subprocess wrapper around tmux for starting and steering Cursor CLI sessions."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

PORTAL_TMUX_CONF = Path("/exec-daemon/tmux.portal.conf")


def _tmux_base_cmd() -> list[str]:
    tmux_bin = shutil.which("tmux") or "/exec-daemon/tmux"
    if PORTAL_TMUX_CONF.exists():
        return [tmux_bin, "-f", str(PORTAL_TMUX_CONF)]
    return [tmux_bin]


def sanitize_session_name(raw: str) -> str:
    clean = re.sub(r"[^a-z0-9-]+", "-", raw.lower().strip()).strip("-")
    return clean[:48] or "cursor-worker"


def default_cursor_command() -> list[str]:
    """Prefer Cursor's `agent` (`cursor-agent`) CLI if installed; fallback to login shell."""
    local_agent = Path.home() / ".local" / "bin" / "agent"
    agent_bin = (
        str(local_agent)
        if local_agent.exists()
        else (shutil.which("agent") or shutil.which("cursor-agent"))
    )
    if agent_bin:
        return [agent_bin, "--trust", "--approve-mcps"]
    shell_bin = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
    return [shell_bin, "-l"]


def has_tmux_session(slug: str) -> bool:
    name = sanitize_session_name(slug)
    res = subprocess.run(
        [*_tmux_base_cmd(), "has-session", "-t", f"={name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def spawn_session(
    slug: str,
    *,
    workdir: str = "/workspace",
    command: list[str] | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    """Start or reuse a detached tmux session and optionally push an opening prompt."""
    name = sanitize_session_name(slug)
    cwd = Path(workdir).expanduser()
    if not cwd.exists():
        cwd = Path.cwd()

    created = False
    if not has_tmux_session(name):
        cmd = command or default_cursor_command()
        env_args = [f"CURSOR_FLEET_SESSION={name}"]
        subprocess.run(
            [
                *_tmux_base_cmd(),
                "new-session",
                "-d",
                "-s",
                name,
                "-c",
                str(cwd),
                "-e",
                env_args[0],
                "--",
                *cmd,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        created = True
        time.sleep(0.15)

    if initial_prompt and initial_prompt.strip():
        push_text_to_session(name, initial_prompt.strip())

    return {
        "slug": name,
        "tmux_target": f"{name}:0.0",
        "workdir": str(cwd),
        "created": created,
    }


def push_text_to_session(slug_or_target: str, text: str) -> bool:
    """Push text directly into a tmux pane using Ctrl+J for newlines and Enter to steer/submit.

    Cursor CLI (`agent`) inside tmux accepts incoming keystrokes + Enter to steer an active
    turn or start a new turn immediately without the session polling any tool.
    """
    base_name = slug_or_target.split(":")[0]
    name = sanitize_session_name(base_name)
    if not has_tmux_session(name):
        return False

    target = f"{name}:0.0"
    lines = text.splitlines() or [text]
    for idx, line in enumerate(lines):
        if line:
            subprocess.run(
                [*_tmux_base_cmd(), "send-keys", "-t", target, "-l", "--", line],
                capture_output=True,
                text=True,
                check=False,
            )
        if idx < len(lines) - 1:
            # Cursor CLI in tmux uses Ctrl+J (0x0a) for multi-line input
            subprocess.run(
                [*_tmux_base_cmd(), "send-keys", "-t", target, "C-j"],
                capture_output=True,
                text=True,
                check=False,
            )

    subprocess.run(
        [*_tmux_base_cmd(), "send-keys", "-t", target, "C-m"],
        capture_output=True,
        text=True,
        check=False,
    )
    return True


def capture_pane(slug_or_target: str, *, lines: int = 120) -> str:
    base_name = slug_or_target.split(":")[0]
    name = sanitize_session_name(base_name)
    if not has_tmux_session(name):
        return ""
    res = subprocess.run(
        [
            *_tmux_base_cmd(),
            "capture-pane",
            "-p",
            "-J",
            "-t",
            f"{name}:0.0",
            "-S",
            f"-{max(int(lines), 20)}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.rstrip() if res.returncode == 0 else ""


def list_tmux_sessions() -> list[dict[str, Any]]:
    res = subprocess.run(
        [
            *_tmux_base_cmd(),
            "list-sessions",
            "-F",
            "#{session_name}\t#{pane_current_path}\t#{session_windows}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return []
    out: list[dict[str, Any]] = []
    for line in res.stdout.strip().splitlines():
        parts = line.split("\t")
        slug = parts[0]
        workdir = parts[1] if len(parts) > 1 else ""
        windows = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
        out.append(
            {
                "slug": slug,
                "tmux_target": f"{slug}:0.0",
                "workdir": workdir,
                "windows": windows,
            }
        )
    return out


def kill_tmux_session(slug: str) -> bool:
    name = sanitize_session_name(slug)
    if not has_tmux_session(name):
        return False
    res = subprocess.run(
        [*_tmux_base_cmd(), "kill-session", "-t", f"={name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0
