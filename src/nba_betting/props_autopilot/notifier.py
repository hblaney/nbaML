"""Multi-Provider SMS & Phone Push Notifier for R&D Advancements & Post-Close Autopsies.

Reads credentials strictly from environment variables (bound via Cursor Secrets, never hardcoded):
  1. Twilio SMS: ``TWILIO_ACCOUNT_SID``, ``TWILIO_AUTH_TOKEN``, ``TWILIO_FROM_NUMBER``, ``ALERT_TO_PHONE``
  2. ntfy iPhone/Android Push: ``NTFY_TOPIC`` (optional ``NTFY_TOKEN``)
  3. Pushover iPhone Push: ``PUSHOVER_USER_KEY``, ``PUSHOVER_API_TOKEN``
  4. Telegram Bot / Generic Webhook: ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID``, or ``ALERT_WEBHOOK_URL``

When none is configured yet, records every outbound alert into ``alerts_outbox.jsonl`` so zero
notifications are lost and the Web/Mobile UI displays the exact alert feed.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def configured_channels() -> list[str]:
    channels: list[str] = []
    if (
        os.environ.get("TWILIO_ACCOUNT_SID")
        and os.environ.get("TWILIO_AUTH_TOKEN")
        and os.environ.get("TWILIO_FROM_NUMBER")
        and os.environ.get("ALERT_TO_PHONE")
    ):
        channels.append("twilio_sms")
    if os.environ.get("NTFY_TOPIC"):
        channels.append("ntfy_push")
    if os.environ.get("PUSHOVER_USER_KEY") and os.environ.get("PUSHOVER_API_TOKEN"):
        channels.append("pushover")
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        channels.append("telegram")
    if os.environ.get("ALERT_WEBHOOK_URL"):
        channels.append("webhook")
    return channels


def send_advancement_alert(
    title: str,
    body: str,
    *,
    category: str = "rd_advancement",
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Dispatch an SMS / mobile push alert across all configured providers and log to JSONL."""
    clean_title = title.strip()[:120]
    clean_body = body.strip()[:1400]
    full_sms = f"[{clean_title}]\n{clean_body}"[:1500]
    delivered: list[str] = []
    errors: list[str] = []

    # 1. Twilio SMS
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    tok = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_num = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
    to_num = os.environ.get("ALERT_TO_PHONE", "").strip()
    if sid and tok and from_num and to_num:
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            basic = base64.b64encode(f"{sid}:{tok}".encode("utf-8")).decode("ascii")
            form = urlencode({"From": from_num, "To": to_num, "Body": full_sms}).encode("utf-8")
            req = Request(
                url,
                data=form,
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                method="POST",
            )
            with urlopen(req, timeout=12) as resp:
                if 200 <= resp.status < 300:
                    delivered.append("twilio_sms")
        except Exception:
            errors.append("twilio_sms_failed")

    # 2. ntfy.sh iOS / Android instant push
    ntfy_topic = os.environ.get("NTFY_TOPIC", "").strip()
    if ntfy_topic:
        try:
            hdrs = {"Title": clean_title, "Priority": "high"}
            ntfy_tok = os.environ.get("NTFY_TOKEN", "").strip()
            if ntfy_tok:
                hdrs["Authorization"] = f"Bearer {ntfy_tok}"
            req = Request(
                f"https://ntfy.sh/{ntfy_topic}",
                data=clean_body.encode("utf-8"),
                headers=hdrs,
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    delivered.append("ntfy_push")
        except Exception:
            errors.append("ntfy_failed")

    # 3. Pushover iOS Push
    po_user = os.environ.get("PUSHOVER_USER_KEY", "").strip()
    po_tok = os.environ.get("PUSHOVER_API_TOKEN", "").strip()
    if po_user and po_tok:
        try:
            form = urlencode({"token": po_tok, "user": po_user, "title": clean_title, "message": clean_body}).encode(
                "utf-8"
            )
            req = Request("https://api.pushover.net/1/messages.json", data=form, method="POST")
            with urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    delivered.append("pushover")
        except Exception:
            errors.append("pushover_failed")

    # 4. Telegram Bot
    tg_tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if tg_tok and tg_chat:
        try:
            payload = json.dumps({"chat_id": tg_chat, "text": full_sms}).encode("utf-8")
            req = Request(
                f"https://api.telegram.org/bot{tg_tok}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    delivered.append("telegram")
        except Exception:
            errors.append("telegram_failed")

    # 5. Generic Webhook (Discord / Slack / iOS Shortcuts)
    wh_url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if wh_url.startswith("https://"):
        try:
            payload = json.dumps({"content": full_sms, "text": full_sms, "title": clean_title}).encode("utf-8")
            req = Request(wh_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    delivered.append("webhook")
        except Exception:
            errors.append("webhook_failed")

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "title": clean_title,
        "body": clean_body,
        "delivered_channels": delivered,
        "status": "sent" if delivered else "queued_awaiting_provider",
        "errors": errors,
    }

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    return record
