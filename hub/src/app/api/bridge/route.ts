import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { sql } from "@/lib/db";
import {
  createOrUpdateSession,
  dispatchAndRecordMessage,
  type OutboxRow,
} from "@/lib/store";

export const dynamic = "force-dynamic";

/**
 * Authenticated bridge endpoint for `tools/cursor_fleet/push_daemon.py` and `mcp_server.py`.
 * Pulls pending tmux push/spawn commands from Postgres so web/phone messages are pushed directly
 * into running `tmux` panes without any Cursor session polling tools, and syncs live pane output
 * and agent replies back into Postgres.
 */
export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  const outbox = await sql<OutboxRow>(
    `select id, session_id, tmux_target, action, payload_text, workdir, status, created_at
     from fleet_tmux_outbox
     where owner_key = $1 and status = 'pending'
     order by created_at asc
     limit 50`,
    [caller.ownerKey],
  );

  return jsonNoStore({ outbox });
}

export async function POST(request: Request) {
  const caller = await authorizeRequest(request, { requireCsrfForWrite: true });
  if (caller instanceof Response) return caller;

  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return jsonNoStore({ error: "Invalid JSON body" }, 400);
  }

  const op = String(body.op ?? "");

  if (op === "ack_outbox") {
    const ids = Array.isArray(body.ids)
      ? body.ids.map((x) => String(x)).slice(0, 50)
      : [];
    for (const id of ids) {
      await sql(
        `update fleet_tmux_outbox
         set status = 'delivered', delivered_at = now()
         where owner_key = $1 and id = $2`,
        [caller.ownerKey, id],
      );
      await sql(
        `update fleet_messages
         set delivery_status = 'pushed_to_tmux'
         where owner_key = $1 and delivery_status = 'queued_for_tmux'`,
        [caller.ownerKey],
      );
    }
    return jsonNoStore({ ok: true, acked: ids.length });
  }

  if (op === "sync_tmux_panes") {
    const panes = Array.isArray(body.panes) ? body.panes.slice(0, 40) : [];
    for (const rawPane of panes) {
      if (!rawPane || typeof rawPane !== "object") continue;
      const p = rawPane as Record<string, unknown>;
      const slug = String(p.slug ?? "").trim();
      if (!slug) continue;
      const snapshot = String(p.paneSnapshot ?? "").slice(0, 32000);
      const status = String(p.status ?? "active").slice(0, 32);
      const workdir = String(p.workdir ?? "/workspace").slice(0, 240);
      const branch = String(p.branch ?? "").slice(0, 120);

      const updated = await sql<{ id: string }>(
        `update fleet_sessions
         set pane_snapshot = $3,
             status = $4,
             workdir = $5,
             branch = case when $6 <> '' then $6 else branch end,
             is_sample = false,
             last_heartbeat_at = now(),
             updated_at = now()
         where owner_key = $1 and slug = $2
         returning id`,
        [caller.ownerKey, slug, snapshot, status, workdir, branch],
      );

      if (updated.length === 0) {
        const created = await createOrUpdateSession({
          ownerKey: caller.ownerKey,
          title: String(p.title ?? slug),
          slug,
          role: slug === "coordinator" ? "coordinator" : "worker",
          kind: "tmux",
          workdir,
          branch,
          enqueueTmuxSpawn: false,
        });
        await sql(
          `update fleet_sessions
           set pane_snapshot = $3, status = $4, last_heartbeat_at = now(), is_sample = false
           where owner_key = $1 and id = $2`,
          [caller.ownerKey, created.id, snapshot, status],
        );
      }
    }
    return jsonNoStore({ ok: true, synced: panes.length });
  }

  if (op === "session_message") {
    const fromSlug = String(body.fromSession ?? "coordinator").trim();
    const toSlugRaw = body.toSession ? String(body.toSession).trim() : null;
    const text = String(body.body ?? "").trim();
    const skipPush = Boolean(body.skipPush);
    if (!text) {
      return jsonNoStore({ error: "body required" }, 400);
    }
    const msg = await dispatchAndRecordMessage({
      ownerKey: caller.ownerKey,
      senderKind: "session",
      fromSessionId: fromSlug,
      fromLabel: fromSlug,
      toSessionId: toSlugRaw,
      toLabel: toSlugRaw ?? "You",
      body: text,
      skipPush,
    });
    return jsonNoStore({ ok: true, message: msg });
  }

  return jsonNoStore({ error: "Unknown bridge op" }, 400);
}
