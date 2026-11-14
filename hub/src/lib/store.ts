import { randomUUID } from "node:crypto";
import { steerCloudAgent } from "./cursor-cloud";
import { sql } from "./db";

export type FleetSessionRow = {
  id: string;
  owner_key: string;
  slug: string;
  title: string;
  role: string;
  kind: "tmux" | "cloud";
  status: string;
  tmux_target: string;
  workdir: string;
  branch: string;
  cloud_agent_id: string;
  repo_url: string;
  pane_snapshot: string;
  is_sample: boolean;
  last_heartbeat_at: string | null;
  created_at: string;
  updated_at: string;
};

export type FleetMessageRow = {
  id: string;
  owner_key: string;
  sender_kind: "user" | "session" | "system";
  from_session_id: string | null;
  from_label: string;
  to_session_id: string | null;
  to_label: string;
  body: string;
  delivery_mode: string;
  delivery_status: string;
  metadata_json: string;
  is_sample: boolean;
  created_at: string;
};

export type OutboxRow = {
  id: string;
  session_id: string;
  tmux_target: string;
  action: string;
  payload_text: string;
  workdir: string;
  status: string;
  created_at: string;
};

export function sanitizeSlug(raw: string): string {
  const clean = raw
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return clean || `session-${randomUUID().slice(0, 6)}`;
}

export async function ensureStarterFleet(ownerKey: string): Promise<void> {
  const existing = await sql<{ cnt: string | number }>(
    "select count(*)::int as cnt from fleet_sessions where owner_key = $1",
    [ownerKey],
  );
  if (Number(existing[0]?.cnt ?? 0) > 0) {
    return;
  }

  const coordId = `ses_${randomUUID()}`;
  const propsId = `ses_${randomUUID()}`;
  const backtestId = `ses_${randomUUID()}`;

  await sql(
    `insert into fleet_sessions
      (id, owner_key, slug, title, role, kind, status, tmux_target, workdir, branch, pane_snapshot, is_sample)
     values
      ($1, $2, 'coordinator', 'Coordinator (Main)', 'coordinator', 'tmux', 'active', 'coordinator:0.0', '/workspace', 'cursor/session-hub-ac4c', $3, true),
      ($4, $2, 'props-scanner', 'PrizePicks Live Scanner', 'worker', 'tmux', 'active', 'props-scanner:0.0', '/workspace', 'cursor/parlay-ev-selector-ac4c', $5, true),
      ($6, $2, 'kelly-backtest', 'Kelly Portfolio Worker', 'worker', 'cloud', 'idle', '', '/workspace', 'cursor/parlay-ev-selector-ac4c', $7, true)`,
    [
      coordId,
      ownerKey,
      [
        "$ tmux ls",
        "coordinator: 1 windows (created) [Cursor agent --trust --approve-mcps]",
        "props-scanner: 1 windows (created) [Cursor agent --workspace /workspace]",
        "",
        "● Coordinator online. MCP fleet tools ready:",
        "  • fleet_spawn_tmux_session  • fleet_message_session",
        "  • fleet_read_tmux_pane      • fleet_search_history",
      ].join("\n"),
      propsId,
      [
        "$ agent --workspace /workspace --approve-mcps",
        "▸ Scanning PrizePicks Standard board vs DraftKings two-way Shin odds...",
        "▸ Found 5 clean .5 half-point discrepancies (+14.6% 5-Flex EV).",
        "▸ Waiting for live push messages from Coordinator or Web UI...",
      ].join("\n"),
      backtestId,
      [
        "Cursor Cloud Agent • idle",
        "Last verified OOS run: 2024-25 $50 -> $178.87 | 2025-26 $50 -> $166.69",
      ].join("\n"),
    ],
  );

  await sql(
    `insert into fleet_messages
      (id, owner_key, sender_kind, from_session_id, from_label, to_session_id, to_label, body, delivery_mode, delivery_status, is_sample)
     values
      ($1, $2, 'user', null, 'You', $3, 'coordinator', 'Spin up a worker to watch live PrizePicks vs DraftKings stale lines and ping me when 3+ games line up.', 'tmux_push', 'delivered', true),
      ($4, $2, 'session', $3, 'coordinator', $5, 'props-scanner', 'Start in tmux session props-scanner: pull PrizePicks league_id=7 standard board, compare against DraftKings Shin fair probs, and exclude Goblins/Demons.', 'inter_session_push', 'delivered', true),
      ($6, $2, 'session', $5, 'props-scanner', $3, 'coordinator', 'Done — locked in 1-pick-per-game rule and verified .5 half-point filter so pushes cannot downgrade 4-Flex / 5-Flex slips.', 'inter_session_push', 'delivered', true),
      ($7, $2, 'session', $3, 'coordinator', null, 'You', 'props-scanner is live in tmux and synced to Postgres. Anything you type here pushes straight into its pane with zero polling, and every session-to-session note is indexed for search.', 'reply', 'delivered', true)`,
    [
      `msg_${randomUUID()}`,
      ownerKey,
      coordId,
      `msg_${randomUUID()}`,
      propsId,
      `msg_${randomUUID()}`,
      `msg_${randomUUID()}`,
    ],
  );
}

export async function listSessions(ownerKey: string): Promise<FleetSessionRow[]> {
  return sql<FleetSessionRow>(
    `select * from fleet_sessions
     where owner_key = $1
     order by case when role = 'coordinator' then 0 else 1 end, updated_at desc
     limit 100`,
    [ownerKey],
  );
}

export async function listMessages(
  ownerKey: string,
  sessionId?: string | null,
  mode: "session" | "bus" | "all" = "all",
): Promise<FleetMessageRow[]> {
  if (mode === "bus") {
    return sql<FleetMessageRow>(
      `select * from fleet_messages
       where owner_key = $1 and sender_kind = 'session' and from_session_id is not null and to_session_id is not null
       order by created_at asc
       limit 200`,
      [ownerKey],
    );
  }
  if (mode === "session" && sessionId) {
    return sql<FleetMessageRow>(
      `select * from fleet_messages
       where owner_key = $1 and (to_session_id = $2 or from_session_id = $2)
       order by created_at asc
       limit 250`,
      [ownerKey, sessionId],
    );
  }
  return sql<FleetMessageRow>(
    `select * from fleet_messages
     where owner_key = $1
     order by created_at asc
     limit 300`,
    [ownerKey],
  );
}

export async function createOrUpdateSession(params: {
  ownerKey: string;
  title: string;
  slug?: string;
  role?: string;
  kind: "tmux" | "cloud";
  workdir?: string;
  branch?: string;
  repoUrl?: string;
  cloudAgentId?: string;
  initialPrompt?: string;
  enqueueTmuxSpawn?: boolean;
}): Promise<FleetSessionRow> {
  const slug = sanitizeSlug(params.slug || params.title);
  const id = `ses_${randomUUID()}`;
  const tmuxTarget = params.kind === "tmux" ? `${slug}:0.0` : "";
  const workdir = (params.workdir?.trim() || "/workspace").slice(0, 240);
  const role = params.role === "coordinator" ? "coordinator" : "worker";
  const status = params.kind === "cloud" ? "active" : "starting";

  const rows = await sql<FleetSessionRow>(
    `insert into fleet_sessions
      (id, owner_key, slug, title, role, kind, status, tmux_target, workdir, branch, cloud_agent_id, repo_url, is_sample, updated_at)
     values
      ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, false, now())
     on conflict (owner_key, slug) do update set
       title = excluded.title,
       kind = excluded.kind,
       status = excluded.status,
       workdir = excluded.workdir,
       branch = case when excluded.branch <> '' then excluded.branch else fleet_sessions.branch end,
       cloud_agent_id = case when excluded.cloud_agent_id <> '' then excluded.cloud_agent_id else fleet_sessions.cloud_agent_id end,
       is_sample = false,
       updated_at = now()
     returning *`,
    [
      id,
      params.ownerKey,
      slug,
      params.title.trim().slice(0, 100),
      role,
      params.kind,
      status,
      tmuxTarget,
      workdir,
      (params.branch?.trim() || "").slice(0, 120),
      (params.cloudAgentId?.trim() || "").slice(0, 120),
      (params.repoUrl?.trim() || "").slice(0, 240),
    ],
  );
  const session = rows[0];

  if (params.kind === "tmux" && params.enqueueTmuxSpawn !== false) {
    await sql(
      `insert into fleet_tmux_outbox
        (id, owner_key, session_id, tmux_target, action, payload_text, workdir, status)
       values ($1, $2, $3, $4, 'spawn_tmux', $5, $6, 'pending')`,
      [
        `out_${randomUUID()}`,
        params.ownerKey,
        session.id,
        session.tmux_target || `${session.slug}:0.0`,
        (params.initialPrompt ?? "").slice(0, 8000),
        workdir,
      ],
    );
  }

  return session;
}

export async function dispatchAndRecordMessage(params: {
  ownerKey: string;
  senderKind: "user" | "session" | "system";
  fromSessionId?: string | null;
  fromLabel: string;
  toSessionId?: string | null;
  toLabel?: string;
  body: string;
  skipPush?: boolean;
}): Promise<FleetMessageRow> {
  const cleanBody = params.body.trim().slice(0, 16000);
  if (!cleanBody) {
    throw new Error("Message body cannot be empty");
  }

  let targetSession: FleetSessionRow | undefined;
  if (params.toSessionId) {
    const found = await sql<FleetSessionRow>(
      "select * from fleet_sessions where owner_key = $1 and (id = $2 or slug = $2) limit 1",
      [params.ownerKey, params.toSessionId],
    );
    targetSession = found[0];
  }

  let sourceSession: FleetSessionRow | undefined;
  if (params.fromSessionId) {
    const found = await sql<FleetSessionRow>(
      "select * from fleet_sessions where owner_key = $1 and (id = $2 or slug = $2) limit 1",
      [params.ownerKey, params.fromSessionId],
    );
    sourceSession = found[0];
  }

  const resolvedFromId = sourceSession?.id ?? params.fromSessionId ?? null;
  const resolvedFromLabel = sourceSession?.slug ?? params.fromLabel.slice(0, 80);
  const resolvedToId = targetSession?.id ?? null;
  const resolvedToLabel = targetSession?.slug ?? params.toLabel?.slice(0, 80) ?? "You";

  let deliveryMode = "logged";
  let deliveryStatus = "delivered";

  const pushFormatted =
    params.senderKind === "session"
      ? `[Inter-Session Message from ${resolvedFromLabel}]: ${cleanBody}`
      : cleanBody;

  if (targetSession && !params.skipPush) {
    if (targetSession.kind === "cloud" && targetSession.cloud_agent_id) {
      deliveryMode = "cloud_steer";
      const steerRes = await steerCloudAgent(targetSession.cloud_agent_id, pushFormatted);
      deliveryStatus = steerRes.ok ? steerRes.delivery : `queued (${steerRes.delivery})`;
    } else if (targetSession.kind === "tmux") {
      deliveryMode = params.senderKind === "session" ? "inter_session_push" : "tmux_push";
      deliveryStatus = "queued_for_tmux";
      await sql(
        `insert into fleet_tmux_outbox
          (id, owner_key, session_id, tmux_target, action, payload_text, workdir, status)
         values ($1, $2, $3, $4, 'push_keys', $5, $6, 'pending')`,
        [
          `out_${randomUUID()}`,
          params.ownerKey,
          targetSession.id,
          targetSession.tmux_target || `${targetSession.slug}:0.0`,
          pushFormatted,
          targetSession.workdir || "/workspace",
        ],
      );
    }
    await sql(
      "update fleet_sessions set updated_at = now() where owner_key = $1 and id = $2",
      [params.ownerKey, targetSession.id],
    );
  }

  const inserted = await sql<FleetMessageRow>(
    `insert into fleet_messages
      (id, owner_key, sender_kind, from_session_id, from_label, to_session_id, to_label, body, delivery_mode, delivery_status, is_sample)
     values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, false)
     returning *`,
    [
      `msg_${randomUUID()}`,
      params.ownerKey,
      params.senderKind,
      resolvedFromId,
      resolvedFromLabel,
      resolvedToId,
      resolvedToLabel,
      cleanBody,
      deliveryMode,
      deliveryStatus,
    ],
  );
  return inserted[0];
}

export async function searchMessageHistory(params: {
  ownerKey: string;
  query: string;
  scope?: "all" | "inter_session" | "user";
  sessionSlugOrId?: string;
  limit?: number;
}): Promise<FleetMessageRow[]> {
  const q = params.query.trim().slice(0, 200);
  const cap = Math.min(Math.max(params.limit ?? 40, 1), 100);
  if (!q) {
    return [];
  }

  const scopeClause =
    params.scope === "inter_session"
      ? "and sender_kind = 'session' and from_session_id is not null and to_session_id is not null"
      : params.scope === "user"
        ? "and sender_kind = 'user'"
        : "";

  const likePattern = `%${q.replace(/[%_\\]/g, "\\$&")}%`;

  if (params.sessionSlugOrId) {
    return sql<FleetMessageRow>(
      `select *
       from fleet_messages
       where owner_key = $1
         and (from_session_id = $4 or to_session_id = $4 or from_label = $4 or to_label = $4)
         ${scopeClause}
         and (
           to_tsvector('english', coalesce(from_label, '') || ' ' || coalesce(to_label, '') || ' ' || coalesce(body, ''))
             @@ websearch_to_tsquery('english', $2)
           or body ilike $3
           or from_label ilike $3
           or to_label ilike $3
         )
       order by created_at desc
       limit $5`,
      [params.ownerKey, q, likePattern, params.sessionSlugOrId, cap],
    );
  }

  return sql<FleetMessageRow>(
    `select *
     from fleet_messages
     where owner_key = $1
       ${scopeClause}
       and (
         to_tsvector('english', coalesce(from_label, '') || ' ' || coalesce(to_label, '') || ' ' || coalesce(body, ''))
           @@ websearch_to_tsquery('english', $2)
         or body ilike $3
         or from_label ilike $3
         or to_label ilike $3
       )
     order by created_at desc
     limit $4`,
    [params.ownerKey, q, likePattern, cap],
  );
}
