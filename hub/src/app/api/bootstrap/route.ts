import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { hasCursorApiKey } from "@/lib/cursor-cloud";
import { isUsingManagedPostgres, sql } from "@/lib/db";
import { ensureStarterFleet, listMessages, listSessions } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  await ensureStarterFleet(caller.ownerKey);
  const [sessions, messages, outboxCount] = await Promise.all([
    listSessions(caller.ownerKey),
    listMessages(caller.ownerKey, null, "all"),
    sql<{ cnt: number }>(
      "select count(*)::int as cnt from fleet_tmux_outbox where owner_key = $1 and status = 'pending'",
      [caller.ownerKey],
    ),
  ]);

  return jsonNoStore({
    viewer: {
      name: caller.displayName,
      email: caller.email,
      simulated: caller.simulated,
      stage: caller.stage,
    },
    runtime: {
      cloudApiConfigured: hasCursorApiKey(),
      managedPostgres: isUsingManagedPostgres(),
      pendingOutbox: Number(outboxCount[0]?.cnt ?? 0),
    },
    sessions,
    messages,
  });
}
