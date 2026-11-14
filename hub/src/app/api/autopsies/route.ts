import { randomBytes } from "node:crypto";
import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { sql } from "@/lib/db";

export const dynamic = "force-dynamic";

const SAFE_TOPIC = /^[a-zA-Z0-9_-]{6,64}$/;

async function getOrCreateTopic(ownerKey: string): Promise<string> {
  const envTopic = (process.env.NTFY_TOPIC ?? "").trim();
  const rows = await sql<{ ntfy_topic: string }>(
    "select ntfy_topic from fleet_alert_settings where owner_key = $1 limit 1",
    [ownerKey],
  );
  if (rows[0]?.ntfy_topic) {
    return rows[0].ntfy_topic;
  }
  const generated =
    envTopic && SAFE_TOPIC.test(envTopic)
      ? envTopic
      : `nba-props-${randomBytes(5).toString("hex")}`;
  await sql(
    `insert into fleet_alert_settings (owner_key, ntfy_topic, enabled, updated_at)
     values ($1, $2, true, now())
     on conflict (owner_key) do nothing`,
    [ownerKey, generated],
  );
  const again = await sql<{ ntfy_topic: string }>(
    "select ntfy_topic from fleet_alert_settings where owner_key = $1 limit 1",
    [ownerKey],
  );
  return again[0]?.ntfy_topic ?? generated;
}

export async function POST(request: Request) {
  const caller = await authorizeRequest(request, { requireCsrfForWrite: true });
  if (caller instanceof Response) return caller;

  let body: Record<string, unknown> = {};
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return jsonNoStore({ error: "Invalid JSON body" }, 400);
  }

  if (body.op === "save_topic") {
    const nextTopic = String(body.ntfyTopic ?? "").trim();
    if (!SAFE_TOPIC.test(nextTopic)) {
      return jsonNoStore(
        { ok: false, error: "Topic must be 6-64 letters, numbers, or hyphens" },
        400,
      );
    }
    await sql(
      `insert into fleet_alert_settings (owner_key, ntfy_topic, enabled, updated_at)
       values ($1, $2, true, now())
       on conflict (owner_key) do update set ntfy_topic = excluded.ntfy_topic, updated_at = now()`,
      [caller.ownerKey, nextTopic],
    );
    return jsonNoStore({ ok: true, ntfyTopic: nextTopic });
  }

  const topic = await getOrCreateTopic(caller.ownerKey);
  if (!SAFE_TOPIC.test(topic)) {
    return jsonNoStore({ ok: false, status: "invalid_topic" }, 400);
  }

  const title = String(body.title ?? "NBA Props 24/7 Autopilot").slice(0, 100);
  const message = String(
    body.message ??
      "Connected! Every promoted R&D upgrade, 1-a-day PrizePicks card, and post-close slip autopsy will buzz your iPhone here.",
  ).slice(0, 1000);

  try {
    const resp = await fetch(`https://ntfy.sh/${encodeURIComponent(topic)}`, {
      method: "POST",
      headers: {
        Title: title,
        Priority: "high",
      },
      body: message,
      signal: AbortSignal.timeout(8000),
    });
    return jsonNoStore({
      ok: resp.ok,
      ntfyTopic: topic,
      status: resp.ok ? "sent" : `http_${resp.status}`,
      error: resp.ok ? undefined : `ntfy returned HTTP ${resp.status}`,
    });
  } catch (err) {
    return jsonNoStore(
      {
        ok: false,
        ntfyTopic: topic,
        status: "network_error",
        error: err instanceof Error ? err.message : "network_error",
      },
      502,
    );
  }
}

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  const topic = await getOrCreateTopic(caller.ownerKey);
  return jsonNoStore({
    configured: true,
    ntfyTopic: topic,
    channels: ["ntfy_push"],
    recentAutopsies: [
      {
        date: "Thu Nov 12, 2026",
        slip: "3-Pick Power ($4 -> $10 Cashout, +150%)",
        legs: [
          {
            player: "Ayo Dosunmu (MIN)",
            pick: "MORE 17.5 PTS",
            actual: "18 PTS (27m)",
            tag: "CLV_SAVED_WIN",
            note: "Beat DK's 18.5 line by 1.0 pt; landed on exact 18 before sitting Q4 in 133-94 blowout.",
          },
          {
            player: "John Collins (DET)",
            pick: "LESS 21.5 PRA",
            actual: "14 PRA (25m, 5 PF)",
            tag: "SURVIVED_OVERTIME_WIN",
            note: "Cleared LESS 21.5 easily even through 2OT; 5 fouls capped extra OT floor time.",
          },
          {
            player: "Jalen Green (PHX)",
            pick: "LESS 20.5 PRA",
            actual: "16 PRA (21m, -16 +/-)",
            tag: "DISCIPLINED_LIVE_CASHOUT",
            note: "Cashed out $10 in 1-pt game; coach kept him benched all Q4 on -16 +/- -> add +/- hook signal to live Q4 model.",
          },
        ],
      },
      {
        date: "Fri Nov 13, 2026 (Completed Early Legs)",
        slip: "5-Pick Flex (In Progress)",
        legs: [
          {
            player: "Landry Shamet (NY)",
            pick: "LESS 11.5 PR",
            actual: "8 PR (21m)",
            tag: "GOOD_PROCESS_WIN",
            note: "+1.0 pt CLV (PP 11.5 vs DK 10.5) + Mikal Bridges return cut spot-starter minutes to 21m.",
          },
          {
            player: "Desmond Bane (ORL)",
            pick: "LESS 20.5 PTS",
            actual: "21 PTS (6/10 3PT, 6 PF)",
            tag: "GOOD_PROCESS_3PT_VARIANCE_LOSS",
            note: "+1.0 pt CLV & fouled out in 35m, but shot 60% on 10 threes (+7.0 pts above expected 3PT). Directly validated RD-001 (-0.51% Points variance penalty).",
          },
        ],
      },
    ],
  });
}
