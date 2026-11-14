import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { dispatchAndRecordMessage, listMessages } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  const url = new URL(request.url);
  const sessionId = url.searchParams.get("sessionId");
  const modeParam = url.searchParams.get("mode");
  const mode =
    modeParam === "bus" ? "bus" : modeParam === "session" ? "session" : "all";

  const messages = await listMessages(caller.ownerKey, sessionId, mode);
  return jsonNoStore({ messages });
}

export async function POST(request: Request) {
  const caller = await authorizeRequest(request, { requireCsrfForWrite: true });
  if (caller instanceof Response) return caller;

  let payload: Record<string, unknown>;
  try {
    payload = (await request.json()) as Record<string, unknown>;
  } catch {
    return jsonNoStore({ error: "Invalid JSON body" }, 400);
  }

  const text = String(payload.body ?? "").trim();
  if (!text) {
    return jsonNoStore({ error: "Message body cannot be empty" }, 400);
  }

  const senderKind = payload.senderKind === "session" ? "session" : "user";
  const fromSessionId =
    typeof payload.fromSessionId === "string" && payload.fromSessionId
      ? payload.fromSessionId
      : null;
  const toSessionId =
    typeof payload.toSessionId === "string" && payload.toSessionId
      ? payload.toSessionId
      : null;

  const message = await dispatchAndRecordMessage({
    ownerKey: caller.ownerKey,
    senderKind,
    fromSessionId,
    fromLabel:
      senderKind === "session"
        ? String(payload.fromLabel ?? "coordinator")
        : caller.displayName,
    toSessionId,
    toLabel: typeof payload.toLabel === "string" ? payload.toLabel : undefined,
    body: text,
  });

  return jsonNoStore({ message });
}
