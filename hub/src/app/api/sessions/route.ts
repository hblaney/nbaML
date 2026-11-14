import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { launchCloudAgent } from "@/lib/cursor-cloud";
import {
  createOrUpdateSession,
  dispatchAndRecordMessage,
  ensureStarterFleet,
  listSessions,
} from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  await ensureStarterFleet(caller.ownerKey);
  const sessions = await listSessions(caller.ownerKey);
  return jsonNoStore({ sessions });
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

  const title = String(body.title ?? "").trim();
  if (!title) {
    return jsonNoStore({ error: "Session title is required" }, 400);
  }

  const kind = body.kind === "cloud" ? "cloud" : "tmux";
  const role = body.role === "coordinator" ? "coordinator" : "worker";
  const slug = typeof body.slug === "string" ? body.slug : title;
  const workdir = typeof body.workdir === "string" ? body.workdir : "/workspace";
  const branch = typeof body.branch === "string" ? body.branch : "";
  const repoUrl = typeof body.repoUrl === "string" ? body.repoUrl : "";
  const initialPrompt =
    typeof body.initialPrompt === "string" ? body.initialPrompt.trim() : "";

  let cloudAgentId = "";
  let cloudLaunchNote = "";

  if (kind === "cloud" && initialPrompt) {
    const launched = await launchCloudAgent({
      name: title,
      promptText: initialPrompt,
      repoUrl,
      startingRef: branch || "main",
    });
    if (launched.ok && launched.agentId) {
      cloudAgentId = launched.agentId;
    } else {
      cloudLaunchNote = launched.error ?? "Cloud API unavailable";
    }
  }

  const session = await createOrUpdateSession({
    ownerKey: caller.ownerKey,
    title,
    slug,
    role,
    kind,
    workdir,
    branch,
    repoUrl,
    cloudAgentId,
    initialPrompt,
    enqueueTmuxSpawn: kind === "tmux",
  });

  if (initialPrompt) {
    await dispatchAndRecordMessage({
      ownerKey: caller.ownerKey,
      senderKind: "user",
      fromLabel: caller.displayName,
      toSessionId: session.id,
      toLabel: session.slug,
      body: initialPrompt,
      skipPush: true, // already included in spawn_tmux or cloud launch payload
    });
  }

  return jsonNoStore({
    session,
    cloudLaunchNote,
  });
}
