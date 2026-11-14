const CURSOR_API_BASE = "https://api.cursor.com";
const VALID_AGENT_ID = /^bc-[a-zA-Z0-9-]+$/;

export function hasCursorApiKey(): boolean {
  return Boolean(
    process.env.FLEET_AGENT_API_KEY?.trim() || process.env.CURSOR_API_KEY?.trim(),
  );
}

function authHeaders(): Record<string, string> {
  const key = (
    process.env.FLEET_AGENT_API_KEY?.trim() ||
    process.env.CURSOR_API_KEY?.trim() ||
    ""
  );
  if (!key) {
    throw new Error("FLEET_AGENT_API_KEY is not configured");
  }
  return {
    Authorization: `Bearer ${key}`,
    "Content-Type": "application/json",
    Accept: "application/json",
  };
}

export type CloudLaunchResult = {
  ok: boolean;
  agentId?: string;
  status?: string;
  url?: string;
  error?: string;
};

export async function launchCloudAgent(params: {
  name: string;
  promptText: string;
  repoUrl?: string;
  startingRef?: string;
}): Promise<CloudLaunchResult> {
  if (!hasCursorApiKey()) {
    return { ok: false, error: "FLEET_AGENT_API_KEY not bound yet" };
  }
  // Cloud Agents API v1: only documented fields — do not send legacy `mode`.
  const body: Record<string, unknown> = {
    name: params.name.slice(0, 100),
    prompt: { text: params.promptText.slice(0, 12000) },
    autoCreatePR: true,
  };
  const trimmedRepo = params.repoUrl?.trim() ?? "";
  if (trimmedRepo.startsWith("https://")) {
    body.repos = [
      {
        url: trimmedRepo,
        startingRef: params.startingRef?.trim() || "main",
      },
    ];
  }
  try {
    const resp = await fetch(`${CURSOR_API_BASE}/v1/agents`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    const rawText = await resp.text();
    let data: {
      agent?: { id?: string; status?: string; url?: string };
      id?: string;
      error?: string;
      message?: string;
    } = {};
    try {
      data = JSON.parse(rawText) as typeof data;
    } catch {
      /* non-JSON body */
    }
    if (!resp.ok) {
      const detail =
        data.error ||
        data.message ||
        (rawText ? rawText.slice(0, 240) : "") ||
        `HTTP ${resp.status}`;
      return { ok: false, error: `Cursor API ${resp.status}: ${detail}` };
    }
    const agentId = data.agent?.id ?? data.id ?? "";
    if (!agentId) {
      return { ok: false, error: "Cursor API returned no agent id" };
    }
    return {
      ok: true,
      agentId,
      status: data.agent?.status?.toLowerCase() ?? "active",
      url: data.agent?.url ?? "",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "request failed";
    return { ok: false, error: `Cursor Cloud API request failed (${msg})` };
  }
}

export async function steerCloudAgent(
  cloudAgentId: string,
  text: string,
): Promise<{ ok: boolean; delivery: string }> {
  if (!VALID_AGENT_ID.test(cloudAgentId)) {
    return { ok: false, delivery: "invalid_agent_id" };
  }
  if (!hasCursorApiKey()) {
    return { ok: false, delivery: "api_key_missing" };
  }
  try {
    const resp = await fetch(
      `${CURSOR_API_BASE}/v1/agents/${encodeURIComponent(cloudAgentId)}/runs`,
      {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          prompt: { text: text.slice(0, 12000) },
          delivery: "steer",
        }),
        signal: AbortSignal.timeout(15000),
      },
    );
    if (!resp.ok) {
      return { ok: false, delivery: `http_${resp.status}` };
    }
    const data = (await resp.json()) as { delivery?: string };
    return { ok: true, delivery: data.delivery ?? "steered" };
  } catch {
    return { ok: false, delivery: "network_error" };
  }
}
