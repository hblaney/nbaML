import { requireIdentity, type VerifiedIdentity } from "../../cursor/identity";

export type AuthorizedCaller = {
  identity: VerifiedIdentity;
  ownerKey: string;
  displayName: string;
  email: string;
  isApiKey: boolean;
  simulated: boolean;
  stage: "production" | "preview";
};

export function jsonNoStore(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: {
      "Cache-Control": "private, no-store",
    },
  });
}

/**
 * Authenticates the caller via Cursor's signed identity header (or simulated dev viewer locally)
 * and resolves a stable stage-isolated `ownerKey` shared only between the owner and the owner's
 * personal Cursor API key (used by the local tmux bridge / MCP server).
 */
export async function authorizeRequest(
  request: Request,
  options: { requireCsrfForWrite?: boolean } = {},
): Promise<AuthorizedCaller | Response> {
  const identity = await requireIdentity(request);
  if (identity instanceof Response) {
    return identity;
  }

  if (options.requireCsrfForWrite && identity.kind === "person") {
    const contentType = request.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().includes("application/json")) {
      return jsonNoStore({ error: "Expected application/json body" }, 415);
    }
    const origin = request.headers.get("origin");
    if (origin) {
      let originHost = "";
      try {
        originHost = new URL(origin).host.toLowerCase();
      } catch {
        return jsonNoStore({ error: "Invalid Origin header" }, 403);
      }
      const candidates = [
        request.headers.get("x-forwarded-host"),
        request.headers.get("host"),
        (() => {
          try {
            return new URL(request.url).host;
          } catch {
            return null;
          }
        })(),
      ]
        .flatMap((value) => (value ?? "").split(","))
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean);
      if (candidates.length > 0 && !candidates.includes(originHost)) {
        return jsonNoStore({ error: "Cross-origin state mutation refused" }, 403);
      }
    }
  }

  if (identity.kind === "key") {
    if (typeof identity.ownerUserId !== "number" || identity.ownerUserId <= 0) {
      return jsonNoStore(
        { error: "Only a personal owner API key may access this workspace" },
        403,
      );
    }
    return {
      identity,
      ownerKey: `${identity.stage}:user:${identity.ownerUserId}`,
      displayName: identity.name || "Bridge Key",
      email: "",
      isApiKey: true,
      simulated: identity.simulated,
      stage: identity.stage,
    };
  }

  const baseId = identity.userId > 0 ? `user:${identity.userId}` : identity.sub;
  return {
    identity,
    ownerKey: `${identity.stage}:${baseId}`,
    displayName: identity.name || identity.email || "Operator",
    email: identity.email,
    isApiKey: false,
    simulated: identity.simulated,
    stage: identity.stage,
  };
}
