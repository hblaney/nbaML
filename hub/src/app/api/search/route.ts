import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { searchMessageHistory } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  const url = new URL(request.url);
  const q = url.searchParams.get("q") ?? "";
  const scopeRaw = url.searchParams.get("scope");
  const scope =
    scopeRaw === "inter_session"
      ? "inter_session"
      : scopeRaw === "user"
        ? "user"
        : "all";
  const sessionFilter = url.searchParams.get("session") ?? undefined;

  const results = await searchMessageHistory({
    ownerKey: caller.ownerKey,
    query: q,
    scope,
    sessionSlugOrId: sessionFilter,
    limit: 50,
  });

  return jsonNoStore({ query: q, count: results.length, results });
}
