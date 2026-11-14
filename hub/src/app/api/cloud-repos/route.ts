import { authorizeRequest, jsonNoStore } from "@/lib/auth";
import { hasCursorApiKey } from "@/lib/cursor-cloud";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  const fallbackRepos = [
    { name: "hblaney/nbaML", url: "https://github.com/hblaney/nbaML" },
    { name: "hblaney/mlb-edge", url: "https://github.com/hblaney/mlb-edge" },
  ];

  if (!hasCursorApiKey()) {
    return jsonNoStore({ configured: false, repositories: fallbackRepos });
  }

  const key =
    process.env.FLEET_AGENT_API_KEY?.trim() || process.env.CURSOR_API_KEY?.trim() || "";
  try {
    const resp = await fetch("https://api.cursor.com/v1/repositories", {
      headers: {
        Authorization: `Bearer ${key}`,
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) {
      return jsonNoStore({ configured: true, repositories: fallbackRepos });
    }
    const data = (await resp.json()) as {
      repositories?: Array<{ owner?: string; name?: string; repository?: string; url?: string }>;
    };
    const mapped = (data.repositories ?? [])
      .map((r) => {
        const url =
          r.url ||
          r.repository ||
          (r.owner && r.name ? `https://github.com/${r.owner}/${r.name}` : "");
        const name = url.replace("https://github.com/", "");
        return { name, url };
      })
      .filter((r) => r.url.startsWith("https://github.com/"));

    return jsonNoStore({
      configured: true,
      repositories: mapped.length > 0 ? mapped : fallbackRepos,
    });
  } catch {
    return jsonNoStore({ configured: true, repositories: fallbackRepos });
  }
}
