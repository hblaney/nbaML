import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { PGlite } from "@electric-sql/pglite";
import pg from "pg";

const databaseUrl = process.env.DATABASE_URL?.trim() ?? "";

declare global {
  // Preserve PGlite / pg Pool across Next.js hot reloads in dev
  var __cursorHubPgPool: pg.Pool | undefined;
  var __cursorHubPglite: Promise<PGlite> | undefined;
}

export async function sql<T = Record<string, unknown>>(
  text: string,
  params: unknown[] = [],
): Promise<T[]> {
  if (databaseUrl !== "") {
    globalThis.__cursorHubPgPool ??= new pg.Pool({
      connectionString: databaseUrl,
      max: 5,
    });
    const res = await globalThis.__cursorHubPgPool.query(text, params);
    return res.rows as T[];
  }
  globalThis.__cursorHubPglite ??= startLocal().catch((error: unknown) => {
    globalThis.__cursorHubPglite = undefined;
    throw error;
  });
  const db = await globalThis.__cursorHubPglite;
  const res = await db.query<T>(text, params);
  return res.rows;
}

async function startLocal(): Promise<PGlite> {
  const db = new PGlite();
  const migDir = path.join(process.cwd(), "migrations");
  const files = (await readdir(migDir).catch(() => []))
    .filter((f) => f.endsWith(".sql"))
    .sort();
  for (const file of files) {
    await db.exec(await readFile(path.join(migDir, file), "utf8"));
  }
  return db;
}

export function isUsingManagedPostgres(): boolean {
  return databaseUrl !== "";
}
