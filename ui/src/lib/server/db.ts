import postgres, { type Sql } from 'postgres';
import { drizzle, type PostgresJsDatabase } from 'drizzle-orm/postgres-js';
import * as schema from '../../../drizzle/schema';

// Resolve DATABASE_URL from SvelteKit's `$env/dynamic/private` when running
// inside the SvelteKit runtime (covers dev server / Vercel deploy automatically),
// falling back to `process.env` for raw-bun script execution.
async function resolveDatabaseUrl(): Promise<string | undefined> {
	try {
		const mod = (await import('$env/dynamic/private')) as { env: Record<string, string | undefined> };
		if (mod.env?.DATABASE_URL) return mod.env.DATABASE_URL;
	} catch {
		/* not in SvelteKit context — falling through to process.env */
	}
	return process.env.DATABASE_URL;
}

// Lazy initialisation — tests import this module for the `DbClient` type but
// don't have a `DATABASE_URL` set. We only fail when something tries to use
// the production client.
let _pg: Sql | null = null;
let _pgPromise: Promise<Sql> | null = null;
function pgClient(): Promise<Sql> {
	if (_pg) return Promise.resolve(_pg);
	if (_pgPromise) return _pgPromise;
	_pgPromise = (async () => {
		const url = await resolveDatabaseUrl();
		if (!url) {
			throw new Error('DATABASE_URL is required (Supabase Postgres pooler URL)');
		}
		// `prepare: false` is mandatory for Supabase's transaction-mode pooler (port 6543).
		// `max: 5` is a compromise — Supabase free pooler aggressively kills idle
		// connections, so a smaller pool keeps reconnect churn down. Production code
		// running on Vercel should bump this back up via env if the plan allows it.
		_pg = postgres(url, {
			prepare: false,
			max: 5,
			idle_timeout: 20,
			connect_timeout: 10
		});
		return _pg;
	})();
	return _pgPromise;
}

let _db: PostgresJsDatabase<typeof schema> | null = null;
export async function getDb(): Promise<PostgresJsDatabase<typeof schema>> {
	if (_db) return _db;
	_db = drizzle(await pgClient(), { schema });
	return _db;
}

/**
 * Minimal libsql-shaped client kept for backward compatibility with the
 * existing `repo/*.ts` modules. New code should prefer the Drizzle `db`
 * export above and write queries against `schema`.
 */
export interface DbClient {
	execute(opts: { sql: string; args: readonly unknown[] }): Promise<DbResult>;
	batch(
		stmts: Array<{ sql: string; args: readonly unknown[] }>,
		mode?: 'write' | 'read'
	): Promise<DbResult[]>;
	close(): Promise<void>;
}

export interface DbResult {
	rows: Record<string, unknown>[];
	rowsAffected: number;
}

// SQLite uses `?` for positional params; Postgres uses `$1, $2, …`. We only
// rewrite literal `?` outside of single-quoted string literals.
function convertPlaceholders(sql: string): string {
	let out = '';
	let i = 0;
	let n = 0;
	while (i < sql.length) {
		const ch = sql[i];
		if (ch === "'") {
			out += ch;
			i++;
			while (i < sql.length) {
				const c = sql[i];
				if (c === "'" && sql[i + 1] === "'") { out += "''"; i += 2; continue; }
				out += c;
				i++;
				if (c === "'") break;
			}
			continue;
		}
		if (ch === '?') {
			n++;
			out += `$${n}`;
		} else {
			out += ch;
		}
		i++;
	}
	return out;
}

async function runOne(
	tx: Sql,
	stmt: { sql: string; args: readonly unknown[] }
): Promise<DbResult> {
	const query = convertPlaceholders(stmt.sql);
	const rows = (await tx.unsafe(
		query,
		stmt.args as Array<string | number | boolean | null>
	)) as unknown as Array<Record<string, unknown>> & { count?: number };
	const arr = Array.from(rows) as Record<string, unknown>[];
	return { rows: arr, rowsAffected: rows.count ?? arr.length };
}

class PostgresClient implements DbClient {
	async execute(opts: { sql: string; args: readonly unknown[] }): Promise<DbResult> {
		const pg = await pgClient();
		return runOne(pg, opts);
	}
	async batch(
		stmts: Array<{ sql: string; args: readonly unknown[] }>,
		_mode?: 'write' | 'read'
	): Promise<DbResult[]> {
		const pg = await pgClient();
		return (await pg.begin(async (tx) => {
			const out: DbResult[] = [];
			for (const s of stmts) out.push(await runOne(tx as unknown as Sql, s));
			return out;
		})) as DbResult[];
	}
	async close(): Promise<void> {
		if (_pg) await _pg.end({ timeout: 5 });
	}
}

const client = new PostgresClient();

export async function getReadyClient(): Promise<DbClient> {
	return client;
}

// Schema is now managed by drizzle-kit (`bunx drizzle-kit push`). This stub
// exists for tests and older callers that imported applySchema directly.
export async function applySchema(_c?: DbClient): Promise<void> {
	/* no-op: schema lives in drizzle/schema.ts */
}
