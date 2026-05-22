import { PGlite } from '@electric-sql/pglite';
import type { DbClient, DbResult } from '$lib/server/db';

// Postgres DDL mirroring drizzle/schema.ts. Kept here (not imported from the
// production schema) so test isolation is explicit and the test surface
// doesn't depend on Drizzle migration plumbing.
const SCHEMA_SQL = `
	CREATE TABLE IF NOT EXISTS meetings (
		meeting_id   TEXT PRIMARY KEY,
		meeting_name TEXT,
		meeting_date TEXT,
		city_id      TEXT,
		audio_url    TEXT,
		audio_cdn_url TEXT,
		youtube_url  TEXT
	);

	CREATE TABLE IF NOT EXISTS corrections (
		edit_id              TEXT PRIMARY KEY,
		utterance_id         TEXT,
		meeting_id           TEXT REFERENCES meetings(meeting_id),
		latest_per_utterance BOOLEAN NOT NULL DEFAULT true,
		edit_timestamp       TEXT NOT NULL,
		edit_updated_at      TEXT,
		before_text          TEXT NOT NULL,
		after_text           TEXT NOT NULL,
		edited_by            TEXT,
		utterance_start      DOUBLE PRECISION NOT NULL,
		utterance_end        DOUBLE PRECISION NOT NULL,
		ingest_category      TEXT,
		cleaning_applied     TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_corrections_ts ON corrections(edit_timestamp);
	CREATE INDEX IF NOT EXISTS idx_corrections_meeting ON corrections(meeting_id);
	CREATE INDEX IF NOT EXISTS idx_corrections_editor ON corrections(edited_by);
	CREATE INDEX IF NOT EXISTS idx_corrections_ingest_category ON corrections(ingest_category);

	CREATE TABLE IF NOT EXISTS review_labels (
		edit_id          TEXT PRIMARY KEY REFERENCES corrections(edit_id),
		error_category   TEXT,
		include_status   TEXT NOT NULL DEFAULT 'unreviewed',
		adjusted_start   DOUBLE PRECISION,
		adjusted_end     DOUBLE PRECISION,
		reviewer_notes   TEXT,
		human_updated_at TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_labels_status ON review_labels(include_status);
	CREATE INDEX IF NOT EXISTS idx_labels_category ON review_labels(error_category);

	CREATE TABLE IF NOT EXISTS events (
		id BIGSERIAL PRIMARY KEY,
		ts TEXT NOT NULL,
		edit_id TEXT NOT NULL,
		field TEXT NOT NULL,
		old_val TEXT,
		new_val TEXT,
		actor TEXT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_events_edit_id ON events(edit_id);

	CREATE TABLE IF NOT EXISTS category_descriptions (
		category TEXT PRIMARY KEY,
		label_el TEXT NOT NULL,
		reason_el TEXT NOT NULL,
		is_rejected INTEGER NOT NULL DEFAULT 0
	);
`;

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

export interface TestClient extends DbClient {
	raw: PGlite;
}

export async function makeTestClient(): Promise<TestClient> {
	const pg = new PGlite();
	await pg.exec(SCHEMA_SQL);

	async function runOne(stmt: { sql: string; args: readonly unknown[] }): Promise<DbResult> {
		const query = convertPlaceholders(stmt.sql);
		const result = await pg.query(query, stmt.args as unknown[]);
		return {
			rows: result.rows as Record<string, unknown>[],
			rowsAffected: result.affectedRows ?? result.rows.length
		};
	}

	return {
		raw: pg,
		async execute(opts) {
			return runOne(opts);
		},
		async batch(stmts) {
			await pg.exec('BEGIN');
			try {
				const out: DbResult[] = [];
				for (const s of stmts) out.push(await runOne(s));
				await pg.exec('COMMIT');
				return out;
			} catch (e) {
				await pg.exec('ROLLBACK');
				throw e;
			}
		},
		async close() {
			await pg.close();
		}
	};
}
