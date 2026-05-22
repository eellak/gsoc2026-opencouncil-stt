#!/usr/bin/env bun
/**
 * Applies data/audio-fix/url-map.json to corrections.audio_cdn_url without modifying audio_url.
 *
 * Usage:
 *   bun scripts/apply-audio-cdn-map.ts
 *   DATABASE_URL=file:/path/to/corrections.sqlite bun scripts/apply-audio-cdn-map.ts
 *   TURSO_DATABASE_URL=libsql://... TURSO_AUTH_TOKEN=... bun scripts/apply-audio-cdn-map.ts
 */

import { createClient } from '@libsql/client';
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

// Anchor default mapPath to the script location, not process.cwd().
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..');

const dbUrl = process.env.DATABASE_URL || process.env.TURSO_DATABASE_URL || 'file:./data/corrections.sqlite';
const authToken = process.env.TURSO_AUTH_TOKEN;
const mapPath = resolve(process.argv[2] ?? resolve(REPO_ROOT, 'data', 'audio-fix', 'url-map.json'));

const client = createClient(authToken ? { url: dbUrl, authToken } : { url: dbUrl });

function isDuplicateColumnError(err: unknown): boolean {
	const msg = err instanceof Error ? err.message : String(err);
	// libsql/sqlite: "duplicate column name: audio_cdn_url"
	return /duplicate column name/i.test(msg);
}

try {
	try {
		await client.execute({ sql: `ALTER TABLE corrections ADD COLUMN audio_cdn_url TEXT`, args: [] });
	} catch (err) {
		if (!isDuplicateColumnError(err)) throw err;
		// Column already present — nothing to do.
	}

	const raw = await readFile(mapPath, 'utf8');
	const urlMap = JSON.parse(raw) as Record<string, string>;
	const entries = Object.entries(urlMap);

	let matchedRows = 0;
	const batchSize = 100;

	for (let i = 0; i < entries.length; i += batchSize) {
		const chunk = entries.slice(i, i + batchSize);
		const result = await client.batch(
			chunk.map(([originalUrl, cdnUrl]) => ({
				sql: `UPDATE corrections SET audio_cdn_url = ? WHERE audio_url = ?`,
				args: [cdnUrl, originalUrl]
			})),
			'write'
		);
		matchedRows += result.reduce((sum, r) => sum + (r.rowsAffected ?? 0), 0);
	}

	console.log(`Applied ${entries.length} CDN URL mappings from ${mapPath}`);
	console.log(`Updated ${matchedRows} correction rows in ${dbUrl}`);
} catch (err) {
	console.error('[apply-audio-cdn-map] failed:', err instanceof Error ? err.stack ?? err.message : err);
	process.exitCode = 1;
} finally {
	try {
		await client.close();
	} catch (closeErr) {
		console.error('[apply-audio-cdn-map] client.close error:', closeErr);
	}
}
