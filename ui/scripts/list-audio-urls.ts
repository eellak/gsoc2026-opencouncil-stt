#!/usr/bin/env bun
/**
 * Prints one distinct correction audio_url per line.
 *
 * DB resolution order:
 *   1. DATABASE_URL
 *   2. TURSO_DATABASE_URL
 *   3. file:./data/corrections.sqlite
 */

import { createClient, type Client } from '@libsql/client';
import { pathToFileURL } from 'url';

export async function listAudioUrls(client: Client): Promise<string[]> {
	const result = await client.execute({
		sql: `SELECT DISTINCT audio_url
		      FROM corrections
		      WHERE audio_url IS NOT NULL AND audio_url != ''
		      ORDER BY audio_url`,
		args: []
	});
	return result.rows.map((row) => String(row.audio_url));
}

function getDbConfig(): { url: string; authToken?: string } {
	const url = process.env.DATABASE_URL || process.env.TURSO_DATABASE_URL || 'file:./data/corrections.sqlite';
	const authToken = process.env.TURSO_AUTH_TOKEN;
	return authToken ? { url, authToken } : { url };
}

async function main(): Promise<void> {
	const client = createClient(getDbConfig());
	try {
		for (const url of await listAudioUrls(client)) {
			console.log(url);
		}
	} finally {
		await client.close();
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	main().catch((err) => {
		console.error(err);
		process.exit(1);
	});
}
