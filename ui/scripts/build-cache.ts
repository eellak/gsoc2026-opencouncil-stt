#!/usr/bin/env bun
/**
 * Build the grouped cache from the v2 corrections CSV.
 *
 * Usage:
 *   bun scripts/build-cache.ts [csv-path] [out-dir]
 *
 * Defaults:
 *   csv-path = ../data-1779206108158.csv (repo-root copy)
 *   out-dir  = ui/.cache
 *
 * The output is two files written atomically:
 *   groups.v1.json — sorted array of Group objects
 *   meta.json      — source fingerprint + counts
 *
 * If the CSV's fingerprint matches the existing meta.json, regeneration is
 * skipped (use REBUILD=1 to force).
 */

import { parse } from 'csv-parse';
import { createReadStream, promises as fs } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildGroups, type V2CsvRow } from '../src/lib/server/cache/build';
import { buildSqlite } from '../src/lib/server/cache/build-sqlite';
import { fingerprintFile } from '../src/lib/server/cache/fingerprint';
import { CACHE_VERSION, type CacheMeta } from '../src/lib/domain/groups';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..');

// Positional args: [csv-path] [out-dir]. Optional --format flag picks output
// kind; default is the legacy JSON cache so existing dev flows keep working.
// Deployment targets explicitly opt into `sqlite` (or `both` for parity tests).
function parseFormat(argv: string[]): 'json' | 'sqlite' | 'both' {
	const ix = argv.findIndex((a) => a === '--format');
	if (ix === -1) return 'json';
	const v = argv[ix + 1];
	if (v !== 'json' && v !== 'sqlite' && v !== 'both') {
		throw new Error(`--format must be one of json|sqlite|both, got ${v}`);
	}
	return v;
}

const positional = process.argv.slice(2).filter((a, i, arr) => {
	if (a.startsWith('--')) return false;
	if (i > 0 && arr[i - 1] === '--format') return false;
	return true;
});

const format = parseFormat(process.argv);
const csvPath = resolve(positional[0] ?? resolve(REPO_ROOT, 'data-1779206108158.csv'));
const outDir = resolve(positional[1] ?? resolve(HERE, '..', '.cache'));
const metaPath = resolve(outDir, 'meta.json');
const dataPath = resolve(outDir, 'groups.v1.json');
const sqlitePath = resolve(outDir, 'groups.v1.sqlite');
const force = process.env.REBUILD === '1';

console.log(`Source CSV: ${csvPath}`);
console.log(`Cache dir : ${outDir}`);

await fs.mkdir(outDir, { recursive: true });

const fp = await fingerprintFile(csvPath);

if (!force) {
	try {
		const existing = JSON.parse(await fs.readFile(metaPath, 'utf8')) as CacheMeta;
		if (
			existing.cache_version === CACHE_VERSION &&
			existing.source_hash === fp.hash &&
			existing.source_size === fp.size
		) {
			console.log(`Cache is up to date (hash=${fp.hash}). Use REBUILD=1 to force.`);
			process.exit(0);
		}
	} catch {
		/* no meta yet — fall through to build */
	}
}

console.log('Parsing CSV …');
const rows: Array<V2CsvRow & { csv_row: number }> = [];

const parser = parse({ columns: true, skip_empty_lines: true, relax_column_count: true });
let i = 0;
await new Promise<void>((res, rej) => {
	parser.on('readable', () => {
		let row: V2CsvRow | null;
		while ((row = parser.read() as V2CsvRow | null) !== null) {
			rows.push({ ...row, csv_row: i++ });
		}
	});
	parser.on('end', res);
	parser.on('error', rej);
	createReadStream(csvPath).pipe(parser);
});

console.log(`Parsed ${rows.length.toLocaleString()} rows. Grouping …`);
const { groups, missingUtteranceIds, invalidTimestamps, editCount } = buildGroups(rows);
console.log(
	`Built ${groups.length.toLocaleString()} groups from ${editCount.toLocaleString()} edits ` +
		`(skipped ${missingUtteranceIds} missing-utterance-id, ${invalidTimestamps} invalid-timestamps).`
);

const meta: CacheMeta = {
	cache_version: CACHE_VERSION,
	source_csv_path: csvPath,
	source_size: fp.size,
	source_mtime_ms: fp.mtime_ms,
	source_hash: fp.hash,
	generated_at: new Date().toISOString(),
	group_count: groups.length,
	edit_count: editCount,
	missing_utterance_id_count: missingUtteranceIds
};

// meta.json is small and cheap — write it regardless of --format so existing
// tooling that reads it (e.g. fingerprint comparison above) keeps working.
await fs.writeFile(`${metaPath}.tmp`, JSON.stringify(meta, null, 2));
await fs.rename(`${metaPath}.tmp`, metaPath);
console.log(`Wrote ${metaPath}.`);

if (format === 'json' || format === 'both') {
	// Atomic write: tmp then rename. Survives crash mid-write.
	await fs.writeFile(`${dataPath}.tmp`, JSON.stringify(groups));
	await fs.rename(`${dataPath}.tmp`, dataPath);
	console.log(`Wrote ${dataPath}.`);
}

if (format === 'sqlite' || format === 'both') {
	console.log(`Building SQLite cache at ${sqlitePath} …`);
	await buildSqlite({ groups, meta, outPath: sqlitePath });
	console.log(`Wrote ${sqlitePath}.`);
}
