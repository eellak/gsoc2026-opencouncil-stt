#!/usr/bin/env tsx
/**
 * Build the grouped cache from the v2 corrections CSV.
 *
 * Usage:
 *   tsx scripts/build-cache.ts [csv-path] [out-dir] --format sqlite
 *
 * NOTE: run with `tsx`/node, not `bun` — the sqlite build uses better-sqlite3,
 * which bun cannot load. The json-only build works under either.
 *
 * Defaults:
 *   csv-path = ../data-1779206108158.csv (repo-root copy)
 *   out-dir  = ui/.cache
 *
 * Hard exclusions (filtered rebuild): groups are physically dropped from the
 * index when their meeting is PRIVATE (from the latest meeting-availability
 * probe report) or their latest edit is a DEGENERATE ingest bin
 * (DROP_INGEST_CATEGORIES, default noop_edit,empty_after,whitespace_only). The
 * CSV stays the source of truth; getGroup() for a dropped utterance returns
 * null. Exclusion inputs are digested into meta.json (exclusions_hash) and
 * gate the rebuild-skip check, so changing them forces a rebuild even when the
 * CSV is unchanged. Set DROP_INGEST_CATEGORIES="" and NO_PRIVATE_EXCLUSION=1 to
 * build the full corpus.
 *
 * Outputs (written atomically, tmp then rename):
 *   groups.v1.json   — sorted array of Group objects (FileRepo)
 *   groups.v1.sqlite — same dataset (SqliteRepo)
 *   meta.json        — source fingerprint + counts + exclusion provenance
 */

import { parse } from 'csv-parse';
import { createReadStream, promises as fs } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { buildGroups, type V2CsvRow } from '../src/lib/server/cache/build';
import { buildSqlite } from '../src/lib/server/cache/build-sqlite';
import { fingerprintFile } from '../src/lib/server/cache/fingerprint';
import { degenerateCategories } from '../src/lib/server/state/ingest-filter';
import { SidecarStore } from '../src/lib/server/state/sidecar';
import { CACHE_VERSION, type CacheMeta } from '../src/lib/domain/groups';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..');
const UI_ROOT = resolve(HERE, '..');

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
const outDir = resolve(positional[1] ?? resolve(UI_ROOT, '.cache'));
const stateDir = process.env.REVIEW_STATE_DIR ?? resolve(UI_ROOT, '.state');
const reportsDir = resolve(REPO_ROOT, 'data', 'reports');
const metaPath = resolve(outDir, 'meta.json');
const dataPath = resolve(outDir, 'groups.v1.json');
const sqlitePath = resolve(outDir, 'groups.v1.sqlite');
const force = process.env.REBUILD === '1';

console.log(`Source CSV: ${csvPath}`);
console.log(`Cache dir : ${outDir}`);

await fs.mkdir(outDir, { recursive: true });

// ── Exclusion inputs ──────────────────────────────────────────────────────
// Private meetings from the newest availability probe (unless disabled), plus
// the degenerate ingest categories. Both digested into exclusions_hash.
interface AvailabilityReport {
	generated_at?: string;
	error?: number;
	private_meeting_keys?: string[];
}

async function loadAvailabilityReport(): Promise<{ file: string; report: AvailabilityReport } | null> {
	if (process.env.NO_PRIVATE_EXCLUSION === '1') return null;
	const explicit = process.env.AVAILABILITY_REPORT;
	if (explicit) {
		const report = JSON.parse(await fs.readFile(explicit, 'utf8')) as AvailabilityReport;
		return { file: explicit, report };
	}
	let entries: string[];
	try {
		entries = (await fs.readdir(reportsDir))
			.filter((f) => /^meeting-availability-.*\.json$/.test(f))
			.sort();
	} catch {
		return null;
	}
	if (entries.length === 0) return null;
	const file = resolve(reportsDir, entries[entries.length - 1]); // newest by name (dated)
	const report = JSON.parse(await fs.readFile(file, 'utf8')) as AvailabilityReport;
	return { file, report };
}

const dropCategories = degenerateCategories();
const avail = await loadAvailabilityReport();
if (avail) {
	const errCount = avail.report.error ?? 0;
	if (errCount > 0 && process.env.ALLOW_REPORT_ERRORS !== '1') {
		throw new Error(
			`Availability report ${avail.file} has ${errCount} errored meetings — ` +
				`re-run the probe, or set ALLOW_REPORT_ERRORS=1 to build anyway (errored meetings are kept).`
		);
	}
	if (!Array.isArray(avail.report.private_meeting_keys)) {
		throw new Error(`Availability report ${avail.file} is missing private_meeting_keys[].`);
	}
}
const privateKeys = new Set<string>(avail?.report.private_meeting_keys ?? []);

const exclusionsHash = createHash('sha256')
	.update(
		JSON.stringify({
			private: [...privateKeys].sort(),
			drop: [...dropCategories].sort()
		})
	)
	.digest('hex')
	.slice(0, 16);

console.log(
	`Exclusions: private_meetings=${privateKeys.size} ` +
		`(report=${avail ? avail.file.replace(REPO_ROOT + '/', '') : 'none'}), ` +
		`drop_categories=[${[...dropCategories].join(',') || 'none'}], hash=${exclusionsHash}`
);

const fp = await fingerprintFile(csvPath);

if (!force) {
	try {
		const existing = JSON.parse(await fs.readFile(metaPath, 'utf8')) as CacheMeta;
		const exclMatch = (existing.exclusions?.exclusions_hash ?? null) === exclusionsHash;
		if (
			existing.cache_version === CACHE_VERSION &&
			existing.source_hash === fp.hash &&
			existing.source_size === fp.size &&
			exclMatch
		) {
			console.log(`Cache is up to date (hash=${fp.hash}, exclusions=${exclusionsHash}). Use REBUILD=1 to force.`);
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
const { groups, missingUtteranceIds, invalidTimestamps, editCount, excluded } = buildGroups(rows, {
	excludeMeetingKeys: privateKeys,
	dropCategories
});
console.log(
	`Built ${groups.length.toLocaleString()} groups from ${editCount.toLocaleString()} edits ` +
		`(skipped ${missingUtteranceIds} missing-utterance-id, ${invalidTimestamps} invalid-timestamps).`
);
console.log(
	`Excluded ${excluded.total.toLocaleString()} utterances ` +
		`(private=${excluded.private.toLocaleString()}, degenerate=${excluded.degenerate.toLocaleString()}, ` +
		`both=${excluded.both.toLocaleString()}).`
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
	missing_utterance_id_count: missingUtteranceIds,
	exclusions: {
		exclusions_hash: exclusionsHash,
		availability_report_file: avail ? avail.file.replace(REPO_ROOT + '/', '') : null,
		availability_generated_at: avail?.report.generated_at ?? null,
		private_meeting_keys: [...privateKeys].sort(),
		drop_categories: [...dropCategories].sort(),
		excluded_private_utterances: excluded.private,
		excluded_degenerate_utterances: excluded.degenerate,
		excluded_both: excluded.both,
		excluded_total: excluded.total
	}
};

// ── Orphan-label report: which dropped utterances had human review work ─────
let orphanLabels: Array<{ utterance_id: string; reasons: string[]; include_status: string }> = [];
try {
	const sidecar = await SidecarStore.load(stateDir);
	const labels = sidecar.all();
	for (const d of excluded.dropped) {
		const lbl = labels.get(d.utterance_id);
		if (lbl && lbl.include_status && lbl.include_status !== 'unreviewed') {
			orphanLabels.push({ utterance_id: d.utterance_id, reasons: d.reasons, include_status: lbl.include_status });
		}
	}
} catch (err) {
	console.warn('[build-cache] could not load sidecar for orphan-label report:', err);
}

const stamp = new Date().toISOString().slice(0, 10);
await fs.mkdir(reportsDir, { recursive: true });
const rebuildReportPath = resolve(reportsDir, `index-rebuild-${stamp}.json`);
await fs.writeFile(
	rebuildReportPath,
	JSON.stringify(
		{
			generated_at: new Date().toISOString(),
			source_csv: csvPath,
			source_hash: fp.hash,
			exclusions_hash: exclusionsHash,
			availability_report: meta.exclusions!.availability_report_file,
			utterances_kept: groups.length,
			excluded: {
				total: excluded.total,
				private: excluded.private,
				degenerate: excluded.degenerate,
				both: excluded.both
			},
			orphaned_labels_count: orphanLabels.length,
			orphaned_labels: orphanLabels
		},
		null,
		2
	)
);
console.log(
	`Orphaned human labels among dropped utterances: ${orphanLabels.length} ` +
		`(preserved in .state, backed up). Report: ${rebuildReportPath.replace(REPO_ROOT + '/', '')}`
);

// ── Backup the current index before overwriting (rollback safety) ───────────
async function backupIfExists(path: string): Promise<string | null> {
	try {
		await fs.access(path);
	} catch {
		return null;
	}
	const bak = `${path}.pre-rebuild-${stamp}`;
	await fs.copyFile(path, bak);
	console.log(`Backed up ${path.replace(REPO_ROOT + '/', '')} → ${bak.replace(REPO_ROOT + '/', '')}`);
	return bak;
}

// meta.json is small and cheap — write last (it's the fingerprint other tooling
// reads), AFTER the data files validate, so a crash mid-build can't leave meta
// claiming a build that didn't complete.
if (format === 'sqlite' || format === 'both') {
	await backupIfExists(sqlitePath);
}

if (format === 'json' || format === 'both') {
	await fs.writeFile(`${dataPath}.tmp`, JSON.stringify(groups));
	await fs.rename(`${dataPath}.tmp`, dataPath);
	console.log(`Wrote ${dataPath}.`);
}

if (format === 'sqlite' || format === 'both') {
	console.log(`Building SQLite cache at ${sqlitePath} …`);
	await buildSqlite({ groups, meta, outPath: sqlitePath });
	// Post-build validation: row count must match the in-memory dataset.
	const Database = (await import('better-sqlite3')).default;
	const db = new Database(sqlitePath, { readonly: true, fileMustExist: true });
	const n = (db.prepare('SELECT COUNT(*) AS n FROM groups').get() as { n: number }).n;
	db.close();
	if (n !== groups.length) {
		throw new Error(`Post-build validation FAILED: sqlite has ${n} groups, expected ${groups.length}.`);
	}
	console.log(`Validated SQLite: ${n.toLocaleString()} groups.`);
}

await fs.writeFile(`${metaPath}.tmp`, JSON.stringify(meta, null, 2));
await fs.rename(`${metaPath}.tmp`, metaPath);
console.log(`Wrote ${metaPath}.`);
