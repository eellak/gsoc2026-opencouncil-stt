/**
 * Materialise grouped corrections as a SQLite database.
 *
 * Runtime trade-off: the JSON cache is ~432 MB and forces every request path
 * to live with a fully-loaded in-memory map (>1 GB heap on real data). The
 * SQLite layout lets the server query a single row per request and keeps the
 * resident set under ~80 MB, which matters on the 1 GB Oracle VM.
 *
 * Schema (kept narrow on purpose — most query patterns hit `utterance_id`
 * directly; `ord` preserves the deterministic insertion order from
 * `buildGroups()` so seeded shuffles are stable across rebuilds):
 *
 *   groups(utterance_id PK, meeting_id, city_id, ord, json TEXT)
 *   edits(edit_id PK, utterance_id)
 *   meta(key PK, value)
 *
 * Notes:
 * - The serialised `json` strips the `label` field — labels live in the
 *   sidecar JSONL. Tests enforce this; if a label leaks in it can silently
 *   override the sidecar's source-of-truth.
 * - WAL is enabled while we write (faster bulk insert) and explicitly
 *   checkpointed + closed before the atomic rename. Runtime opens the DB
 *   read-only, so we never want WAL/SHM sidecar files following the rename.
 */

import Database from 'better-sqlite3';
import { promises as fs } from 'node:fs';
import type { CacheMeta, Group } from '$lib/domain/groups';

export interface BuildSqliteInput {
	groups: Group[];
	meta: CacheMeta;
	/** Final on-disk path (the file we'll `fs.rename` into at the end). */
	outPath: string;
}

export async function buildSqlite(input: BuildSqliteInput): Promise<void> {
	const { groups, meta, outPath } = input;
	const tmpPath = `${outPath}.tmp`;

	// Clean any leftover tmp from a previous failed build before opening.
	await fs.rm(tmpPath, { force: true });

	const db = new Database(tmpPath);
	try {
		db.pragma('journal_mode = WAL');
		db.pragma('synchronous = NORMAL');

		db.exec(`
			CREATE TABLE groups (
				utterance_id TEXT PRIMARY KEY,
				meeting_id   TEXT NOT NULL,
				city_id      TEXT NOT NULL,
				ord          INTEGER NOT NULL,
				json         TEXT NOT NULL
			);
			CREATE INDEX idx_groups_ord     ON groups(ord);
			CREATE INDEX idx_groups_meeting ON groups(meeting_id);

			CREATE TABLE edits (
				edit_id      TEXT PRIMARY KEY,
				utterance_id TEXT NOT NULL REFERENCES groups(utterance_id)
			);
			CREATE INDEX idx_edits_utterance ON edits(utterance_id);

			CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
		`);

		const insertGroup = db.prepare(
			'INSERT INTO groups (utterance_id, meeting_id, city_id, ord, json) VALUES (?, ?, ?, ?, ?)'
		);
		const insertEdit = db.prepare('INSERT INTO edits (edit_id, utterance_id) VALUES (?, ?)');
		const insertMeta = db.prepare('INSERT INTO meta (key, value) VALUES (?, ?)');

		const seenEditIds = new Set<string>();
		const tx = db.transaction((items: Group[]) => {
			let ord = 0;
			for (const g of items) {
				// Strip the label before serialisation so the sidecar remains
				// the only source of truth. Spreading like this avoids mutating
				// the caller's group object.
				const { label: _label, ...rest } = g;
				void _label; // discard
				const json = JSON.stringify(rest);
				insertGroup.run(g.utterance_id, g.meeting_id ?? '', g.city_id ?? '', ord, json);
				for (const e of g.edits) {
					if (seenEditIds.has(e.edit_id)) {
						throw new Error(
							`duplicate edit_id ${JSON.stringify(e.edit_id)} ` +
								`(utterance_id=${g.utterance_id}, csv_row=${e.csv_row})`
						);
					}
					seenEditIds.add(e.edit_id);
					insertEdit.run(e.edit_id, g.utterance_id);
				}
				ord++;
			}
		});
		tx(groups);

		const metaRows: Array<[string, string]> = [
			['cache_version', String(meta.cache_version)],
			['source_csv_path', meta.source_csv_path],
			['source_size', String(meta.source_size)],
			['source_mtime_ms', String(meta.source_mtime_ms)],
			['source_hash', meta.source_hash],
			['generated_at', meta.generated_at],
			['group_count', String(meta.group_count)],
			['edit_count', String(meta.edit_count)],
			['missing_utterance_id_count', String(meta.missing_utterance_id_count)]
		];
		// Filtered rebuild: persist the exclusion digest so the runtime cache_hash
		// reflects the index CONTENT (not just the source CSV), forcing dependent
		// snapshots (stats/category/eligibility) to invalidate when exclusions change.
		if (meta.exclusions) {
			metaRows.push(['exclusions_hash', meta.exclusions.exclusions_hash]);
		}
		const metaTx = db.transaction((rows: Array<[string, string]>) => {
			for (const [k, v] of rows) insertMeta.run(k, v);
		});
		metaTx(metaRows);

		// Force a full checkpoint so WAL/SHM are merged before close.
		// Runtime opens read-only and any leftover -wal/-shm files would be
		// invalid relative to the renamed db file.
		db.pragma('wal_checkpoint(TRUNCATE)');
	} catch (err) {
		db.close();
		await fs.rm(tmpPath, { force: true });
		await fs.rm(`${tmpPath}-journal`, { force: true });
		await fs.rm(`${tmpPath}-wal`, { force: true });
		await fs.rm(`${tmpPath}-shm`, { force: true });
		throw err;
	}
	db.close();

	// Final sanity: any sidecar files at this point are stale (we truncated
	// the WAL above and closed), but if they exist remove them so the rename
	// produces a clean single-file DB.
	await fs.rm(`${tmpPath}-wal`, { force: true });
	await fs.rm(`${tmpPath}-shm`, { force: true });

	await fs.rename(tmpPath, outPath);
}
