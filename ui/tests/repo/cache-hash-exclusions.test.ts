import { describe, it, expect, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { SqliteRepo } from '../../src/lib/server/repo/sqlite-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import { buildSqlite } from '../../src/lib/server/cache/build-sqlite';
import type { CacheMeta } from '../../src/lib/domain/groups';

const roots: string[] = [];
afterEach(async () => {
	await Promise.all(roots.splice(0).map((r) => rm(r, { recursive: true, force: true })));
});

async function buildFixture(exclusionsHash: string | null): Promise<{ cacheDir: string; stateDir: string }> {
	const root = await mkdtemp(join(tmpdir(), 'cache-hash-'));
	roots.push(root);
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	const rows: Array<V2CsvRow & { csv_row: number }> = Array.from({ length: 4 }, (_, i) => ({
		edit_id: `e${i}`,
		utterance_id: `u${i}`,
		edit_timestamp: `2026-05-01T00:00:0${i}Z`,
		edit_updated_at: '',
		before_text: `b${i}`,
		after_text: `b${i} x`,
		edited_by: 'user',
		utterance_start: '0',
		utterance_end: '1',
		audio_url: '',
		youtube_url: '',
		meeting_name: 'm',
		meeting_date: '2026-05-01',
		meeting_id: 'm1',
		city_id: 'athens',
		csv_row: i
	}));
	const { groups } = buildGroups(rows);
	await fs.writeFile(join(cacheDir, 'groups.v1.json'), JSON.stringify(groups));

	const meta: CacheMeta = {
		cache_version: 1,
		source_csv_path: 'synthetic.csv',
		source_size: 0,
		source_mtime_ms: 0,
		source_hash: 'SRC',
		generated_at: new Date().toISOString(),
		group_count: groups.length,
		edit_count: groups.reduce((a, g) => a + g.edits.length, 0),
		missing_utterance_id_count: 0,
		...(exclusionsHash
			? {
					exclusions: {
						exclusions_hash: exclusionsHash,
						availability_report_file: null,
						availability_generated_at: null,
						private_meeting_keys: [],
						drop_categories: [],
						excluded_private_utterances: 0,
						excluded_degenerate_utterances: 0,
						excluded_both: 0,
						excluded_total: 0
					}
				}
			: {})
	};
	await fs.writeFile(join(cacheDir, 'meta.json'), JSON.stringify(meta));
	await buildSqlite({ groups, meta, outPath: join(cacheDir, 'groups.v1.sqlite') });
	return { cacheDir, stateDir };
}

describe('repo.hash reflects exclusion provenance', () => {
	it('SqliteRepo: bare source hash when no exclusions', async () => {
		const { cacheDir, stateDir } = await buildFixture(null);
		const repo = await SqliteRepo.load({ cacheDir, stateDir, meetingMinHumanUtterances: 0 });
		expect(repo.hash).toBe('SRC');
	});

	it('SqliteRepo + FileRepo: fold exclusions_hash into cache_hash, identically', async () => {
		const { cacheDir, stateDir } = await buildFixture('EXCL1');
		const sql = await SqliteRepo.load({ cacheDir, stateDir, meetingMinHumanUtterances: 0 });
		const file = await FileRepo.load({ cacheDir, stateDir, meetingMinHumanUtterances: 0 });
		expect(sql.hash).toBe('SRC+xEXCL1');
		expect(file.hash).toBe('SRC+xEXCL1'); // both repos agree → snapshots are portable
	});

	it('same CSV, different exclusions ⇒ different cache_hash (forces snapshot recompute)', async () => {
		const a = await buildFixture('EXCL1');
		const b = await buildFixture('EXCL2');
		const ra = await SqliteRepo.load({ cacheDir: a.cacheDir, stateDir: a.stateDir, meetingMinHumanUtterances: 0 });
		const rb = await SqliteRepo.load({ cacheDir: b.cacheDir, stateDir: b.stateDir, meetingMinHumanUtterances: 0 });
		expect(ra.hash).not.toBe(rb.hash);
	});
});
