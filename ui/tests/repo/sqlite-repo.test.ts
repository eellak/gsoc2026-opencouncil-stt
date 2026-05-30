import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { SqliteRepo } from '../../src/lib/server/repo/sqlite-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import { buildSqlite } from '../../src/lib/server/cache/build-sqlite';
import type { CacheMeta, Group } from '../../src/lib/domain/groups';

interface Fixture {
	cacheDir: string;
	stateDir: string;
	groups: Group[];
	meta: CacheMeta;
}

async function setupFixture(n: number): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'sqlite-repo-'));
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	const rows: Array<V2CsvRow & { csv_row: number }> = [];
	for (let i = 0; i < n; i++) {
		rows.push({
			edit_id: `e${i.toString().padStart(4, '0')}`,
			utterance_id: `u${i.toString().padStart(4, '0')}`,
			edit_timestamp: `2026-05-01T00:00:${(i % 60).toString().padStart(2, '0')}Z`,
			edit_updated_at: '',
			before_text: `before-${i}`,
			after_text: `after-${i}`,
			edited_by: 'tester',
			utterance_start: '0',
			utterance_end: '1',
			audio_url: 'https://example.com/a.mp3',
			youtube_url: '',
			meeting_name: 'm',
			meeting_date: '2026-05-01',
			meeting_id: i < n / 2 ? 'm1' : 'm2',
			city_id: 'athens',
			csv_row: i
		});
	}
	const { groups } = buildGroups(rows);

	// Write BOTH JSON (so FileRepo can load same fixture for parity checks)
	// AND SQLite (for SqliteRepo).
	await fs.writeFile(join(cacheDir, 'groups.v1.json'), JSON.stringify(groups));

	const meta: CacheMeta = {
		cache_version: 1,
		source_csv_path: 'synthetic.csv',
		source_size: 0,
		source_mtime_ms: 0,
		source_hash: 'test-hash',
		generated_at: new Date().toISOString(),
		group_count: groups.length,
		edit_count: groups.reduce((acc, g) => acc + g.edits.length, 0),
		missing_utterance_id_count: 0
	};
	await fs.writeFile(join(cacheDir, 'meta.json'), JSON.stringify(meta));

	await buildSqlite({
		groups,
		meta,
		outPath: join(cacheDir, 'groups.v1.sqlite')
	});

	return { cacheDir, stateDir, groups, meta };
}

let fx: Fixture;
beforeEach(async () => {
	fx = await setupFixture(20);
});
afterEach(async () => {
	await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

function openRepo(): Promise<SqliteRepo> {
	return SqliteRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
}

describe('SqliteRepo — basic interface parity with FileRepo', () => {
	it('loads cache and exposes the source hash and total', async () => {
		const repo = await openRepo();
		expect(repo.hash).toBe('test-hash');
		expect(repo.total).toBe(20);
	});

	it('getGroup returns the same group as FileRepo for the same fixture', async () => {
		const file = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const sql = await openRepo();
		const idA = file.getGroup('u0005');
		const idB = sql.getGroup('u0005');
		expect(idB).toEqual(idA);
	});

	it('getGroup returns null for unknown ids', async () => {
		const repo = await openRepo();
		expect(repo.getGroup('does-not-exist')).toBeNull();
	});
});

describe('SqliteRepo — queue ordering', () => {
	it('preserves shuffled order even after batched WHERE IN SELECT', async () => {
		// This is THE test codex flagged: SQLite IN(...) returns rows in storage
		// (primary-key) order, not in the order requested. SqliteRepo must
		// reorder results in JS to match the seeded shuffle.
		const sql = await openRepo();
		const file = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });

		const expected = file.queue(42, 0, 20).groups.map((g) => g.utterance_id);
		const got = sql.queue(42, 0, 20).groups.map((g) => g.utterance_id);
		expect(got).toEqual(expected);
	});

	it('seeded queue is reproducible for the same seed', async () => {
		const repo = await openRepo();
		const a = repo.queue(42, 0, 20).groups.map((g) => g.utterance_id);
		const b = repo.queue(42, 0, 20).groups.map((g) => g.utterance_id);
		expect(b).toEqual(a);
	});

	it('different seeds produce different orders (same set)', async () => {
		const repo = await openRepo();
		const a = repo.queue(1, 0, 20).groups.map((g) => g.utterance_id);
		const b = repo.queue(2, 0, 20).groups.map((g) => g.utterance_id);
		expect(b).not.toEqual(a);
		expect(b.slice().sort()).toEqual(a.slice().sort());
	});

	it('queue clamps negative `from` to 0 and caps `n` at 50', async () => {
		const repo = await openRepo();
		const p = repo.queue(7, -10, 999);
		expect(p.groups.length).toBeLessThanOrEqual(50);
		// First page → cursor should advance past page or be null if fewer
		expect(p.next_cursor === null || p.next_cursor === 50).toBe(true);
	});

	it('queue returns next_cursor=null past the end', async () => {
		const repo = await openRepo();
		const p = repo.queue(1, 18, 5);
		expect(p.groups).toHaveLength(2);
		expect(p.next_cursor).toBeNull();
	});

	it('does not retain per-seed shuffle caches between calls', async () => {
		// Codex flag: arbitrary ?seed= values must not accumulate memory.
		// We assert via an internal hook: SqliteRepo exposes a debug counter.
		const repo = await openRepo();
		for (let s = 0; s < 100; s++) repo.queue(s, 0, 1);
		expect(repo._debugSeedCacheSize()).toBe(0);
	});
});

describe('SqliteRepo — label overlay', () => {
	it('returns DEFAULT_LABEL when sidecar has no label', async () => {
		const repo = await openRepo();
		const g = repo.getGroup('u0003');
		expect(g?.label.include_status).toBe('unreviewed');
		expect(g?.label.error_categories).toEqual([]);
	});

	it('overlays the sidecar label on top of the sqlite group', async () => {
		const repo = await openRepo();
		await repo.patchLabel('u0007', { include_status: 'include', error_categories: ['homophone'] });
		await repo.flush();

		const g = repo.getGroup('u0007');
		expect(g?.label.include_status).toBe('include');
		expect(g?.label.error_categories).toEqual(['homophone']);
	});

	it('iterGroups also overlays sidecar labels', async () => {
		const repo = await openRepo();
		await repo.patchLabel('u0009', { include_status: 'exclude' });
		await repo.flush();
		const all = [...repo.iterGroups()];
		const target = all.find((g) => g.utterance_id === 'u0009');
		expect(target?.label.include_status).toBe('exclude');
	});

	it('patchLabel on unknown utterance_id returns null', async () => {
		const repo = await openRepo();
		const out = await repo.patchLabel('nope', { include_status: 'include' });
		expect(out).toBeNull();
	});
});

describe('SqliteRepo — stored JSON does not embed labels', () => {
	it('the raw `json` column has no `label` field', async () => {
		// If labels leak into the DB, sidecar overlay can be silently overridden.
		const db = new Database(join(fx.cacheDir, 'groups.v1.sqlite'), { readonly: true });
		const row = db.prepare('SELECT json FROM groups WHERE utterance_id = ?').get('u0001') as
			| { json: string }
			| undefined;
		db.close();
		expect(row).toBeTruthy();
		const parsed = JSON.parse(row!.json);
		expect(parsed.label).toBeUndefined();
		// Sanity: other fields ARE present
		expect(parsed.utterance_id).toBe('u0001');
		expect(parsed.edits).toBeInstanceOf(Array);
	});
});

describe('SqliteRepo — edit_id lookup', () => {
	it('utteranceIdForEdit resolves known edits', async () => {
		const repo = await openRepo();
		expect(repo.utteranceIdForEdit('e0005')).toBe('u0005');
	});

	it('utteranceIdForEdit returns null for unknown edits', async () => {
		const repo = await openRepo();
		expect(repo.utteranceIdForEdit('missing-edit')).toBeNull();
	});
});

describe('SqliteRepo — groupsByErrorCategory', () => {
	it('returns only groups whose sidecar label contains the category', async () => {
		const repo = await openRepo();
		await repo.patchLabel('u0001', { error_categories: ['homophone'] });
		await repo.patchLabel('u0002', { error_categories: ['punctuation'] });
		await repo.patchLabel('u0003', { error_categories: ['homophone', 'punctuation'] });
		await repo.flush();

		const ids = repo.groupsByErrorCategory('homophone').map((g) => g.utterance_id);
		expect(new Set(ids)).toEqual(new Set(['u0001', 'u0003']));
	});

	it('idsByErrorCategory matches groupsByErrorCategory but materialises nothing', async () => {
		const repo = await openRepo();
		await repo.patchLabel('u0001', { error_categories: ['homophone'] });
		await repo.patchLabel('u0003', { error_categories: ['homophone', 'punctuation'] });
		await repo.flush();

		const ids = repo.idsByErrorCategory('homophone');
		expect(new Set(ids)).toEqual(new Set(['u0001', 'u0003']));
		// Same match set as the materialising variant.
		expect(new Set(ids)).toEqual(
			new Set(repo.groupsByErrorCategory('homophone').map((g) => g.utterance_id))
		);
		// Canonical (orderedIds) order: u0001 before u0003.
		expect(ids).toEqual(['u0001', 'u0003']);
	});
});

describe('SqliteRepo — iterGroups parity with FileRepo.allGroups', () => {
	it('iterGroups yields the same set in the same `ord` order', async () => {
		const file = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const sql = await openRepo();
		const fileOrder = file.allGroups().map((g) => g.utterance_id);
		const sqlOrder = [...sql.iterGroups()].map((g) => g.utterance_id);
		expect(sqlOrder).toEqual(fileOrder);
	});
});

describe('buildSqlite — metadata + atomic write', () => {
	it('rejects fixtures with duplicate edit_id loudly', async () => {
		const { groups, meta } = fx;
		const tampered = JSON.parse(JSON.stringify(groups)) as Group[];
		// Duplicate edit_id across two groups should fail loudly, not silently.
		tampered[0].edits[0].edit_id = 'dup-id';
		tampered[1].edits[0].edit_id = 'dup-id';

		const outPath = join(fx.cacheDir, 'dup.sqlite');
		await expect(
			buildSqlite({ groups: tampered, meta, outPath })
		).rejects.toThrow(/duplicate edit_id/i);
	});

	it('writes via .tmp + rename (atomic) and leaves no stray tmp on success', async () => {
		const outPath = join(fx.cacheDir, 'atomic.sqlite');
		await buildSqlite({ groups: fx.groups, meta: fx.meta, outPath });
		await expect(fs.stat(outPath)).resolves.toBeTruthy();
		await expect(fs.stat(outPath + '.tmp')).rejects.toThrow();
	});

	it('SqliteRepo.load rejects when cache_version does not match', async () => {
		// Tamper meta row directly in the sqlite (simulate a stale rebuild).
		const dbPath = join(fx.cacheDir, 'groups.v1.sqlite');
		const db = new Database(dbPath);
		db.prepare("UPDATE meta SET value = '999' WHERE key = 'cache_version'").run();
		db.close();

		await expect(openRepo()).rejects.toThrow(/cache_version/i);
	});
});
