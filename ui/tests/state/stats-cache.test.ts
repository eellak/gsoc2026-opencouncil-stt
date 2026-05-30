import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import type { CacheMeta } from '../../src/lib/domain/groups';
import { StatsCache, computeStats } from '../../src/lib/server/state/stats-cache';

interface Fixture {
	cacheDir: string;
	stateDir: string;
}

async function setupFixture(n: number): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'stats-cache-'));
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
			before_text: 'before',
			after_text: 'after',
			edited_by: i % 2 === 0 ? 'alice' : 'bob',
			utterance_start: '0',
			utterance_end: '1',
			audio_url: 'https://example.com/a.mp3',
			youtube_url: '',
			meeting_name: 'm',
			meeting_date: '2026-05-01',
			meeting_id: 'm1',
			city_id: 'athens',
			csv_row: i
		});
	}
	const { groups } = buildGroups(rows);
	await fs.writeFile(join(cacheDir, 'groups.v1.json'), JSON.stringify(groups));
	const meta: CacheMeta = {
		cache_version: 1,
		source_csv_path: 'synthetic.csv',
		source_size: 0,
		source_mtime_ms: 0,
		source_hash: 'test-hash',
		generated_at: new Date().toISOString(),
		group_count: groups.length,
		edit_count: groups.length,
		missing_utterance_id_count: 0
	};
	await fs.writeFile(join(cacheDir, 'meta.json'), JSON.stringify(meta));
	return { cacheDir, stateDir };
}

let fx: Fixture;
beforeEach(async () => {
	fx = await setupFixture(20);
});
afterEach(async () => {
	await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

describe('computeStats', () => {
	it('aggregates groups, edits, status, editors, meetings', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const stats = await computeStats(repo);
		expect(stats.total).toBe(20);
		expect(stats.groups).toBe(20);
		expect(stats.total_edits).toBe(20);
		expect(stats.by_status.unreviewed).toBe(20);
		expect(stats.cache_hash).toBe('test-hash');
		const editors = Object.fromEntries(stats.by_editor.map((e) => [e.edited_by, e.count]));
		expect(editors).toEqual({ alice: 10, bob: 10 });
		expect(stats.by_meeting[0]?.count).toBe(20);
	});
});

describe('StatsCache', () => {
	it('first call computes and persists the snapshot to disk', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new StatsCache(fx.stateDir);
		const got = await cache.get(repo);
		expect(got.stats.total).toBe(20);

		// Snapshot file exists with the same shape.
		const onDisk = JSON.parse(
			await fs.readFile(join(fx.stateDir, 'stats.snapshot.json'), 'utf8')
		);
		expect(onDisk.stats.total).toBe(20);
		expect(onDisk.stats.cache_hash).toBe('test-hash');
		expect(typeof onDisk.computedAt).toBe('number');
	});

	it('subsequent calls return the same cached snapshot instance', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new StatsCache(fx.stateDir);
		const a = await cache.get(repo);
		const b = await cache.get(repo);
		expect(b).toBe(a);
	});

	it('cold start hydrates from disk without recomputing', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		// Pre-seed the on-disk snapshot directly.
		const synthetic = {
			stats: { ...(await computeStats(repo)), total: 42 },
			computedAt: Date.now()
		};
		await fs.writeFile(join(fx.stateDir, 'stats.snapshot.json'), JSON.stringify(synthetic));

		const cache = new StatsCache(fx.stateDir);
		const got = await cache.get(repo);
		// Loaded from disk verbatim (didn't re-aggregate, so total stays at 42).
		expect(got.stats.total).toBe(42);
	});

	it('rejects snapshots whose cache_hash no longer matches the repo', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const stale = {
			stats: { ...(await computeStats(repo)), cache_hash: 'old-hash' },
			computedAt: Date.now()
		};
		await fs.writeFile(join(fx.stateDir, 'stats.snapshot.json'), JSON.stringify(stale));

		const cache = new StatsCache(fx.stateDir);
		const got = await cache.get(repo);
		// Recomputed against the real repo hash.
		expect(got.stats.cache_hash).toBe('test-hash');
	});

	it('get() returns the existing snapshot without recomputing when the cache is stale (cron-driven model)', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new StatsCache(fx.stateDir);

		// Seed an old-but-valid snapshot.
		const seeded = {
			stats: { ...(await computeStats(repo)), total: 999 },
			computedAt: Date.now() - 60 * 60 * 1000 // 1 h ago — way past TTL
		};
		await fs.writeFile(join(fx.stateDir, 'stats.snapshot.json'), JSON.stringify(seeded));

		// First get() rehydrates from disk; we expect the stale total to come
		// back unchanged. Under the old stale-while-revalidate model this
		// would have fired a background recompute; under the cron model
		// `get()` is purely a read.
		const got = await cache.get(repo);
		expect(got.stats.total).toBe(999);
	});

	it('startBackgroundRefresh kicks off an immediate refresh and is idempotent', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new StatsCache(fx.stateDir);

		const repoFactory = async () => repo;
		const stop = cache.startBackgroundRefresh(repoFactory, 60_000);
		// The immediate tick is async — give it a moment to land.
		await new Promise((r) => setTimeout(r, 30));
		expect(cache._peek()?.stats.total).toBe(20);
		expect(cache._hasInterval()).toBe(true);

		// Second call is a no-op (returns a stopper but doesn't add a timer).
		const stop2 = cache.startBackgroundRefresh(repoFactory, 60_000);
		expect(cache._hasInterval()).toBe(true);

		stop();
		stop2();
		expect(cache._hasInterval()).toBe(false);
	});
});
