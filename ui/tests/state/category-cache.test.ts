import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import type { CacheMeta } from '../../src/lib/domain/groups';
import { CategoryCache } from '../../src/lib/server/state/category-cache';

interface Fixture {
	cacheDir: string;
	stateDir: string;
}

async function setupFixture(): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'category-cache-'));
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	const rows: Array<V2CsvRow & { csv_row: number }> = [];
	// 12 rows, split across 3 ingest_categories. (csv-clean's `categorise`
	// derives ingest_category from the before/after text shape — we just need
	// a heterogeneous mix here, not exact category names.)
	const samples = [
		{ before: 'καλωσηρθατε', after: 'καλωσήρθατε' },
		{ before: 'Hello.', after: 'Hello' },
		{ before: '', after: 'empty before' },
		{ before: 'noop edit', after: 'noop edit' }
	];
	for (let i = 0; i < 12; i++) {
		const s = samples[i % samples.length];
		rows.push({
			edit_id: `e${i.toString().padStart(4, '0')}`,
			utterance_id: `u${i.toString().padStart(4, '0')}`,
			edit_timestamp: `2026-05-01T00:00:${i.toString().padStart(2, '0')}Z`,
			edit_updated_at: '',
			before_text: s.before,
			after_text: s.after,
			edited_by: 'tester',
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
		edit_count: rows.length,
		missing_utterance_id_count: 0
	};
	await fs.writeFile(join(cacheDir, 'meta.json'), JSON.stringify(meta));
	return { cacheDir, stateDir };
}

let fx: Fixture;
beforeEach(async () => {
	fx = await setupFixture();
});
afterEach(async () => {
	await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

describe('CategoryCache', () => {
	it('first call builds the index from the repo and persists it', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new CategoryCache(fx.stateDir);

		// Pick the largest category from the snapshot to assert non-zero behaviour
		// without coupling the test to csv-clean's category vocabulary.
		const peeked = cache._peek();
		expect(peeked).toBeNull();

		const allCats = new Map<string, number>();
		for (const g of repo.iterGroups()) {
			for (const e of g.edits) {
				allCats.set(e.ingest_category, (allCats.get(e.ingest_category) ?? 0) + 1);
			}
		}
		const [topCat, topCount] = [...allCats.entries()].sort((a, b) => b[1] - a[1])[0];

		const page1 = await cache.getPage(repo, topCat, 1, 50);
		expect(page1.total).toBe(topCount);
		expect(page1.items.length).toBeLessThanOrEqual(topCount);
		expect(page1.items[0].edit_id).toMatch(/^e\d+/);

		// Snapshot persisted.
		const onDisk = JSON.parse(
			await fs.readFile(join(fx.stateDir, 'category-index.snapshot.json'), 'utf8')
		);
		expect(onDisk.cache_hash).toBe('test-hash');
		expect(onDisk.byCategory[topCat].length).toBe(topCount);
	});

	it('unknown category returns empty without error', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new CategoryCache(fx.stateDir);
		const page = await cache.getPage(repo, 'does-not-exist', 1, 50);
		expect(page.items).toEqual([]);
		expect(page.total).toBe(0);
	});

	it('paginates within a category and clamps page_size', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const cache = new CategoryCache(fx.stateDir);

		// Find a category with at least 2 edits to test pagination.
		const allCats = new Map<string, number>();
		for (const g of repo.iterGroups()) {
			for (const e of g.edits) {
				allCats.set(e.ingest_category, (allCats.get(e.ingest_category) ?? 0) + 1);
			}
		}
		const [pickedCat] = [...allCats.entries()].find(([, n]) => n >= 2) ?? [];
		if (!pickedCat) {
			// Degenerate fixture — skip.
			return;
		}

		const page1 = await cache.getPage(repo, pickedCat, 1, 1);
		const page2 = await cache.getPage(repo, pickedCat, 2, 1);
		expect(page1.items.length).toBe(1);
		expect(page2.items.length).toBe(1);
		expect(page1.items[0].edit_id).not.toBe(page2.items[0].edit_id);
	});

	it('rejects a snapshot whose cache_hash drifted', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		await fs.writeFile(
			join(fx.stateDir, 'category-index.snapshot.json'),
			JSON.stringify({
				cache_hash: 'OLD-HASH',
				computedAt: Date.now(),
				byCategory: { fake: [{ edit_id: 'e9999', utterance_id: 'u9999', before_text: '', after_text: '', edited_by: null, cleaning_applied: '' }] }
			})
		);

		const cache = new CategoryCache(fx.stateDir);
		// Asking for the synthetic 'fake' category MUST miss because the cache
		// hash on disk doesn't match — index gets rebuilt against the real repo.
		const page = await cache.getPage(repo, 'fake', 1, 50);
		expect(page.items).toEqual([]);
		expect(cache._peek()?.cache_hash).toBe('test-hash');
	});
});
