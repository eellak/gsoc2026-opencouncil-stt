import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import type { CacheMeta } from '../../src/lib/domain/groups';

interface Fixture {
	cacheDir: string;
	stateDir: string;
}

async function setupFixture(n: number): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'file-repo-'));
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	const rows: Array<V2CsvRow & { csv_row: number }> = [];
	for (let i = 0; i < n; i++) {
		rows.push({
			edit_id: `e${i}`,
			utterance_id: `u${i.toString().padStart(4, '0')}`,
			edit_timestamp: `2026-05-01T00:00:${i.toString().padStart(2, '0')}Z`,
			edit_updated_at: '',
			before_text: 'before',
			after_text: 'after',
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

describe('FileRepo', () => {
	it('loads cache and exposes the source hash and total', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		expect(repo.hash).toBe('test-hash');
		expect(repo.total).toBe(20);
	});

	it('seeded queue order is reproducible for the same seed', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const a = repo.queue(42, 0, 20).groups.map((g) => g.utterance_id);
		const b = repo.queue(42, 0, 20).groups.map((g) => g.utterance_id);
		expect(b).toEqual(a);
	});

	it('different seeds produce different orders (for known seeds)', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const a = repo.queue(1, 0, 20).groups.map((g) => g.utterance_id);
		const b = repo.queue(2, 0, 20).groups.map((g) => g.utterance_id);
		expect(b).not.toEqual(a);
		expect(b.slice().sort()).toEqual(a.slice().sort()); // same set
	});

	it('seeded queue is a permutation of all utterance_ids', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const out = repo.queue(7, 0, 100).groups.map((g) => g.utterance_id);
		expect(new Set(out).size).toBe(20);
	});

	it('queue cursor advances across pages', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const p1 = repo.queue(3, 0, 5);
		expect(p1.groups).toHaveLength(5);
		expect(p1.next_cursor).toBe(5);
		const p2 = repo.queue(3, p1.next_cursor!, 5);
		expect(p2.groups[0].utterance_id).not.toBe(p1.groups[0].utterance_id);
	});

	it('queue returns next_cursor=null past the end', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const p = repo.queue(1, 18, 5);
		expect(p.groups).toHaveLength(2);
		expect(p.next_cursor).toBeNull();
	});

	it('patchLabel writes to the sidecar and is visible on subsequent reads', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		await repo.patchLabel('u0005', { include_status: 'include', error_categories: ['homophone'] });
		await repo.flush();
		const g = repo.getGroup('u0005');
		expect(g?.label.include_status).toBe('include');
		expect(g?.label.error_categories).toEqual(['homophone']);
	});

	it('patchLabel on unknown utterance_id returns null', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const out = await repo.patchLabel('nope', { include_status: 'include' });
		expect(out).toBeNull();
	});
});
