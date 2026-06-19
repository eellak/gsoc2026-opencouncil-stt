import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { SqliteRepo } from '../../src/lib/server/repo/sqlite-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import { buildSqlite } from '../../src/lib/server/cache/build-sqlite';
import {
	degenerateCategories,
	isDegenerate
} from '../../src/lib/server/state/ingest-filter';
import type { CacheMeta, Group } from '../../src/lib/domain/groups';

/**
 * Rows hand-crafted so categorise() assigns a known ingest_category to each
 * utterance's single (latest) edit. All edited_by:'user' so they count for
 * meeting eligibility. `kind` is the expected latest-edit category.
 */
const ROWS: Array<{ id: string; before: string; after: string; kind: string }> = [
	{ id: 'u00_clean1', before: 'alpha', after: 'alpha beta', kind: 'clean' },
	{ id: 'u01_clean2', before: 'gamma', after: 'gamma delta', kind: 'clean' },
	{ id: 'u02_clean3', before: 'epsilon', after: 'epsilon zeta', kind: 'clean' },
	{ id: 'u03_noop', before: 'same', after: 'same', kind: 'noop_edit' },
	{ id: 'u04_emptyA', before: 'deleted', after: '', kind: 'empty_after' },
	{ id: 'u05_ws', before: 'a b', after: 'a  b', kind: 'whitespace_only' },
	{ id: 'u06_emptyB', before: '', after: 'inserted', kind: 'empty_before' }
];

const DEGENERATE_IDS = ['u03_noop', 'u04_emptyA', 'u05_ws'];
const KEPT_IDS = ['u00_clean1', 'u01_clean2', 'u02_clean3', 'u06_emptyB'];

interface Fixture {
	cacheDir: string;
	stateDir: string;
	groups: Group[];
}

async function setupFixture(): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'ingest-filter-'));
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	const rows: Array<V2CsvRow & { csv_row: number }> = ROWS.map((r, i) => ({
		edit_id: `e-${r.id}`,
		utterance_id: r.id,
		edit_timestamp: `2026-05-01T00:00:${i.toString().padStart(2, '0')}Z`,
		edit_updated_at: '',
		before_text: r.before,
		after_text: r.after,
		edited_by: 'user',
		utterance_start: '0',
		utterance_end: '1',
		audio_url: 'https://example.com/a.mp3',
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
		source_hash: 'ingest-filter-hash',
		generated_at: new Date().toISOString(),
		group_count: groups.length,
		edit_count: groups.reduce((a, g) => a + g.edits.length, 0),
		missing_utterance_id_count: 0
	};
	await fs.writeFile(join(cacheDir, 'meta.json'), JSON.stringify(meta));
	await buildSqlite({ groups, meta, outPath: join(cacheDir, 'groups.v1.sqlite') });

	return { cacheDir, stateDir, groups };
}

let fx: Fixture;
const ORIG = process.env.DROP_INGEST_CATEGORIES;
beforeEach(async () => {
	delete process.env.DROP_INGEST_CATEGORIES; // use the default degenerate set
	fx = await setupFixture();
});
afterEach(async () => {
	if (ORIG === undefined) delete process.env.DROP_INGEST_CATEGORIES;
	else process.env.DROP_INGEST_CATEGORIES = ORIG;
	await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

// The fixture exercises categorise() itself — assert our category assumptions
// hold before testing the filter built on them.
describe('fixture sanity — categorise assigns expected bins', () => {
	it('each utterance lands in its expected ingest_category', () => {
		const byId = new Map(fx.groups.map((g) => [g.utterance_id, g]));
		for (const r of ROWS) {
			const g = byId.get(r.id)!;
			expect(g.edits[g.edits.length - 1].ingest_category).toBe(r.kind);
		}
	});
});

// Run the same expectations against both repo flavours — they must agree.
const flavours: Array<{ name: string; load: (fx: Fixture) => Promise<FileRepo | SqliteRepo> }> = [
	{ name: 'SqliteRepo', load: (f) => SqliteRepo.load({ cacheDir: f.cacheDir, stateDir: f.stateDir, meetingMinHumanUtterances: 0 }) },
	{ name: 'FileRepo', load: (f) => FileRepo.load({ cacheDir: f.cacheDir, stateDir: f.stateDir, meetingMinHumanUtterances: 0 }) }
];

for (const { name, load } of flavours) {
	describe(`${name} — degenerate ids excluded from review navigation`, () => {
		it('eligibleOrderedIds drops degenerate bins, keeps clean + empty_before', async () => {
			const repo = await load(fx);
			const elig = new Set(repo.eligibleOrderedIds());
			for (const id of DEGENERATE_IDS) expect(elig.has(id)).toBe(false);
			for (const id of KEPT_IDS) expect(elig.has(id)).toBe(true);
			expect(repo.total).toBe(KEPT_IDS.length);
		});

		it('getGroup still resolves a dropped utterance (reversible, nothing deleted)', async () => {
			const repo = await load(fx);
			const g = repo.getGroup('u04_emptyA');
			expect(g).not.toBeNull();
			expect(g!.utterance_id).toBe('u04_emptyA');
		});

		it('the seeded queue never serves a degenerate utterance', async () => {
			const repo = await load(fx);
			const ids = repo.queue(7, 0, 50).groups.map((g) => g.utterance_id);
			for (const id of DEGENERATE_IDS) expect(ids).not.toContain(id);
			expect(ids.sort()).toEqual([...KEPT_IDS].sort());
		});

		it('loading twice yields an identical eligible set (deterministic)', async () => {
			const a = [...(await load(fx)).eligibleOrderedIds()];
			const b = [...(await load(fx)).eligibleOrderedIds()];
			expect(b).toEqual(a);
		});

		it('DROP_INGEST_CATEGORIES="" disables the filter (all back in)', async () => {
			process.env.DROP_INGEST_CATEGORIES = '';
			const repo = await load(fx);
			const elig = new Set(repo.eligibleOrderedIds());
			for (const id of [...DEGENERATE_IDS, ...KEPT_IDS]) expect(elig.has(id)).toBe(true);
		});
	});
}

describe('export layer — same policy, applied independently of navigation', () => {
	it('export predicate skips exactly the degenerate groups', async () => {
		const drop = degenerateCategories();
		const exported = fx.groups.filter((g) => !isDegenerate(g, drop)).map((g) => g.utterance_id);
		expect(exported.sort()).toEqual([...KEPT_IDS].sort());
	});
});
