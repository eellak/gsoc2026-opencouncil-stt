/**
 * Evaluation for the meeting-eligibility filter (Change 2).
 *
 * Fixture: three meetings —
 *   m1: 9 distinct human-corrected utterances (one carries duplicate human
 *       edits, to prove distinct-utterance de-duplication) → ineligible (<10).
 *   m2: 10 distinct human-corrected utterances → eligible.
 *   m3: 10 utterances, all edits machine ('task') → ineligible (0 human).
 *
 * The single big regression this guards against: applying eligibility to the
 * seeded order but forgetting the sidecar-derived paths (idsByStatus reviewed
 * bucket, idsByErrorCategory). Every list path must agree on the eligible set;
 * getGroup() must stay unfiltered.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { SqliteRepo } from '../../src/lib/server/repo/sqlite-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import { buildSqlite } from '../../src/lib/server/cache/build-sqlite';
import type { CacheMeta } from '../../src/lib/domain/groups';

interface Fixture {
	cacheDir: string;
	stateDir: string;
	m1Ids: string[];
	m2Ids: string[];
	m3Ids: string[];
	// Colliding meeting: same meeting_id slug 'm2' but a DIFFERENT city. Must be
	// counted separately from athens/m2 (which is eligible).
	collisionIds: string[];
}

let editSeq = 0;
function makeRow(
	utterance_id: string,
	meeting_id: string,
	edited_by: 'user' | 'task',
	city_id = 'athens'
): V2CsvRow & { csv_row: number } {
	const i = editSeq++;
	return {
		edit_id: `e${i.toString().padStart(5, '0')}`,
		utterance_id,
		edit_timestamp: `2026-05-01T00:00:${(i % 60).toString().padStart(2, '0')}Z`,
		edit_updated_at: '',
		before_text: `before-${i}`,
		after_text: `after-${i}`,
		edited_by,
		utterance_start: '0',
		utterance_end: '1',
		audio_url: 'https://example.com/a.mp3',
		youtube_url: '',
		meeting_name: `${city_id}-${meeting_id}`,
		meeting_date: '2026-05-01',
		meeting_id,
		city_id,
		csv_row: i
	};
}

async function setupFixture(): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'meeting-elig-'));
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	editSeq = 0;
	const rows: Array<V2CsvRow & { csv_row: number }> = [];

	const m1Ids: string[] = [];
	for (let k = 0; k < 9; k++) {
		const id = `m1u${k}`;
		m1Ids.push(id);
		rows.push(makeRow(id, 'm1', 'user'));
	}
	// Extra human edits on one m1 utterance — must still count it once.
	rows.push(makeRow('m1u0', 'm1', 'user'));
	rows.push(makeRow('m1u0', 'm1', 'user'));

	const m2Ids: string[] = [];
	for (let k = 0; k < 10; k++) {
		const id = `m2u${k}`;
		m2Ids.push(id);
		rows.push(makeRow(id, 'm2', 'user'));
	}

	const m3Ids: string[] = [];
	for (let k = 0; k < 10; k++) {
		const id = `m3u${k}`;
		m3Ids.push(id);
		rows.push(makeRow(id, 'm3', 'task'));
	}

	// Collision: city 'sparta', meeting_id 'm2' (same slug as athens/m2), but
	// only 4 human utterances → must be ineligible and NOT merge with athens/m2.
	const collisionIds: string[] = [];
	for (let k = 0; k < 4; k++) {
		const id = `spm2u${k}`;
		collisionIds.push(id);
		rows.push(makeRow(id, 'm2', 'user', 'sparta'));
	}

	const { groups } = buildGroups(rows);
	await fs.writeFile(join(cacheDir, 'groups.v1.json'), JSON.stringify(groups));

	const meta: CacheMeta = {
		cache_version: 1,
		source_csv_path: 'synthetic.csv',
		source_size: 0,
		source_mtime_ms: 0,
		source_hash: 'elig-hash',
		generated_at: new Date().toISOString(),
		group_count: groups.length,
		edit_count: groups.reduce((acc, g) => acc + g.edits.length, 0),
		missing_utterance_id_count: 0
	};
	await fs.writeFile(join(cacheDir, 'meta.json'), JSON.stringify(meta));
	await buildSqlite({ groups, meta, outPath: join(cacheDir, 'groups.v1.sqlite') });

	return { cacheDir, stateDir, m1Ids, m2Ids, m3Ids, collisionIds };
}

let fx: Fixture;
beforeEach(async () => {
	fx = await setupFixture();
});
afterEach(async () => {
	await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

function loadSqlite(threshold: number): Promise<SqliteRepo> {
	return SqliteRepo.load({
		cacheDir: fx.cacheDir,
		stateDir: fx.stateDir,
		meetingMinHumanUtterances: threshold
	});
}

describe('meeting-eligibility — SqliteRepo with threshold 10', () => {
	it('total and queue include only the eligible meeting (m2)', async () => {
		const repo = await loadSqlite(10);
		expect(repo.total).toBe(10);

		const q = repo.queue(42, 0, 50);
		expect(q.total).toBe(10);
		const got = q.groups.map((g) => g.utterance_id);
		expect(new Set(got)).toEqual(new Set(fx.m2Ids));
		// every returned group really is from m2
		expect(q.groups.every((g) => g.meeting_id === 'm2')).toBe(true);
	});

	it('eligibleOrderedIds is exactly the athens/m2 set', async () => {
		const repo = await loadSqlite(10);
		expect(new Set(repo.eligibleOrderedIds())).toEqual(new Set(fx.m2Ids));
	});

	it('does NOT merge a colliding meeting_id from another city', async () => {
		// sparta/m2 shares the slug 'm2' with the eligible athens/m2 but has only
		// 4 human utterances. Keyed by (city, meeting) it stays ineligible.
		const repo = await loadSqlite(10);
		const eligible = new Set(repo.eligibleOrderedIds());
		for (const id of fx.collisionIds) expect(eligible.has(id)).toBe(false);
		// total is still just athens/m2 (10), not 10+4.
		expect(repo.total).toBe(10);
		// …and the colliding group still resolves directly (nothing deleted).
		expect(repo.getGroup(fx.collisionIds[0])?.utterance_id).toBe(fx.collisionIds[0]);
	});

	it('idsByStatus (reviewed) excludes labeled ids from m1 and m3', async () => {
		const repo = await loadSqlite(10);
		// Label one utterance per meeting as include.
		await repo.patchLabel(fx.m1Ids[0], { include_status: 'include' });
		await repo.patchLabel(fx.m2Ids[0], { include_status: 'include' });
		await repo.patchLabel(fx.m3Ids[0], { include_status: 'include' });
		await repo.flush();

		expect(new Set(repo.idsByStatus('include'))).toEqual(new Set([fx.m2Ids[0]]));
	});

	it('idsByStatus (unreviewed) is the eligible set minus reviewed', async () => {
		const repo = await loadSqlite(10);
		await repo.patchLabel(fx.m2Ids[0], { include_status: 'include' });
		await repo.flush();

		const unreviewed = new Set(repo.idsByStatus('unreviewed'));
		expect(unreviewed).toEqual(new Set(fx.m2Ids.slice(1)));
	});

	it('idsByErrorCategory excludes m1 and m3', async () => {
		const repo = await loadSqlite(10);
		await repo.patchLabel(fx.m1Ids[1], { error_categories: ['homophone'] });
		await repo.patchLabel(fx.m2Ids[1], { error_categories: ['homophone'] });
		await repo.patchLabel(fx.m3Ids[1], { error_categories: ['homophone'] });
		await repo.flush();

		expect(repo.idsByErrorCategory('homophone')).toEqual([fx.m2Ids[1]]);
		expect(new Set(repo.groupsByErrorCategory('homophone').map((g) => g.utterance_id))).toEqual(
			new Set([fx.m2Ids[1]])
		);
	});

	it('getGroup still resolves an ineligible utterance (nothing deleted)', async () => {
		const repo = await loadSqlite(10);
		expect(repo.getGroup(fx.m1Ids[0])?.utterance_id).toBe(fx.m1Ids[0]);
		expect(repo.getGroup(fx.m3Ids[0])?.utterance_id).toBe(fx.m3Ids[0]);
	});

	it('persists a snapshot and a reload reuses it with the same result', async () => {
		await loadSqlite(10);
		const snapPath = join(fx.stateDir, 'meeting-eligibility.snapshot.json');
		const snap = JSON.parse(await fs.readFile(snapPath, 'utf8'));
		expect(snap.cache_hash).toBe('elig-hash');
		expect(snap.threshold).toBe(10);
		// Keyed by (city, meeting): only athens/m2 qualifies (sparta/m2 has 4).
		expect(new Set(snap.eligible_meeting_keys)).toEqual(new Set(['athens m2']));

		const reloaded = await loadSqlite(10);
		expect(new Set(reloaded.eligibleOrderedIds())).toEqual(new Set(fx.m2Ids));
	});
});

describe('meeting-eligibility — disabled (threshold 0)', () => {
	it('includes all meetings when the filter is off', async () => {
		const repo = await loadSqlite(0);
		const allIds = [...fx.m1Ids, ...fx.m2Ids, ...fx.m3Ids, ...fx.collisionIds];
		expect(repo.total).toBe(allIds.length);
		expect(new Set(repo.eligibleOrderedIds())).toEqual(new Set(allIds));
	});
});

describe('meeting-eligibility — FileRepo parity', () => {
	it('FileRepo applies the same eligible set as SqliteRepo', async () => {
		const file = await FileRepo.load({
			cacheDir: fx.cacheDir,
			stateDir: fx.stateDir,
			meetingMinHumanUtterances: 10
		});
		expect(file.total).toBe(10);
		expect(new Set(file.eligibleOrderedIds())).toEqual(new Set(fx.m2Ids));
	});
});
