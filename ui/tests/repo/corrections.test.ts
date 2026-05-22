import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { makeTestClient, type TestClient } from '../_helpers/pg-test-client';
import {
	insertCorrection,
	getCorrection,
	listCorrections,
	getFirstUnreviewed,
	getNeighbors
} from '$lib/server/repo/corrections';
import { patchLabel } from '$lib/server/repo/labels';
import { getStats } from '$lib/server/repo/stats';
import type { Correction } from '$lib/domain/types';

function makeCorrection(overrides: Partial<Correction> = {}): Correction {
	return {
		edit_id: 'id-001',
		utterance_id: null,
		meeting_id: null,
		latest_per_utterance: true,
		edit_timestamp: '2026-01-01T10:00:00Z',
		edit_updated_at: null,
		before_text: 'παλιο κείμενο',
		after_text: 'νέο κείμενο',
		edited_by: 'user1',
		utterance_start: 10.5,
		utterance_end: 15.0,
		city_id: null,
		audio_url: 'https://cdn.example.com/audio/meeting1.mp3',
		audio_cdn_url: null,
		youtube_url: null,
		meeting_name: 'Δημοτικό Συμβούλιο',
		meeting_date: '2026-01-01',
		ingest_category: null,
		cleaning_applied: null,
		...overrides
	};
}

describe('corrections repo', () => {
	let client: TestClient;

	beforeEach(async () => { client = await makeTestClient(); });
	afterEach(async () => { await client.close(); });

	it('inserts and retrieves a correction', async () => {
		await insertCorrection(client, makeCorrection());
		const c = await getCorrection(client, 'id-001');
		expect(c).not.toBeNull();
		expect(c!.edit_id).toBe('id-001');
		expect(c!.before_text).toBe('παλιο κείμενο');
		expect(c!.include_status).toBe('unreviewed');
	});

	it('insert is idempotent (ON CONFLICT DO UPDATE)', async () => {
		await insertCorrection(client, makeCorrection());
		await insertCorrection(client, makeCorrection({ before_text: 'αλλαγή' }));
		const { total } = await listCorrections(client);
		expect(total).toBe(1);
		expect((await getCorrection(client, 'id-001'))!.before_text).toBe('αλλαγή');
	});

	it('returns null for unknown edit_id', async () => {
		expect(await getCorrection(client, 'nonexistent')).toBeNull();
	});

	it('list returns paginated results', async () => {
		for (let i = 1; i <= 5; i++) {
			await insertCorrection(client, makeCorrection({ edit_id: `id-${i.toString().padStart(3, '0')}`, edit_timestamp: `2026-01-0${i}T10:00:00Z` }));
		}
		const { items, total } = await listCorrections(client, { page: 1, page_size: 3 });
		expect(total).toBe(5);
		expect(items).toHaveLength(3);
	});

	it('list filters by status', async () => {
		await insertCorrection(client, makeCorrection({ edit_id: 'a' }));
		await insertCorrection(client, makeCorrection({ edit_id: 'b', edit_timestamp: '2026-01-02T10:00:00Z' }));
		await patchLabel(client, 'a', { include_status: 'include' });

		const { total } = await listCorrections(client, { status: 'include' });
		expect(total).toBe(1);
	});

	it('getFirstUnreviewed returns the earliest unreviewed', async () => {
		for (let i = 1; i <= 3; i++) {
			await insertCorrection(client, makeCorrection({ edit_id: `id-${i}`, edit_timestamp: `2026-01-0${i}T10:00:00Z` }));
		}
		await patchLabel(client, 'id-1', { include_status: 'include' });
		expect(await getFirstUnreviewed(client)).toBe('id-2');
	});

	it('getFirstUnreviewed returns null when all reviewed', async () => {
		await insertCorrection(client, makeCorrection());
		await patchLabel(client, 'id-001', { include_status: 'include' });
		expect(await getFirstUnreviewed(client)).toBeNull();
	});

	it('getNeighbors returns prev and next by timestamp', async () => {
		for (let i = 1; i <= 3; i++) {
			await insertCorrection(client, makeCorrection({ edit_id: `id-${i}`, edit_timestamp: `2026-01-0${i}T10:00:00Z` }));
		}
		const { prev, next } = await getNeighbors(client, 'id-2');
		expect(prev?.edit_id).toBe('id-1');
		expect(next?.edit_id).toBe('id-3');
	});

	it('getNeighbors at boundaries returns null', async () => {
		for (let i = 1; i <= 3; i++) {
			await insertCorrection(client, makeCorrection({ edit_id: `id-${i}`, edit_timestamp: `2026-01-0${i}T10:00:00Z` }));
		}
		expect((await getNeighbors(client, 'id-1')).prev).toBeNull();
		expect((await getNeighbors(client, 'id-3')).next).toBeNull();
	});
});

describe('labels repo', () => {
	let client: TestClient;

	beforeEach(async () => {
		client = await makeTestClient();
		await insertCorrection(client, makeCorrection());
	});
	afterEach(async () => { await client.close(); });

	it('patches include_status', async () => {
		const { updated } = await patchLabel(client, 'id-001', { include_status: 'include' });
		expect(updated).toBe(true);
		expect((await getCorrection(client, 'id-001'))!.include_status).toBe('include');
	});

	it('patches error_category', async () => {
		await patchLabel(client, 'id-001', { error_category: 'named_entity' });
		expect((await getCorrection(client, 'id-001'))!.error_category).toBe('named_entity');
	});

	it('patches reviewer_notes', async () => {
		await patchLabel(client, 'id-001', { reviewer_notes: 'Σημείωση ελέγχου' });
		expect((await getCorrection(client, 'id-001'))!.reviewer_notes).toBe('Σημείωση ελέγχου');
	});

	it('returns updated: false for nonexistent edit_id', async () => {
		expect((await patchLabel(client, 'nope', { include_status: 'include' })).updated).toBe(false);
	});

	it('sets human_updated_at on patch', async () => {
		await patchLabel(client, 'id-001', { include_status: 'include' });
		expect((await getCorrection(client, 'id-001'))!.human_updated_at).toBeTruthy();
	});

	it('rejects unknown columns in patch', async () => {
		await expect(
			// @ts-expect-error — intentionally unknown column to test the whitelist
			patchLabel(client, 'id-001', { evil_drop: 'x' })
		).rejects.toThrow(/not patchable/);
	});
});

describe('stats repo', () => {
	let client: TestClient;

	beforeEach(async () => {
		client = await makeTestClient();
		for (let i = 1; i <= 4; i++) {
			await insertCorrection(client, makeCorrection({
				edit_id: `id-${i}`,
				edit_timestamp: `2026-01-0${i}T10:00:00Z`,
				utterance_start: i * 2,
				utterance_end: i * 2 + 3,
				edited_by: i % 2 === 0 ? 'user1' : 'user2'
			}));
		}
		await patchLabel(client, 'id-1', { include_status: 'include', error_category: 'named_entity' });
		await patchLabel(client, 'id-2', { include_status: 'exclude' });
		await patchLabel(client, 'id-3', { include_status: 'include', error_category: 'named_entity' });
	});
	afterEach(async () => { await client.close(); });

	it('returns correct total', async () => {
		expect((await getStats(client)).total).toBe(4);
	});

	it('counts by_status correctly', async () => {
		const { by_status } = await getStats(client);
		expect(by_status.include).toBe(2);
		expect(by_status.exclude).toBe(1);
		expect(by_status.unreviewed).toBe(1);
	});

	it('counts by_category', async () => {
		const { by_category } = await getStats(client);
		const named = by_category.find((r) => r.category === 'named_entity');
		expect(named?.count).toBe(2);
	});

	it('counts by_editor', async () => {
		const { by_editor } = await getStats(client);
		expect(by_editor.length).toBeGreaterThan(0);
	});

	it('has duration buckets', async () => {
		const { by_duration_bucket } = await getStats(client);
		expect(by_duration_bucket.length).toBeGreaterThan(0);
	});
});
