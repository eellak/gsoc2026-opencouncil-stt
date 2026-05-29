import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FileRepo } from '../../src/lib/server/repo/file-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import type { CacheMeta } from '../../src/lib/domain/groups';
import {
	buildBatch,
	ingestBatch,
	getStats
} from '../../src/lib/server/llm-batch/service';

interface Fixture {
	cacheDir: string;
	stateDir: string;
}

async function setupFixture(n: number): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'llm-batch-'));
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
			before_text: `before-${i}`,
			after_text: `after-${i}`,
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
	fx = await setupFixture(5);
});
afterEach(async () => {
	await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

describe('llm-batch service', () => {
	it('ingests valid items, rejects unknown categories and out-of-batch ids, is idempotent', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });

		// Pre-label one utterance so it's excluded from /next.
		await repo.patchLabel('u0000', { error_categories: ['homophone'] });
		await repo.flush();

		// Issue a batch — should pick from the four remaining unlabeled ids.
		const batch = await buildBatch(repo, {
			n: 3,
			model: 'Test Model 🤖',
			stateDir: fx.stateDir
		});
		expect(batch.items.length).toBe(3);
		expect(batch.items.every((i) => i.id !== 'u0000')).toBe(true);
		expect(batch.batch_id).toMatch(/^[a-f0-9]+$/);

		// Pasted reply: 1 valid, 1 unknown category, 1 not-in-batch.
		const issuedIds = batch.items.map((i) => i.id);
		const validId = issuedIds[0];
		const otherIssued = issuedIds[1]; // we'll just include valid; omit means []
		const raw = JSON.stringify([
			{ id: validId, c: ['homophone'] },
			{ id: otherIssued, c: ['totally_bogus'] },
			{ id: 'u9999', c: ['homophone'] }
		]);

		const res = await ingestBatch(repo, {
			batch_id: batch.batch_id,
			raw,
			stateDir: fx.stateDir
		});
		await repo.flush();

		expect(res.accepted).toBe(1);
		expect(res.rejected.length).toBe(2);
		const reasons = new Set(res.rejected.map((r) => r.reason));
		expect(reasons.has('not_in_batch')).toBe(true);
		expect(reasons.has('no_valid_categories')).toBe(true);

		// The third issued id (not mentioned in the paste) becomes an explicit decline.
		expect(res.declined.length).toBe(1);

		// Verify the source landed as the sanitized slug — accepted + declines all
		// carry the same `ext-test-model` source.
		const eventsRaw = await fs.readFile(
			join(fx.stateDir, 'review-events.jsonl'),
			'utf8'
		);
		const events = eventsRaw
			.split('\n')
			.filter(Boolean)
			.map((l) => JSON.parse(l) as { utterance_id: string; source: unknown; patch: { error_categories?: string[] } });
		const extEvents = events.filter((e) => e.source === 'ext-test-model');
		expect(extEvents.length).toBe(2);
		const accepted = extEvents.find((e) => e.patch.error_categories?.length);
		expect(accepted?.utterance_id).toBe(validId);
		expect(accepted?.patch.error_categories).toEqual(['homophone']);

		// Stats reflect the new ext source.
		const stats = await getStats(repo, fx.stateDir);
		expect(stats.by_source['ext-test-model']).toBe(2); // 1 accepted + 1 declined
		// labeled = u0000 (local) + validId (accepted) + 1 declined.
		expect(stats.labeled).toBe(3);

		// Idempotency: re-post the same body.
		const res2 = await ingestBatch(repo, {
			batch_id: batch.batch_id,
			raw,
			stateDir: fx.stateDir
		});
		await repo.flush();
		expect(res2.accepted).toBe(0);
		expect(res2.duplicates.length).toBe(1);
		expect(res2.duplicates[0]).toBe(validId);

		// No new events appended.
		const eventsRaw2 = await fs.readFile(
			join(fx.stateDir, 'review-events.jsonl'),
			'utf8'
		);
		const eventLines2 = eventsRaw2.split('\n').filter(Boolean);
		expect(eventLines2.length).toBe(events.length);
	});

	it('rejects ingest when batch_id is unknown', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		await expect(
			ingestBatch(repo, {
				batch_id: 'deadbeef',
				raw: '[]',
				stateDir: fx.stateDir
			})
		).rejects.toThrow(/batch/i);
	});

	it('omitted ids in the response are interpreted as empty categories (explicit decline)', async () => {
		const repo = await FileRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir });
		const batch = await buildBatch(repo, { n: 3, model: 'gemini-2.5-flash', stateDir: fx.stateDir });
		const issuedIds = batch.items.map((i) => i.id);

		// Response only mentions the first id; the other two are omitted.
		const raw = JSON.stringify([{ id: issuedIds[0], c: ['accent_tonos'] }]);
		const res = await ingestBatch(repo, {
			batch_id: batch.batch_id,
			raw,
			stateDir: fx.stateDir
		});
		await repo.flush();

		expect(res.accepted).toBe(1);
		expect(res.declined.length).toBe(2);
		// Omitted ids should now be excluded from future /next calls.
		const batch2 = await buildBatch(repo, { n: 5, model: 'gemini-2.5-flash', stateDir: fx.stateDir });
		expect(batch2.items.every((i) => !issuedIds.includes(i.id))).toBe(true);
	});
});
