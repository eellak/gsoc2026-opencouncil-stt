import { describe, it, expect } from 'vitest';
import { mergeEventLogs, parseEventLog, type ReviewEventLike } from '../scripts/merge-state';

function ev(
	id: number,
	utterance_id: string,
	ts: string,
	patch: Record<string, unknown>,
	username = 'ang'
): ReviewEventLike {
	return { id, ts, utterance_id, source: { kind: 'local', username }, patch };
}

describe('parseEventLog', () => {
	it('tolerates a truncated final line', () => {
		const a = JSON.stringify(ev(1, 'u1', '2026-05-01T00:00:00.000Z', { include_status: 'include' }));
		const raw = a + '\n{"id":2,"ts":"2026'; // truncated
		const parsed = parseEventLog(raw);
		expect(parsed).toHaveLength(1);
		expect(parsed[0].utterance_id).toBe('u1');
	});

	it('throws on corruption before the final line', () => {
		const good = JSON.stringify(ev(2, 'u2', '2026-05-01T00:00:01.000Z', {}));
		const raw = 'not-json\n' + good;
		expect(() => parseEventLog(raw)).toThrow(/corrupt event at line 1/);
	});
});

describe('mergeEventLogs', () => {
	it('orders by ts and renumbers ids sequentially', () => {
		const a = [
			ev(1, 'u1', '2026-05-01T10:00:00.000Z', { include_status: 'include' }),
			ev(2, 'u2', '2026-05-01T12:00:00.000Z', { include_status: 'exclude' })
		];
		const b = [ev(1, 'u3', '2026-05-01T11:00:00.000Z', { include_status: 'uncertain' })];

		const { merged, stats } = mergeEventLogs(a, b);

		expect(merged.map((e) => e.id)).toEqual([1, 2, 3]);
		// chronological: u1 (10:00) → u3 (11:00) → u2 (12:00)
		expect(merged.map((e) => e.utterance_id)).toEqual(['u1', 'u3', 'u2']);
		expect(stats.mergedCount).toBe(3);
		expect(stats.conflicts).toBe(0);
	});

	it('last-write-wins for the same utterance decided in both logs', () => {
		const a = [ev(1, 'u1', '2026-05-01T10:00:00.000Z', { include_status: 'include' })];
		const b = [ev(1, 'u1', '2026-05-01T15:00:00.000Z', { include_status: 'exclude' })];

		const { merged, stats } = mergeEventLogs(a, b);

		expect(stats.conflicts).toBe(1);
		// The later event (15:00, exclude) must come last so replay leaves exclude.
		expect(merged[merged.length - 1].patch.include_status).toBe('exclude');
	});

	it('drops exact-duplicate events (shared history is not double-counted)', () => {
		const shared = ev(1, 'u1', '2026-05-01T10:00:00.000Z', { include_status: 'include' });
		const a = [shared, ev(2, 'u2', '2026-05-01T11:00:00.000Z', { include_status: 'exclude' })];
		// b was copied from a (shares the u1 event) and added one more.
		const b = [shared, ev(2, 'u3', '2026-05-01T12:00:00.000Z', { include_status: 'uncertain' })];

		const { merged, stats } = mergeEventLogs(a, b);

		expect(stats.duplicatesDropped).toBe(1);
		expect(stats.mergedCount).toBe(3);
		const u1Events = merged.filter((e) => e.utterance_id === 'u1');
		expect(u1Events).toHaveLength(1);
	});

	it('merges partial patches (category then status) by chronological replay', () => {
		const a = [ev(1, 'u1', '2026-05-01T10:00:00.000Z', { error_categories: ['homophone'] })];
		const b = [ev(1, 'u1', '2026-05-01T11:00:00.000Z', { include_status: 'include' })];

		const { merged } = mergeEventLogs(a, b);

		expect(merged.map((e) => e.patch)).toEqual([
			{ error_categories: ['homophone'] },
			{ include_status: 'include' }
		]);
	});
});
