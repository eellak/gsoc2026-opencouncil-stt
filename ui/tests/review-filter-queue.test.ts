import { describe, it, expect } from 'vitest';
import {
	scanFilteredQueue,
	firstFilteredId,
	type SeededQueueSource
} from '../src/lib/server/state/review-filter-queue';
import type { Group, GroupEdit, QueueResponse } from '../src/lib/domain/groups';
import { DEFAULT_LABEL } from '../src/lib/domain/groups';
import type { ReviewFilterSpec } from '../src/lib/shared/review-filters';

function mkGroup(id: string, source: 'task' | 'user'): Group {
	const e: GroupEdit = {
		edit_id: id + '-e',
		edit_timestamp: '2024-01-01T00:00:00Z',
		edit_updated_at: null,
		before_text: 'να γίνη',
		after_text: 'να γίνει',
		edited_by: source,
		utterance_start: 0,
		utterance_end: 1,
		ingest_category: 'clean',
		cleaning_applied: '',
		csv_row: 0
	};
	return {
		utterance_id: id,
		meeting_id: 'm',
		city_id: 'c',
		meeting_name: null,
		meeting_date: null,
		audio_url: '',
		audio_cdn_url: null,
		youtube_url: null,
		start: 0,
		end: 1,
		initial_before_text: 'να γίνη',
		final_after_text: 'να γίνει',
		edits: [e],
		chain_consistent: true,
		label: { ...DEFAULT_LABEL }
	};
}

/** A fake seeded source over a fixed ordered list, paging in batches of ≤50. */
function fakeRepo(order: Group[]): SeededQueueSource {
	return {
		queue(_seed: number, from: number, n: number): QueueResponse {
			const start = Math.max(0, Math.min(order.length, from));
			const count = Math.max(0, Math.min(50, n));
			const slice = order.slice(start, start + count);
			const next = start + count < order.length ? start + count : null;
			return { cache_hash: 'h', total: order.length, groups: slice, next_cursor: next };
		}
	};
}

const userOnly: ReviewFilterSpec = { dropPunctOnly: false, sources: ['user'] };

describe('scanFilteredQueue cursor contract', () => {
	it('collects up to n matches and advances cursor (not exhausted)', () => {
		// 200 groups alternating task/user; user matches.
		const order = Array.from({ length: 200 }, (_, i) =>
			mkGroup('g' + i, i % 2 === 0 ? 'task' : 'user')
		);
		const res = scanFilteredQueue(fakeRepo(order), 1, 0, 11, userOnly);
		// Stops once a batch pushes the count to ≥ want; may overshoot within the
		// batch (no data loss), but never returns fewer than requested mid-stream.
		expect(res.groups.length).toBeGreaterThanOrEqual(11);
		expect(res.groups.length).toBeLessThanOrEqual(50);
		expect(res.groups.every((g) => g.edits[0].edited_by === 'user')).toBe(true);
		expect(res.exhausted).toBe(false);
		expect(res.next_cursor).not.toBeNull();
	});

	it('sets exhausted=true and null cursor at the true end', () => {
		const order = Array.from({ length: 10 }, (_, i) => mkGroup('g' + i, 'user'));
		const res = scanFilteredQueue(fakeRepo(order), 1, 0, 50, userOnly);
		expect(res.groups).toHaveLength(10);
		expect(res.exhausted).toBe(true);
		expect(res.next_cursor).toBeNull();
	});

	it('cap hit before a match → empty groups, non-null cursor, not exhausted', () => {
		// All task; user filter matches nothing. cap=50 → one batch, no match.
		const order = Array.from({ length: 500 }, (_, i) => mkGroup('g' + i, 'task'));
		const res = scanFilteredQueue(fakeRepo(order), 1, 0, 11, userOnly, { cap: 50 });
		expect(res.groups).toHaveLength(0);
		expect(res.exhausted).toBe(false);
		expect(res.next_cursor).toBe(50);
	});

	it('empty params spec returns everything (passthrough semantics)', () => {
		const order = Array.from({ length: 5 }, (_, i) => mkGroup('g' + i, i % 2 ? 'user' : 'task'));
		const res = scanFilteredQueue(fakeRepo(order), 1, 0, 50, {
			dropPunctOnly: false,
			sources: null
		});
		expect(res.groups).toHaveLength(5);
		expect(res.exhausted).toBe(true);
	});

	it('accept gate narrows further (e.g. skip classified)', () => {
		const order = Array.from({ length: 6 }, (_, i) => mkGroup('g' + i, 'user'));
		order[0].label = { ...DEFAULT_LABEL, include_status: 'include' };
		const res = scanFilteredQueue(fakeRepo(order), 1, 0, 50, userOnly, {
			accept: (g) => g.label.include_status === 'unreviewed'
		});
		expect(res.groups.map((g) => g.utterance_id)).not.toContain('g0');
		expect(res.groups).toHaveLength(5);
	});
});

describe('firstFilteredId', () => {
	it('finds the first match across capped passes', () => {
		// 120 task then 1 user: cap=50 means it takes 3 passes to reach the user.
		const order = [
			...Array.from({ length: 120 }, (_, i) => mkGroup('t' + i, 'task')),
			mkGroup('the-user', 'user')
		];
		expect(firstFilteredId(fakeRepo(order), 1, userOnly, { cap: 50 })).toBe('the-user');
	});

	it('returns null when nothing matches', () => {
		const order = Array.from({ length: 30 }, (_, i) => mkGroup('t' + i, 'task'));
		expect(firstFilteredId(fakeRepo(order), 1, userOnly)).toBeNull();
	});

	it('maxScan bounds the search and gives up early', () => {
		// A matching user item exists only at index 300; with maxScan=100 we bail.
		const order = [
			...Array.from({ length: 300 }, (_, i) => mkGroup('t' + i, 'task')),
			mkGroup('late-user', 'user')
		];
		expect(firstFilteredId(fakeRepo(order), 1, userOnly, { maxScan: 100 })).toBeNull();
		// Without the bound it is found.
		expect(firstFilteredId(fakeRepo(order), 1, userOnly)).toBe('late-user');
	});
});
