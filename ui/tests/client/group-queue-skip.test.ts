/**
 * Skip-classified navigation over the seeded queue (resume / skip-flagged work).
 *
 * Locks in: next/prev jump over already-classified items, the tail of the queue
 * yields no navigation (never falls back to a classified neighbour), and prev
 * skipping works within the retained window.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import type { Group } from '../../src/lib/domain/groups';
import {
	nextUnreviewedId,
	nextUnreviewedIdLoaded,
	prevUnreviewedId,
	nextIdOf,
	_resetForTest,
	_loadSeededForTest
} from '../../src/lib/client/group-queue.svelte';
import type { IncludeStatus } from '../../src/lib/domain/types';

function g(id: string, status: IncludeStatus): Group {
	return {
		utterance_id: id,
		label: {
			include_status: status,
			error_categories: [],
			adjusted_start: null,
			adjusted_end: null,
			reviewer_notes: null,
			human_updated_at: null
		}
	} as unknown as Group;
}

beforeEach(() => _resetForTest());

describe('seeded skip navigation', () => {
	it('next jumps over classified items to the next unreviewed', async () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'include'), g('C', 'exclude'), g('D', 'unreviewed')],
			null
		);
		expect(await nextUnreviewedId('A')).toBe('D');
		expect(nextUnreviewedIdLoaded('A')).toBe('D');
	});

	it('prev jumps back over classified items to the previous unreviewed', () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'include'), g('C', 'exclude'), g('D', 'unreviewed')],
			null
		);
		expect(prevUnreviewedId('D')).toBe('A');
	});

	it('returns undefined at the tail instead of a classified neighbour', async () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'include'), g('C', 'exclude')],
			null
		);
		// Nothing unreviewed ahead of A and paging exhausted → no navigation.
		expect(await nextUnreviewedId('A')).toBe(undefined);
		expect(nextUnreviewedIdLoaded('A')).toBe(undefined);
	});

	it('prev returns undefined when there is no earlier unreviewed item', () => {
		_loadSeededForTest(
			[g('A', 'include'), g('B', 'exclude'), g('C', 'unreviewed')],
			null
		);
		expect(prevUnreviewedId('C')).toBe(undefined);
	});

	// The loaded-window skip (drives href/gating) is distinct from the raw
	// neighbour: when everything ahead is classified, the skip-aware target is
	// undefined while the raw neighbour still exists. The 404 auto-skip relies on
	// the async paging walk (nextUnreviewedId), NOT on the raw neighbour, so it
	// never lands on a classified item — see autoSkipPrivate / resolveAutoSkipTargetId.
	it('loaded-window skip is empty while the raw neighbour still exists', () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'include'), g('C', 'exclude')],
			null
		);
		expect(nextUnreviewedIdLoaded('A')).toBe(undefined);
		expect(nextIdOf('A')).toBe('B'); // raw neighbour — deliberately NOT used by auto-skip
	});
});
