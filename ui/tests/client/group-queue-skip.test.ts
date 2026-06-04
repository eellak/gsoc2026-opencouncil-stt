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
});
