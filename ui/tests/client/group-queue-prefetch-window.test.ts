/**
 * prefetchWindow: forward-biased, skip-aware warm window for audio/transcript
 * prefetch. Locks in that prefetch follows the navigation path (not raw ±radius)
 * so it warms what the reviewer actually lands on, walks-to-fill past
 * classified/holes, and never skips on the backward side.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import type { Group } from '../../src/lib/domain/groups';
import {
	prefetchWindow,
	_resetForTest,
	_loadSeededForTest,
	_setSeededOrderAndCacheForTest
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

const ids = (gs: Group[]) => gs.map((x) => x.utterance_id);

beforeEach(() => _resetForTest());

describe('prefetchWindow', () => {
	it('forward-biased: returns up to `forward` ahead then `back` behind, in that order', () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'unreviewed'), g('C', 'unreviewed'), g('D', 'unreviewed'), g('E', 'unreviewed')],
			null
		);
		// current C: forward 2 (D,E), back 1 (B)
		expect(ids(prefetchWindow('C', 2, 1, false))).toEqual(['D', 'E', 'B']);
	});

	it('forward skips classified only when skipClassified=true, walking to fill the budget', () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'include'), g('C', 'exclude'), g('D', 'unreviewed'), g('E', 'unreviewed')],
			null
		);
		// skip on: walk past B,C to fill 2 → D,E
		expect(ids(prefetchWindow('A', 2, 0, true))).toEqual(['D', 'E']);
		// skip off: raw forward → B,C
		expect(ids(prefetchWindow('A', 2, 0, false))).toEqual(['B', 'C']);
	});

	it('backward never skips classified, even when skipClassified=true', () => {
		_loadSeededForTest(
			[g('A', 'unreviewed'), g('B', 'include'), g('C', 'exclude'), g('D', 'unreviewed')],
			null
		);
		// from D, back 2 → C, B (classified, but prev never skips)
		expect(ids(prefetchWindow('D', 0, 2, true))).toEqual(['C', 'B']);
	});

	it('stops cleanly at queue boundaries', () => {
		_loadSeededForTest([g('A', 'unreviewed'), g('B', 'unreviewed')], null);
		expect(ids(prefetchWindow('B', 3, 3, false))).toEqual(['A']); // nothing ahead, one behind
		expect(prefetchWindow('A', 3, 3, false).map((x) => x.utterance_id)).toEqual(['B']);
	});

	it('cached-only: walks past cache holes to fill the forward budget', () => {
		// order has ids 1..8; only 2,3,5,6,8 are cached; 4 and 6 are classified.
		_setSeededOrderAndCacheForTest(
			['1', '2', '3', '4', '5', '6', '7', '8'],
			[g('2', 'unreviewed'), g('3', 'unreviewed'), g('5', 'unreviewed'), g('6', 'include'), g('8', 'unreviewed')]
		);
		// from 3, skip on, forward 3: 4 hole→skip, 5 cached unrev→keep, 6 classified→skip,
		// 7 hole→skip, 8 cached unrev→keep. Only 2 eligible cached ahead.
		expect(ids(prefetchWindow('3', 3, 0, true))).toEqual(['5', '8']);
	});

	it('returns [] when the id is not in the order', () => {
		_loadSeededForTest([g('A', 'unreviewed')], null);
		expect(prefetchWindow('ZZZ', 3, 3, false)).toEqual([]);
	});
});
