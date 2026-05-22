/**
 * @vitest-environment jsdom
 *
 * Client-side meeting-context tests. Use jsdom so we have a real `fetch`-able
 * global and the module under test can construct URLs the way it would in a
 * browser. Network is mocked via `globalThis.fetch`.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
	prefetch,
	getContext,
	mergeBySpeaker,
	hasMeeting,
	_reset,
	_stats
} from '../../src/lib/client/meeting-context.svelte';
import type { ContextUtterance } from '../../src/lib/domain/meeting-context';

const FAKE_MEETING = {
	transcript: [
		{
			id: 'seg-1',
			startTimestamp: 0,
			endTimestamp: 30,
			speakerTagId: 'tag-A',
			utterances: [
				{ id: 'u-p2', startTimestamp: 1, endTimestamp: 2, text: 'before-2', speakerSegmentId: 'seg-1' },
				{ id: 'u-p1', startTimestamp: 3, endTimestamp: 4, text: 'before-1', speakerSegmentId: 'seg-1' },
				{ id: 'u-target', startTimestamp: 12.5, endTimestamp: 15, text: 'current', speakerSegmentId: 'seg-1' }
			]
		},
		{
			id: 'seg-2',
			startTimestamp: 30,
			endTimestamp: 60,
			speakerTagId: 'tag-B',
			utterances: [
				{ id: 'u-n1', startTimestamp: 31, endTimestamp: 32, text: 'after-1', speakerSegmentId: 'seg-2' }
			]
		}
	],
	people: [
		{ id: 'p1', name: 'Maria Papadopoulos', name_short: 'Μ. Παπαδ.' }
	],
	speakerTags: [
		{ id: 'tag-A', label: 'SPEAKER_1', personId: 'p1' },
		{ id: 'tag-B', label: 'SPEAKER_2', personId: null }
	]
};

function installFetch(impl: (url: string) => Promise<Response>) {
	(globalThis as { fetch: typeof fetch }).fetch = impl as unknown as typeof fetch;
}

beforeEach(() => {
	_reset();
});

describe('mergeBySpeaker', () => {
	const baseUtt = (over: Partial<ContextUtterance>): ContextUtterance => ({
		utterance_id: 'x',
		start: 0,
		end: 1,
		text: 'hi',
		speaker_label: null,
		speaker_person_id: null,
		speaker_name: null,
		is_current: false,
		same_speaker_as_current: false,
		...over
	});

	it('merges adjacent utterances sharing the same person_id', () => {
		const utts = [
			baseUtt({ utterance_id: 'a', text: 'foo', speaker_person_id: 'p1' }),
			baseUtt({ utterance_id: 'b', text: 'bar', speaker_person_id: 'p1' }),
			baseUtt({ utterance_id: 'c', text: 'baz', speaker_person_id: 'p2' })
		];
		const runs = mergeBySpeaker(utts);
		expect(runs).toHaveLength(2);
		expect(runs[0].text).toBe('foo bar');
		expect(runs[0].parts).toHaveLength(2);
		expect(runs[1].text).toBe('baz');
	});

	it('falls back to speaker_label when person_id is null', () => {
		const utts = [
			baseUtt({ utterance_id: 'a', text: 'foo', speaker_label: 'SPEAKER_2' }),
			baseUtt({ utterance_id: 'b', text: 'bar', speaker_label: 'SPEAKER_2' })
		];
		const runs = mergeBySpeaker(utts);
		expect(runs).toHaveLength(1);
		expect(runs[0].text).toBe('foo bar');
	});

	it('does NOT merge across speaker boundaries even if labels match by coincidence in null cases', () => {
		const utts = [
			baseUtt({ utterance_id: 'a', text: 'foo', speaker_person_id: 'p1', speaker_label: 'SPEAKER_1' }),
			baseUtt({ utterance_id: 'b', text: 'bar', speaker_person_id: null, speaker_label: 'SPEAKER_1' })
		];
		const runs = mergeBySpeaker(utts);
		// `a` has person_id, `b` does not — the merge rule requires both to
		// share a non-null person_id OR both to be null (and share label).
		expect(runs).toHaveLength(2);
	});
});

describe('getContext (client cache)', () => {
	it('fetches once for the same meeting, slices ±radius around the target', async () => {
		let calls = 0;
		installFetch(async () => {
			calls++;
			return new Response(JSON.stringify(FAKE_MEETING), {
				status: 200,
				headers: { 'content-type': 'application/json' }
			});
		});

		const a = await getContext('athens', 'feb01_2026', 'u-target', 3);
		const b = await getContext('athens', 'feb01_2026', 'u-target', 3);

		expect(a.error).toBeNull();
		expect(a.current?.utterance_id).toBe('u-target');
		expect(a.prev.map((u) => u.utterance_id)).toEqual(['u-p2', 'u-p1']);
		expect(a.next.map((u) => u.utterance_id)).toEqual(['u-n1']);
		expect(b.current?.utterance_id).toBe('u-target');
		expect(calls).toBe(1);
	});

	it('resolves speaker_name via speakerTag → person', async () => {
		installFetch(async () =>
			new Response(JSON.stringify(FAKE_MEETING), { status: 200 })
		);
		const ctx = await getContext('athens', 'feb01_2026', 'u-target', 2);
		expect(ctx.current?.speaker_name).toBe('Μ. Παπαδ.');
		expect(ctx.current?.speaker_label).toBe('SPEAKER_1');
		expect(ctx.next[0]?.speaker_name).toBeNull();
		expect(ctx.next[0]?.speaker_label).toBe('SPEAKER_2');
	});

	it('returns a context with an error when the upstream is unreachable', async () => {
		installFetch(async () => new Response('boom', { status: 502 }));
		const ctx = await getContext('athens', 'feb01_2026', 'u-target', 2);
		expect(ctx.error).toMatch(/upstream/);
		expect(ctx.prev).toEqual([]);
		expect(ctx.next).toEqual([]);
	});

	it('returns a context with an error when the utterance is unknown', async () => {
		installFetch(async () =>
			new Response(JSON.stringify(FAKE_MEETING), { status: 200 })
		);
		const ctx = await getContext('athens', 'feb01_2026', 'no-such-id', 2);
		expect(ctx.error).toMatch(/not found/);
	});

	it('prefetch warms the LRU without blocking', async () => {
		const seen = vi.fn();
		installFetch(async () => {
			seen();
			return new Response(JSON.stringify(FAKE_MEETING), { status: 200 });
		});
		prefetch('athens', 'feb01_2026');
		// Yield once so the queued fetch runs.
		await new Promise((r) => setTimeout(r, 0));
		expect(seen).toHaveBeenCalled();
		expect(_stats().size).toBe(1);
	});

	it('flags same_speaker_as_current across segment boundaries', async () => {
		installFetch(async () => new Response(JSON.stringify(FAKE_MEETING), { status: 200 }));
		const ctx = await getContext('athens', 'feb01_2026', 'u-target', 3);
		// Both prev share tag-A with current.
		expect(ctx.prev.every((u) => u.same_speaker_as_current)).toBe(true);
		// next is tag-B → not same speaker.
		expect(ctx.next.every((u) => !u.same_speaker_as_current)).toBe(true);
	});

	it('hasMeeting reports false before the fetch settles and true after a successful fetch', async () => {
		installFetch(async () =>
			new Response(JSON.stringify(FAKE_MEETING), { status: 200 })
		);
		expect(hasMeeting('athens', 'feb01_2026')).toBe(false);
		// Fire the fetch but check immediately — settle hasn't happened yet.
		prefetch('athens', 'feb01_2026');
		expect(hasMeeting('athens', 'feb01_2026')).toBe(false);
		// Yield to let the fetch resolve.
		await new Promise((r) => setTimeout(r, 0));
		expect(hasMeeting('athens', 'feb01_2026')).toBe(true);
	});

	it('hasMeeting stays false on upstream failure even though the LRU has the in-flight entry', async () => {
		installFetch(async () => new Response('boom', { status: 502 }));
		await getContext('athens', 'feb01_2026', 'u-target', 1);
		expect(hasMeeting('athens', 'feb01_2026')).toBe(false);
		expect(_stats().size).toBe(1); // dedup entry exists
		expect(_stats().resolved).toBe(0); // but resolved map is empty
	});

	it('LRU cap evicts both maps in lockstep', async () => {
		installFetch(async (url: string) => {
			// Echo a per-meeting payload so each cache entry is distinct.
			const meetingId = decodeURIComponent(url.split('/').pop() ?? 'unknown');
			return new Response(
				JSON.stringify({
					...FAKE_MEETING,
					transcript: [
						{
							...FAKE_MEETING.transcript[0],
							utterances: [
								{
									id: `u-${meetingId}`,
									startTimestamp: 0,
									endTimestamp: 1,
									text: meetingId,
									speakerSegmentId: 'seg-1'
								}
							]
						}
					]
				}),
				{ status: 200 }
			);
		});

		// Push 110 distinct meetings through the cache; expect only LRU_CAP=100
		// to stay around. Use small awaits so settles happen in order.
		for (let i = 0; i < 110; i++) {
			await getContext('athens', `meeting-${i}`, `u-meeting-${i}`, 0);
		}
		expect(_stats().size).toBeLessThanOrEqual(100);
		expect(_stats().resolved).toBeLessThanOrEqual(100);
		// The very first meeting should have been evicted by the time meeting-109 lands.
		expect(hasMeeting('athens', 'meeting-0')).toBe(false);
		// A late meeting should still be present.
		expect(hasMeeting('athens', 'meeting-109')).toBe(true);
	});
});
