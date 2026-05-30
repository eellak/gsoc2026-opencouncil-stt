/**
 * @vitest-environment jsdom
 *
 * Client-side meeting-context tests. The module now fetches per-utterance
 * context from /api/oc-context/{id}?before&after (a few KB) instead of the
 * whole meeting JSON. Network is mocked via `globalThis.fetch`.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
	prefetch,
	getContext,
	mergeBySpeaker,
	hasContext,
	_reset,
	_stats
} from '../../src/lib/client/meeting-context.svelte';
import type { ContextUtterance } from '../../src/lib/domain/meeting-context';

const FAKE_CONTEXT = {
	meeting: { id: 'feb01_2026', cityId: 'athens', name: 'Council', dateTime: '2026-02-01T10:00:00.000Z' },
	before: [
		{ id: 'u-p2', text: 'before-2', start: 1, end: 2, speakerTagId: 'tag-A' },
		{ id: 'u-p1', text: 'before-1', start: 3, end: 4, speakerTagId: 'tag-A' }
	],
	after: [{ id: 'u-n1', text: 'after-1', start: 31, end: 32, speakerTagId: 'tag-B' }]
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

	it('merges adjacent utterances sharing the same person_id (tag id)', () => {
		const utts = [
			baseUtt({ utterance_id: 'a', text: 'foo', speaker_person_id: 'tag-A' }),
			baseUtt({ utterance_id: 'b', text: 'bar', speaker_person_id: 'tag-A' }),
			baseUtt({ utterance_id: 'c', text: 'baz', speaker_person_id: 'tag-B' })
		];
		const runs = mergeBySpeaker(utts);
		expect(runs).toHaveLength(2);
		expect(runs[0].text).toBe('foo bar');
		expect(runs[0].parts).toHaveLength(2);
		expect(runs[1].text).toBe('baz');
	});

	it('does NOT merge across different tag ids', () => {
		const utts = [
			baseUtt({ utterance_id: 'a', text: 'foo', speaker_person_id: 'tag-A' }),
			baseUtt({ utterance_id: 'b', text: 'bar', speaker_person_id: 'tag-B' })
		];
		expect(mergeBySpeaker(utts)).toHaveLength(2);
	});
});

describe('getContext (per-utterance endpoint)', () => {
	it('maps before→prev / after→next and caches per (id,before,after)', async () => {
		let calls = 0;
		installFetch(async () => {
			calls++;
			return new Response(JSON.stringify(FAKE_CONTEXT), {
				status: 200,
				headers: { 'content-type': 'application/json' }
			});
		});

		const a = await getContext('u-target', 3, 3);
		const b = await getContext('u-target', 3, 3);

		expect(a.error).toBeNull();
		expect(a.current).toBeNull(); // endpoint does not return the anchor
		expect(a.prev.map((u) => u.utterance_id)).toEqual(['u-p2', 'u-p1']);
		expect(a.next.map((u) => u.utterance_id)).toEqual(['u-n1']);
		expect(a.city_id).toBe('athens');
		expect(a.meeting_id).toBe('feb01_2026');
		expect(b.prev).toHaveLength(2);
		expect(calls).toBe(1); // second call served from cache
	});

	it('re-fetches when the window (before/after) grows', async () => {
		let calls = 0;
		installFetch(async () => {
			calls++;
			return new Response(JSON.stringify(FAKE_CONTEXT), { status: 200 });
		});
		await getContext('u-target', 5, 5);
		await getContext('u-target', 10, 10);
		expect(calls).toBe(2);
	});

	it('stashes speakerTagId in speaker_person_id and leaves names null', async () => {
		installFetch(async () => new Response(JSON.stringify(FAKE_CONTEXT), { status: 200 }));
		const ctx = await getContext('u-target', 2, 2);
		expect(ctx.prev[0]?.speaker_person_id).toBe('tag-A');
		expect(ctx.prev[0]?.speaker_name).toBeNull();
		expect(ctx.prev[0]?.speaker_label).toBeNull();
		expect(ctx.next[0]?.speaker_person_id).toBe('tag-B');
	});

	it('returns an error context when the upstream is unreachable', async () => {
		installFetch(async () => new Response('boom', { status: 502 }));
		const ctx = await getContext('u-target', 2, 2);
		expect(ctx.error).toMatch(/upstream/);
		expect(ctx.prev).toEqual([]);
		expect(ctx.next).toEqual([]);
	});

	it('maps 404 to a not-found error', async () => {
		installFetch(async () => new Response('nope', { status: 404 }));
		const ctx = await getContext('no-such-id', 2, 2);
		expect(ctx.error).toMatch(/not found/);
	});

	it('prefetch warms the cache without blocking', async () => {
		const seen = vi.fn();
		installFetch(async () => {
			seen();
			return new Response(JSON.stringify(FAKE_CONTEXT), { status: 200 });
		});
		prefetch('u-target');
		await new Promise((r) => setTimeout(r, 0));
		expect(seen).toHaveBeenCalled();
		expect(_stats().size).toBe(1);
	});

	it('hasContext is false before settle and true after a successful fetch', async () => {
		installFetch(async () => new Response(JSON.stringify(FAKE_CONTEXT), { status: 200 }));
		expect(hasContext('u-target', 5, 5)).toBe(false);
		prefetch('u-target'); // default radius 5/5
		expect(hasContext('u-target', 5, 5)).toBe(false);
		await new Promise((r) => setTimeout(r, 0));
		expect(hasContext('u-target', 5, 5)).toBe(true);
	});

	it('evicts a failed lookup so it stays retryable', async () => {
		let calls = 0;
		installFetch(async () => {
			calls++;
			// First call fails, second succeeds.
			return calls === 1
				? new Response('boom', { status: 502 })
				: new Response(JSON.stringify(FAKE_CONTEXT), { status: 200 });
		});
		const first = await getContext('u-target', 1, 1);
		expect(first.error).toMatch(/upstream/);
		expect(hasContext('u-target', 1, 1)).toBe(false);
		expect(_stats().size).toBe(0); // failed entry evicted, not cached
		expect(_stats().resolved).toBe(0);
		// Retry refetches and succeeds.
		const second = await getContext('u-target', 1, 1);
		expect(second.error).toBeNull();
		expect(calls).toBe(2);
	});

	it('LRU cap evicts both maps in lockstep', async () => {
		installFetch(async (url: string) => {
			const id = decodeURIComponent(url.split('/').pop()?.split('?')[0] ?? 'unknown');
			return new Response(
				JSON.stringify({
					meeting: { id: 'm', cityId: 'athens' },
					before: [{ id: `b-${id}`, text: id, start: 0, end: 1, speakerTagId: 'tag-A' }],
					after: []
				}),
				{ status: 200 }
			);
		});
		for (let i = 0; i < 210; i++) {
			await getContext(`u-${i}`, 0, 0);
		}
		expect(_stats().size).toBeLessThanOrEqual(200);
		expect(_stats().resolved).toBeLessThanOrEqual(200);
		expect(hasContext('u-0', 0, 0)).toBe(false);
		expect(hasContext('u-209', 0, 0)).toBe(true);
	});
});
