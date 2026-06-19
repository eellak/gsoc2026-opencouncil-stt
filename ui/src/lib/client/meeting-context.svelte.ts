/**
 * Client-side surrounding-utterance context.
 *
 * Fetches just the N utterances before/after the target from the OpenCouncil
 * per-utterance context endpoint, relayed through /api/oc-context/{id} (a CORS
 * bridge — no slicing happens there). This replaces the old approach of
 * downloading the entire ~600 KB meeting transcript and slicing it locally;
 * payloads are now a few KB.
 *
 * The endpoint returns only `speakerTagId` per neighbour (no speaker names),
 * so we surface no human-readable speaker names. We still group adjacent
 * same-speaker utterances by stashing the tag id in `speaker_person_id`
 * (used purely as a grouping key by `mergeBySpeaker`; never rendered as a
 * name — the panel falls back to "—").
 *
 * Pairs with the audio prefetch sliding window: as the review page advances we
 * `prefetch(id)` upcoming utterances so their context is warm on arrival.
 *
 * SSR-safe: `fetch` works server-side too; the module never touches `window`.
 */

import type { MeetingContext, ContextUtterance } from '$lib/domain/meeting-context';

// Each context response is a few KB. 200 entries is a comfortably small cache
// that survives crossing a seeded window without re-fetching.
const LRU_CAP = 200;
const MAX_NEIGHBOURS = 50;
/** Radius used by background prefetch (matches the page's initial window). */
const PREFETCH_RADIUS = 5;

interface OcNeighbour {
	id: string;
	text: string;
	start: number;
	end: number;
	speakerTagId: string | null;
}

interface OcContextResponse {
	meeting?: { id?: string; cityId?: string; name?: string; dateTime?: string };
	before?: OcNeighbour[];
	after?: OcNeighbour[];
}

const lru = new Map<string, Promise<MeetingContext>>();
const resolved = new Map<string, MeetingContext>();

function clampRadius(n: number): number {
	if (!Number.isFinite(n)) return 0;
	return Math.max(0, Math.min(MAX_NEIGHBOURS, Math.floor(n)));
}

function key(utteranceId: string, before: number, after: number): string {
	return `${utteranceId}|${before}|${after}`;
}

function touchLru(k: string, value: Promise<MeetingContext>): void {
	lru.delete(k);
	lru.set(k, value);
	while (lru.size > LRU_CAP) {
		const oldest = lru.keys().next().value;
		if (oldest === undefined) break;
		lru.delete(oldest);
		resolved.delete(oldest);
	}
}

function toCtx(n: OcNeighbour): ContextUtterance {
	return {
		utterance_id: n.id,
		start: n.start,
		end: n.end,
		text: n.text,
		// Tag id drives same-speaker grouping only — never displayed as a name.
		speaker_label: null,
		speaker_person_id: n.speakerTagId ?? null,
		speaker_name: null,
		is_current: false,
		same_speaker_as_current: false
	};
}

// Transient failures (502/timeout/network) are usually a momentary upstream
// hiccup under a navigation burst and self-heal within ~1s (verified: a 502'd
// context returns 200 on the next fetch). Auto-retry a couple of times with a
// short backoff so the panel loads instead of flashing "context unavailable".
// A 404/401/403 (private) is permanent — return immediately, never retry.
const CONTEXT_MAX_ATTEMPTS = 3; // 1 try + 2 retries
const CONTEXT_RETRY_BASE_MS = 200;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function fetchContext(
	utteranceId: string,
	before: number,
	after: number
): Promise<MeetingContext> {
	const base: MeetingContext = {
		city_id: '',
		meeting_id: '',
		current: null,
		prev: [],
		next: [],
		error: null,
		error_kind: null
	};
	const url = `/api/oc-context/${encodeURIComponent(utteranceId)}?before=${before}&after=${after}`;

	for (let attempt = 0; attempt < CONTEXT_MAX_ATTEMPTS; attempt++) {
		if (attempt > 0) await sleep(CONTEXT_RETRY_BASE_MS * attempt);
		try {
			const resp = await fetch(url);
			if (!resp.ok) {
				// The bridge passes through 401/403/404 only for genuine upstream
				// auth/not-found responses — those are "private" (skippable) and must
				// NOT be retried. Everything else (502/timeout/network) is transient.
				const isPrivate = resp.status === 401 || resp.status === 403 || resp.status === 404;
				if (isPrivate) {
					return {
						...base,
						error: resp.status === 404 ? 'utterance not found' : 'upstream context fetch failed',
						error_kind: 'private'
					};
				}
				continue; // transient → retry
			}
			const json = (await resp.json()) as OcContextResponse;
			return {
				...base,
				city_id: json.meeting?.cityId ?? '',
				meeting_id: json.meeting?.id ?? '',
				// `before`/`after` arrive chronological (oldest→newest); map straight
				// to prev/next around the (locally-known) current utterance.
				prev: (json.before ?? []).map(toCtx),
				next: (json.after ?? []).map(toCtx)
			};
		} catch {
			// Network error / abort — never private; fall through to retry.
		}
	}
	// Exhausted retries — transient failure, evicted by loadOnce so it stays retryable.
	return { ...base, error: 'upstream context fetch failed', error_kind: 'transient' };
}

function loadOnce(utteranceId: string, before: number, after: number): Promise<MeetingContext> {
	if (!utteranceId) {
		return Promise.resolve({
			city_id: '', meeting_id: '', current: null, prev: [], next: [], error: 'missing utterance id', error_kind: 'transient'
		});
	}
	const b = clampRadius(before);
	const a = clampRadius(after);
	const k = key(utteranceId, b, a);
	const existing = lru.get(k);
	if (existing) {
		touchLru(k, existing);
		return existing;
	}
	const promise = fetchContext(utteranceId, b, a).then((ctx) => {
		if (!ctx.error && lru.has(k)) {
			resolved.set(k, ctx);
		} else if (ctx.error && lru.get(k) === promise) {
			// Evict the failed lookup so a later getContext/prefetch refetches
			// instead of replaying this cached error. Guard on identity so we
			// don't clobber a newer in-flight fetch for the same key.
			lru.delete(k);
		}
		return ctx;
	});
	touchLru(k, promise);
	return promise;
}

/**
 * Synchronously check whether this exact (id, before, after) context is already
 * resolved in cache — lets the review page skip the "loading…" flicker on hits.
 */
export function hasContext(utteranceId: string, before: number, after: number): boolean {
	if (!utteranceId) return false;
	return resolved.has(key(utteranceId, clampRadius(before), clampRadius(after)));
}

/** Kick off a background fetch of an utterance's default-radius context. */
export function prefetch(utteranceId: string | null | undefined): void {
	if (!utteranceId) return;
	void loadOnce(utteranceId, PREFETCH_RADIUS, PREFETCH_RADIUS);
}

/**
 * Fetch (or reuse cached) context: `before` utterances before and `after`
 * after the target. Resolves with `error` populated on failure.
 */
export function getContext(
	utteranceId: string,
	before: number,
	after: number
): Promise<MeetingContext> {
	return loadOnce(utteranceId, before, after);
}

/** Test/debug only. */
export function _stats(): { size: number; resolved: number } {
	return { size: lru.size, resolved: resolved.size };
}
export function _reset(): void {
	lru.clear();
	resolved.clear();
}

/**
 * Merge runs of adjacent same-speaker utterances into one entry. Two
 * utterances merge when they share a `speaker_person_id` (when both are
 * non-null) or, as a fallback, a `speaker_label`. Texts are joined with a
 * space; the run carries the earliest start timestamp.
 */
export interface MergedRun {
	speaker_name: string | null;
	speaker_label: string | null;
	speaker_person_id: string | null;
	start: number;
	text: string;
	same_speaker_as_current: boolean;
	parts: ContextUtterance[];
}

export function mergeBySpeaker(utts: ContextUtterance[]): MergedRun[] {
	const out: MergedRun[] = [];
	for (const u of utts) {
		const last = out[out.length - 1];
		const sameRunKey =
			(u.speaker_person_id && last?.speaker_person_id === u.speaker_person_id) ||
			(!u.speaker_person_id &&
				!last?.speaker_person_id &&
				last?.speaker_label === u.speaker_label);
		if (last && sameRunKey) {
			last.text = `${last.text} ${u.text}`.trim();
			last.parts.push(u);
		} else {
			out.push({
				speaker_name: u.speaker_name,
				speaker_label: u.speaker_label,
				speaker_person_id: u.speaker_person_id,
				start: u.start,
				text: u.text,
				same_speaker_as_current: u.same_speaker_as_current,
				parts: [u]
			});
		}
	}
	return out;
}
