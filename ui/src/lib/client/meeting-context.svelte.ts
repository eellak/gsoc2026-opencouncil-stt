/**
 * Client-side surrounding-utterance context.
 *
 * The browser fetches the full meeting JSON for each meeting we need
 * (relayed via /api/oc-meeting/{city}/{meeting}, which is just a CORS
 * bridge — no slicing or caching happens there), caches it in memory under
 * an LRU cap, and slices it locally to produce ±radius context for any
 * `utterance_id` known to be in that meeting.
 *
 * Pairs with the audio prefetch sliding window: every time the review page
 * advances, we also call `prefetch(neighbours)` so that the meeting JSON for
 * upcoming utterances is already in memory when the user lands on them.
 *
 * SSR-safe: `fetch` works server-side too, but the singleton is only
 * meaningful in the browser. The module never touches `window` directly so
 * importing it during SSR is harmless.
 */

import type { MeetingContext, ContextUtterance } from '$lib/domain/meeting-context';

// Each meeting JSON is ~600 KB. 100 entries ≈ 60 MB browser memory — well
// inside a modern Chromium tab budget but big enough that a reviewer
// crossing a whole seeded random window never sees a re-fetch within one
// session. The LRU still evicts oldest beyond the cap so memory is bounded.
const LRU_CAP = 100;

interface OcUtterance {
	id: string;
	startTimestamp: number;
	endTimestamp: number;
	text: string;
	speakerSegmentId: string;
}

interface OcSpeakerTag {
	id: string;
	label: string;
	personId: string | null;
}

interface OcSegment {
	id: string;
	startTimestamp: number;
	endTimestamp: number;
	speakerTagId: string;
	speakerTag?: OcSpeakerTag;
	utterances: OcUtterance[];
}

interface OcPerson {
	id: string;
	name?: string;
	name_short?: string;
}

interface OcMeeting {
	transcript: OcSegment[];
	people?: OcPerson[];
	speakerTags?: OcSpeakerTag[];
}

interface MeetingHandle {
	json: OcMeeting;
	utteranceIndex: Map<string, { segIdx: number; uttIdx: number }>;
	peopleById: Map<string, OcPerson>;
	speakerTagsById: Map<string, OcSpeakerTag>;
}

const lru = new Map<string, Promise<MeetingHandle | null>>();
// Parallel resolved-handle cache so callers (notably the review page's
// context effect) can synchronously decide whether a meeting is ready and
// avoid showing a `loading…` flicker for cache hits. Kept in lockstep with
// `lru` — both are populated on settle and evicted in touchLru below.
const resolved = new Map<string, MeetingHandle>();

function key(cityId: string, meetingId: string): string {
	return `${cityId}|${meetingId}`;
}

function touchLru(k: string, value: Promise<MeetingHandle | null>): void {
	lru.delete(k);
	lru.set(k, value);
	while (lru.size > LRU_CAP) {
		const oldest = lru.keys().next().value;
		if (oldest === undefined) break;
		lru.delete(oldest);
		resolved.delete(oldest);
	}
}

async function fetchMeeting(cityId: string, meetingId: string): Promise<MeetingHandle | null> {
	const url = `/api/oc-meeting/${encodeURIComponent(cityId)}/${encodeURIComponent(meetingId)}`;
	try {
		const resp = await fetch(url);
		if (!resp.ok) return null;
		const json = (await resp.json()) as OcMeeting;
		if (!json?.transcript) return null;

		const utteranceIndex = new Map<string, { segIdx: number; uttIdx: number }>();
		for (let s = 0; s < json.transcript.length; s++) {
			const seg = json.transcript[s];
			const utts = seg.utterances ?? [];
			for (let u = 0; u < utts.length; u++) {
				utteranceIndex.set(utts[u].id, { segIdx: s, uttIdx: u });
			}
		}
		const peopleById = new Map<string, OcPerson>();
		for (const p of json.people ?? []) peopleById.set(p.id, p);
		const speakerTagsById = new Map<string, OcSpeakerTag>();
		for (const t of json.speakerTags ?? []) speakerTagsById.set(t.id, t);

		return { json, utteranceIndex, peopleById, speakerTagsById };
	} catch {
		return null;
	}
}

function loadOnce(cityId: string, meetingId: string): Promise<MeetingHandle | null> {
	if (!cityId || !meetingId) return Promise.resolve(null);
	const k = key(cityId, meetingId);
	const existing = lru.get(k);
	if (existing) {
		// Touch order even on hit so the meeting stays warm.
		touchLru(k, existing);
		return existing;
	}
	const promise = fetchMeeting(cityId, meetingId).then((handle) => {
		// Keep the parallel resolved cache in sync. `null` results (network
		// failures) are intentionally not stored so a retry path stays open
		// — the lru entry is still there to dedup concurrent retries.
		if (handle) resolved.set(k, handle);
		return handle;
	});
	touchLru(k, promise);
	return promise;
}

/**
 * Synchronously check whether a meeting is already resolved in the cache.
 * Returns true only when a successful fetch has landed AND the entry hasn't
 * been evicted. Used by the review page to skip the `loading…` UI flicker on
 * cache hits.
 */
export function hasMeeting(
	cityId: string | null | undefined,
	meetingId: string | null | undefined
): boolean {
	if (!cityId || !meetingId) return false;
	return resolved.has(key(cityId, meetingId));
}

/**
 * Kick off a background fetch of the meeting JSON without waiting on the
 * result. Safe to call repeatedly — concurrent calls share the same
 * underlying promise.
 */
export function prefetch(cityId: string | null | undefined, meetingId: string | null | undefined): void {
	if (!cityId || !meetingId) return;
	// Force-load into the LRU; the returned promise is intentionally ignored.
	void loadOnce(cityId, meetingId);
}

function resolveSpeaker(
	handle: MeetingHandle,
	speakerTagId: string | null
): { speaker_label: string | null; speaker_person_id: string | null; speaker_name: string | null } {
	if (!speakerTagId) return { speaker_label: null, speaker_person_id: null, speaker_name: null };
	const tag = handle.speakerTagsById.get(speakerTagId) ?? null;
	const person = tag?.personId ? handle.peopleById.get(tag.personId) ?? null : null;
	return {
		speaker_label: tag?.label ?? null,
		speaker_person_id: tag?.personId ?? null,
		speaker_name: person?.name_short ?? person?.name ?? null
	};
}

/**
 * Fetch (or reuse cached) meeting JSON and return ±radius context around
 * `utteranceId`. Resolves to a context with `error` populated if the meeting
 * can't be loaded or the utterance isn't in it.
 */
export async function getContext(
	cityId: string | null | undefined,
	meetingId: string | null | undefined,
	utteranceId: string,
	radius: number
): Promise<MeetingContext> {
	const ctx: MeetingContext = {
		city_id: cityId ?? '',
		meeting_id: meetingId ?? '',
		current: null,
		prev: [],
		next: [],
		error: null
	};

	if (!cityId || !meetingId) {
		ctx.error = 'missing city/meeting id';
		return ctx;
	}

	const handle = await loadOnce(cityId, meetingId);
	if (!handle) {
		ctx.error = 'upstream meeting fetch failed';
		return ctx;
	}

	const ix = handle.utteranceIndex.get(utteranceId);
	if (!ix) {
		ctx.error = 'utterance not found in meeting transcript';
		return ctx;
	}

	// Build a flat list in transcript order so neighbour slicing is linear
	// across segment boundaries. Cheap on a single meeting (~hundreds of
	// utterances) and avoids building the same flat array on every call.
	interface Flat {
		utt: OcUtterance;
		speakerTagId: string | null;
	}
	const flat: Flat[] = [];
	let targetIdx = -1;
	for (const seg of handle.json.transcript) {
		const tagId = seg.speakerTagId ?? seg.speakerTag?.id ?? null;
		for (const u of seg.utterances ?? []) {
			if (u.id === utteranceId) targetIdx = flat.length;
			flat.push({ utt: u, speakerTagId: tagId });
		}
	}
	if (targetIdx < 0) {
		ctx.error = 'utterance not found in meeting transcript';
		return ctx;
	}

	const safeRadius = Math.max(0, Math.min(20, Math.floor(radius)));
	const lo = Math.max(0, targetIdx - safeRadius);
	const hi = Math.min(flat.length, targetIdx + safeRadius + 1);
	const currentTagId = flat[targetIdx].speakerTagId;

	const toCtx = (f: Flat, isCurrent: boolean): ContextUtterance => {
		const sp = resolveSpeaker(handle, f.speakerTagId);
		return {
			utterance_id: f.utt.id,
			start: f.utt.startTimestamp,
			end: f.utt.endTimestamp,
			text: f.utt.text,
			speaker_label: sp.speaker_label,
			speaker_person_id: sp.speaker_person_id,
			speaker_name: sp.speaker_name,
			is_current: isCurrent,
			same_speaker_as_current: !!(currentTagId && f.speakerTagId === currentTagId)
		};
	};

	for (let i = lo; i < targetIdx; i++) ctx.prev.push(toCtx(flat[i], false));
	ctx.current = toCtx(flat[targetIdx], true);
	for (let i = targetIdx + 1; i < hi; i++) ctx.next.push(toCtx(flat[i], false));

	return ctx;
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
