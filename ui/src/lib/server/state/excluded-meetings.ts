/**
 * Unreviewed-meeting denylist (the 13 meetings with human-edit fraction <5% as of
 * 2026-06-23 — essentially never reviewed, so their transcripts and even their
 * individually-`include`d corrections are untrustworthy). Canonical source of
 * truth: `data/exclusions/unreviewed_meetings.json` at the repo root, shared with
 * the Python pipeline (`eval/exclusions.py`) — no duplicated constants.
 *
 * Keyed strictly by (city_id, meeting_id) via `meetingKey` — meeting_id slugs
 * collide across cities, so `rhodes/jul17_2025` must be excluded without touching
 * a different city that reuses the same slug.
 *
 * Reversible like the other filters: set `DISABLE_UNREVIEWED_MEETING_EXCLUSIONS=1`
 * to return an empty set (raw build), nothing is deleted, `getGroup(id)` still
 * resolves an excluded utterance.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { meetingKey } from './meeting-eligibility';

interface ExclusionFile {
	meetings?: Array<{ city_id: string; meeting_id: string }>;
}

function exclusionsDisabled(): boolean {
	const raw = (process.env.DISABLE_UNREVIEWED_MEETING_EXCLUSIONS ?? '').trim().toLowerCase();
	return !['', '0', 'false', 'no', 'off'].includes(raw);
}

function exclusionsPath(): string {
	return (
		process.env.MEETING_EXCLUSIONS_PATH ??
		// .cache is resolved from cwd (= ui/), so the repo root is one level up.
		resolve(process.cwd(), '..', 'data', 'exclusions', 'unreviewed_meetings.json')
	);
}

let cached: Set<string> | null = null;

/** Set of excluded `meetingKey(city, meeting)` strings. Loaded once, memoised. */
export function excludedMeetingKeys(): Set<string> {
	if (cached) return cached;
	if (exclusionsDisabled()) {
		cached = new Set();
		return cached;
	}
	try {
		const parsed = JSON.parse(readFileSync(exclusionsPath(), 'utf8')) as ExclusionFile;
		cached = new Set((parsed.meetings ?? []).map((m) => meetingKey(m.city_id, m.meeting_id)));
		console.log(`[excluded-meetings] loaded ${cached.size} denylisted meetings`);
	} catch (err) {
		// Fail open but loud: the Python consumers also filter, so a missing file
		// here doesn't silently ship junk to training — but we want to notice.
		console.warn('[excluded-meetings] could not load denylist, none excluded', err);
		cached = new Set();
	}
	return cached;
}

export function isExcludedMeeting(
	city_id: string | null | undefined,
	meeting_id: string | null | undefined
): boolean {
	return excludedMeetingKeys().has(meetingKey(city_id, meeting_id));
}
