/**
 * Meeting eligibility filter.
 *
 * A meeting only enters the review queue once it has at least THRESHOLD
 * *human-corrected utterances* — distinct utterances carrying at least one edit
 * with `edited_by === 'user'`. Meetings whose edits are all machine/`task`
 * generated (or that have fewer than THRESHOLD human-touched utterances) are
 * filtered out of navigation, stats, and filter queues. Nothing is deleted:
 * `getGroup(id)` still resolves an excluded utterance, and label state in
 * `.state/` is never touched — so the filter is fully reversible (change the
 * threshold, or set it to 0 to disable).
 *
 * Computing per-meeting human-utterance counts means parsing every group's
 * edits (~17 s over ~287 k groups on the Oracle VM), so we do it once, persist
 * to `ui/.state/meeting-eligibility.snapshot.json`, and reload thereafter. The
 * snapshot is keyed by BOTH `cache_hash` (source data identity) AND `threshold`
 * (policy) — a rebuilt corpus or a changed threshold invalidates it.
 *
 * Mirrors the category-cache / stats-cache scan discipline: point queries (not
 * a long-lived cursor), yield to the event loop periodically, and serialise
 * against the other heavy scans via runExclusiveScan so the 1 GB VM never holds
 * two big scans at once.
 */

import { promises as fs } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Group } from '$lib/domain/groups';
import { runExclusiveScan } from './scan-lock';
import { excludedMeetingKeys } from './excluded-meetings';

/**
 * Remove denylisted (unreviewed, <5% human-edit) meetings from an eligible set.
 * Applied on read, not baked into the persisted snapshot, so the denylist can
 * change without invalidating the expensive eligibility scan.
 */
function dropExcluded(eligible: Set<string>): Set<string> {
	const excluded = excludedMeetingKeys();
	if (excluded.size === 0) return eligible;
	let removed = 0;
	for (const k of excluded) if (eligible.delete(k)) removed++;
	if (removed > 0) console.log(`[meeting-eligibility] dropped ${removed} denylisted meetings`);
	return eligible;
}

const YIELD_EVERY = 5_000;
const yieldToEventLoop = () => new Promise<void>((r) => setImmediate(r));

const DEFAULT_THRESHOLD = 10;

/**
 * Stable key for a meeting. `meeting_id` is NOT globally unique — the same slug
 * (e.g. "feb26_2025") is reused across cities (105 such collisions in the
 * corpus), so two different councils can share it. The real meeting identity is
 * the (city_id, meeting_id) pair; keying by `meeting_id` alone merges distinct
 * meetings and miscounts. The ` ` separator can't appear in a slug.
 */
export function meetingKey(
	city_id: string | null | undefined,
	meeting_id: string | null | undefined
): string {
	return `${city_id ?? ''} ${meeting_id ?? ''}`;
}

/** Minimal repo surface the eligibility scan needs. Both repos satisfy it. */
export interface EligibilityScanRepo {
	readonly hash: string;
	allOrderedIds(): readonly string[];
	getGroup(utterance_id: string): Group | null;
}

interface EligibilitySnapshot {
	cache_hash: string;
	threshold: number;
	computedAt: number;
	/** Eligible meeting keys (see `meetingKey` — `${city_id} ${meeting_id}`). */
	eligible_meeting_keys: string[];
}

/**
 * Threshold from `MEETING_MIN_HUMAN_UTTERANCES` (default 10). A value of 0 (or
 * any negative / non-numeric junk that resolves to ≤ 0) disables the filter.
 */
export function meetingEligibilityThreshold(): number {
	const raw = process.env.MEETING_MIN_HUMAN_UTTERANCES;
	if (raw == null || raw === '') return DEFAULT_THRESHOLD;
	const n = Number(raw);
	if (!Number.isFinite(n)) return DEFAULT_THRESHOLD;
	return Math.max(0, Math.floor(n));
}

function snapshotPath(stateDir: string): string {
	return resolve(stateDir, 'meeting-eligibility.snapshot.json');
}

async function loadSnapshot(
	stateDir: string,
	cacheHash: string,
	threshold: number
): Promise<Set<string> | null> {
	try {
		const text = await fs.readFile(snapshotPath(stateDir), 'utf8');
		const parsed = JSON.parse(text) as EligibilitySnapshot;
		if (!parsed || !Array.isArray(parsed.eligible_meeting_keys)) return null;
		// Both inputs are part of the cache key: a rebuilt corpus (cache_hash) or
		// a changed policy (threshold) must force a recompute.
		if (parsed.cache_hash !== cacheHash || parsed.threshold !== threshold) return null;
		return new Set(parsed.eligible_meeting_keys);
	} catch {
		return null;
	}
}

async function persist(stateDir: string, snap: EligibilitySnapshot): Promise<void> {
	const p = snapshotPath(stateDir);
	// stateDir may not exist yet on a fresh checkout — create before atomic rename.
	await fs.mkdir(dirname(p), { recursive: true });
	const tmp = `${p}.tmp`;
	await fs.writeFile(tmp, JSON.stringify(snap));
	await fs.rename(tmp, p);
}

async function computeEligibleMeetings(
	repo: EligibilityScanRepo,
	threshold: number
): Promise<Set<string>> {
	// (city_id, meeting_id) → count of distinct human-corrected utterances. Each
	// ordered id is one utterance (utterance_id is the groups PK), so counting
	// once per group that has any `edited_by === 'user'` edit is already
	// de-duplicated. Keyed by the (city, meeting) pair because meeting_id slugs
	// collide across cities (see meetingKey).
	const humanUtterancesByMeeting = new Map<string, number>();
	let nullMeetingHumanUtterances = 0;
	let n = 0;
	for (const id of repo.allOrderedIds()) {
		if (++n % YIELD_EVERY === 0) await yieldToEventLoop();
		const g = repo.getGroup(id);
		if (!g) continue;
		const hasHuman = g.edits.some((e) => e.edited_by === 'user');
		if (!hasHuman) continue;
		if (!g.meeting_id) {
			// Can't attribute an utterance with no meeting_id to a meeting; it's
			// unreviewable as part of a "corrected meeting", so it's excluded.
			nullMeetingHumanUtterances++;
			continue;
		}
		const key = meetingKey(g.city_id, g.meeting_id);
		humanUtterancesByMeeting.set(key, (humanUtterancesByMeeting.get(key) ?? 0) + 1);
	}
	const eligible = new Set<string>();
	for (const [key, count] of humanUtterancesByMeeting) {
		if (count >= threshold) eligible.add(key);
	}
	console.log(
		`[meeting-eligibility] threshold=${threshold} ` +
			`eligible_meetings=${eligible.size}/${humanUtterancesByMeeting.size} ` +
			`(meetings with ≥1 human utterance); ` +
			`null_meeting_human_utterances=${nullMeetingHumanUtterances}`
	);
	return eligible;
}

/**
 * Set of meeting_ids that meet the human-correction threshold. Loads the
 * snapshot when it matches (cache_hash + threshold), otherwise runs the
 * one-time scan and persists. Serialised against stats/category scans.
 */
export async function loadOrComputeEligibleMeetings(
	repo: EligibilityScanRepo,
	stateDir: string,
	threshold: number
): Promise<Set<string>> {
	const cached = await loadSnapshot(stateDir, repo.hash, threshold);
	if (cached) {
		console.log(
			`[meeting-eligibility] loaded snapshot: ${cached.size} eligible meetings ` +
				`(threshold=${threshold})`
		);
		return dropExcluded(cached);
	}
	const eligible = await runExclusiveScan(() => computeEligibleMeetings(repo, threshold));
	try {
		// Persist the full eligibility set; exclusions are applied on read.
		await persist(stateDir, {
			cache_hash: repo.hash,
			threshold,
			computedAt: Date.now(),
			eligible_meeting_keys: [...eligible]
		});
	} catch (err) {
		console.warn('[meeting-eligibility] persist failed', err);
	}
	return dropExcluded(eligible);
}
