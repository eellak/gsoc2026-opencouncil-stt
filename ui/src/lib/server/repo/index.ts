/**
 * Repo selector — picks SqliteRepo when the cache .sqlite is present, else
 * falls back to FileRepo so existing dev environments (and tests) keep
 * working without forcing a rebuild.
 *
 * Both implementations share the same public surface (Group queries, seeded
 * queue, sidecar-backed labels). The selector returns the narrower union so
 * call sites stay typed against the common interface.
 */

import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { getFileRepo } from './file-repo';
import { getSqliteRepo } from './sqlite-repo';
import type {
	Group,
	GroupLabel,
	GroupPatchBody,
	QueueResponse
} from '$lib/domain/groups';
import type { IncludeStatus } from '$lib/domain/types';
import type { UserCounts } from '../state/sidecar';
import type { LocalContextResult } from '../cache/transcript-extract';

export interface ReviewRepo {
	readonly hash: string;
	readonly total: number;
	/** Monotonic label revision (sidecar event id). Use as cache key. */
	readonly labelsRevision: number;
	getGroup(utterance_id: string): Group | null;
	queue(seed: number, from: number, n: number): QueueResponse;
	patchLabel(utterance_id: string, patch: GroupPatchBody, username?: string): Promise<GroupLabel | null>;
	patchLabelExt(utterance_id: string, patch: GroupPatchBody, source: string): Promise<GroupLabel | null>;
	listUsernames(): string[];
	userCounts(): Record<string, UserCounts>;
	allLabels(): ReadonlyMap<string, GroupLabel>;
	allGroups(): Group[];
	iterGroups(): IterableIterator<Group>;
	/**
	 * All utterance ids in canonical order. Lets long scans fetch groups one at
	 * a time via getGroup() (a point query) instead of holding a single
	 * better-sqlite3 cursor open — the latter throws "statement is busy" if the
	 * scan yields to the event loop and another scan/request touches the DB.
	 */
	allOrderedIds(): readonly string[];
	/**
	 * Navigation universe after the meeting-eligibility filter (meetings with
	 * ≥ MEETING_MIN_HUMAN_UTTERANCES human-corrected utterances), canonical
	 * order. The seeded queue, status/category filters, and stats all scope to
	 * this set; `getGroup(id)` stays unfiltered so direct links still resolve.
	 */
	eligibleOrderedIds(): readonly string[];
	utteranceIdForEdit(edit_id: string): string | null;
	groupsByErrorCategory(category: string): Group[];
	/** Just the ids whose label carries `category`, in canonical order — cheap
	 *  (in-memory label scan, no group materialisation). Use for paged listings
	 *  and the review filter queue. */
	idsByErrorCategory(category: string): string[];
	idsByStatus(status: IncludeStatus): string[];
	/**
	 * Local surrounding context for an utterance, or null to fall back to the
	 * live upstream /context proxy. sqlite serves it from the transcript table
	 * when present + current; file-repo always returns null.
	 */
	getContext(utterance_id: string, before: number, after: number): LocalContextResult | null;
	flush(): Promise<void>;
}

let cachedFlavour: 'sqlite' | 'file' | null = null;

export async function getRepo(): Promise<ReviewRepo> {
	const cacheDir = process.env.REVIEW_CACHE_DIR ?? resolve(process.cwd(), '.cache');
	const sqlitePath = resolve(cacheDir, 'groups.v1.sqlite');

	// `REVIEW_REPO=file` lets us pin the legacy path explicitly when both
	// caches exist side-by-side (CI parity checks).
	const override = process.env.REVIEW_REPO;
	const wantSqlite =
		override === 'sqlite' || (override !== 'file' && existsSync(sqlitePath));

	if (wantSqlite) {
		if (cachedFlavour && cachedFlavour !== 'sqlite') {
			throw new Error('REVIEW_REPO changed at runtime — restart the process');
		}
		cachedFlavour = 'sqlite';
		return getSqliteRepo();
	}
	if (cachedFlavour && cachedFlavour !== 'file') {
		throw new Error('REVIEW_REPO changed at runtime — restart the process');
	}
	cachedFlavour = 'file';
	return getFileRepo();
}
