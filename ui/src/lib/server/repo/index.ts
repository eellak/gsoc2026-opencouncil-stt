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

export interface ReviewRepo {
	readonly hash: string;
	readonly total: number;
	getGroup(utterance_id: string): Group | null;
	queue(seed: number, from: number, n: number): QueueResponse;
	patchLabel(utterance_id: string, patch: GroupPatchBody): Promise<GroupLabel | null>;
	allLabels(): ReadonlyMap<string, GroupLabel>;
	allGroups(): Group[];
	iterGroups(): IterableIterator<Group>;
	utteranceIdForEdit(edit_id: string): string | null;
	groupsByErrorCategory(category: string): Group[];
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
