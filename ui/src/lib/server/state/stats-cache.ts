/**
 * Stats aggregation cache — full-corpus stats are O(N) over the 287 k groups
 * and take ~17 s on the Oracle VM. We compute once, persist to
 * `ui/.state/stats.snapshot.json`, and serve from cache for `TTL_MS`. After
 * the TTL we serve the stale snapshot synchronously while a background
 * recompute runs (stale-while-revalidate).
 *
 * Invalidation is intentionally time-based, not write-based. Label PATCHes
 * happen many times per second during an active review; recomputing on each
 * would turn the server into a slideshow. The lag (TTL_MS, 10 min) is the
 * explicit trade-off; the /stats page also has a manual "refresh now" button
 * for when you need it sooner. The scan yields to the event loop so it never
 * freezes concurrent requests.
 *
 * Single-process assumption: snapshot file is the single source of truth
 * across restarts, and there is no cross-process coordination. See
 * decisions/storage.md for the broader prototype scope.
 */

import { promises as fs } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Group } from '$lib/domain/groups';
import type { IncludeStatus, StatsResponse } from '$lib/domain/types';
import type { ReviewRepo } from '../repo';

const TTL_MS = 10 * 60 * 1000;

// Yield to the event loop every this many groups during the full-corpus scan
// so the ~17 s aggregation doesn't block other requests (e.g. a reviewer
// navigating /review while a recompute runs). ~287 k groups ≈ 57 yields.
const YIELD_EVERY = 5_000;
const yieldToEventLoop = () => new Promise<void>((r) => setImmediate(r));

export interface ExtendedStats extends StatsResponse {
	groups: number;
	total_edits: number;
	multi_edit_groups: number;
	cache_hash: string;
}

export interface CachedStats {
	stats: ExtendedStats;
	computedAt: number; // epoch ms
}

function bucketFor(seconds: number): string {
	if (seconds < 2) return '0–2s';
	if (seconds < 5) return '2–5s';
	if (seconds < 10) return '5–10s';
	if (seconds < 30) return '10–30s';
	return '30s+';
}

export async function computeStats(repo: ReviewRepo): Promise<ExtendedStats> {
	let groupCount = 0;
	let totalEdits = 0;
	let multiEditGroups = 0;
	const byStatus: Record<IncludeStatus, number> = {
		unreviewed: 0, include: 0, exclude: 0, uncertain: 0
	};
	const byCategory = new Map<string | null, number>();
	const byIngestCategory = new Map<string | null, number>();
	const byEditor = new Map<string | null, number>();
	const byDuration = new Map<string, number>();
	const byMeeting = new Map<
		string,
		{ meeting_name: string | null; meeting_date: string | null; count: number }
	>();

	for (const g of repo.iterGroups() as IterableIterator<Group>) {
		groupCount += 1;
		if (groupCount % YIELD_EVERY === 0) await yieldToEventLoop();
		if (g.edits.length > 1) multiEditGroups++;
		byStatus[g.label.include_status] += 1;
		const cats = g.label.error_categories;
		if (cats.length === 0) {
			byCategory.set(null, (byCategory.get(null) ?? 0) + 1);
		} else {
			for (const c of cats) byCategory.set(c, (byCategory.get(c) ?? 0) + 1);
		}
		const mkey = `${g.meeting_id ?? ''}|${g.meeting_name ?? ''}|${g.meeting_date ?? ''}`;
		const cur = byMeeting.get(mkey);
		if (cur) cur.count += 1;
		else
			byMeeting.set(mkey, {
				meeting_name: g.meeting_name,
				meeting_date: g.meeting_date,
				count: 1
			});
		for (const e of g.edits) {
			totalEdits += 1;
			byIngestCategory.set(e.ingest_category, (byIngestCategory.get(e.ingest_category) ?? 0) + 1);
			byEditor.set(e.edited_by, (byEditor.get(e.edited_by) ?? 0) + 1);
			const duration = Math.max(0, e.utterance_end - e.utterance_start);
			byDuration.set(bucketFor(duration), (byDuration.get(bucketFor(duration)) ?? 0) + 1);
		}
	}

	return {
		total: groupCount,
		groups: groupCount,
		total_edits: totalEdits,
		multi_edit_groups: multiEditGroups,
		cache_hash: repo.hash,
		by_status: byStatus,
		by_category: [...byCategory.entries()]
			.sort((a, b) => b[1] - a[1])
			.map(([category, count]) => ({ category, count })),
		by_ingest_category: [...byIngestCategory.entries()]
			.sort((a, b) => b[1] - a[1])
			.map(([ingest_category, count]) => ({
				ingest_category,
				label_el: null,
				reason_el: null,
				is_rejected: 0,
				count
			})),
		by_editor: [...byEditor.entries()]
			.sort((a, b) => b[1] - a[1])
			.map(([edited_by, count]) => ({ edited_by, count })),
		by_duration_bucket: [...byDuration.entries()]
			.sort()
			.map(([bucket, count]) => ({ bucket, count })),
		by_meeting: [...byMeeting.values()].sort((a, b) => b.count - a.count).slice(0, 20)
	};
}

export class StatsCache {
	private current: CachedStats | null = null;
	private inFlight: Promise<CachedStats> | null = null;
	private snapshotPath: string;
	private intervalHandle: ReturnType<typeof setInterval> | null = null;

	constructor(stateDir: string) {
		this.snapshotPath = resolve(stateDir, 'stats.snapshot.json');
	}

	/** Read snapshot from disk if present and still matches `repo.hash`. */
	private async loadSnapshot(repoHash: string): Promise<CachedStats | null> {
		try {
			const text = await fs.readFile(this.snapshotPath, 'utf8');
			const parsed = JSON.parse(text) as CachedStats;
			if (!parsed?.stats || parsed.stats.cache_hash !== repoHash) return null;
			if (typeof parsed.computedAt !== 'number') return null;
			return parsed;
		} catch {
			return null;
		}
	}

	private async persist(snap: CachedStats): Promise<void> {
		// stateDir may not exist yet on a fresh checkout (or after someone
		// blew it away) — mirror writeDisk in meeting-context and create it
		// before the atomic rename.
		await fs.mkdir(dirname(this.snapshotPath), { recursive: true });
		const tmp = `${this.snapshotPath}.tmp`;
		await fs.writeFile(tmp, JSON.stringify(snap));
		await fs.rename(tmp, this.snapshotPath);
	}

	private async recompute(repo: ReviewRepo): Promise<CachedStats> {
		if (this.inFlight) return this.inFlight;
		const task = (async () => {
			const stats = await computeStats(repo);
			const snap: CachedStats = { stats, computedAt: Date.now() };
			this.current = snap;
			try {
				await this.persist(snap);
			} catch (err) {
				console.warn('[stats-cache] persist failed', err);
			}
			return snap;
		})();
		this.inFlight = task.finally(() => {
			this.inFlight = null;
		});
		return task;
	}

	/**
	 * Return whatever snapshot we currently have. The background timer set up
	 * by `startBackgroundRefresh()` is responsible for keeping the snapshot
	 * fresh; request handlers never trigger a recompute themselves, so the
	 * client never pays for the aggregation.
	 *
	 * The one exception is a true cold start (fresh process, no disk
	 * snapshot, cron hasn't fired yet) — we block once on a compute so the
	 * first request doesn't 500.
	 */
	async get(repo: ReviewRepo): Promise<CachedStats> {
		// Hydrate from disk on first access if we haven't yet.
		if (!this.current) {
			const disk = await this.loadSnapshot(repo.hash);
			if (disk) this.current = disk;
		}

		// Repo hash changed (the source data was rebuilt while the server
		// was running). Treat the existing snapshot as invalid and let the
		// next cron tick refresh; meanwhile compute once synchronously.
		if (this.current && this.current.stats.cache_hash !== repo.hash) {
			this.current = null;
		}

		if (!this.current) {
			return this.recompute(repo);
		}
		return this.current;
	}

	/**
	 * Schedule a background recompute every `intervalMs`. Idempotent — a
	 * second call is a no-op so SvelteKit's hot reload + repeated module
	 * imports won't pile up timers.
	 *
	 * Kicks off an immediate refresh if we're already stale (or have nothing
	 * cached) so the first request after deploy doesn't have to do it.
	 *
	 * Returns a stop function for tests and lifecycle teardown.
	 */
	startBackgroundRefresh(
		repoFactory: () => Promise<ReviewRepo>,
		intervalMs: number = TTL_MS
	): () => void {
		if (this.intervalHandle) return () => this.stopBackgroundRefresh();

		const refresh = async (): Promise<void> => {
			try {
				const repo = await repoFactory();
				if (!this.current) {
					const disk = await this.loadSnapshot(repo.hash);
					if (disk) this.current = disk;
				}
				const age = this.current ? Date.now() - this.current.computedAt : Infinity;
				if (
					!this.current ||
					this.current.stats.cache_hash !== repo.hash ||
					age >= intervalMs
				) {
					await this.recompute(repo);
				}
			} catch (err) {
				console.warn('[stats-cache] background refresh failed', err);
			}
		};
		// One immediate tick at startup, then on the interval.
		void refresh();
		this.intervalHandle = setInterval(() => void refresh(), intervalMs);
		// `unref` exists on Node's Timeout — don't block process shutdown on it.
		const h = this.intervalHandle as unknown as { unref?: () => void };
		if (typeof h.unref === 'function') h.unref();

		return () => this.stopBackgroundRefresh();
	}

	/**
	 * Force a fresh recompute now, regardless of TTL. Backs the manual
	 * "refresh" button on /stats so a reviewer can confirm their labels
	 * registered without waiting up to 5 minutes for the background tick.
	 * Dedupes with any in-flight compute via `recompute`.
	 */
	async forceRecompute(repo: ReviewRepo): Promise<CachedStats> {
		return this.recompute(repo);
	}

	stopBackgroundRefresh(): void {
		if (this.intervalHandle) {
			clearInterval(this.intervalHandle);
			this.intervalHandle = null;
		}
	}

	/** Test hook. */
	_peek(): CachedStats | null {
		return this.current;
	}
	_hasInterval(): boolean {
		return this.intervalHandle !== null;
	}
	_reset(): void {
		this.stopBackgroundRefresh();
		this.current = null;
		this.inFlight = null;
	}
}

let _statsCache: StatsCache | null = null;
export function getStatsCache(): StatsCache {
	if (!_statsCache) {
		const stateDir = process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');
		_statsCache = new StatsCache(stateDir);
	}
	return _statsCache;
}

/** Test-only reset to start a fresh cache instance. */
export function _resetStatsCacheForTests(): void {
	_statsCache = null;
}
