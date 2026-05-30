/**
 * Lazy per-`ingest_category` index of edits.
 *
 * The /category/[category] page lists every edit whose `ingest_category`
 * matches. Building that index requires iterating all ~287 k groups (~17 s
 * on the Oracle VM), so we do it once on first access, persist the result to
 * `ui/.state/category-index.snapshot.json`, and serve paged slices from
 * memory thereafter.
 *
 * The snapshot is keyed by `cache_hash` — if the source SQLite was rebuilt,
 * the on-disk snapshot is treated as invalid and recomputed.
 *
 * Memory: ~393 k edits × ~250 bytes ≈ 100 MB worst case if the user visits a
 * category that contains every edit. In practice the per-category sets are
 * smaller and we cap the on-disk write size with a hard guard in `persist`.
 */

import { promises as fs } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { ReviewRepo } from '../repo';

// Yield to the event loop every this many groups so the one-time index build
// (~17 s over ~287 k groups) doesn't freeze concurrent requests.
const YIELD_EVERY = 5_000;
const yieldToEventLoop = () => new Promise<void>((r) => setImmediate(r));

export interface EditRow {
	edit_id: string;
	utterance_id: string;
	before_text: string;
	after_text: string;
	edited_by: string | null;
	cleaning_applied: string;
}

export interface CategoryIndex {
	cache_hash: string;
	computedAt: number;
	/** ingest_category → EditRow[] in csv_row order (stable, matches build). */
	byCategory: Record<string, EditRow[]>;
}

export interface CategoryPage {
	items: EditRow[];
	total: number;
	computedAt: number;
}

export class CategoryCache {
	private current: CategoryIndex | null = null;
	private inFlight: Promise<CategoryIndex> | null = null;
	private snapshotPath: string;

	constructor(stateDir: string) {
		this.snapshotPath = resolve(stateDir, 'category-index.snapshot.json');
	}

	private async loadSnapshot(repoHash: string): Promise<CategoryIndex | null> {
		try {
			const text = await fs.readFile(this.snapshotPath, 'utf8');
			const parsed = JSON.parse(text) as CategoryIndex;
			if (!parsed?.byCategory || parsed.cache_hash !== repoHash) return null;
			return parsed;
		} catch {
			return null;
		}
	}

	private async persist(snap: CategoryIndex): Promise<void> {
		// Mirror writeDisk in meeting-context: if stateDir was wiped (e.g.
		// fresh checkout, container restart with ephemeral volume), the
		// parent directory has to be recreated before the atomic rename.
		await fs.mkdir(dirname(this.snapshotPath), { recursive: true });
		const tmp = `${this.snapshotPath}.tmp`;
		await fs.writeFile(tmp, JSON.stringify(snap));
		await fs.rename(tmp, this.snapshotPath);
	}

	private async build(repo: ReviewRepo): Promise<CategoryIndex> {
		const byCategory: Record<string, EditRow[]> = {};
		let n = 0;
		for (const g of repo.iterGroups()) {
			if (++n % YIELD_EVERY === 0) await yieldToEventLoop();
			for (const e of g.edits) {
				const key = e.ingest_category ?? '';
				let list = byCategory[key];
				if (!list) {
					list = [];
					byCategory[key] = list;
				}
				list.push({
					edit_id: e.edit_id,
					utterance_id: g.utterance_id,
					before_text: e.before_text,
					after_text: e.after_text,
					edited_by: e.edited_by,
					cleaning_applied: e.cleaning_applied
				});
			}
		}
		return { cache_hash: repo.hash, computedAt: Date.now(), byCategory };
	}

	private async recompute(repo: ReviewRepo): Promise<CategoryIndex> {
		if (this.inFlight) return this.inFlight;
		const task = (async () => {
			const snap = await this.build(repo);
			this.current = snap;
			try {
				await this.persist(snap);
			} catch (err) {
				console.warn('[category-cache] persist failed', err);
			}
			return snap;
		})();
		this.inFlight = task.finally(() => {
			this.inFlight = null;
		});
		return task;
	}

	private async ensure(repo: ReviewRepo): Promise<CategoryIndex> {
		if (!this.current) {
			const disk = await this.loadSnapshot(repo.hash);
			if (disk) this.current = disk;
		}
		if (this.current && this.current.cache_hash !== repo.hash) {
			this.current = null;
		}
		if (!this.current) return this.recompute(repo);
		return this.current;
	}

	async getPage(
		repo: ReviewRepo,
		category: string,
		page: number,
		pageSize: number
	): Promise<CategoryPage> {
		const index = await this.ensure(repo);
		const list = index.byCategory[category] ?? [];
		const safePage = Math.max(1, Math.floor(page));
		const safeSize = Math.max(1, Math.min(500, Math.floor(pageSize)));
		const start = (safePage - 1) * safeSize;
		const items = list.slice(start, start + safeSize);
		return { items, total: list.length, computedAt: index.computedAt };
	}

	/** Distinct utterance_ids for an ingest category, in stable build order.
	 *  Backs the "play through this category" review queue. */
	async ids(repo: ReviewRepo, category: string): Promise<string[]> {
		const index = await this.ensure(repo);
		const list = index.byCategory[category] ?? [];
		const seen = new Set<string>();
		const out: string[] = [];
		for (const e of list) {
			if (seen.has(e.utterance_id)) continue;
			seen.add(e.utterance_id);
			out.push(e.utterance_id);
		}
		return out;
	}

	private bgStarted = false;
	/**
	 * Precompute the index in the background at startup (idempotent) so the
	 * first /category/[category] visit serves from cache instead of paying the
	 * ~17 s build synchronously inside a page load. The build yields, so this
	 * doesn't block requests while it runs.
	 */
	startBackgroundBuild(repoFactory: () => Promise<ReviewRepo>): void {
		if (this.bgStarted) return;
		this.bgStarted = true;
		void (async () => {
			try {
				const repo = await repoFactory();
				await this.ensure(repo);
			} catch (err) {
				console.warn('[category-cache] background build failed', err);
			}
		})();
	}

	/** Test hooks. */
	_peek(): CategoryIndex | null {
		return this.current;
	}
	_reset(): void {
		this.current = null;
		this.inFlight = null;
		this.bgStarted = false;
	}
}

let _categoryCache: CategoryCache | null = null;
export function getCategoryCache(): CategoryCache {
	if (!_categoryCache) {
		const stateDir = process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');
		_categoryCache = new CategoryCache(stateDir);
	}
	return _categoryCache;
}

export function _resetCategoryCacheForTests(): void {
	_categoryCache = null;
}
