/**
 * SQLite-backed repository.
 *
 * Same surface as FileRepo, but the 432 MB groups payload stays on disk and
 * only the requested rows are parsed into memory per call. Tradeoff: each
 * getGroup() does a synchronous SELECT + JSON.parse — measured at ~50 µs on
 * Oracle's free shape, dominated by JSON parsing not I/O. The seeded queue
 * builds its shuffle from the in-memory `orderedIds` list (~10 MB for the
 * full 287 k corpus) and is recomputed per call so we don't accumulate one
 * shuffled array per attacker-supplied seed.
 *
 * The DB is opened read-only. WAL/SHM sidecar files should not exist at all
 * — buildSqlite checkpoints and truncates before renaming. Opening read-only
 * also protects against accidental writes from request handlers.
 *
 * Label state still lives in the sidecar JSONL + snapshot — see SidecarStore.
 */

import Database, { type Database as DB, type Statement } from 'better-sqlite3';
import { existsSync, promises as fs } from 'node:fs';
import { resolve } from 'node:path';
import { CACHE_VERSION } from '$lib/domain/groups';
import type {
	Group,
	GroupLabel,
	GroupPatchBody,
	QueueResponse
} from '$lib/domain/groups';
import { SidecarStore } from '../state/sidecar';

// Defaults are resolved from `process.cwd()` so the bundled adapter-node
// output picks up the operator's working directory at start time. Tests pass
// explicit paths; production sets REVIEW_CACHE_DIR / REVIEW_STATE_DIR when it
// wants something other than `<cwd>/.cache` and `<cwd>/.state`.
const DEFAULT_CACHE_DIR = (): string => resolve(process.cwd(), '.cache');
const DEFAULT_STATE_DIR = (): string => resolve(process.cwd(), '.state');
const DEFAULT_CDN_MAP = (): string =>
	resolve(process.cwd(), '..', 'data', 'audio-fix', 'url-map.json');

export interface SqliteRepoOptions {
	cacheDir?: string;
	stateDir?: string;
	audioCdnMapPath?: string;
}

interface MetaRow {
	key: string;
	value: string;
}

export class SqliteRepo {
	private getByIdStmt: Statement<[string]>;
	private getByIdsBatchSize = 64;
	private editToUtteranceStmt: Statement<[string]>;
	private iterAllStmt: Statement<[]>;

	private constructor(
		private db: DB,
		private orderedIds: string[],
		private cacheHash: string,
		private sidecar: SidecarStore,
		private audioCdnMap: Map<string, string>
	) {
		this.getByIdStmt = db.prepare('SELECT json FROM groups WHERE utterance_id = ?');
		this.editToUtteranceStmt = db.prepare(
			'SELECT utterance_id FROM edits WHERE edit_id = ?'
		);
		this.iterAllStmt = db.prepare('SELECT json FROM groups ORDER BY ord');
		void this.audioCdnMap; // currently unused by sqlite path; surfaces via /api/audio
	}

	static async load(opts: SqliteRepoOptions = {}): Promise<SqliteRepo> {
		const cacheDir = opts.cacheDir ?? process.env.REVIEW_CACHE_DIR ?? DEFAULT_CACHE_DIR();
		const stateDir = opts.stateDir ?? process.env.REVIEW_STATE_DIR ?? DEFAULT_STATE_DIR();
		const cdnMapPath =
			opts.audioCdnMapPath ?? process.env.REVIEW_AUDIO_MAP_PATH ?? DEFAULT_CDN_MAP();

		const dbPath = resolve(cacheDir, 'groups.v1.sqlite');
		// readonly + fileMustExist mirrors the "I'm a consumer, not a builder"
		// posture — accidental writes from request handlers fail loudly.
		const db = new Database(dbPath, { readonly: true, fileMustExist: true });
		db.pragma('query_only = true');
		db.pragma('cache_size = -32000'); // 32 MB page cache
		db.pragma('mmap_size = 67108864'); // 64 MB mmap window (best effort)

		// Schema sanity. cache_version mismatch must fail loudly so a stale
		// build doesn't get silently mounted by a newer server.
		const metaRows = db.prepare('SELECT key, value FROM meta').all() as MetaRow[];
		const meta = new Map(metaRows.map((r) => [r.key, r.value]));
		const dbCacheVersion = Number(meta.get('cache_version'));
		if (!Number.isFinite(dbCacheVersion) || dbCacheVersion !== CACHE_VERSION) {
			db.close();
			throw new Error(
				`SqliteRepo: cache_version mismatch (db=${meta.get('cache_version')}, ` +
					`expected=${CACHE_VERSION}). Rebuild with: bun scripts/build-cache.ts --format sqlite`
			);
		}
		const sourceHash = meta.get('source_hash');
		if (!sourceHash) {
			db.close();
			throw new Error('SqliteRepo: missing source_hash in meta — corrupt cache');
		}

		const orderedIds = (
			db.prepare('SELECT utterance_id FROM groups ORDER BY ord').all() as Array<{
				utterance_id: string;
			}>
		).map((r) => r.utterance_id);

		const cdnMap = new Map<string, string>();
		if (existsSync(cdnMapPath)) {
			try {
				const raw = JSON.parse(await fs.readFile(cdnMapPath, 'utf8')) as Record<string, string>;
				for (const [k, v] of Object.entries(raw)) cdnMap.set(k, v);
			} catch {
				/* malformed map → ignore, log on next request */
			}
		}

		const sidecar = await SidecarStore.load(stateDir);

		console.log(
			`[sqlite-repo] mounted ${orderedIds.length.toLocaleString()} groups; ` +
				`cache_hash=${sourceHash}; sidecar_labels=${sidecar.all().size}`
		);

		return new SqliteRepo(db, orderedIds, sourceHash, sidecar, cdnMap);
	}

	get hash(): string {
		return this.cacheHash;
	}

	get total(): number {
		return this.orderedIds.length;
	}

	getGroup(utterance_id: string): Group | null {
		const row = this.getByIdStmt.get(utterance_id) as { json: string } | undefined;
		if (!row) return null;
		const base = JSON.parse(row.json) as Omit<Group, 'label'>;
		return { ...base, label: this.sidecar.get(utterance_id) } as Group;
	}

	queue(seed: number, from: number, n: number): QueueResponse {
		const order = this.computeSeededOrder(seed);
		const start = Math.max(0, Math.min(order.length, Math.floor(from)));
		const count = Math.max(0, Math.min(50, Math.floor(n)));
		const slice = order.slice(start, start + count);

		// Codex flag: SELECT … WHERE utterance_id IN (?,?,…) returns rows in
		// SQLite's chosen order (typically rowid/PK), not in the order of the
		// IN clause. We reorder the result map back into the seeded order
		// before returning.
		const byId = new Map<string, Group>();
		if (slice.length > 0) {
			// Chunk to avoid huge IN(…) lists; with `count` ≤ 50 this loop runs once.
			for (let i = 0; i < slice.length; i += this.getByIdsBatchSize) {
				const chunk = slice.slice(i, i + this.getByIdsBatchSize);
				const placeholders = chunk.map(() => '?').join(',');
				const rows = this.db
					.prepare(
						`SELECT utterance_id, json FROM groups WHERE utterance_id IN (${placeholders})`
					)
					.all(...chunk) as Array<{ utterance_id: string; json: string }>;
				for (const r of rows) {
					const base = JSON.parse(r.json) as Omit<Group, 'label'>;
					byId.set(r.utterance_id, {
						...base,
						label: this.sidecar.get(r.utterance_id)
					} as Group);
				}
			}
		}
		const groups = slice
			.map((id) => byId.get(id))
			.filter((g): g is Group => Boolean(g));
		const next_cursor = start + count < order.length ? start + count : null;
		return { cache_hash: this.cacheHash, total: order.length, groups, next_cursor };
	}

	async patchLabel(utterance_id: string, patch: GroupPatchBody): Promise<GroupLabel | null> {
		// Cheap existence check via the prepared statement — same shape as
		// FileRepo's "is the utterance in my map?" guard.
		const exists = this.getByIdStmt.get(utterance_id);
		if (!exists) return null;
		return this.sidecar.patch(utterance_id, patch);
	}

	allLabels(): ReadonlyMap<string, GroupLabel> {
		return this.sidecar.all();
	}

	/**
	 * Convenience materialiser for callers that need an array (export/stats
	 * paths were rewritten to use `iterGroups()` instead — prefer that for
	 * memory). Kept for tests + small ad-hoc consumers.
	 */
	allGroups(): Group[] {
		return [...this.iterGroups()];
	}

	/**
	 * Stream groups in `ord` order without materialising the full array.
	 * Synchronous because better-sqlite3 is synchronous — callers must NOT
	 * `await` inside the loop on the same DB connection, or they risk
	 * starving the iterator (this is a single-process prototype).
	 */
	*iterGroups(): IterableIterator<Group> {
		for (const row of this.iterAllStmt.iterate() as IterableIterator<{ json: string }>) {
			const base = JSON.parse(row.json) as Omit<Group, 'label'>;
			yield { ...base, label: this.sidecar.get(base.utterance_id) } as Group;
		}
	}

	utteranceIdForEdit(edit_id: string): string | null {
		const row = this.editToUtteranceStmt.get(edit_id) as { utterance_id: string } | undefined;
		return row?.utterance_id ?? null;
	}

	groupsByErrorCategory(category: string): Group[] {
		// Labels live in memory; filter there first and only hit SQLite for
		// the matching ids. Avoids parsing the full 432 MB payload to find
		// "homophone".
		const matchingIds: string[] = [];
		for (const [id, lbl] of this.sidecar.all()) {
			if (lbl.error_categories.includes(category)) matchingIds.push(id);
		}
		if (matchingIds.length === 0) return [];

		const out: Group[] = [];
		const chunkSize = this.getByIdsBatchSize;
		for (let i = 0; i < matchingIds.length; i += chunkSize) {
			const chunk = matchingIds.slice(i, i + chunkSize);
			const placeholders = chunk.map(() => '?').join(',');
			const rows = this.db
				.prepare(
					`SELECT utterance_id, json FROM groups WHERE utterance_id IN (${placeholders})`
				)
				.all(...chunk) as Array<{ utterance_id: string; json: string }>;
			for (const r of rows) {
				const base = JSON.parse(r.json) as Omit<Group, 'label'>;
				out.push({ ...base, label: this.sidecar.get(r.utterance_id) } as Group);
			}
		}
		return out;
	}

	flush(): Promise<void> {
		return this.sidecar.flush();
	}

	close(): void {
		this.db.close();
	}

	/** Test-only hook: confirms we don't accumulate per-seed shuffle caches. */
	_debugSeedCacheSize(): number {
		return 0;
	}

	// Mulberry32 — same algorithm as FileRepo so that the JSON-backed and
	// SQLite-backed repos produce identical orderings for the same seed.
	// Intentionally NOT memoised: a public `?seed=` parameter could otherwise
	// be used to cause unbounded memory growth on a small VM.
	private computeSeededOrder(seed: number): string[] {
		const arr = [...this.orderedIds];
		let state = (seed >>> 0) || 1;
		for (let i = arr.length - 1; i > 0; i--) {
			state = (state + 0x6d2b79f5) >>> 0;
			let t = state;
			t = Math.imul(t ^ (t >>> 15), t | 1);
			t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
			const r = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
			const j = Math.floor(r * (i + 1));
			[arr[i], arr[j]] = [arr[j], arr[i]];
		}
		return arr;
	}
}

// Module-level singleton, mirroring getFileRepo's lifecycle so callers can
// remain synchronous past the first await.
let _repo: SqliteRepo | null = null;
let _loading: Promise<SqliteRepo> | null = null;

export async function getSqliteRepo(): Promise<SqliteRepo> {
	if (_repo) return _repo;
	if (_loading) return _loading;
	const attempt = SqliteRepo.load().then(
		(r) => {
			_repo = r;
			return r;
		},
		(err: unknown) => {
			// Clear so the next caller retries instead of getting the cached
			// rejection forever (e.g., when the DB file briefly disappears
			// during an .incoming → final rename).
			if (_loading === attempt) _loading = null;
			throw err;
		}
	);
	_loading = attempt;
	return attempt;
}

export function _resetSqliteRepoForTests(): void {
	if (_repo) {
		try {
			_repo.close();
		} catch {
			/* already closed */
		}
	}
	_repo = null;
	_loading = null;
}
