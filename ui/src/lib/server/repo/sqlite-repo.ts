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
import { CACHE_VERSION, cacheHashWithExclusions } from '$lib/domain/groups';
import type {
	Group,
	GroupLabel,
	GroupPatchBody,
	QueueResponse
} from '$lib/domain/groups';
import type { IncludeStatus } from '$lib/domain/types';
import { SidecarStore } from '../state/sidecar';
import { TRANSCRIPT_SCHEMA_VERSION } from '../cache/transcript-extract';
import type { ContextNeighbour, LocalContextResult } from '../cache/transcript-extract';
import {
	loadOrComputeEligibleMeetings,
	meetingEligibilityThreshold,
	meetingKey
} from '../state/meeting-eligibility';
import { degenerateCategories } from '../state/ingest-filter';

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
	/** Overrides MEETING_MIN_HUMAN_UTTERANCES (tests pass this explicitly). */
	meetingMinHumanUtterances?: number;
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

	// Navigation universe after the meeting-eligibility filter. Starts as the
	// full corpus (filter inactive) and is narrowed by applyMeetingEligibility()
	// during load(). `eligibleIdSet` is null when filtering is disabled
	// (threshold ≤ 0), meaning "everything is eligible" without a 287 k-entry Set.
	private eligibleIds: string[];
	private eligibleIdSet: Set<string> | null = null;

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
		this.eligibleIds = orderedIds;
		void this.audioCdnMap; // currently unused by sqlite path; surfaces via /api/audio
		this.initTranscript();
	}

	// --- Surrounding-context index (optional table; null when absent/stale) ---
	private ctxAnchorStmt: Statement<[string]> | null = null;
	private ctxMeetingStmt: Statement<[string, string]> | null = null;
	private ctxSliceStmt: Statement<[string, string, number, number]> | null = null;

	/**
	 * Wire up the transcript-context statements iff the table exists AND its
	 * schema version matches. Any mismatch leaves the statements null, so
	 * getContext() returns null and the bridge falls back to live upstream.
	 */
	private initTranscript(): void {
		const meta = new Map(
			(this.db.prepare('SELECT key, value FROM meta').all() as MetaRow[]).map((r) => [
				r.key,
				r.value
			])
		);
		const hasTable =
			(
				this.db
					.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='transcript'")
					.get() as { name: string } | undefined
			)?.name === 'transcript';
		const versionOk = Number(meta.get('transcript_schema_version')) === TRANSCRIPT_SCHEMA_VERSION;
		const hashPresent = !!meta.get('transcript_manifest_hash');
		if (!hasTable || !versionOk || !hashPresent) return;

		this.ctxAnchorStmt = this.db.prepare(
			'SELECT city_id, meeting_id, seq FROM transcript WHERE utterance_id = ? LIMIT 1'
		);
		this.ctxMeetingStmt = this.db.prepare(
			'SELECT 1 FROM transcript_meeting WHERE city_id = ? AND meeting_id = ?'
		);
		// Inclusive window around the anchor seq; anchor row excluded by the caller
		// (by seq, not id, so a duplicate id can't slip through). ORDER BY seq keeps
		// both before/after chronological ascending — matches the upstream contract.
		// Scoped by (city_id, meeting_id): meeting_id slugs collide across cities.
		this.ctxSliceStmt = this.db.prepare(
			`SELECT utterance_id, text, start, "end", speaker_tag, seq
			 FROM transcript WHERE city_id = ? AND meeting_id = ? AND seq >= ? AND seq <= ? ORDER BY seq`
		);
	}

	/**
	 * Local surrounding context for an utterance, or null to signal "fall back to
	 * live upstream". Null covers: transcript table absent / stale version /
	 * meeting not indexed (per-meeting manifest miss) / utterance unknown.
	 * `before`/`after` are chronological ascending, same shape as upstream.
	 */
	getContext(utterance_id: string, before: number, after: number): LocalContextResult | null {
		if (!this.ctxAnchorStmt || !this.ctxMeetingStmt || !this.ctxSliceStmt) return null;
		const anchor = this.ctxAnchorStmt.get(utterance_id) as
			| { city_id: string; meeting_id: string; seq: number }
			| undefined;
		if (!anchor) return null;
		// Per-meeting presence: a partial build may lack this meeting.
		if (!this.ctxMeetingStmt.get(anchor.city_id, anchor.meeting_id)) return null;

		const b = Math.max(0, Math.floor(Number.isFinite(before) ? before : 0));
		const a = Math.max(0, Math.floor(Number.isFinite(after) ? after : 0));
		const rows = this.ctxSliceStmt.all(
			anchor.city_id,
			anchor.meeting_id,
			anchor.seq - b,
			anchor.seq + a
		) as Array<{
			utterance_id: string;
			text: string;
			start: number | null;
			end: number | null;
			speaker_tag: string | null;
			seq: number;
		}>;

		const toNeighbour = (r: (typeof rows)[number]): ContextNeighbour => ({
			id: r.utterance_id,
			text: r.text,
			start: r.start,
			end: r.end,
			speakerTagId: r.speaker_tag
		});
		return {
			meeting: { id: anchor.meeting_id, cityId: anchor.city_id },
			before: rows.filter((r) => r.seq < anchor.seq).map(toNeighbour),
			after: rows.filter((r) => r.seq > anchor.seq).map(toNeighbour)
		};
	}

	/** True when `utterance_id` survives the meeting-eligibility filter. */
	private isEligible(utterance_id: string): boolean {
		return this.eligibleIdSet ? this.eligibleIdSet.has(utterance_id) : true;
	}

	/**
	 * Narrow the navigation universe to utterances whose meeting clears the
	 * human-correction threshold. Runs once during load(): loads the snapshot
	 * when it matches, else pays the one-time scan. Threshold ≤ 0 disables it.
	 */
	private async applyMeetingEligibility(stateDir: string, thresholdOverride?: number): Promise<void> {
		const threshold = thresholdOverride ?? meetingEligibilityThreshold();
		if (threshold <= 0) {
			this.eligibleIds = this.orderedIds;
			this.eligibleIdSet = null;
			return;
		}
		const eligibleMeetings = await loadOrComputeEligibleMeetings(this, stateDir, threshold);
		// Build the eligible order by streaming ordered (utterance_id, city_id,
		// meeting_id) rows — both are columns, so no JSON parse. utterance_id is
		// the PK, so rows are 1:1 with `orderedIds`; ORDER BY ord keeps canonical
		// order. Keyed by (city, meeting) since meeting_id slugs collide.
		const rows = this.db
			.prepare('SELECT utterance_id, city_id, meeting_id FROM groups ORDER BY ord')
			.all() as Array<{ utterance_id: string; city_id: string | null; meeting_id: string | null }>;
		const elig: string[] = [];
		for (const r of rows) {
			if (r.meeting_id && eligibleMeetings.has(meetingKey(r.city_id, r.meeting_id))) {
				elig.push(r.utterance_id);
			}
		}
		this.eligibleIds = elig;
		this.eligibleIdSet = new Set(elig);
		console.log(
			`[sqlite-repo] meeting-eligibility: ${elig.length.toLocaleString()}/` +
				`${this.orderedIds.length.toLocaleString()} utterances eligible (threshold=${threshold})`
		);
	}

	/**
	 * Second eligibility layer: drop utterances whose LATEST edit is a degenerate
	 * ingest bin (no real correction signal). Runs after applyMeetingEligibility,
	 * so it narrows whatever set that left. A no-op when the drop set is empty
	 * (DROP_INGEST_CATEGORIES=""). Reversible: only the navigation set shrinks —
	 * getGroup() still resolves dropped ids. The degenerate ids are computed in
	 * one SQL pass via json_extract on the last edit (no JS group parsing).
	 */
	private applyIngestCategoryFilter(): void {
		const drop = degenerateCategories();
		if (drop.size === 0) return;
		const cats = [...drop];
		const placeholders = cats.map(() => '?').join(',');
		// json_array_length(...)-1 indexes the last edit; matches build.ts ordering
		// (edits sorted ascending, latest is last) and the dataset's final_after_text.
		const rows = this.db
			.prepare(
				`SELECT utterance_id FROM groups
				 WHERE json_extract(
				   json,
				   '$.edits[' || (json_array_length(json, '$.edits') - 1) || '].ingest_category'
				 ) IN (${placeholders})`
			)
			.all(...cats) as Array<{ utterance_id: string }>;
		const degenerate = new Set(rows.map((r) => r.utterance_id));
		if (degenerate.size === 0) return;
		const before = this.eligibleIds.length;
		const kept = this.eligibleIds.filter((id) => !degenerate.has(id));
		this.eligibleIds = kept;
		this.eligibleIdSet = new Set(kept);
		console.log(
			`[sqlite-repo] ingest-filter: dropped ${(before - kept.length).toLocaleString()} ` +
				`degenerate-latest-edit utterances (categories: ${cats.join(',')}); ` +
				`${kept.length.toLocaleString()} remain`
		);
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
		// Filtered rebuild: fold the exclusion digest into the runtime cache_hash
		// so dependent snapshots invalidate when exclusions change (the CSV hash
		// alone can't distinguish a filtered index from the full one).
		const cacheHash = cacheHashWithExclusions(sourceHash, meta.get('exclusions_hash'));

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
				`cache_hash=${cacheHash}; sidecar_labels=${sidecar.all().size}`
		);

		const repo = new SqliteRepo(db, orderedIds, cacheHash, sidecar, cdnMap);
		await repo.applyMeetingEligibility(stateDir, opts.meetingMinHumanUtterances);
		repo.applyIngestCategoryFilter();
		return repo;
	}

	get hash(): string {
		return this.cacheHash;
	}

	get total(): number {
		return this.eligibleIds.length;
	}

	/** Navigation universe after the meeting-eligibility filter, canonical order. */
	eligibleOrderedIds(): readonly string[] {
		return this.eligibleIds;
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

	async patchLabel(utterance_id: string, patch: GroupPatchBody, username?: string): Promise<GroupLabel | null> {
		const exists = this.getByIdStmt.get(utterance_id);
		if (!exists) return null;
		return this.sidecar.patch(utterance_id, patch, username);
	}

	async patchLabelExt(utterance_id: string, patch: GroupPatchBody, source: string): Promise<GroupLabel | null> {
		const exists = this.getByIdStmt.get(utterance_id);
		if (!exists) return null;
		return this.sidecar.patchWithSource(utterance_id, patch, source);
	}

	listUsernames(): string[] {
		return this.sidecar.listUsernames();
	}

	userCounts() {
		return this.sidecar.userCounts();
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

	allOrderedIds(): readonly string[] {
		return this.orderedIds;
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

	get labelsRevision(): number {
		return this.sidecar.revision;
	}

	idsByStatus(status: IncludeStatus): string[] {
		const out: string[] = [];
		if (status === 'unreviewed') {
			const labels = this.sidecar.all();
			// Iterate the eligible universe directly — ineligible meetings are
			// never offered for review, reviewed or not.
			for (const id of this.eligibleIds) {
				const lbl = labels.get(id);
				if (!lbl || lbl.include_status === 'unreviewed') out.push(id);
			}
			return out;
		}
		// Reviewed buckets enumerate the (smaller) sidecar label map, but a label
		// on an ineligible meeting must not leak back into the queue.
		for (const [id, lbl] of this.sidecar.all()) {
			if (lbl.include_status === status && this.isEligible(id)) out.push(id);
		}
		return out;
	}

	/** Ids whose label.error_categories contains `category`, canonical order.
	 *  Cheap: in-memory label scan only, no SQLite/group materialisation. */
	idsByErrorCategory(category: string): string[] {
		const out: string[] = [];
		for (const id of this.eligibleIds) {
			if (this.sidecar.get(id).error_categories.includes(category)) out.push(id);
		}
		return out;
	}

	groupsByErrorCategory(category: string): Group[] {
		// Labels live in memory; filter there first and only hit SQLite for
		// the matching ids. Avoids parsing the full 432 MB payload to find
		// "homophone".
		const matchingIds: string[] = [];
		for (const [id, lbl] of this.sidecar.all()) {
			if (lbl.error_categories.includes(category) && this.isEligible(id)) matchingIds.push(id);
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
		const arr = [...this.eligibleIds];
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
