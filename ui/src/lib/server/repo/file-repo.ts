/**
 * In-memory, file-backed repository.
 *
 * - Reads ui/.cache/groups.v1.json once at startup (or lazily on first use).
 * - Reads label state from the sidecar JSONL+snapshot.
 * - Applies an optional audio CDN URL map (data/audio-fix/url-map.json).
 * - Computes seeded review order via mulberry32 — same seed → same order.
 *
 * Single-process assumption. See decisions/storage.md.
 */

import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';
import type { Group, GroupLabel, GroupPatchBody, CacheMeta, QueueResponse } from '$lib/domain/groups';
import { cacheHashWithExclusions } from '$lib/domain/groups';
import type { IncludeStatus } from '$lib/domain/types';
import { SidecarStore } from '../state/sidecar';
import {
	loadOrComputeEligibleMeetings,
	meetingEligibilityThreshold,
	meetingKey
} from '../state/meeting-eligibility';
import {
	degenerateCategories,
	filterDegenerateIds,
	latestEditCategory
} from '../state/ingest-filter';

// Same convention as SqliteRepo: defaults resolve from `process.cwd()` so the
// bundled adapter-node output picks up the operator's working directory.
const DEFAULT_CACHE_DIR = (): string => resolve(process.cwd(), '.cache');
const DEFAULT_STATE_DIR = (): string => resolve(process.cwd(), '.state');
const DEFAULT_CDN_MAP = (): string =>
	resolve(process.cwd(), '..', 'data', 'audio-fix', 'url-map.json');

export interface RepoOptions {
	cacheDir?: string;
	stateDir?: string;
	audioCdnMapPath?: string;
	/** Overrides MEETING_MIN_HUMAN_UTTERANCES (tests pass this explicitly). */
	meetingMinHumanUtterances?: number;
}

export class FileRepo {
	/** edit_id → utterance_id index. Built lazily on first use. */
	private editToUtteranceIndex: Map<string, string> | null = null;

	// Navigation universe after the meeting-eligibility filter (see
	// meeting-eligibility.ts). `eligibleIdSet` is null when the filter is
	// disabled (threshold ≤ 0), meaning "everything is eligible".
	private eligibleIds: string[];
	private eligibleIdSet: Set<string> | null = null;

	private constructor(
		private groupsById: Map<string, Group>,
		private orderedIds: string[],
		private cacheHash: string,
		private sidecar: SidecarStore,
		private audioCdnMap: Map<string, string>
	) {
		this.eligibleIds = orderedIds;
	}

	/** True when `utterance_id` survives the meeting-eligibility filter. */
	private isEligible(utterance_id: string): boolean {
		return this.eligibleIdSet ? this.eligibleIdSet.has(utterance_id) : true;
	}

	/**
	 * Narrow the navigation universe to utterances whose meeting clears the
	 * human-correction threshold. Runs once during load(). Threshold ≤ 0
	 * disables it. Shares the same snapshot as SqliteRepo (keyed by cache_hash +
	 * threshold), so the two repos agree on the eligible set for the same data.
	 */
	private async applyMeetingEligibility(stateDir: string, thresholdOverride?: number): Promise<void> {
		const threshold = thresholdOverride ?? meetingEligibilityThreshold();
		if (threshold <= 0) {
			this.eligibleIds = this.orderedIds;
			this.eligibleIdSet = null;
			return;
		}
		const eligibleMeetings = await loadOrComputeEligibleMeetings(this, stateDir, threshold);
		const elig: string[] = [];
		for (const id of this.orderedIds) {
			const g = this.groupsById.get(id);
			if (g?.meeting_id && eligibleMeetings.has(meetingKey(g.city_id, g.meeting_id))) {
				elig.push(id);
			}
		}
		this.eligibleIds = elig;
		this.eligibleIdSet = new Set(elig);
		console.log(
			`[file-repo] meeting-eligibility: ${elig.length.toLocaleString()}/` +
				`${this.orderedIds.length.toLocaleString()} utterances eligible (threshold=${threshold})`
		);
	}

	/**
	 * Second eligibility layer: drop utterances whose LATEST edit is a degenerate
	 * ingest bin. Mirrors SqliteRepo.applyIngestCategoryFilter; runs after the
	 * meeting filter and narrows whatever it left. No-op when the drop set is
	 * empty. Reversible — getGroup() still resolves dropped ids. Groups are
	 * already in memory here, so the latest-edit category is a cheap lookup.
	 */
	private applyIngestCategoryFilter(): void {
		const drop = degenerateCategories();
		if (drop.size === 0) return;
		const before = this.eligibleIds.length;
		const { kept } = filterDegenerateIds(
			this.eligibleIds,
			(id) => {
				const g = this.groupsById.get(id);
				return g ? latestEditCategory(g) : null;
			},
			drop
		);
		if (kept.length === before) return;
		this.eligibleIds = kept;
		this.eligibleIdSet = new Set(kept);
		console.log(
			`[file-repo] ingest-filter: dropped ${(before - kept.length).toLocaleString()} ` +
				`degenerate-latest-edit utterances (categories: ${[...drop].join(',')}); ` +
				`${kept.length.toLocaleString()} remain`
		);
	}

	static async load(opts: RepoOptions = {}): Promise<FileRepo> {
		const cacheDir = opts.cacheDir ?? process.env.REVIEW_CACHE_DIR ?? DEFAULT_CACHE_DIR();
		const stateDir = opts.stateDir ?? process.env.REVIEW_STATE_DIR ?? DEFAULT_STATE_DIR();
		const cdnMapPath =
			opts.audioCdnMapPath ?? process.env.REVIEW_AUDIO_MAP_PATH ?? DEFAULT_CDN_MAP();

		const meta = JSON.parse(await fs.readFile(resolve(cacheDir, 'meta.json'), 'utf8')) as CacheMeta;
		const groups = JSON.parse(
			await fs.readFile(resolve(cacheDir, 'groups.v1.json'), 'utf8')
		) as Group[];

		// Load the CDN/mirror map but DO NOT preemptively use it as the primary
		// playback URL. The client tries the original opencouncil URL first;
		// the mirror is only consulted by the waveform's fallback chain on real
		// decode failures. See decisions/audio.md.
		const cdnMap = new Map<string, string>();
		try {
			const raw = JSON.parse(await fs.readFile(cdnMapPath, 'utf8')) as Record<string, string>;
			for (const [k, v] of Object.entries(raw)) cdnMap.set(k, v);
		} catch {
			/* no mapping file — fine */
		}
		// audio_cdn_url stays null on every group — the mirror surfaces only
		// through /api/audio/mirror-map → audio-source.ts.

		const groupsById = new Map<string, Group>();
		const orderedIds: string[] = [];
		for (const g of groups) {
			groupsById.set(g.utterance_id, g);
			orderedIds.push(g.utterance_id);
		}

		const sidecar = await SidecarStore.load(stateDir);

		// Filtered rebuild: fold the exclusion digest into the runtime cache_hash
		// (see cacheHashWithExclusions) so dependent snapshots invalidate.
		const cacheHash = cacheHashWithExclusions(meta.source_hash, meta.exclusions?.exclusions_hash);

		console.log(
			`[file-repo] loaded ${groups.length.toLocaleString()} groups; ` +
				`cache_hash=${cacheHash}; sidecar_labels=${sidecar.all().size}`
		);

		const repo = new FileRepo(groupsById, orderedIds, cacheHash, sidecar, cdnMap);
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
		const g = this.groupsById.get(utterance_id);
		if (!g) return null;
		return { ...g, label: this.sidecar.get(utterance_id) };
	}

	queue(seed: number, from: number, n: number): QueueResponse {
		const order = this.seededOrder(seed);
		const start = Math.max(0, Math.min(order.length, Math.floor(from)));
		const count = Math.max(0, Math.min(50, Math.floor(n)));
		const slice = order.slice(start, start + count);
		const groups = slice.map((id) => this.getGroup(id)!).filter(Boolean);
		const next_cursor = start + count < order.length ? start + count : null;
		return { cache_hash: this.cacheHash, total: order.length, groups, next_cursor };
	}

	async patchLabel(utterance_id: string, patch: GroupPatchBody, username?: string): Promise<GroupLabel | null> {
		if (!this.groupsById.has(utterance_id)) return null;
		return this.sidecar.patch(utterance_id, patch, username);
	}

	/**
	 * Write a label patch with an explicit source slug (e.g. "ext-gemini-2.5-pro").
	 * Used by the external-LLM ingest path. Returns null when the utterance is
	 * unknown to the repo.
	 */
	async patchLabelExt(utterance_id: string, patch: GroupPatchBody, source: string): Promise<GroupLabel | null> {
		if (!this.groupsById.has(utterance_id)) return null;
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

	allGroups(): Group[] {
		return this.orderedIds.map((id) => this.getGroup(id)!);
	}

	allOrderedIds(): readonly string[] {
		return this.orderedIds;
	}

	*iterGroups(): IterableIterator<Group> {
		for (const id of this.orderedIds) {
			const g = this.getGroup(id);
			if (g) yield g;
		}
	}

	/** edit_id → utterance_id lookup; returns null when the edit is unknown. */
	utteranceIdForEdit(edit_id: string): string | null {
		if (!this.editToUtteranceIndex) {
			const map = new Map<string, string>();
			for (const g of this.groupsById.values()) {
				for (const e of g.edits) map.set(e.edit_id, g.utterance_id);
			}
			this.editToUtteranceIndex = map;
		}
		return this.editToUtteranceIndex.get(edit_id) ?? null;
	}

	get labelsRevision(): number {
		return this.sidecar.revision;
	}

	/** utterance_ids whose label.include_status matches `status`. */
	idsByStatus(status: IncludeStatus): string[] {
		const out: string[] = [];
		if (status === 'unreviewed') {
			const labels = this.sidecar.all();
			for (const id of this.eligibleIds) {
				const lbl = labels.get(id);
				if (!lbl || lbl.include_status === 'unreviewed') out.push(id);
			}
			return out;
		}
		for (const [id, lbl] of this.sidecar.all()) {
			if (lbl.include_status === status && this.isEligible(id)) out.push(id);
		}
		return out;
	}

	/** Ids whose label.error_categories contains `category`, canonical order.
	 *  Cheap: in-memory label scan, no group materialisation. */
	idsByErrorCategory(category: string): string[] {
		const out: string[] = [];
		for (const id of this.eligibleIds) {
			if (this.sidecar.get(id).error_categories.includes(category)) out.push(id);
		}
		return out;
	}

	/** Groups whose label.error_categories array contains `category`. */
	groupsByErrorCategory(category: string): Group[] {
		const out: Group[] = [];
		for (const id of this.eligibleIds) {
			const lbl = this.sidecar.get(id);
			if (lbl.error_categories.includes(category)) {
				const g = this.getGroup(id);
				if (g) out.push(g);
			}
		}
		return out;
	}

	flush(): Promise<void> {
		return this.sidecar.flush();
	}

	// Mulberry32: small, deterministic 32-bit PRNG. Sufficient for shuffle.
	private seededOrder(seed: number): string[] {
		const cached = this.orderCache.get(seed);
		if (cached) {
			// LRU refresh — keep recently-used seeds (paging callers reuse the same seed).
			this.orderCache.delete(seed);
			this.orderCache.set(seed, cached);
			return cached;
		}
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
		// Bounded LRU: prevents memory DoS via /api/review/queue?seed=<random>.
		// Each entry holds a full id array (~10k strings); cap keeps it bounded.
		if (this.orderCache.size >= ORDER_CACHE_MAX) {
			const oldest = this.orderCache.keys().next().value;
			if (oldest !== undefined) this.orderCache.delete(oldest);
		}
		this.orderCache.set(seed, arr);
		return arr;
	}

	private orderCache = new Map<number, string[]>();
}

const ORDER_CACHE_MAX = 8;

// Module-level singleton wired into SvelteKit endpoints. Lazy: first call
// loads cache and sidecar; later calls return the same instance.
let _repo: FileRepo | null = null;
let _loading: Promise<FileRepo> | null = null;

export async function getFileRepo(): Promise<FileRepo> {
	if (_repo) return _repo;
	if (_loading) return _loading;
	const attempt = FileRepo.load().then(
		(r) => {
			_repo = r;
			return r;
		},
		(err: unknown) => {
			// Don't cache the rejection forever — let the next caller retry.
			if (_loading === attempt) _loading = null;
			throw err;
		}
	);
	_loading = attempt;
	return attempt;
}

/** Test helper — reset the singleton so tests can load alternative paths. */
export function _resetFileRepoForTests(): void {
	_repo = null;
	_loading = null;
}
