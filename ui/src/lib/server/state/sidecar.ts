/**
 * Sidecar review state — append-only JSONL event log plus a periodically
 * rewritten snapshot.
 *
 * State on disk:
 *   ui/.state/review-events.jsonl       — every PATCH, in order
 *   ui/.state/review-labels.snapshot.json — latest label per utterance_id
 *
 * On boot we load the snapshot (fast path) then replay any events that
 * arrived after the snapshot's last_event_id. If the snapshot is missing or
 * corrupt we replay from event 0.
 *
 * Writes are serialised through a single in-process queue. We assume a single
 * node process — the prototype is local-only; serverless deploy is explicitly
 * out of scope (see decisions/storage.md).
 */

import { promises as fs } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { GroupLabel, GroupPatchBody } from '$lib/domain/groups';
import { DEFAULT_LABEL } from '$lib/domain/groups';
import { canonicalizeLabel, normalizeLabelCategories, validateCategoriesForWrite } from '$lib/domain/labels';

const SNAPSHOT_EVERY = 100; // flush snapshot after this many appended events

export interface SidecarPaths {
	eventsPath: string;
	snapshotPath: string;
}

export interface ReviewEventSource {
	kind: 'local';
	username: string | null;
}

/** Per-reviewer decision tallies. Counts decision *actions* (status-setting
 *  events), so re-deciding the same utterance counts again — this is an
 *  activity meter, not a distinct-utterance count. */
export interface UserCounts {
	include: number;
	exclude: number;
	uncertain: number;
	total: number;
}

const DECISION_STATUSES = new Set(['include', 'exclude', 'uncertain']);

export interface ReviewEvent {
	id: number;
	ts: string;
	utterance_id: string;
	/** Structured source. Legacy events written before username support have source: "local" (string). */
	source: ReviewEventSource | string;
	patch: GroupPatchBody;
}

interface Snapshot {
	last_event_id: number;
	labels: Record<string, GroupLabel>;
}

export class SidecarStore {
	private labels = new Map<string, GroupLabel>();
	private lastEventId = 0;
	private writeQueue: Promise<void> = Promise.resolve();
	private eventsAppendedSinceSnapshot = 0;

	constructor(private paths: SidecarPaths) {}

	static async load(stateDir: string): Promise<SidecarStore> {
		const paths: SidecarPaths = {
			eventsPath: resolve(stateDir, 'review-events.jsonl'),
			snapshotPath: resolve(stateDir, 'review-labels.snapshot.json')
		};
		await fs.mkdir(dirname(paths.eventsPath), { recursive: true });
		const store = new SidecarStore(paths);
		await store.rehydrate();
		return store;
	}

	private async rehydrate(): Promise<void> {
		// 1. Try snapshot. Build into locals first and commit atomically so a
		//    mid-iteration failure can't leave us with half-loaded labels and
		//    a stale lastEventId.
		try {
			const raw = await fs.readFile(this.paths.snapshotPath, 'utf8');
			const snap = JSON.parse(raw) as Snapshot;
			if (typeof snap.last_event_id !== 'number' || !Number.isFinite(snap.last_event_id) || snap.last_event_id < 0) {
				throw new Error('snapshot has invalid last_event_id');
			}
			const staged = new Map<string, GroupLabel>();
			for (const [id, lbl] of Object.entries(snap.labels)) {
				// Canonicalize on load so legacy single-value labels show up as arrays.
				staged.set(id, canonicalizeLabel(lbl));
			}
			this.labels = staged;
			this.lastEventId = snap.last_event_id;
		} catch {
			/* missing or corrupt snapshot — replay from scratch */
		}

		// 2. Replay events. Tolerate a truncated final line (likely a crash
		//    mid-write) but fail loudly on earlier corruption — that points at
		//    something we shouldn't silently paper over.
		let raw: string;
		try {
			raw = await fs.readFile(this.paths.eventsPath, 'utf8');
		} catch {
			return; // no event log yet
		}
		const lines = raw.split('\n');
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i];
			if (!line) continue;
			let ev: ReviewEvent;
			try {
				ev = JSON.parse(line) as ReviewEvent;
			} catch (err) {
				const isLast = i === lines.length - 1 || lines.slice(i + 1).every((l) => !l);
				if (isLast) {
					console.warn(`[sidecar] dropping truncated final event line at offset ${i}`);
					break;
				}
				throw new Error(`[sidecar] corrupt event at line ${i + 1}: ${(err as Error).message}`);
			}
			// Username + counts accumulate over the FULL history, independent of
			// the snapshot — every line is parsed here anyway, so tally before the
			// skip. (Doing it after the `continue` would miss every reviewer whose
			// events predate the snapshot.)
			this.recordUsername(ev);
			this.bumpUserCounts(ev);
			if (ev.id <= this.lastEventId) continue; // already folded into snapshot — don't re-apply
			this.apply(ev);
			this.lastEventId = ev.id;
		}
	}

	private apply(ev: ReviewEvent): void {
		const current = this.labels.get(ev.utterance_id) ?? { ...DEFAULT_LABEL };
		const next: GroupLabel = { ...current, error_categories: [...current.error_categories] };
		const patch = ev.patch as Record<string, unknown>;

		// Categories: normalize from the patch (accepts legacy single-field too).
		// If either category field is present in the patch we replace the whole array.
		if ('error_categories' in patch || 'error_category' in patch) {
			next.error_categories = normalizeLabelCategories(patch);
		}

		// Scalar fields: omitted = keep current, null = clear, otherwise set.
		for (const k of ['include_status', 'adjusted_start', 'adjusted_end', 'reviewer_notes'] as const) {
			if (!(k in patch)) continue;
			(next as unknown as Record<string, unknown>)[k] = patch[k];
		}
		next.human_updated_at = ev.ts;
		this.labels.set(ev.utterance_id, next);
	}

	get(utterance_id: string): GroupLabel {
		return this.labels.get(utterance_id) ?? { ...DEFAULT_LABEL };
	}

	all(): ReadonlyMap<string, GroupLabel> {
		return this.labels;
	}

	/** Monotonic revision — last appended event id. Use as cache key. */
	get revision(): number {
		return this.lastEventId;
	}

	listUsernames(): string[] {
		const seen = new Set<string>();
		// We can't read the JSONL here synchronously — usernames are collected
		// in-memory during rehydrate via a separate pass. For simplicity we keep
		// a live set updated on every write.
		return [...this._usernames].sort();
	}

	private _usernames = new Set<string>();
	private _userCounts = new Map<string, UserCounts>();

	private recordUsername(ev: ReviewEvent): void {
		if (typeof ev.source === 'object' && ev.source.username) {
			this._usernames.add(ev.source.username);
		}
	}

	/**
	 * Tally a decision against its reviewer. O(1) per event, run during both
	 * boot replay and live writes, so /api/users can surface per-user counts
	 * without re-scanning the log (no UI lag). Only local human reviews that
	 * set a real decision status are counted (LLM `ext-*` writes are skipped).
	 */
	private bumpUserCounts(ev: ReviewEvent): void {
		if (typeof ev.source !== 'object' || !ev.source.username) return;
		const status = (ev.patch as Record<string, unknown>).include_status;
		if (typeof status !== 'string' || !DECISION_STATUSES.has(status)) return;
		const u = ev.source.username;
		const c = this._userCounts.get(u) ?? { include: 0, exclude: 0, uncertain: 0, total: 0 };
		c[status as 'include' | 'exclude' | 'uncertain'] += 1;
		c.total += 1;
		this._userCounts.set(u, c);
	}

	/** Per-reviewer decision tallies. Keyed by username. */
	userCounts(): Record<string, UserCounts> {
		return Object.fromEntries(this._userCounts);
	}

	async patch(utterance_id: string, patch: GroupPatchBody, username?: string): Promise<GroupLabel> {
		// Validate before queueing. `async` ensures the throw becomes a
		// rejected Promise rather than a synchronous exception inside the
		// SvelteKit handler.
		const canonicalPatch = canonicalizePatchForWrite(patch);
		validatePatch(canonicalPatch);
		const run = () => this.doPatch(utterance_id, canonicalPatch, { kind: 'local', username: username ?? null });
		const op = this.writeQueue.then(run, run);
		this.writeQueue = op.catch(() => {});
		await op;
		return this.get(utterance_id);
	}

	/**
	 * Write a patch with an explicit string source (e.g. "ext-gemini-2.5-pro").
	 * Used by the external-LLM ingest path. Source must already be sanitized
	 * by the caller; this method does not slugify.
	 */
	async patchWithSource(utterance_id: string, patch: GroupPatchBody, source: string): Promise<GroupLabel> {
		if (typeof source !== 'string' || !/^[a-z0-9._-]{1,64}$/i.test(source)) {
			throw new Error(`invalid source slug: ${source}`);
		}
		const canonicalPatch = canonicalizePatchForWrite(patch);
		validatePatch(canonicalPatch);
		const run = () => this.doPatch(utterance_id, canonicalPatch, source);
		const op = this.writeQueue.then(run, run);
		this.writeQueue = op.catch(() => {});
		await op;
		return this.get(utterance_id);
	}

	private async doPatch(utterance_id: string, patch: GroupPatchBody, source: ReviewEventSource | string): Promise<void> {
		const tentativeId = this.lastEventId + 1;
		const ev: ReviewEvent = {
			id: tentativeId,
			ts: new Date().toISOString(),
			utterance_id,
			source,
			patch
		};
		const handle = await fs.open(this.paths.eventsPath, 'a');
		try {
			await handle.appendFile(JSON.stringify(ev) + '\n');
			await handle.datasync();
		} finally {
			await handle.close();
		}
		this.lastEventId = tentativeId;
		this.apply(ev);
		this.recordUsername(ev);
		this.bumpUserCounts(ev);
		this.eventsAppendedSinceSnapshot += 1;
		if (this.eventsAppendedSinceSnapshot >= SNAPSHOT_EVERY) {
			await this.flushSnapshot();
		}
	}

	private async flushSnapshot(): Promise<void> {
		const snap: Snapshot = {
			last_event_id: this.lastEventId,
			labels: Object.fromEntries(this.labels)
		};
		const tmp = `${this.paths.snapshotPath}.tmp`;
		const handle = await fs.open(tmp, 'w');
		try {
			await handle.writeFile(JSON.stringify(snap));
			await handle.sync();
		} finally {
			await handle.close();
		}
		await fs.rename(tmp, this.paths.snapshotPath);
		// fsync parent dir so the rename is durable across power loss.
		// Best-effort: directory fsync is Linux-canonical but inconsistent on
		// macOS dev machines, so swallow unsupported errors silently.
		try {
			const dir = await fs.open(dirname(this.paths.snapshotPath), 'r');
			try { await dir.sync(); } finally { await dir.close(); }
		} catch {
			/* dev/macOS may not support directory fsync — best-effort only */
		}
		this.eventsAppendedSinceSnapshot = 0;
	}

	/** Test helper — wait for all queued writes to settle. */
	flush(): Promise<void> {
		return this.writeQueue;
	}
}

/**
 * Convert a wire-shaped patch (which may use legacy `error_category`) into the
 * canonical shape we persist: a single `error_categories: string[]` field, no
 * `error_category` key. Other fields pass through unchanged.
 */
function canonicalizePatchForWrite(patch: GroupPatchBody): GroupPatchBody {
	const out: GroupPatchBody = { ...patch };
	if ('error_categories' in out || 'error_category' in out) {
		const cats = normalizeLabelCategories({
			error_categories: out.error_categories,
			error_category: out.error_category
		});
		delete out.error_category;
		out.error_categories = cats;
	}
	return out;
}

function validatePatch(patch: GroupPatchBody): void {
	if (patch.error_categories !== undefined) {
		validateCategoriesForWrite(patch.error_categories);
	}
	if ('include_status' in patch && patch.include_status !== undefined) {
		const ok = ['unreviewed', 'include', 'exclude', 'uncertain'];
		if (!ok.includes(patch.include_status)) {
			throw new Error(`invalid include_status: ${patch.include_status}`);
		}
	}
	for (const key of ['adjusted_start', 'adjusted_end'] as const) {
		if (patch[key] === undefined || patch[key] === null) continue;
		const v = patch[key]!;
		if (!Number.isFinite(v) || v < 0) {
			throw new Error(`invalid ${key}: ${v}`);
		}
	}
	if (
		patch.adjusted_start != null &&
		patch.adjusted_end != null &&
		patch.adjusted_start >= patch.adjusted_end
	) {
		throw new Error('adjusted_start must be < adjusted_end');
	}
}
