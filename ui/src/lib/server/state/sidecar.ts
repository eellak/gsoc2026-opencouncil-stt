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

export interface ReviewEvent {
	id: number;
	ts: string;
	utterance_id: string;
	source: string; // "local" for now; future imports can use a different tag
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
			if (ev.id <= this.lastEventId) continue; // already in snapshot
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

	async patch(utterance_id: string, patch: GroupPatchBody): Promise<GroupLabel> {
		// Validate before queueing. `async` ensures the throw becomes a
		// rejected Promise rather than a synchronous exception inside the
		// SvelteKit handler.
		const canonicalPatch = canonicalizePatchForWrite(patch);
		validatePatch(canonicalPatch);
		// Pass `null` as onRejected so a poisoned previous op does NOT skip the
		// next doPatch. The chain's tail swallows errors so subsequent callers
		// see a fresh, fulfilled writeQueue; the current caller still observes
		// this op's outcome via `await op`.
		const run = () => this.doPatch(utterance_id, canonicalPatch);
		const op = this.writeQueue.then(run, run);
		this.writeQueue = op.catch(() => {});
		await op;
		return this.get(utterance_id);
	}

	private async doPatch(utterance_id: string, patch: GroupPatchBody): Promise<void> {
		// Tentatively assign the next id, but DO NOT mutate in-memory state until
		// the durable append succeeds. If the write fails we leave lastEventId
		// untouched so the next patch retries with the same id and there is no
		// in-memory/disk divergence.
		const tentativeId = this.lastEventId + 1;
		const ev: ReviewEvent = {
			id: tentativeId,
			ts: new Date().toISOString(),
			utterance_id,
			source: 'local',
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
