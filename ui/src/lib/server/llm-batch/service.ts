/**
 * External-LLM categorization orchestrator service.
 *
 * Used by the `/api/llm-batch/*` endpoints. Has no SvelteKit dependency so it
 * can be tested with a real FileRepo against a temp state dir.
 *
 * Contract summary:
 *   - `buildBatch` issues a batch of unlabeled utterance ids, persists the
 *     issued-id set to disk under `<stateDir>/llm-batches/<batch_id>.json`,
 *     and returns the prompt + items ready to copy into Gemini/AI Studio.
 *   - `ingestBatch` accepts the operator's pasted reply, verifies the
 *     `BATCH_ID:` echo, validates each element strictly, and appends
 *     `ext-<model>` events through the existing sidecar path.
 *   - `getStats` reports current label coverage by source.
 *
 * Already-labeled utterances are excluded from both /next issuance and
 * /ingest acceptance, regardless of source. There is no overwrite path.
 */

import { promises as fs } from 'node:fs';
import { randomBytes, createHash } from 'node:crypto';
import { resolve } from 'node:path';
import type { ReviewRepo } from '../repo';
import { TAXONOMY_MAP, normalizeTaxonomyId } from '$lib/shared/taxonomy';
import { renderPrompt, type BatchItem } from './prompt';

const BATCHES_SUBDIR = 'llm-batches';
const MAX_CATS_PER_ITEM = 3;
const MAX_INGEST_BYTES = 4 * 1024 * 1024;

export interface BuildBatchOpts {
	n: number;
	model: string;
	stateDir: string;
	/** Optional seed for deterministic ordering — defaults to Date.now() so successive calls return different slices. */
	seed?: number;
	/**
	 * When true, items whose current label has an empty `error_categories` set
	 * (e.g. declined by a previous external model) are eligible again. Default
	 * false so the human review UI doesn't loop on the same indecisive items.
	 */
	include_empty?: boolean;
}

export interface BuildBatchResult {
	batch_id: string;
	model: string;
	items: BatchItem[];
	prompt: string;
	remaining: number;
}

interface BatchFile {
	batch_id: string;
	model: string;
	created_at: string;
	ids: string[];
	include_empty?: boolean;
}

export interface IngestOpts {
	batch_id: string;
	raw: string;
	stateDir: string;
}

export interface RejectedItem {
	id: string;
	reason:
		| 'not_in_batch'
		| 'no_valid_categories'
		| 'unknown_utterance'
		| 'too_many_categories'
		| 'malformed_element';
}

export interface IngestResult {
	batch_id: string;
	model: string;
	accepted: number;
	declined: string[];
	duplicates: string[];
	rejected: RejectedItem[];
	missing_from_batch: string[];
	extra_ids: string[];
}

export interface Stats {
	total: number;
	labeled: number;
	remaining: number;
	by_source: Record<string, number>;
}

/** Slugify a model name to `[a-z0-9._-]{1,40}`. */
export function slugifyModel(raw: string): string {
	if (typeof raw !== 'string') return 'unknown';
	const cleaned = raw
		.toLowerCase()
		.replace(/[^a-z0-9._-]+/g, '-')
		.replace(/^-+|-+$/g, '')
		.replace(/-{2,}/g, '-')
		.slice(0, 40);
	return cleaned || 'unknown';
}

function batchPath(stateDir: string, batch_id: string): string {
	return resolve(stateDir, BATCHES_SUBDIR, `${batch_id}.json`);
}

/** Short, easy-to-eyeball id. 6 hex chars = 16 M possibilities, collisions vanishingly rare in practice. */
function newBatchId(): string {
	return randomBytes(3).toString('hex');
}

/** Stable-content batch_id (not used right now but handy for canary/probe batches). */
export function hashIds(ids: readonly string[]): string {
	const h = createHash('sha256');
	for (const id of [...ids].sort()) h.update(id).update('\n');
	return h.digest('hex').slice(0, 8);
}

export async function buildBatch(repo: ReviewRepo, opts: BuildBatchOpts): Promise<BuildBatchResult> {
	const model = slugifyModel(opts.model);
	const n = Math.max(1, Math.min(500, Math.floor(opts.n)));

	const labels = repo.allLabels();
	const allGroups = repo.allGroups();
	const includeEmpty = opts.include_empty === true;

	// Pick the first N eligible groups in seeded order so parallel batches
	// requested back-to-back don't all return the same items. We use a small
	// deterministic shuffle keyed by Date.now() so a refresh gets a fresh slice.
	const seed = (opts.seed ?? Date.now()) >>> 0 || 1;
	const order = seededOrder(allGroups.length, seed);
	const items: BatchItem[] = [];
	for (const idx of order) {
		if (items.length === n) break;
		const g = allGroups[idx];
		const lbl = labels.get(g.utterance_id);
		if (lbl) {
			// Already has a sidecar entry. Skip unless caller opts into retrying
			// items whose only label is an empty error_categories (a previous
			// decline).
			if (!includeEmpty) continue;
			if (lbl.error_categories.length > 0) continue;
		}
		items.push({
			id: g.utterance_id,
			b: g.initial_before_text,
			a: g.final_after_text
		});
	}

	let batch_id = '';
	for (let attempt = 0; attempt < 8; attempt++) {
		batch_id = newBatchId();
		try {
			await fs.mkdir(resolve(opts.stateDir, BATCHES_SUBDIR), { recursive: true });
			await fs.writeFile(
				batchPath(opts.stateDir, batch_id),
				JSON.stringify({
					batch_id,
					model,
					created_at: new Date().toISOString(),
					ids: items.map((i) => i.id),
					include_empty: includeEmpty
				} satisfies BatchFile),
				{ flag: 'wx' }
			);
			break;
		} catch (err: unknown) {
			// `wx` flag means we collided — try again with a fresh id.
			if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err;
		}
	}
	if (!batch_id) throw new Error('failed to allocate batch_id after 8 attempts');

	const prompt = renderPrompt(batch_id, items);

	// Remaining = unlabeled across the whole repo, not just outside this batch.
	let remaining = 0;
	for (const g of allGroups) if (!labels.has(g.utterance_id)) remaining += 1;

	return { batch_id, model, items, prompt, remaining };
}

export async function ingestBatch(repo: ReviewRepo, opts: IngestOpts): Promise<IngestResult> {
	if (typeof opts.raw !== 'string') throw new Error('raw must be a string');
	if (opts.raw.length > MAX_INGEST_BYTES) throw new Error('paste too large (> 4 MB)');

	const batchFile = await readBatchFile(opts.stateDir, opts.batch_id);
	if (!batchFile) throw new Error(`unknown batch: ${opts.batch_id}`);

	const echoed = extractBatchIdMarker(opts.raw);
	if (echoed && echoed !== batchFile.batch_id) {
		throw new Error(
			`paste mismatch: response carries BATCH_ID ${echoed} but you targeted ${batchFile.batch_id}`
		);
	}

	const parsed = parseRawArray(opts.raw);

	const source = `ext-${batchFile.model}`;
	const issuedSet = new Set(batchFile.ids);
	const labels = repo.allLabels();

	const accepted: string[] = [];
	const duplicates: string[] = [];
	const rejected: RejectedItem[] = [];
	const extras: string[] = [];
	const seen = new Set<string>();
	const declared = new Set<string>(); // ids declared with non-empty c

	for (const raw of parsed) {
		if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
			rejected.push({ id: '(malformed)', reason: 'malformed_element' });
			continue;
		}
		const r = raw as Record<string, unknown>;
		const id = typeof r.id === 'string' ? r.id : typeof r.utterance_id === 'string' ? (r.utterance_id as string) : null;
		const cRaw = (r.c ?? r.categories ?? r.labels) as unknown;
		if (!id) {
			rejected.push({ id: '(no-id)', reason: 'malformed_element' });
			continue;
		}
		if (seen.has(id)) continue; // dedup repeats in the same paste
		seen.add(id);

		if (!issuedSet.has(id)) {
			extras.push(id);
			rejected.push({ id, reason: 'not_in_batch' });
			continue;
		}

		const cats = normalizeCategories(cRaw);
		if (cats.length === 0) {
			rejected.push({ id, reason: 'no_valid_categories' });
			continue;
		}
		if (cats.length > MAX_CATS_PER_ITEM) {
			rejected.push({ id, reason: 'too_many_categories' });
			continue;
		}

		const existing = labels.get(id);
		if (existing) {
			// If the batch was issued with include_empty, overwrite an empty
			// label with the new categories; otherwise treat as duplicate.
			if (!batchFile.include_empty || existing.error_categories.length > 0) {
				duplicates.push(id);
				continue;
			}
		}

		const result = await repo.patchLabelExt(id, { error_categories: cats }, source);
		if (!result) {
			rejected.push({ id, reason: 'unknown_utterance' });
			continue;
		}
		accepted.push(id);
		declared.add(id);
	}

	// Omitted ids = explicit decline (the model had no opinion). Persist as
	// empty-categories events so they're never re-issued.
	//
	// When the batch was issued with include_empty (retrying prior declines)
	// AND the model declined again, we skip writing a new empty event — the
	// existing empty label already records that decision and we don't want to
	// inflate the event log with no-op duplicates.
	const declined: string[] = [];
	for (const id of batchFile.ids) {
		if (declared.has(id)) continue;
		// Also skip ids the operator's paste mentioned but were rejected; we'll
		// re-issue them in a later batch so we don't lock in a bad signal.
		if (seen.has(id)) continue;
		const existing = labels.get(id);
		if (existing) continue; // any existing label (empty or not) means we've already recorded a decision
		const result = await repo.patchLabelExt(id, { error_categories: [] }, source);
		if (result) declined.push(id);
	}

	return {
		batch_id: batchFile.batch_id,
		model: batchFile.model,
		accepted: accepted.length,
		declined,
		duplicates,
		rejected,
		missing_from_batch: batchFile.ids.filter((id) => !seen.has(id) && !declined.includes(id)),
		extra_ids: extras
	};
}

export async function getStats(repo: ReviewRepo, stateDir: string): Promise<Stats> {
	const total = repo.total;
	const labels = repo.allLabels();
	const by_source = await countBySource(stateDir);
	return {
		total,
		labeled: labels.size,
		remaining: total - labels.size,
		by_source
	};
}

/**
 * Last-write-wins per-utterance source counts derived from the JSONL event
 * log. We scan once per call — the page only refreshes after ingest, so this
 * is cheap enough and avoids carrying source on every in-memory label.
 */
async function countBySource(stateDir: string): Promise<Record<string, number>> {
	const path = resolve(stateDir, 'review-events.jsonl');
	const latestSource = new Map<string, string>();
	let raw: string;
	try {
		raw = await fs.readFile(path, 'utf8');
	} catch {
		return {};
	}
	for (const line of raw.split('\n')) {
		if (!line) continue;
		let ev: { utterance_id?: string; source?: unknown };
		try {
			ev = JSON.parse(line);
		} catch {
			continue;
		}
		if (!ev.utterance_id) continue;
		const src = typeof ev.source === 'string'
			? ev.source
			: typeof ev.source === 'object' && ev.source !== null && (ev.source as { kind?: string }).kind === 'local'
				? 'local'
				: 'unknown';
		latestSource.set(ev.utterance_id, src);
	}
	const counts: Record<string, number> = {};
	for (const src of latestSource.values()) counts[src] = (counts[src] ?? 0) + 1;
	return counts;
}

async function readBatchFile(stateDir: string, batch_id: string): Promise<BatchFile | null> {
	if (!/^[a-f0-9]{4,32}$/.test(batch_id)) return null;
	try {
		const raw = await fs.readFile(batchPath(stateDir, batch_id), 'utf8');
		return JSON.parse(raw) as BatchFile;
	} catch {
		return null;
	}
}

function extractBatchIdMarker(raw: string): string | null {
	const m = /BATCH_ID\s*:\s*([a-f0-9]{4,32})/i.exec(raw);
	return m ? m[1].toLowerCase() : null;
}

function parseRawArray(raw: string): unknown[] {
	// Strip ```json fences if present.
	let body = raw.trim();
	body = body.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/i, '');
	// Drop leading BATCH_ID: line and any preamble before the first `[`.
	const lb = body.indexOf('[');
	if (lb < 0) throw new Error('no JSON array found in paste');
	body = body.slice(lb);
	// Trim trailing prose after the last `]`.
	const rb = body.lastIndexOf(']');
	if (rb < 0) throw new Error('JSON array is unclosed');
	body = body.slice(0, rb + 1);
	let parsed: unknown;
	try {
		parsed = JSON.parse(body);
	} catch (err) {
		throw new Error(`paste did not parse as JSON array: ${(err as Error).message}`);
	}
	if (!Array.isArray(parsed)) throw new Error('paste root must be an array');
	return parsed;
}

function normalizeCategories(raw: unknown): string[] {
	if (!Array.isArray(raw)) return [];
	const out: string[] = [];
	const seen = new Set<string>();
	for (const item of raw) {
		if (typeof item !== 'string') continue;
		const norm = normalizeTaxonomyId(item);
		if (!norm) continue;
		if (!(norm in TAXONOMY_MAP)) continue;
		if (seen.has(norm)) continue;
		seen.add(norm);
		out.push(norm);
	}
	return out;
}

function seededOrder(n: number, seed: number): number[] {
	const arr = Array.from({ length: n }, (_, i) => i);
	let state = seed;
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
