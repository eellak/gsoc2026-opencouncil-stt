/**
 * Build the local surrounding-context index: fetch each eligible meeting's full
 * JSON once, extract transcript rows, and assemble the manifest that
 * `buildSqlite` writes into the `transcript` / `transcript_meeting` tables.
 *
 * Resumable by meeting: each fetched meeting JSON is cached as a blob under
 * `blobDir`; a re-run reads the blob instead of re-fetching. Concurrency-limited
 * so a 1 GB VM (or a build box) never buffers hundreds of bodies at once.
 * Failure policy: 4xx = permanent (recorded, skipped), 5xx/network = transient
 * (bounded retries, exponential backoff + jitter). Partial success is allowed
 * and surfaced via `manifest.build_status` + `failed_count`; the per-meeting
 * `manifest.meetings` list is what lets the runtime fall back live ONLY for the
 * meetings that didn't make it.
 */

import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';
import {
	extractTranscriptRows,
	TRANSCRIPT_SCHEMA_VERSION,
	type TranscriptRow
} from './transcript-extract';
import type { TranscriptManifest } from './build-sqlite';

export interface MeetingRef {
	city_id: string;
	meeting_id: string;
}

export interface FetchOutcome {
	ok: boolean;
	status: number;
	json?: unknown;
}

export interface BuildTranscriptOptions {
	meetings: MeetingRef[];
	/** Directory for resumable raw-JSON blobs (e.g. `.cache/meetings`). */
	blobDir: string;
	/** Injectable for tests; defaults to the live OpenCouncil meeting endpoint. */
	fetchMeeting?: (m: MeetingRef) => Promise<FetchOutcome>;
	concurrency?: number;
	maxRetries?: number;
	log?: (msg: string) => void;
	/** Injectable for deterministic backoff in tests. */
	sleep?: (ms: number) => Promise<void>;
}

export interface BuildTranscriptResult {
	rows: TranscriptRow[];
	manifest: TranscriptManifest;
	failed: MeetingRef[];
}

const OPENCOUNCIL_BASE = 'https://opencouncil.gr/api/cities';
const FETCH_TIMEOUT_MS = 20_000;
const MAX_BODY_BYTES = 25 * 1024 * 1024;

function blobName(m: MeetingRef): string {
	// city/meeting ids are slugs/cuids; sanitise just in case for a flat filename.
	const safe = (s: string) => s.replace(/[^A-Za-z0-9._-]/g, '_');
	return `${safe(m.city_id)}__${safe(m.meeting_id)}.json`;
}

async function liveFetchMeeting(m: MeetingRef): Promise<FetchOutcome> {
	const url = `${OPENCOUNCIL_BASE}/${encodeURIComponent(m.city_id)}/meetings/${encodeURIComponent(m.meeting_id)}`;
	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
	try {
		const resp = await fetch(url, { signal: ctrl.signal });
		if (!resp.ok) return { ok: false, status: resp.status };
		const cl = resp.headers.get('content-length');
		if (cl && Number(cl) > MAX_BODY_BYTES) return { ok: false, status: 413 };
		// Malformed JSON on a 200 is a PERMANENT error (don't retry) — distinct
		// from the network/timeout failures the outer catch handles as transient.
		try {
			return { ok: true, status: 200, json: await resp.json() };
		} catch {
			return { ok: false, status: 422 };
		}
	} catch {
		return { ok: false, status: 0 }; // network/timeout → transient
	} finally {
		clearTimeout(timer);
	}
}

const defaultSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** 4xx (except 408/429) is permanent; 0/408/429/5xx are transient. */
function isPermanent(status: number): boolean {
	return status >= 400 && status < 500 && status !== 408 && status !== 429;
}

/**
 * Read a meeting from its blob cache, or fetch it (with retries) and write the
 * blob. Returns parsed JSON or null on permanent/exhausted failure.
 */
async function loadMeetingJson(
	m: MeetingRef,
	blobDir: string,
	fetchMeeting: (m: MeetingRef) => Promise<FetchOutcome>,
	maxRetries: number,
	sleep: (ms: number) => Promise<void>,
	log: (msg: string) => void
): Promise<unknown | null> {
	const blobPath = resolve(blobDir, blobName(m));
	try {
		return JSON.parse(await fs.readFile(blobPath, 'utf8'));
	} catch {
		/* no/invalid blob → fetch */
	}
	for (let attempt = 0; attempt <= maxRetries; attempt++) {
		if (attempt > 0) {
			const backoff = Math.min(8000, 250 * 2 ** (attempt - 1));
			await sleep(backoff + Math.floor(Math.random() * 200));
		}
		const out = await fetchMeeting(m);
		if (out.ok && out.json !== undefined) {
			// Blob caching is an optimisation — a write failure (disk full, perms)
			// must NOT abort the whole build, just lose resumability for this one.
			try {
				await fs.writeFile(blobPath, JSON.stringify(out.json));
			} catch (err) {
				log(`[transcript] ${m.city_id}/${m.meeting_id} blob write failed (non-fatal): ${err}`);
			}
			return out.json;
		}
		if (isPermanent(out.status)) {
			log(`[transcript] ${m.city_id}/${m.meeting_id} permanent fail (status=${out.status})`);
			return null;
		}
		// transient → loop and retry
	}
	log(`[transcript] ${m.city_id}/${m.meeting_id} gave up after ${maxRetries} retries`);
	return null;
}

export async function buildTranscriptIndex(
	opts: BuildTranscriptOptions
): Promise<BuildTranscriptResult> {
	const {
		meetings,
		blobDir,
		fetchMeeting = liveFetchMeeting,
		concurrency = 5,
		maxRetries = 3,
		log = () => {},
		sleep = defaultSleep
	} = opts;

	await fs.mkdir(blobDir, { recursive: true });

	const rows: TranscriptRow[] = [];
	const okMeetings: Array<{ city_id: string; meeting_id: string; utt_count: number }> = [];
	const failed: MeetingRef[] = [];

	// Bounded worker pool over the meeting list.
	let cursor = 0;
	async function worker(): Promise<void> {
		while (cursor < meetings.length) {
			const m = meetings[cursor++];
			const json = await loadMeetingJson(m, blobDir, fetchMeeting, maxRetries, sleep, log);
			if (json == null) {
				failed.push(m);
				continue;
			}
			const { rows: mRows, skipped } = extractTranscriptRows(json);
			if (mRows.length === 0) {
				// A reachable meeting with no usable utterances is recorded as failed
				// (so the runtime falls back live) rather than a silent empty entry.
				log(`[transcript] ${m.city_id}/${m.meeting_id} yielded 0 rows (skipped=${skipped})`);
				failed.push(m);
				continue;
			}
			rows.push(...mRows);
			okMeetings.push({ city_id: m.city_id, meeting_id: m.meeting_id, utt_count: mRows.length });
			if (skipped > 0) log(`[transcript] ${m.city_id}/${m.meeting_id}: skipped ${skipped} malformed`);
		}
	}
	await Promise.all(Array.from({ length: Math.max(1, concurrency) }, () => worker()));

	// Workers complete out of order — sort for a deterministic manifest.
	okMeetings.sort((a, b) =>
		a.city_id === b.city_id
			? a.meeting_id.localeCompare(b.meeting_id)
			: a.city_id.localeCompare(b.city_id)
	);

	// Manifest hash covers WHICH meetings succeeded + the extractor/schema version,
	// so a schema/ordering change can't leave a "valid" hash over incompatible data.
	const manifest_hash = createHash('sha256')
		.update(
			JSON.stringify({
				schema: TRANSCRIPT_SCHEMA_VERSION,
				meetings: okMeetings
					.map((m) => `${m.city_id}/${m.meeting_id}`)
					.sort()
			})
		)
		.digest('hex')
		.slice(0, 16);

	const build_status: TranscriptManifest['build_status'] =
		okMeetings.length === 0 ? 'absent' : failed.length === 0 ? 'complete' : 'partial';

	return {
		rows,
		failed,
		manifest: {
			meetings: okMeetings,
			schema_version: TRANSCRIPT_SCHEMA_VERSION,
			manifest_hash,
			build_status,
			failed_count: failed.length
		}
	};
}
