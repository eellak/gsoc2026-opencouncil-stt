#!/usr/bin/env tsx
/**
 * Probe each meeting against the live OpenCouncil API to flag PRIVATE meetings.
 *
 * Privacy is per-MEETING and not derivable from our CSV/index: a meeting can be
 * public in one city and private in another, and a public city (e.g. sparta)
 * can contain private meetings (e.g. sparta/may26_2025 → 404). The only
 * authoritative signal is the upstream context endpoint:
 *   https://opencouncil.gr/api/utterance/{id}/context  → 200 public, 401/403/404 private
 *
 * For each distinct (city_id, meeting_id) we pick ONE representative utterance
 * and probe it once. The result feeds the hard index rebuild, which physically
 * drops private meetings so the server needs no runtime 404 auto-skip.
 *
 * Read-only: touches the index and the live API only, writes a report. Polite:
 * bounded concurrency + per-request timeout + one retry on transient errors.
 *
 * Usage:  npx tsx scripts/probe-meeting-availability.ts
 */

import Database from 'better-sqlite3';
import { promises as fs } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const UI_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(UI_ROOT, '..');

const cacheDir = process.env.REVIEW_CACHE_DIR ?? resolve(UI_ROOT, '.cache');
const dbPath = resolve(cacheDir, 'groups.v1.sqlite');
const reportsDir = resolve(REPO_ROOT, 'data', 'reports');
const BASE = 'https://opencouncil.gr/api/utterance';

const CONCURRENCY = Number(process.env.PROBE_CONCURRENCY ?? 5);
const TIMEOUT_MS = Number(process.env.PROBE_TIMEOUT_MS ?? 8000);
const stamp = new Date().toISOString().slice(0, 10);

interface MeetingRow {
	city_id: string;
	meeting_id: string;
	utterance_id: string;
	utt: number;
}

const db = new Database(dbPath, { readonly: true, fileMustExist: true });
db.pragma('query_only = true');
// One representative utterance per (city, meeting) + the meeting's utterance count.
const meetings = db
	.prepare(
		`SELECT city_id, meeting_id, MIN(utterance_id) AS utterance_id, COUNT(*) AS utt
		 FROM groups
		 GROUP BY city_id, meeting_id
		 ORDER BY utt DESC`
	)
	.all() as MeetingRow[];
db.close();

console.log(`Probing ${meetings.length} meetings against ${BASE} (concurrency=${CONCURRENCY}) …`);

type Status = 'public' | 'private' | 'error';
interface Result extends MeetingRow {
	status: Status;
	http: number | null;
	note?: string;
}

async function probeOne(m: MeetingRow): Promise<Result> {
	const url = `${BASE}/${encodeURIComponent(m.utterance_id)}/context?before=1&after=1`;
	for (let attempt = 0; attempt < 2; attempt++) {
		const ctrl = new AbortController();
		const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
		try {
			const resp = await fetch(url, { signal: ctrl.signal });
			clearTimeout(timer);
			if (resp.status === 200) return { ...m, status: 'public', http: 200 };
			if (resp.status === 401 || resp.status === 403 || resp.status === 404) {
				return { ...m, status: 'private', http: resp.status };
			}
			// 5xx / unexpected → retry once, then mark error (keep the meeting on error).
			if (attempt === 1) return { ...m, status: 'error', http: resp.status, note: `http ${resp.status}` };
		} catch (err) {
			clearTimeout(timer);
			if (attempt === 1) {
				return { ...m, status: 'error', http: null, note: err instanceof Error ? err.message : String(err) };
			}
		}
		await new Promise((r) => setTimeout(r, 400)); // small backoff before retry
	}
	return { ...m, status: 'error', http: null, note: 'unreachable' };
}

// Bounded worker pool.
const results: Result[] = [];
let next = 0;
let done = 0;
async function worker() {
	while (next < meetings.length) {
		const i = next++;
		const r = await probeOne(meetings[i]);
		results[i] = r;
		done++;
		if (done % 25 === 0 || done === meetings.length) {
			const priv = results.filter((x) => x?.status === 'private').length;
			const err = results.filter((x) => x?.status === 'error').length;
			console.log(`  ${done}/${meetings.length}  (private=${priv}, error=${err})`);
		}
	}
}
await Promise.all(Array.from({ length: Math.min(CONCURRENCY, meetings.length) }, worker));

const privateMeetings = results.filter((r) => r.status === 'private');
const errorMeetings = results.filter((r) => r.status === 'error');
const publicMeetings = results.filter((r) => r.status === 'public');
const privateUtterances = privateMeetings.reduce((a, r) => a + r.utt, 0);

await fs.mkdir(reportsDir, { recursive: true });
const jsonPath = resolve(reportsDir, `meeting-availability-${stamp}.json`);
await fs.writeFile(
	jsonPath,
	JSON.stringify(
		{
			generated_at: new Date().toISOString(),
			base: BASE,
			total_meetings: meetings.length,
			public: publicMeetings.length,
			private: privateMeetings.length,
			error: errorMeetings.length,
			private_utterances: privateUtterances,
			// The rebuild consumes this: keys are `${city_id} ${meeting_id}` (meetingKey).
			private_meeting_keys: privateMeetings.map((r) => `${r.city_id} ${r.meeting_id}`),
			error_meeting_keys: errorMeetings.map((r) => `${r.city_id} ${r.meeting_id}`),
			meetings: results.map((r) => ({
				key: `${r.city_id} ${r.meeting_id}`,
				city_id: r.city_id,
				meeting_id: r.meeting_id,
				utterances: r.utt,
				status: r.status,
				http: r.http,
				probe_utterance_id: r.utterance_id,
				note: r.note
			}))
		},
		null,
		2
	)
);

console.log(
	`\nDone. public=${publicMeetings.length} private=${privateMeetings.length} ` +
		`error=${errorMeetings.length}; private utterances=${privateUtterances.toLocaleString()}`
);
console.log(`Wrote ${jsonPath}`);
if (errorMeetings.length) {
	console.log(`\n⚠️ ${errorMeetings.length} meetings errored (kept, not flagged). Re-run to resolve:`);
	for (const e of errorMeetings.slice(0, 20)) console.log(`   ${e.city_id} ${e.meeting_id} — ${e.note}`);
}
