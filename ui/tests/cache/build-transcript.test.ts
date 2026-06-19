import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
	buildTranscriptIndex,
	type MeetingRef,
	type FetchOutcome
} from '../../src/lib/server/cache/build-transcript';
import { TRANSCRIPT_SCHEMA_VERSION } from '../../src/lib/server/cache/transcript-extract';

function meetingJson(city: string, id: string, utts: string[]) {
	return {
		meeting: { id, cityId: city },
		transcript: [
			{
				speakerTagId: 'tag',
				utterances: utts.map((t, i) => ({
					id: `${id}-u${i}`,
					text: t,
					startTimestamp: i,
					endTimestamp: i + 0.5
				}))
			}
		]
	};
}

const M: MeetingRef[] = [
	{ city_id: 'athens', meeting_id: 'm1' },
	{ city_id: 'patra', meeting_id: 'm2' }
];

let blobDir: string;
beforeEach(async () => {
	blobDir = await mkdtemp(join(tmpdir(), 'tr-build-'));
});
afterEach(async () => {
	await rm(blobDir, { recursive: true, force: true });
});

const noSleep = async () => {};

describe('buildTranscriptIndex', () => {
	it('fetches each meeting once, extracts rows, manifest=complete', async () => {
		const calls: string[] = [];
		const fetchMeeting = async (m: MeetingRef): Promise<FetchOutcome> => {
			calls.push(m.meeting_id);
			return { ok: true, status: 200, json: meetingJson(m.city_id, m.meeting_id, ['a', 'b']) };
		};
		const res = await buildTranscriptIndex({ meetings: M, blobDir, fetchMeeting, sleep: noSleep });
		expect(res.rows).toHaveLength(4);
		expect(res.manifest.build_status).toBe('complete');
		expect(res.manifest.failed_count).toBe(0);
		expect(res.manifest.schema_version).toBe(TRANSCRIPT_SCHEMA_VERSION);
		expect(res.manifest.meetings).toEqual([
			{ city_id: 'athens', meeting_id: 'm1', utt_count: 2 },
			{ city_id: 'patra', meeting_id: 'm2', utt_count: 2 }
		]);
		expect(calls.sort()).toEqual(['m1', 'm2']);
		expect((await readdir(blobDir)).length).toBe(2); // blobs cached
	});

	it('resumes from blob cache without re-fetching', async () => {
		const fetchOnce = async (m: MeetingRef): Promise<FetchOutcome> => ({
			ok: true,
			status: 200,
			json: meetingJson(m.city_id, m.meeting_id, ['a'])
		});
		await buildTranscriptIndex({ meetings: M, blobDir, fetchMeeting: fetchOnce, sleep: noSleep });

		let calls = 0;
		const fetchAgain = async (m: MeetingRef): Promise<FetchOutcome> => {
			calls++;
			return { ok: true, status: 200, json: meetingJson(m.city_id, m.meeting_id, ['a']) };
		};
		const res = await buildTranscriptIndex({ meetings: M, blobDir, fetchMeeting: fetchAgain, sleep: noSleep });
		expect(calls).toBe(0); // served entirely from blobs
		expect(res.manifest.build_status).toBe('complete');
	});

	it('treats 4xx as permanent → partial build, failed recorded', async () => {
		const fetchMeeting = async (m: MeetingRef): Promise<FetchOutcome> =>
			m.meeting_id === 'm2'
				? { ok: false, status: 404 }
				: { ok: true, status: 200, json: meetingJson(m.city_id, m.meeting_id, ['a']) };
		const res = await buildTranscriptIndex({ meetings: M, blobDir, fetchMeeting, sleep: noSleep });
		expect(res.manifest.build_status).toBe('partial');
		expect(res.manifest.failed_count).toBe(1);
		expect(res.failed).toEqual([{ city_id: 'patra', meeting_id: 'm2' }]);
		expect(res.manifest.meetings.map((x) => x.meeting_id)).toEqual(['m1']);
	});

	it('retries transient 5xx then succeeds', async () => {
		let attempts = 0;
		const fetchMeeting = async (m: MeetingRef): Promise<FetchOutcome> => {
			if (m.meeting_id === 'm1') {
				attempts++;
				if (attempts < 3) return { ok: false, status: 503 }; // fail twice
				return { ok: true, status: 200, json: meetingJson(m.city_id, m.meeting_id, ['a']) };
			}
			return { ok: true, status: 200, json: meetingJson(m.city_id, m.meeting_id, ['a']) };
		};
		const res = await buildTranscriptIndex({
			meetings: [M[0]],
			blobDir,
			fetchMeeting,
			maxRetries: 3,
			sleep: noSleep
		});
		expect(attempts).toBe(3);
		expect(res.manifest.build_status).toBe('complete');
	});

	it('marks a reachable-but-empty meeting as failed (live fallback)', async () => {
		const fetchMeeting = async (m: MeetingRef): Promise<FetchOutcome> => ({
			ok: true,
			status: 200,
			json: { meeting: { id: m.meeting_id, cityId: m.city_id }, transcript: [] }
		});
		const res = await buildTranscriptIndex({ meetings: [M[0]], blobDir, fetchMeeting, sleep: noSleep });
		expect(res.rows).toHaveLength(0);
		expect(res.manifest.build_status).toBe('absent');
		expect(res.manifest.failed_count).toBe(1);
	});
});
