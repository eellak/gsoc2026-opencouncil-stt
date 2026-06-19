import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { SqliteRepo } from '../../src/lib/server/repo/sqlite-repo';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';
import { buildSqlite, type TranscriptManifest } from '../../src/lib/server/cache/build-sqlite';
import {
	TRANSCRIPT_SCHEMA_VERSION,
	type TranscriptRow
} from '../../src/lib/server/cache/transcript-extract';
import type { CacheMeta } from '../../src/lib/domain/groups';

// One meeting, 5 utterances seq 0..4. u2 is the reviewed/edited one (also in
// the groups table); all 5 live in the transcript table as neighbours.
const TR: TranscriptRow[] = [0, 1, 2, 3, 4].map((seq) => ({
	utterance_id: `u${seq}`,
	city_id: 'athens',
	meeting_id: 'm1',
	seq,
	text: `text-${seq}`,
	start: seq * 10,
	end: seq * 10 + 5,
	speaker_tag: 'tagA'
}));

function manifest(status: TranscriptManifest['build_status'] = 'complete'): TranscriptManifest {
	return {
		meetings: [{ city_id: 'athens', meeting_id: 'm1', utt_count: TR.length }],
		schema_version: TRANSCRIPT_SCHEMA_VERSION,
		manifest_hash: 'test-hash',
		build_status: status,
		failed_count: 0
	};
}

interface Fixture {
	cacheDir: string;
	stateDir: string;
}

async function setup(withTranscript: boolean): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), 'tr-ctx-'));
	const cacheDir = join(root, 'cache');
	const stateDir = join(root, 'state');
	await fs.mkdir(cacheDir, { recursive: true });
	await fs.mkdir(stateDir, { recursive: true });

	// Edited utterance u2 → one group (enough to mount the repo).
	const rows: Array<V2CsvRow & { csv_row: number }> = [
		{
			edit_id: 'e-u2',
			utterance_id: 'u2',
			edit_timestamp: '2026-05-01T00:00:00Z',
			edit_updated_at: '',
			before_text: 'text-2 raw',
			after_text: 'text-2',
			edited_by: 'user',
			utterance_start: '20',
			utterance_end: '25',
			audio_url: 'https://example.com/a.mp3',
			youtube_url: '',
			meeting_name: 'm',
			meeting_date: '2026-05-01',
			meeting_id: 'm1',
			city_id: 'athens',
			csv_row: 0
		}
	];
	const { groups } = buildGroups(rows);
	const meta: CacheMeta = {
		cache_version: 1,
		source_csv_path: 'synthetic.csv',
		source_size: 0,
		source_mtime_ms: 0,
		source_hash: 'tr-ctx-hash',
		generated_at: new Date().toISOString(),
		group_count: groups.length,
		edit_count: 1,
		missing_utterance_id_count: 0
	};
	await buildSqlite({
		groups,
		meta,
		outPath: join(cacheDir, 'groups.v1.sqlite'),
		transcript: withTranscript ? { rows: TR, manifest: manifest() } : undefined
	});
	return { cacheDir, stateDir };
}

function load(fx: Fixture) {
	return SqliteRepo.load({ cacheDir: fx.cacheDir, stateDir: fx.stateDir, meetingMinHumanUtterances: 0 });
}

let fx: Fixture;
afterEach(async () => {
	if (fx) await rm(join(fx.cacheDir, '..'), { recursive: true, force: true });
});

describe('SqliteRepo.getContext', () => {
	beforeEach(async () => {
		fx = await setup(true);
	});

	it('returns chronological before/after windows around the anchor', async () => {
		const repo = await load(fx);
		const ctx = repo.getContext('u2', 2, 2)!;
		expect(ctx).not.toBeNull();
		expect(ctx.meeting).toEqual({ id: 'm1', cityId: 'athens' });
		expect(ctx.before.map((u) => u.id)).toEqual(['u0', 'u1']); // ascending
		expect(ctx.after.map((u) => u.id)).toEqual(['u3', 'u4']);
		expect(ctx.before[0]).toEqual({
			id: 'u0',
			text: 'text-0',
			start: 0,
			end: 5,
			speakerTagId: 'tagA'
		});
	});

	it('excludes the anchor and clamps at meeting boundaries', async () => {
		const repo = await load(fx);
		const first = repo.getContext('u0', 3, 3)!;
		expect(first.before).toEqual([]); // anchor is first
		expect(first.after.map((u) => u.id)).toEqual(['u1', 'u2', 'u3']);
		const last = repo.getContext('u4', 3, 3)!;
		expect(last.after).toEqual([]);
		expect(last.before.map((u) => u.id)).toEqual(['u1', 'u2', 'u3']);
	});

	it('honours before=0 / after=0', async () => {
		const repo = await load(fx);
		const onlyAfter = repo.getContext('u2', 0, 2)!;
		expect(onlyAfter.before).toEqual([]);
		expect(onlyAfter.after.map((u) => u.id)).toEqual(['u3', 'u4']);
	});

	it('returns null for an unknown utterance', async () => {
		const repo = await load(fx);
		expect(repo.getContext('does-not-exist', 2, 2)).toBeNull();
	});
});

describe('SqliteRepo.getContext — no transcript table', () => {
	it('returns null so the bridge falls back to live', async () => {
		fx = await setup(false);
		const repo = await load(fx);
		expect(repo.getContext('u2', 2, 2)).toBeNull();
	});
});
