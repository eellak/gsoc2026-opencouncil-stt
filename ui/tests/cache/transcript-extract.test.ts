import { describe, it, expect } from 'vitest';
import { extractTranscriptRows } from '../../src/lib/server/cache/transcript-extract';

const meeting = {
	meeting: { id: 'mar18_2026', cityId: 'vrilissia' },
	transcript: [
		{
			speakerTagId: 'tagA',
			utterances: [
				{ id: 'u1', text: 'Συνάδελφοι,', startTimestamp: 25.4, endTimestamp: 26.1 },
				{ id: 'u2', text: 'καλησπέρα.', startTimestamp: 26.2, endTimestamp: 27.0 }
			]
		},
		{
			speakerTagId: 'tagB',
			utterances: [
				// missing text → skipped, must NOT advance seq
				{ id: 'u3', startTimestamp: 27.1, endTimestamp: 27.5 },
				{ id: 'u4', text: 'Ναι.', startTimestamp: 27.6, endTimestamp: 28.0 }
			]
		}
	]
};

describe('extractTranscriptRows', () => {
	it('walks segments+utterances in array order with contiguous seq', () => {
		const { rows } = extractTranscriptRows(meeting);
		expect(rows.map((r) => r.utterance_id)).toEqual(['u1', 'u2', 'u4']);
		expect(rows.map((r) => r.seq)).toEqual([0, 1, 2]); // no hole where u3 was skipped
	});

	it('maps fields and segment speaker tag onto each utterance', () => {
		const { rows } = extractTranscriptRows(meeting);
		expect(rows[0]).toEqual({
			utterance_id: 'u1',
			city_id: 'vrilissia',
			meeting_id: 'mar18_2026',
			seq: 0,
			text: 'Συνάδελφοι,',
			start: 25.4,
			end: 26.1,
			speaker_tag: 'tagA'
		});
		expect(rows[2].speaker_tag).toBe('tagB');
	});

	it('counts malformed (missing-text) utterances as skipped', () => {
		const { skipped } = extractTranscriptRows(meeting);
		expect(skipped).toBe(1);
	});

	it('stores non-finite timestamps as null', () => {
		const { rows } = extractTranscriptRows({
			meeting: { id: 'm', cityId: 'c' },
			transcript: [{ speakerTagId: null, utterances: [{ id: 'x', text: 'hi' }] }]
		});
		expect(rows[0].start).toBeNull();
		expect(rows[0].end).toBeNull();
		expect(rows[0].speaker_tag).toBeNull();
	});

	it('returns empty when meeting id or city id is missing', () => {
		expect(extractTranscriptRows({ transcript: [] }).rows).toEqual([]);
		expect(extractTranscriptRows({ meeting: { id: 'm' }, transcript: [] }).rows).toEqual([]);
		expect(extractTranscriptRows(null).rows).toEqual([]);
	});

	it('falls back to city.id when meeting.cityId is absent', () => {
		const { rows } = extractTranscriptRows({
			meeting: { id: 'm' },
			city: { id: 'fallback-city' },
			transcript: [{ speakerTagId: 't', utterances: [{ id: 'x', text: 'hi' }] }]
		});
		expect(rows[0].city_id).toBe('fallback-city');
	});
});
