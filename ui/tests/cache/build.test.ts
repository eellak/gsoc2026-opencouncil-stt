import { describe, it, expect } from 'vitest';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';

function row(over: Partial<V2CsvRow & { csv_row: number }> = {}): V2CsvRow & { csv_row: number } {
	const base: V2CsvRow & { csv_row: number } = {
		edit_id: 'e1',
		utterance_id: 'u1',
		edit_timestamp: '2026-05-01T00:00:00Z',
		edit_updated_at: '',
		before_text: 'before',
		after_text: 'after',
		edited_by: 'tester',
		utterance_start: '1.0',
		utterance_end: '2.0',
		audio_url: 'https://example.com/a.mp3',
		youtube_url: '',
		meeting_name: 'Test meeting',
		meeting_date: '2026-05-01',
		meeting_id: 'm1',
		city_id: 'athens',
		csv_row: 0
	};
	return { ...base, ...over };
}

describe('buildGroups', () => {
	it('groups multiple edits under the same utterance_id and sorts chronologically', () => {
		const rows = [
			row({ edit_id: 'e2', edit_timestamp: '2026-05-02T00:00:00Z', before_text: 'after', after_text: 'after-2', csv_row: 1 }),
			row({ edit_id: 'e1', edit_timestamp: '2026-05-01T00:00:00Z', before_text: 'before', after_text: 'after', csv_row: 0 })
		];
		const { groups } = buildGroups(rows);
		expect(groups).toHaveLength(1);
		const g = groups[0];
		expect(g.edits.map((e) => e.edit_id)).toEqual(['e1', 'e2']);
		expect(g.initial_before_text).toBe('before');
		expect(g.final_after_text).toBe('after-2');
		expect(g.chain_consistent).toBe(true);
	});

	it('flags chain inconsistencies', () => {
		const rows = [
			row({ edit_id: 'e1', csv_row: 0, before_text: 'a', after_text: 'b', edit_timestamp: 't1' }),
			row({ edit_id: 'e2', csv_row: 1, before_text: 'c', after_text: 'd', edit_timestamp: 't2' })
		];
		const { groups } = buildGroups(rows);
		expect(groups[0].chain_consistent).toBe(false);
	});

	it('uses csv_row as a deterministic tiebreaker on equal timestamps', () => {
		const rows = [
			row({ edit_id: 'B', csv_row: 1, edit_timestamp: 't', after_text: 'B-after' }),
			row({ edit_id: 'A', csv_row: 0, edit_timestamp: 't', before_text: 'before', after_text: 'A-after' })
		];
		const { groups } = buildGroups(rows);
		expect(groups[0].edits.map((e) => e.edit_id)).toEqual(['A', 'B']);
		expect(groups[0].final_after_text).toBe('B-after');
	});

	it('counts missing utterance_id rows separately without dropping silently', () => {
		const rows = [row({ utterance_id: '', csv_row: 0 }), row({ csv_row: 1 })];
		const { groups, missingUtteranceIds } = buildGroups(rows);
		expect(missingUtteranceIds).toBe(1);
		expect(groups).toHaveLength(1);
	});

	it('returns groups sorted by utterance_id for diffable on-disk cache', () => {
		const rows = [
			row({ utterance_id: 'u-z', csv_row: 0 }),
			row({ utterance_id: 'u-a', csv_row: 1 })
		];
		const { groups } = buildGroups(rows);
		expect(groups.map((g) => g.utterance_id)).toEqual(['u-a', 'u-z']);
	});
});
