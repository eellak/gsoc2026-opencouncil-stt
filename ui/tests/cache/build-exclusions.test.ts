import { describe, it, expect } from 'vitest';
import { buildGroups, type V2CsvRow } from '../../src/lib/server/cache/build';

/** One-edit row whose categorise() bin we control via before/after text. */
function row(
	utterance_id: string,
	city_id: string,
	meeting_id: string,
	before: string,
	after: string,
	i: number
): V2CsvRow & { csv_row: number } {
	return {
		edit_id: `e-${utterance_id}`,
		utterance_id,
		edit_timestamp: `2026-05-01T00:00:0${i}Z`,
		edit_updated_at: '',
		before_text: before,
		after_text: after,
		edited_by: 'user',
		utterance_start: '0',
		utterance_end: '1',
		audio_url: 'https://example.com/a.mp3',
		youtube_url: '',
		meeting_name: 'm',
		meeting_date: '2026-05-01',
		meeting_id,
		city_id,
		csv_row: i
	};
}

// public-normal | private (clean) | degenerate (noop, public) | both (noop, private)
const ROWS = [
	row('u_public', 'athens', 'm_pub', 'alpha', 'alpha beta', 0), // clean, public
	row('u_private', 'athens', 'm_priv', 'gamma', 'gamma delta', 1), // clean, private meeting
	row('u_degen', 'athens', 'm_pub', 'same', 'same', 2), // noop_edit, public
	row('u_both', 'athens', 'm_priv', 'same', 'same', 3) // noop_edit, private meeting
];

describe('buildGroups — hard exclusions (private + degenerate)', () => {
	it('keeps only the public, non-degenerate group; counts dedupe overlap', () => {
		const res = buildGroups(ROWS, {
			excludeMeetingKeys: new Set(['athens m_priv']),
			dropCategories: new Set(['noop_edit'])
		});
		expect(res.groups.map((g) => g.utterance_id)).toEqual(['u_public']);
		expect(res.excluded.total).toBe(3);
		expect(res.excluded.private).toBe(1); // u_private: private only
		expect(res.excluded.degenerate).toBe(1); // u_degen: degenerate only
		expect(res.excluded.both).toBe(1); // u_both: counted once, both reasons
		const byId = new Map(res.excluded.dropped.map((d) => [d.utterance_id, d.reasons]));
		expect(byId.get('u_private')).toEqual(['private']);
		expect(byId.get('u_degen')).toEqual(['degenerate']);
		expect(byId.get('u_both')).toEqual(['private', 'degenerate']);
	});

	it('no exclusions → full corpus, empty excluded report', () => {
		const res = buildGroups(ROWS);
		expect(res.groups.length).toBe(4);
		expect(res.excluded.total).toBe(0);
		expect(res.excluded.dropped).toEqual([]);
	});

	it('private filter is keyed by (city, meeting) — same slug in another city is kept', () => {
		// Reuse slug 'm_priv' in a different city: must NOT be excluded.
		const rows = [...ROWS, row('u_other_city', 'sparta', 'm_priv', 'x', 'x y', 4)];
		const res = buildGroups(rows, { excludeMeetingKeys: new Set(['athens m_priv']) });
		expect(res.groups.map((g) => g.utterance_id).sort()).toEqual(
			['u_degen', 'u_other_city', 'u_public'].sort()
		);
	});
});
