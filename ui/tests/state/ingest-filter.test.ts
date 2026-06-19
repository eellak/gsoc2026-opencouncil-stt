import { describe, it, expect, afterEach } from 'vitest';
import {
	ALL_INGEST_CATEGORIES,
	DEFAULT_DEGENERATE_CATEGORIES,
	degenerateCategories,
	latestEditCategory,
	isDegenerate,
	filterDegenerateIds
} from '../../src/lib/server/state/ingest-filter';
import type { Group, GroupEdit } from '../../src/lib/domain/groups';
import { DEFAULT_LABEL } from '../../src/lib/domain/groups';

/** Minimal group whose edits carry the given ingest categories (in order). */
function groupWith(id: string, cats: Array<string | undefined>): Group {
	const edits: GroupEdit[] = cats.map((c, i) => ({
		edit_id: `${id}-e${i}`,
		edit_timestamp: `2026-05-01T00:00:0${i}Z`,
		edit_updated_at: null,
		before_text: 'b',
		after_text: 'a',
		edited_by: 'user',
		utterance_start: 0,
		utterance_end: 1,
		ingest_category: c as string,
		cleaning_applied: '',
		csv_row: i
	}));
	return {
		utterance_id: id,
		meeting_id: 'm1',
		city_id: 'athens',
		meeting_name: 'm',
		meeting_date: '2026-05-01',
		audio_url: '',
		audio_cdn_url: null,
		youtube_url: null,
		start: 0,
		end: 1,
		initial_before_text: 'b',
		final_after_text: 'a',
		edits,
		chain_consistent: true,
		label: { ...DEFAULT_LABEL }
	};
}

const ORIG = process.env.DROP_INGEST_CATEGORIES;
afterEach(() => {
	if (ORIG === undefined) delete process.env.DROP_INGEST_CATEGORIES;
	else process.env.DROP_INGEST_CATEGORIES = ORIG;
});

describe('degenerateCategories — env policy', () => {
	it('unset → the default degenerate set', () => {
		delete process.env.DROP_INGEST_CATEGORIES;
		expect(degenerateCategories()).toEqual(new Set(DEFAULT_DEGENERATE_CATEGORIES));
	});

	it('default set is exactly noop_edit + empty_after + whitespace_only', () => {
		expect(new Set(DEFAULT_DEGENERATE_CATEGORIES)).toEqual(
			new Set(['noop_edit', 'empty_after', 'whitespace_only'])
		);
	});

	it('empty string → drop nothing (explicit opt-out)', () => {
		process.env.DROP_INGEST_CATEGORIES = '';
		expect(degenerateCategories().size).toBe(0);
	});

	it('custom list → exactly that set, trimmed', () => {
		process.env.DROP_INGEST_CATEGORIES = ' noop_edit , multiline_text ';
		expect(degenerateCategories()).toEqual(new Set(['noop_edit', 'multiline_text']));
	});

	it('fail-closed: an unknown category throws (no silent unfiltered ship)', () => {
		process.env.DROP_INGEST_CATEGORIES = 'noop_edit,garbage';
		expect(() => degenerateCategories()).toThrow(/unknown category "garbage"/i);
	});

	it('every default degenerate category is a real category', () => {
		for (const c of DEFAULT_DEGENERATE_CATEGORIES) {
			expect(ALL_INGEST_CATEGORIES).toContain(c);
		}
	});
});

describe('latestEditCategory — judged by the last edit', () => {
	it('returns the LAST edit category, not the first', () => {
		const g = groupWith('u1', ['clean', 'empty_after']);
		expect(latestEditCategory(g)).toBe('empty_after');
	});

	it('returns null when a group has no edits', () => {
		const g = groupWith('u1', []);
		expect(latestEditCategory(g)).toBeNull();
	});
});

describe('isDegenerate', () => {
	const drop = new Set(DEFAULT_DEGENERATE_CATEGORIES);

	it('true when the latest edit is in the drop set', () => {
		expect(isDegenerate(groupWith('u1', ['clean', 'noop_edit']), drop)).toBe(true);
	});

	it('false when the latest edit is clean even if history was degenerate', () => {
		expect(isDegenerate(groupWith('u1', ['empty_after', 'clean']), drop)).toBe(false);
	});

	it('false when drop set is empty regardless of category', () => {
		expect(isDegenerate(groupWith('u1', ['noop_edit']), new Set())).toBe(false);
	});

	it('keeps non-degenerate bins (empty_before/multiline/embedded)', () => {
		expect(isDegenerate(groupWith('u1', ['empty_before']), drop)).toBe(false);
		expect(isDegenerate(groupWith('u1', ['multiline_text']), drop)).toBe(false);
		expect(isDegenerate(groupWith('u1', ['embedded_reasoning']), drop)).toBe(false);
	});

	it('false when the group has no edits (null category)', () => {
		expect(isDegenerate(groupWith('u1', []), drop)).toBe(false);
	});
});

describe('filterDegenerateIds', () => {
	const catMap: Record<string, string> = {
		a: 'clean',
		b: 'noop_edit',
		c: 'empty_after',
		d: 'empty_before',
		e: 'whitespace_only'
	};
	const catOf = (id: string) => catMap[id] ?? null;

	it('drops degenerate ids, keeps the rest, preserves order', () => {
		const { kept, dropped } = filterDegenerateIds(
			['a', 'b', 'c', 'd', 'e'],
			catOf,
			new Set(DEFAULT_DEGENERATE_CATEGORIES)
		);
		expect(kept).toEqual(['a', 'd']);
		expect(dropped).toEqual(['b', 'c', 'e']);
	});

	it('empty drop set is a no-op (keeps everything)', () => {
		const { kept, dropped } = filterDegenerateIds(['a', 'b', 'c'], catOf, new Set());
		expect(kept).toEqual(['a', 'b', 'c']);
		expect(dropped).toEqual([]);
	});

	it('keeps ids whose category is null (e.g. a missing group), never drops them', () => {
		const { kept, dropped } = filterDegenerateIds(
			['a', 'unknown', 'd'],
			catOf, // catOf('unknown') === null
			new Set(DEFAULT_DEGENERATE_CATEGORIES)
		);
		expect(kept).toEqual(['a', 'unknown', 'd']);
		expect(dropped).toEqual([]);
	});
});
