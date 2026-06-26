import { describe, it, expect } from 'vitest';
import {
	parseReviewFilter,
	serializeReviewFilter,
	isFilterActive,
	matchesReviewFilter,
	sourceProfile,
	isPunctuationOnly,
	EMPTY_FILTER,
	type ReviewFilterSpec
} from '../src/lib/shared/review-filters';
import type { Group, GroupEdit } from '../src/lib/domain/groups';
import { DEFAULT_LABEL } from '../src/lib/domain/groups';

function edit(over: Partial<GroupEdit>): GroupEdit {
	return {
		edit_id: 'e',
		edit_timestamp: '2024-01-01T00:00:00Z',
		edit_updated_at: null,
		before_text: '',
		after_text: '',
		edited_by: null,
		utterance_start: 0,
		utterance_end: 1,
		ingest_category: 'clean',
		cleaning_applied: '',
		csv_row: 0,
		...over
	};
}

function group(over: Partial<Group>): Group {
	return {
		utterance_id: 'u',
		meeting_id: 'm',
		city_id: 'c',
		meeting_name: null,
		meeting_date: null,
		audio_url: '',
		audio_cdn_url: null,
		youtube_url: null,
		start: 0,
		end: 1,
		initial_before_text: '',
		final_after_text: '',
		edits: [],
		chain_consistent: true,
		label: { ...DEFAULT_LABEL },
		...over
	};
}

describe('parse / serialize (canonical form)', () => {
	it('empty params → empty inactive spec', () => {
		const spec = parseReviewFilter(new URLSearchParams(''));
		expect(spec).toEqual(EMPTY_FILTER);
		expect(isFilterActive(spec)).toBe(false);
		expect(serializeReviewFilter(spec)).toBe('');
	});

	it('drop punct only', () => {
		const spec = parseReviewFilter(new URLSearchParams('punct=drop'));
		expect(spec.dropPunctOnly).toBe(true);
		expect(serializeReviewFilter(spec)).toBe('punct=drop');
		expect(isFilterActive(spec)).toBe(true);
	});

	it('source order is canonicalized so equivalent specs serialize identically', () => {
		const a = parseReviewFilter(new URLSearchParams('src=user,both'));
		const b = parseReviewFilter(new URLSearchParams('src=both,user'));
		expect(serializeReviewFilter(a)).toBe('src=both,user');
		expect(serializeReviewFilter(a)).toBe(serializeReviewFilter(b));
	});

	it('all three sources collapses to null (no constraint)', () => {
		const spec = parseReviewFilter(new URLSearchParams('src=task,both,user'));
		expect(spec.sources).toBeNull();
		expect(isFilterActive(spec)).toBe(false);
	});

	it('unknown source token is ignored', () => {
		const spec = parseReviewFilter(new URLSearchParams('src=task,bogus'));
		expect(spec.sources).toEqual(['task']);
	});

	it('combined filters round-trip', () => {
		const sig = 'punct=drop&src=both,user';
		const spec = parseReviewFilter(new URLSearchParams(sig));
		expect(serializeReviewFilter(spec)).toBe(sig);
	});
});

describe('sourceProfile', () => {
	it('task-only', () => {
		expect(sourceProfile(group({ edits: [edit({ edited_by: 'task' })] }))).toBe('task');
	});
	it('user-only', () => {
		expect(sourceProfile(group({ edits: [edit({ edited_by: 'user' })] }))).toBe('user');
	});
	it('both', () => {
		expect(
			sourceProfile(group({ edits: [edit({ edited_by: 'task' }), edit({ edited_by: 'user' })] }))
		).toBe('both');
	});
	it('unknown when no recognized edited_by', () => {
		expect(sourceProfile(group({ edits: [edit({ edited_by: null })] }))).toBe('unknown');
	});
});

describe('isPunctuationOnly', () => {
	it('true for pure punctuation/capitalization change', () => {
		expect(
			isPunctuationOnly(
				group({ initial_before_text: 'ναι κυριε προεδρε', final_after_text: 'Ναι, κύριε πρόεδρε.' })
			)
		).toBe(false); // accent diff present → NOT punct-only
	});
	it('true when only punctuation + case differ', () => {
		expect(
			isPunctuationOnly(group({ initial_before_text: 'ναι κύριε', final_after_text: 'Ναι, κύριε.' }))
		).toBe(true);
	});
	it('false for substantive change', () => {
		expect(
			isPunctuationOnly(group({ initial_before_text: 'να γίνη', final_after_text: 'να γίνει' }))
		).toBe(false);
	});
});

describe('matchesReviewFilter', () => {
	const taskG = group({ edits: [edit({ edited_by: 'task' })] });
	const userG = group({ edits: [edit({ edited_by: 'user' })] });
	const bothG = group({ edits: [edit({ edited_by: 'task' }), edit({ edited_by: 'user' })] });

	it('empty spec keeps everything', () => {
		for (const g of [taskG, userG, bothG]) {
			expect(matchesReviewFilter(g, EMPTY_FILTER)).toBe(true);
		}
	});

	it('source constraint both+user hides task-only', () => {
		const spec: ReviewFilterSpec = { dropPunctOnly: false, sources: ['both', 'user'] };
		expect(matchesReviewFilter(taskG, spec)).toBe(false);
		expect(matchesReviewFilter(userG, spec)).toBe(true);
		expect(matchesReviewFilter(bothG, spec)).toBe(true);
	});

	it('punct-only drop hides punctuation-only nets', () => {
		const punct = group({
			edits: [edit({ edited_by: 'user' })],
			initial_before_text: 'ναι κύριε',
			final_after_text: 'Ναι, κύριε.'
		});
		const subst = group({
			edits: [edit({ edited_by: 'user' })],
			initial_before_text: 'να γίνη',
			final_after_text: 'να γίνει'
		});
		const spec: ReviewFilterSpec = { dropPunctOnly: true, sources: null };
		expect(matchesReviewFilter(punct, spec)).toBe(false);
		expect(matchesReviewFilter(subst, spec)).toBe(true);
	});

	it('unknown source excluded when source constraint active', () => {
		const spec: ReviewFilterSpec = { dropPunctOnly: false, sources: ['user'] };
		expect(matchesReviewFilter(group({ edits: [edit({ edited_by: null })] }), spec)).toBe(false);
	});
});
