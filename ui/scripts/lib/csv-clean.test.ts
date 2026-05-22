import { describe, expect, test } from 'bun:test';
import {
	stripReasoningTail,
	normaliseWhitespace,
	isWhitespaceOnlyDiff,
	fixReversedTimestamps,
	categorise
} from './csv-clean';

describe('stripReasoningTail', () => {
	test('leaves clean text unchanged', () => {
		const { result, changed } = stripReasoningTail('Αθήνα, 2026.');
		expect(changed).toBe(false);
		expect(result).toBe('Αθήνα, 2026.');
	});

	test('strips "Wait, let me reconsider" tail', () => {
		const { result, changed } = stripReasoningTail(
			'Καλημέρα.\n\nWait, let me reconsider. "επαρτέρα" could be...'
		);
		expect(changed).toBe(true);
		expect(result).toBe('Καλημέρα.');
	});

	test('strips </think> tail', () => {
		const { result, changed } = stripReasoningTail('Κείμενο.\n\n</think>more garbage');
		expect(changed).toBe(true);
		expect(result).toBe('Κείμενο.');
	});

	test('strips mid-sentence reasoning', () => {
		const { result, changed } = stripReasoningTail(
			'Μεταγραφή.\n\nActually, let me re-read this...'
		);
		expect(changed).toBe(true);
		expect(result).toBe('Μεταγραφή.');
	});
});

describe('normaliseWhitespace', () => {
	test('trims and collapses spaces', () => {
		const { result, changed } = normaliseWhitespace('  foo   bar  ');
		expect(changed).toBe(true);
		expect(result).toBe('foo bar');
	});

	test('no change on clean text', () => {
		const { result, changed } = normaliseWhitespace('Αθήνα');
		expect(changed).toBe(false);
		expect(result).toBe('Αθήνα');
	});
});

describe('isWhitespaceOnlyDiff', () => {
	test('same words, different spacing → true', () => {
		expect(isWhitespaceOnlyDiff('foo  bar', 'foo bar')).toBe(true);
	});

	test('different words → false', () => {
		expect(isWhitespaceOnlyDiff('Αθήνα', 'Πειραιάς')).toBe(false);
	});
});

describe('fixReversedTimestamps', () => {
	test('normal timestamps unchanged', () => {
		const r = fixReversedTimestamps('100', '110');
		expect(r).toEqual({ start: 100, end: 110, swapped: false, invalid: false });
	});

	test('small reversal (<0.05s) swapped', () => {
		const r = fixReversedTimestamps('6685.328625', '6685.319625');
		expect(r.swapped).toBe(true);
		expect(r.start).toBeLessThan(r.end);
	});

	test('large reversal flagged invalid', () => {
		const r = fixReversedTimestamps('200', '100');
		expect(r.invalid).toBe(true);
		expect(r.swapped).toBe(false);
	});
});

describe('categorise', () => {
	const base = {
		edit_id: 'x', edit_timestamp: '', edit_updated_at: '',
		edited_by: 'user', audio_url: 'https://x.mp3',
		youtube_url: '', meeting_name: '', meeting_date: '',
		utterance_start: '100', utterance_end: '110'
	};

	test('clean row', () => {
		const r = categorise({ ...base, before_text: 'Αθήνα', after_text: 'Αθήναι' });
		expect(r.ingest_category).toBe('clean');
	});

	test('noop_edit', () => {
		const r = categorise({ ...base, before_text: 'Αθήνα', after_text: 'Αθήνα' });
		expect(r.ingest_category).toBe('noop_edit');
	});

	test('whitespace_only', () => {
		const r = categorise({ ...base, before_text: 'Αθήνα ', after_text: 'Αθήνα' });
		expect(r.ingest_category).toBe('whitespace_only');
	});

	test('empty_before', () => {
		const r = categorise({ ...base, before_text: '', after_text: 'Κείμενο' });
		expect(r.ingest_category).toBe('empty_before');
	});

	test('empty_after', () => {
		const r = categorise({ ...base, before_text: 'Κείμενο', after_text: '' });
		expect(r.ingest_category).toBe('empty_after');
	});

	test('embedded_reasoning strips the tail', () => {
		const r = categorise({
			...base,
			before_text: 'Μεταγραφή.\n\nWait, let me reconsider. Something...',
			after_text: 'Μεταγραφή.'
		});
		expect(r.ingest_category).toBe('embedded_reasoning');
		expect(r.cleaning_applied).toContain('reasoning_stripped');
		expect(r.before_text).toBe('Μεταγραφή.');
	});

	test('reversed_timestamps (large gap) flagged', () => {
		const r = categorise({
			...base,
			before_text: 'Α', after_text: 'Β',
			utterance_start: '200', utterance_end: '100'
		});
		expect(r.ingest_category).toBe('reversed_timestamps');
	});

	test('small reversed timestamps swapped and category stays clean', () => {
		const r = categorise({
			...base,
			before_text: 'Α', after_text: 'Β',
			utterance_start: '100.01', utterance_end: '100.005'
		});
		expect(r.cleaning_applied).toContain('timestamps_swapped');
		expect(r.utterance_start).toBeLessThan(r.utterance_end);
		expect(r.ingest_category).toBe('clean');
	});
});
