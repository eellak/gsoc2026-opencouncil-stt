/**
 * Pure CSV cleaning transforms. No I/O — import and test freely.
 */

export interface RawCsvRow {
	edit_id: string;
	edit_timestamp: string;
	edit_updated_at: string;
	before_text: string;
	after_text: string;
	edited_by: string;
	utterance_start: string;
	utterance_end: string;
	audio_url: string;
	youtube_url: string;
	meeting_name: string;
	meeting_date: string;
}

export interface CleanResult {
	ingest_category: string;
	cleaning_applied: string; // comma-separated tags or ''
	before_text: string;      // possibly cleaned
	after_text: string;       // possibly cleaned
	utterance_start: number;
	utterance_end: number;
}

// LLM reasoning artefacts that leak into before_text from task-generated transcripts.
const REASONING_MARKERS = [
	/\n\nWait,?\s+let\s+me\s+reconsider/i,
	/\n\n<\/?think>/i,
	/\n\nActually,?\s+let\s+me/i,
	/\n\nHmm,?\s+let\s+me/i,
	/\n\nLet\s+me\s+re(consider|think|check)/i,
];

export function stripReasoningTail(text: string): { result: string; changed: boolean } {
	for (const marker of REASONING_MARKERS) {
		const match = marker.exec(text);
		if (match) {
			return { result: text.slice(0, match.index).trimEnd(), changed: true };
		}
	}
	return { result: text, changed: false };
}

export function normaliseWhitespace(text: string): { result: string; changed: boolean } {
	const result = text.replace(/\s+/g, ' ').trim();
	return { result, changed: result !== text };
}

export function isWhitespaceOnlyDiff(a: string, b: string): boolean {
	return a.replace(/\s+/g, ' ').trim() === b.replace(/\s+/g, ' ').trim();
}

export function fixReversedTimestamps(
	startStr: string,
	endStr: string
): { start: number; end: number; swapped: boolean; invalid: boolean } {
	const start = parseFloat(startStr);
	const end = parseFloat(endStr);
	if (isNaN(start) || isNaN(end)) return { start: 0, end: 0, swapped: false, invalid: true };
	if (end >= start) return { start, end, swapped: false, invalid: false };
	const gap = start - end;
	if (gap < 0.05) return { start: end, end: start, swapped: true, invalid: false };
	return { start, end, swapped: false, invalid: true };
}

export function categorise(row: RawCsvRow): CleanResult {
	const cleaning: string[] = [];
	const rawBefore = row.before_text ?? '';
	const rawAfter = row.after_text ?? '';

	// Timestamps — fix early so we can use the result
	const ts = fixReversedTimestamps(row.utterance_start, row.utterance_end);
	if (ts.swapped) cleaning.push('timestamps_swapped');

	// Strip LLM reasoning tail from before_text
	const { result: beforeStripped, changed: reasoningStripped } = stripReasoningTail(rawBefore);
	const before = reasoningStripped ? beforeStripped : rawBefore;
	if (reasoningStripped) cleaning.push('reasoning_stripped');

	// Normalise whitespace on both sides
	const { result: beforeNorm, changed: bNorm } = normaliseWhitespace(before);
	const { result: afterNorm, changed: aNorm } = normaliseWhitespace(rawAfter);
	if (bNorm || aNorm) cleaning.push('whitespace_normalised');

	// Categorise — decisions made on RAW (pre-normalisation) values except for reasoning which was stripped
	let ingest_category: string;

	if (reasoningStripped) {
		ingest_category = 'embedded_reasoning';
	} else if (!rawBefore.trim() && !rawAfter.trim()) {
		ingest_category = 'noop_edit';
	} else if (!rawBefore.trim()) {
		ingest_category = 'empty_before';
	} else if (!rawAfter.trim()) {
		ingest_category = 'empty_after';
	} else if (rawBefore === rawAfter) {
		ingest_category = 'noop_edit';
	} else if (isWhitespaceOnlyDiff(rawBefore, rawAfter)) {
		ingest_category = 'whitespace_only';
	} else if (ts.invalid) {
		ingest_category = 'reversed_timestamps';
	} else if (/\n/.test(rawBefore + rawAfter)) {
		ingest_category = 'multiline_text';
	} else {
		ingest_category = 'clean';
	}

	return {
		ingest_category,
		cleaning_applied: cleaning.join(','),
		before_text: beforeNorm,
		after_text: afterNorm,
		utterance_start: ts.swapped ? ts.start : parseFloat(row.utterance_start) || 0,
		utterance_end: ts.swapped ? ts.end : parseFloat(row.utterance_end) || 0
	};
}
