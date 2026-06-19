/**
 * Pure CSV → groups builder. No file I/O — feed it parsed CSV rows.
 *
 * Splits into a separate module so vitest can exercise grouping rules on
 * small synthetic fixtures without touching the 200 MB real CSV.
 */

import { categorise, type RawCsvRow } from '../../../../scripts/lib/csv-clean';
import type { Group, GroupEdit } from '$lib/domain/groups';
import { DEFAULT_LABEL } from '$lib/domain/groups';
import { meetingKey } from '../state/meeting-eligibility';

export interface V2CsvRow extends RawCsvRow {
	utterance_id: string;
	meeting_id: string;
	city_id: string;
}

/**
 * Hard exclusions applied at build time (filtered rebuild). A group is dropped
 * if its meeting is private OR its latest edit is a degenerate ingest bin.
 * Dropped utterances are physically absent from the index (getGroup → null);
 * the CSV stays the source of truth. Omit/empty → no exclusions (full corpus).
 */
export interface BuildExclusions {
	/** `meetingKey(city, meeting)` of private meetings (from the availability probe). */
	excludeMeetingKeys?: ReadonlySet<string>;
	/** Degenerate ingest categories to drop by a group's LATEST edit. */
	dropCategories?: ReadonlySet<string>;
}

export interface DroppedGroup {
	utterance_id: string;
	/** Any of: 'private', 'degenerate'. Both when a private meeting's group is also degenerate. */
	reasons: string[];
}

export interface BuildResult {
	groups: Group[];
	missingUtteranceIds: number;
	invalidTimestamps: number;
	editCount: number;
	excluded: {
		total: number;
		private: number;
		degenerate: number;
		both: number;
		dropped: DroppedGroup[];
	};
}

/**
 * Build groups from an in-memory array of CSV rows. The cache script streams
 * rows in batches and concatenates; tests pass small arrays directly.
 */
export function buildGroups(
	rows: Array<V2CsvRow & { csv_row: number }>,
	exclusions: BuildExclusions = {}
): BuildResult {
	const excludeMeetingKeys = exclusions.excludeMeetingKeys ?? new Set<string>();
	const dropCategories = exclusions.dropCategories ?? new Set<string>();
	const byUtterance = new Map<string, Array<GroupEdit & { _meeting: MeetingFields }>>();
	let missingUtteranceIds = 0;
	let invalidTimestamps = 0;
	let editCount = 0;

	for (const r of rows) {
		const utterance_id = r.utterance_id?.trim();
		if (!utterance_id) {
			missingUtteranceIds++;
			continue;
		}
		const cleaned = categorise(r);
		if (!Number.isFinite(cleaned.utterance_start) || !Number.isFinite(cleaned.utterance_end)) {
			invalidTimestamps++;
			continue;
		}
		const edit: GroupEdit & { _meeting: MeetingFields } = {
			edit_id: r.edit_id,
			edit_timestamp: r.edit_timestamp,
			edit_updated_at: r.edit_updated_at || null,
			before_text: cleaned.before_text,
			after_text: cleaned.after_text,
			edited_by: r.edited_by || null,
			utterance_start: cleaned.utterance_start,
			utterance_end: cleaned.utterance_end,
			ingest_category: cleaned.ingest_category,
			cleaning_applied: cleaned.cleaning_applied,
			csv_row: r.csv_row,
			_meeting: {
				meeting_id: r.meeting_id || null,
				city_id: r.city_id || null,
				meeting_name: r.meeting_name || null,
				meeting_date: r.meeting_date || null,
				audio_url: r.audio_url || '',
				youtube_url: r.youtube_url || null
			}
		};
		editCount++;
		const list = byUtterance.get(utterance_id);
		if (list) list.push(edit);
		else byUtterance.set(utterance_id, [edit]);
	}

	const groups: Group[] = [];
	const dropped: DroppedGroup[] = [];
	let droppedPrivate = 0;
	let droppedDegenerate = 0;
	let droppedBoth = 0;
	for (const [utterance_id, edits] of byUtterance) {
		// Deterministic sort: by edit_timestamp asc, then csv_row asc (handles
		// equal/missing timestamps without flakiness across runs).
		edits.sort((a, b) => {
			const ta = a.edit_timestamp ?? '';
			const tb = b.edit_timestamp ?? '';
			if (ta < tb) return -1;
			if (ta > tb) return 1;
			return a.csv_row - b.csv_row;
		});
		const first = edits[0];
		const last = edits[edits.length - 1];
		// Chain is consistent iff each edit picks up from the previous one's after_text.
		let chain_consistent = true;
		for (let i = 1; i < edits.length; i++) {
			if (edits[i].before_text !== edits[i - 1].after_text) {
				chain_consistent = false;
				break;
			}
		}
		// Latest meeting metadata + audio wins (an utterance shouldn't move
		// meetings, but if it does, "latest CSV row wins" is the least surprising rule).
		const meeting = last._meeting;

		// Hard exclusions (filtered rebuild). Judge by the same signals the repos
		// would: meeting privacy (key) and the latest edit's ingest category. A
		// group can hit both reasons — count it once per reason, drop it once.
		const isPrivate = excludeMeetingKeys.has(meetingKey(meeting.city_id, meeting.meeting_id));
		const isDegenerate = dropCategories.has(last.ingest_category);
		if (isPrivate || isDegenerate) {
			const reasons: string[] = [];
			if (isPrivate) reasons.push('private');
			if (isDegenerate) reasons.push('degenerate');
			dropped.push({ utterance_id, reasons });
			if (isPrivate && isDegenerate) droppedBoth++;
			else if (isPrivate) droppedPrivate++;
			else droppedDegenerate++;
			continue;
		}

		groups.push({
			utterance_id,
			meeting_id: meeting.meeting_id,
			city_id: meeting.city_id,
			meeting_name: meeting.meeting_name,
			meeting_date: meeting.meeting_date,
			audio_url: meeting.audio_url,
			audio_cdn_url: null,
			youtube_url: meeting.youtube_url,
			start: last.utterance_start,
			end: last.utterance_end,
			initial_before_text: first.before_text,
			final_after_text: last.after_text,
			edits: edits.map(stripInternal),
			chain_consistent,
			label: { ...DEFAULT_LABEL }
		});
	}

	// Stable ordering of groups by utterance_id so the on-disk cache is diffable.
	groups.sort((a, b) => (a.utterance_id < b.utterance_id ? -1 : a.utterance_id > b.utterance_id ? 1 : 0));

	return {
		groups,
		missingUtteranceIds,
		invalidTimestamps,
		editCount,
		excluded: {
			total: dropped.length,
			private: droppedPrivate,
			degenerate: droppedDegenerate,
			both: droppedBoth,
			dropped
		}
	};
}

interface MeetingFields {
	meeting_id: string | null;
	city_id: string | null;
	meeting_name: string | null;
	meeting_date: string | null;
	audio_url: string;
	youtube_url: string | null;
}

function stripInternal(e: GroupEdit & { _meeting: MeetingFields }): GroupEdit {
	const { _meeting, ...rest } = e;
	void _meeting;
	return rest;
}
