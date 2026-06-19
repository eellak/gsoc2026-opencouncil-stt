/**
 * Group-based domain types for the file-backed review prototype.
 *
 * A `Group` is one utterance with the full chain of edits that touched it.
 * Review decisions are stored at group level (not per edit), using the
 * initial `before_text` and final `after_text` as the canonical pair.
 */

import type { IncludeStatus } from './types';

export interface GroupEdit {
	edit_id: string;
	edit_timestamp: string;
	edit_updated_at: string | null;
	before_text: string;
	after_text: string;
	edited_by: string | null;
	utterance_start: number;
	utterance_end: number;
	ingest_category: string;
	cleaning_applied: string;
	/** Original CSV row index — preserved for deterministic tie-breaks and export. */
	csv_row: number;
}

export interface GroupLabel {
	/**
	 * Canonical multi-category storage. Empty array = no category assigned.
	 * Legacy on-disk records with a single `error_category` field are normalized
	 * to a one-element array on read. See `$lib/domain/labels.ts`.
	 */
	error_categories: string[];
	include_status: IncludeStatus;
	adjusted_start: number | null;
	adjusted_end: number | null;
	reviewer_notes: string | null;
	human_updated_at: string | null;
}

export interface Group {
	utterance_id: string;
	meeting_id: string | null;
	city_id: string | null;
	meeting_name: string | null;
	meeting_date: string | null;
	audio_url: string;
	audio_cdn_url: string | null;
	youtube_url: string | null;
	/** Final timestamps — taken from the latest edit in the chain. */
	start: number;
	end: number;
	/** Earliest before_text in the chain. */
	initial_before_text: string;
	/** Latest after_text in the chain. */
	final_after_text: string;
	/** Edits sorted by (edit_timestamp, csv_row) ascending. */
	edits: GroupEdit[];
	/** True if every edit's before_text matches the previous edit's after_text. */
	chain_consistent: boolean;
	label: GroupLabel;
}

export interface CacheMeta {
	cache_version: number;
	source_csv_path: string;
	source_size: number;
	source_mtime_ms: number;
	source_hash: string;
	generated_at: string;
	group_count: number;
	edit_count: number;
	missing_utterance_id_count: number;
	/**
	 * Hard-exclusion provenance (present only on a filtered rebuild). The index
	 * physically omits these utterances; the CSV remains the source of truth.
	 * `exclusions_hash` digests the exact exclusion inputs so the build's
	 * skip-check invalidates when they change even if the source CSV does not.
	 */
	exclusions?: {
		exclusions_hash: string;
		availability_report_file: string | null;
		availability_generated_at: string | null;
		private_meeting_keys: string[];
		drop_categories: string[];
		excluded_private_utterances: number;
		excluded_degenerate_utterances: number;
		excluded_both: number;
		excluded_total: number;
	};
}

export interface QueueResponse {
	cache_hash: string;
	total: number;
	groups: Group[];
	next_cursor: number | null;
}

export interface GroupPatchBody {
	/** Canonical write field — full array replaces the existing one. */
	error_categories?: string[];
	/** Legacy write field (single value). Accepted on the API for back-compat; normalized server-side. */
	error_category?: string | null;
	include_status?: IncludeStatus;
	adjusted_start?: number | null;
	adjusted_end?: number | null;
	reviewer_notes?: string | null;
	/** Reviewer identity — required for new annotations; stored in the event log. */
	username?: string;
}

export const DEFAULT_LABEL: GroupLabel = {
	error_categories: [],
	include_status: 'unreviewed',
	adjusted_start: null,
	adjusted_end: null,
	reviewer_notes: null,
	human_updated_at: null
};

export const CACHE_VERSION = 1;

/**
 * Runtime cache identity. For an unfiltered index this is just the source CSV
 * fingerprint. For a FILTERED rebuild the index content no longer maps 1:1 to
 * the CSV (same CSV, different exclusion set ⇒ different served dataset), so the
 * exclusion digest is folded in. This is what stats/category/eligibility caches
 * key on, so they invalidate correctly when exclusions change. Kept as a
 * human-debuggable `${source}+x${excl}` string rather than a re-hash.
 */
export function cacheHashWithExclusions(
	sourceHash: string,
	exclusionsHash: string | null | undefined
): string {
	return exclusionsHash ? `${sourceHash}+x${exclusionsHash}` : sourceHash;
}
