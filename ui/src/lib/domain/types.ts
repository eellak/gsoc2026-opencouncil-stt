export type IncludeStatus = 'unreviewed' | 'include' | 'exclude' | 'uncertain';

export const INCLUDE_STATUSES: IncludeStatus[] = ['unreviewed', 'include', 'exclude', 'uncertain'];

export interface Correction {
	edit_id: string;
	utterance_id: string | null;
	meeting_id: string | null;
	latest_per_utterance: boolean;
	edit_timestamp: string;
	edit_updated_at: string | null;
	before_text: string;
	after_text: string;
	edited_by: string | null;
	utterance_start: number;
	utterance_end: number;
	// Joined from the `meetings` table.
	city_id: string | null;
	audio_url: string;
	audio_cdn_url: string | null;
	youtube_url: string | null;
	meeting_name: string | null;
	meeting_date: string | null;
	ingest_category: string | null;
	cleaning_applied: string | null;
}

export interface ReviewLabel {
	edit_id: string;
	error_category: string | null;
	include_status: IncludeStatus;
	adjusted_start: number | null;
	adjusted_end: number | null;
	reviewer_notes: string | null;
	human_updated_at: string | null;
}

export interface CorrectionWithLabel extends Correction, ReviewLabel {}

export interface PatchLabelBody {
	error_category?: string | null;
	include_status?: IncludeStatus;
	adjusted_start?: number | null;
	adjusted_end?: number | null;
	reviewer_notes?: string | null;
}

export interface NeighborRef {
	edit_id: string;
	audio_url: string;
	audio_cdn_url: string | null;
}

export interface ListOptions {
	status?: IncludeStatus | 'all';
	category?: string | 'all';
	editor?: string | 'all';
	meeting?: string | 'all';
	page?: number;
	page_size?: number;
}

export interface ListResponse {
	items: CorrectionWithLabel[];
	total: number;
	page: number;
	page_size: number;
}

export interface StatsResponse {
	total: number;
	by_status: Record<IncludeStatus, number>;
	// include/exclude/uncertain are per-category review-decision counts; optional
	// because snapshots persisted by older builds lack them until a recompute.
	by_category: Array<{
		category: string | null;
		count: number;
		include?: number;
		exclude?: number;
		uncertain?: number;
	}>;
	by_ingest_category: Array<{
		ingest_category: string | null;
		label_el: string | null;
		reason_el: string | null;
		is_rejected: number;
		count: number;
	}>;
	by_editor: Array<{ edited_by: string | null; count: number }>;
	by_duration_bucket: Array<{ bucket: string; count: number }>;
	by_meeting: Array<{ meeting_name: string | null; meeting_date: string | null; count: number }>;
}
