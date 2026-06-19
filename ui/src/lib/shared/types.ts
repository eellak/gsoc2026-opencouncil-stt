import type { TaxonomyId } from './taxonomy';

export type IncludeStatus = 'unreviewed' | 'include' | 'exclude' | 'uncertain';

export interface Correction {
	edit_id: string;
	edit_timestamp: string;
	edit_updated_at: string | null;
	before_text: string;
	after_text: string;
	edited_by: string | null;
	utterance_start: number;
	utterance_end: number;
	audio_url: string;
	youtube_url: string | null;
	meeting_name: string | null;
	meeting_date: string | null;
}

export interface ReviewLabel {
	edit_id: string;
	error_category: TaxonomyId | null;
	include_status: IncludeStatus;
	adjusted_start: number | null;
	adjusted_end: number | null;
	reviewer_notes: string | null;
	human_updated_at: string | null;
}

export interface CorrectionWithLabel extends Correction, ReviewLabel {}

export interface ListResponse {
	items: CorrectionWithLabel[];
	total: number;
	page: number;
	page_size: number;
}

export interface StatsResponse {
	total: number;
	by_status: Record<IncludeStatus, number>;
	by_category: Array<{
		category: string | null;
		count: number;
		include?: number;
		exclude?: number;
		uncertain?: number;
	}>;
	by_editor: Array<{ edited_by: string | null; count: number }>;
	by_duration_bucket: Array<{ bucket: string; count: number }>;
	by_meeting: Array<{ meeting_name: string | null; meeting_date: string | null; count: number }>;
}

export interface PatchLabelBody {
	error_category?: TaxonomyId | null;
	include_status?: IncludeStatus;
	adjusted_start?: number | null;
	adjusted_end?: number | null;
	reviewer_notes?: string | null;
}
