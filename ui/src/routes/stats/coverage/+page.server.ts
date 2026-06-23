import { error } from '@sveltejs/kit';

export interface CoverageCity {
	city: string;
	n_meetings_public: number;
	n_meetings_private: number;
	n_reviewed: number;
	hours: number;
	hours_reviewed: number;
	n_speakers: number;
	hir: number | null;
}

export interface CoverageMeeting {
	city: string;
	meeting: string;
	date: string | null;
	public: boolean;
	reviewed: boolean | null;
	n_utts: number | null;
	hours: number | null;
	n_speakers: number | null;
	hir: number | null;
	pct_none?: number;
	pct_user?: number;
	pct_task?: number;
}

export interface CoveragePayload {
	summary: Record<string, number | number[]>;
	cities: CoverageCity[];
	meetings: CoverageMeeting[];
}

// Static dataset-coverage snapshot, produced offline by the eval harness
// (eval/fetch_speakers.py + fetch_meeting_meta.py → data/reports/coverage.json,
// copied to ui/static/coverage.json). Refreshed manually when the corpus
// changes; not tied to the live SQLite review repo.
export async function load({ fetch }) {
	let resp: Response;
	try {
		resp = await fetch('/coverage.json');
	} catch (err) {
		console.error('[coverage] fetch failed', err);
		throw error(502, 'coverage snapshot unreachable');
	}
	if (!resp.ok) {
		throw error(404, 'coverage snapshot missing (run eval/fetch_speakers.py)');
	}
	try {
		const coverage = (await resp.json()) as CoveragePayload;
		return { coverage };
	} catch (err) {
		console.error('[coverage] parse failed', err);
		throw error(502, 'coverage snapshot invalid');
	}
}
