import { error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import type { IncludeStatus } from '$lib/domain/types';
import type { PageServerLoad } from './$types';

const VALID: ReadonlySet<IncludeStatus> = new Set([
	'include',
	'exclude',
	'uncertain'
]);

const PREVIEW_CHARS = 120;

function preview(s: string): string {
	const trimmed = s.replace(/\s+/g, ' ').trim();
	if (trimmed.length <= PREVIEW_CHARS) return trimmed;
	return trimmed.slice(0, PREVIEW_CHARS - 1) + '…';
}

export const load: PageServerLoad = async ({ params }) => {
	const status = params.status as IncludeStatus;
	if (!VALID.has(status)) {
		throw error(404, `Unknown status: ${params.status}`);
	}
	const repo = await getRepo();
	const ids = repo.idsByStatus(status);
	const rows = ids
		.map((id) => {
			const g = repo.getGroup(id);
			if (!g) return null;
			return {
				utterance_id: g.utterance_id,
				meeting_name: g.meeting_name,
				meeting_date: g.meeting_date,
				city_id: g.city_id,
				before_preview: preview(g.initial_before_text),
				after_preview: preview(g.final_after_text),
				edited_by: g.edits[g.edits.length - 1]?.edited_by ?? null,
				categories: g.label.error_categories
			};
		})
		.filter((r): r is NonNullable<typeof r> => r !== null);
	return { status, rows, total: rows.length };
};
