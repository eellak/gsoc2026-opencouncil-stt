/**
 * GET /error-category/[category] — list groups whose label.error_categories
 * array contains [category]. Mirrors /category/[ingest_category] but pivots
 * on the human-assigned taxonomy id instead of the ingest classifier.
 */

import { error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { TAXONOMY_MAP, normalizeTaxonomyId } from '$lib/shared/taxonomy';
import { parseSeedParam } from '$lib/shared/urls';
import type { PageServerLoad } from './$types';

const PAGE_SIZE = 50;

export const load: PageServerLoad = async ({ params, url }) => {
	const raw = params.category;
	const normalized = normalizeTaxonomyId(raw);
	const taxonomy = normalized ? TAXONOMY_MAP[normalized] : null;
	// We allow unknown ids in the URL — historic labels may carry them — but
	// then we don't have pretty names to show.

	const repo = await getRepo();
	// Cheap id scan + materialise only the current page (not every match) — the
	// old groupsByErrorCategory materialised all matches and took ~60 s for big
	// categories on the VM.
	const ids = repo.idsByErrorCategory(normalized ?? raw);
	const total = ids.length;
	const rawPage = Number.parseInt(url.searchParams.get('page') ?? '1', 10);
	const page = Number.isFinite(rawPage) && rawPage >= 1 ? rawPage : 1;
	const start = (page - 1) * PAGE_SIZE;
	const items = ids
		.slice(start, start + PAGE_SIZE)
		.map((id) => repo.getGroup(id))
		.filter((g): g is NonNullable<typeof g> => g !== null)
		.map((g) => ({
			utterance_id: g.utterance_id,
			meeting_name: g.meeting_name,
			meeting_date: g.meeting_date,
			city_id: g.city_id,
			before: g.initial_before_text.slice(0, 140),
			after: g.final_after_text.slice(0, 140),
			all_categories: g.label.error_categories
		}));

	const seedParam = parseSeedParam(url.searchParams.get('seed'));

	if (total === 0 && !taxonomy) {
		// Distinguish "no such category" from "this category exists but is empty".
		// If the id is totally unknown AND there are no rows that carry it,
		// give the user a 404 instead of an empty table.
		throw error(404, `unknown error category: ${raw}`);
	}

	return {
		category: normalized ?? raw,
		taxonomy,
		items,
		total,
		page,
		page_size: PAGE_SIZE,
		seed: seedParam
	};
};
