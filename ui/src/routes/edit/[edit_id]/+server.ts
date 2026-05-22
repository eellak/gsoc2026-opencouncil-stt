/**
 * GET /edit/[edit_id] — resolve an edit_id to its utterance and 302 to the
 * canonical review URL. Generates a random seed if the caller doesn't supply
 * one so the reviewer always lands somewhere reproducible.
 *
 * Query params:
 *   - `seed` — preserved if provided and valid; otherwise a fresh random seed.
 */

import { error, redirect } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { parseSeedParam, randomSeed, reviewHref } from '$lib/shared/urls';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, url }) => {
	const repo = await getRepo();
	const utteranceId = repo.utteranceIdForEdit(params.edit_id);
	if (!utteranceId) throw error(404, `edit_id not found: ${params.edit_id}`);

	const seedParam = parseSeedParam(url.searchParams.get('seed'));
	const seed = seedParam ?? randomSeed();
	throw redirect(302, reviewHref({ utterance_id: utteranceId, seed, highlight: params.edit_id }));
};
