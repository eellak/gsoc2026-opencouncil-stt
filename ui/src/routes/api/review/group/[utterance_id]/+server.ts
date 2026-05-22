/**
 * GET    /api/review/group/[utterance_id] → one Group (404 if unknown)
 * PATCH  /api/review/group/[utterance_id] → apply a GroupPatchBody, returns
 *                                            updated label
 *
 * Patch semantics:
 *   - omitted fields = no change
 *   - explicit null  = clear the field
 */

import { json, error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { INCLUDE_STATUSES, type IncludeStatus } from '$lib/domain/types';
import type { GroupPatchBody } from '$lib/domain/groups';
import type { RequestHandler } from './$types';

const isIncludeStatus = (v: unknown): v is IncludeStatus =>
	typeof v === 'string' && (INCLUDE_STATUSES as string[]).includes(v);

function parsePatch(raw: unknown): GroupPatchBody {
	if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
		throw error(400, 'body must be a JSON object');
	}
	const r = raw as Record<string, unknown>;
	const out: GroupPatchBody = {};

	if ('error_categories' in r) {
		const v = r.error_categories;
		if (!Array.isArray(v) || !v.every((x) => typeof x === 'string')) {
			throw error(400, 'error_categories must be an array of strings');
		}
		out.error_categories = v as string[];
	} else if ('error_category' in r) {
		// Legacy single-value accepted for back-compat. Normalized in the sidecar.
		const v = r.error_category;
		if (v !== null && typeof v !== 'string') throw error(400, 'error_category must be string or null');
		out.error_category = v as string | null;
	}
	if ('include_status' in r) {
		if (!isIncludeStatus(r.include_status)) throw error(400, 'include_status invalid');
		out.include_status = r.include_status;
	}
	for (const k of ['adjusted_start', 'adjusted_end'] as const) {
		if (k in r) {
			const v = r[k];
			if (v !== null && (typeof v !== 'number' || !Number.isFinite(v))) {
				throw error(400, `${k} must be finite number or null`);
			}
			out[k] = v as number | null;
		}
	}
	if ('reviewer_notes' in r) {
		const v = r.reviewer_notes;
		if (v !== null && typeof v !== 'string') throw error(400, 'reviewer_notes must be string or null');
		out.reviewer_notes = v as string | null;
	}
	return out;
}

export const GET: RequestHandler = async ({ params }) => {
	const repo = await getRepo();
	const group = repo.getGroup(params.utterance_id);
	if (!group) throw error(404, 'utterance not found');
	return json(group);
};

export const PATCH: RequestHandler = async ({ params, request }) => {
	let raw: unknown;
	try {
		raw = await request.json();
	} catch {
		throw error(400, 'invalid JSON');
	}
	const patch = parsePatch(raw);

	const repo = await getRepo();
	try {
		const label = await repo.patchLabel(params.utterance_id, patch);
		if (!label) throw error(404, 'utterance not found');
		return json({ ok: true, label });
	} catch (e) {
		// Validation errors from the sidecar surface as plain Error — return 400
		// so the client can show a useful message. SvelteKit's HttpError carries
		// its own status, so re-throw it untouched.
		if (e instanceof Error && !('status' in e)) throw error(400, e.message);
		throw e;
	}
};
