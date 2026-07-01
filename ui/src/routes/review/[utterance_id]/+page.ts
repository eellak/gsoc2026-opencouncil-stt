import { error } from '@sveltejs/kit';
import * as queue from '$lib/client/group-queue.svelte';
import { ensureMirrorMapLoaded } from '$lib/client/audio-source';
import { parseSeedParam } from '$lib/shared/urls';
import { parseReviewFilter, serializeReviewFilter } from '$lib/shared/review-filters';
import type { IncludeStatus } from '$lib/domain/types';
import type { PageLoad } from './$types';

export const ssr = false;

const STATUS_SET: ReadonlySet<IncludeStatus> = new Set([
	'include',
	'exclude',
	'uncertain',
	'unreviewed'
]);

export interface QueueFilter {
	/** /api/review/ids query string, e.g. "status=include" or "category=akronymio". */
	query: string;
	/** Display value for the filter badge. */
	label: string;
}

/** Resolve the active queue filter from URL params, if any. */
function readFilter(url: URL): QueueFilter | null {
	const status = url.searchParams.get('status');
	if (status && STATUS_SET.has(status as IncludeStatus)) {
		return { query: `status=${encodeURIComponent(status)}`, label: status };
	}
	const category = url.searchParams.get('category');
	if (category) {
		return { query: `category=${encodeURIComponent(category)}`, label: category };
	}
	const errorCategory = url.searchParams.get('errorCategory');
	if (errorCategory) {
		return { query: `errorCategory=${encodeURIComponent(errorCategory)}`, label: errorCategory };
	}
	const queueName = url.searchParams.get('queue');
	if (queueName === 'nb2') {
		return { query: 'queue=nb2', label: 'ήχος OK (batch-2)' };
	}
	return null;
}

export const load: PageLoad = async ({ params, url }) => {
	const id = params.utterance_id;
	const filter = readFilter(url);
	// Canonical review-filter signature from the URL (seeded mode only; empty in
	// status/category filter mode). Returned so the page can key its resume pointer.
	const reviewFilter = serializeReviewFilter(parseReviewFilter(url.searchParams));

	if (filter) {
		// Populate the filter id list up-front so j/k works on the first render.
		// Errors are non-fatal — ensureLoaded below still resolves the group
		// directly via /api/review/group/{id}.
		try {
			const resp = await queue.fetchFilterIds(filter.query);
			queue.setFilterOrder(resp.filter, resp.ids, resp.revision, resp.cache_hash);
		} catch (e) {
			console.warn('[review] could not fetch filter ids', e);
		}
	} else {
		// Seeded mode. Start-time review filters (punct/src) compose only here —
		// status/category queues above take precedence and ignore them. Only an
		// EXPLICIT seed enters/refreshes seeded mode; we never fabricate one from
		// client module state. Our own links always carry the seed, so a filtered
		// queue is always reproducible.
		const seedParam = url.searchParams.get('seed');
		const s = seedParam ? parseSeedParam(seedParam) : null;
		if (s !== null) queue.setSeed(s, reviewFilter);
	}

	void ensureMirrorMapLoaded();

	const item = await queue.ensureLoaded(id);
	if (!item) throw error(404, 'utterance not found');

	void queue.topUp(id);

	// `reviewFilter` (canonical punct/src sig, '' when none) lets the page scope
	// its resume pointer to the active filtered queue. Empty in status/category mode.
	return { item, seed: queue.seed(), filter, reviewFilter };
};
