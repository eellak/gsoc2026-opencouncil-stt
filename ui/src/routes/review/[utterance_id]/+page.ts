import { error } from '@sveltejs/kit';
import * as queue from '$lib/client/group-queue.svelte';
import { ensureMirrorMapLoaded } from '$lib/client/audio-source';
import type { IncludeStatus } from '$lib/domain/types';
import type { PageLoad } from './$types';

export const ssr = false;

const STATUS_SET: ReadonlySet<IncludeStatus> = new Set(['include', 'exclude', 'uncertain']);

export const load: PageLoad = async ({ params, url }) => {
	const id = params.utterance_id;
	const statusParam = url.searchParams.get('status');
	const statusFilter: IncludeStatus | null =
		statusParam && STATUS_SET.has(statusParam as IncludeStatus)
			? (statusParam as IncludeStatus)
			: null;

	if (statusFilter) {
		// Populate the filter id list up-front so j/k works on the first render.
		// Errors are non-fatal — ensureLoaded below will still resolve the group
		// directly via /api/review/group/{id}.
		try {
			const resp = await queue.fetchStatusIds(statusFilter);
			queue.setStatusOrder(statusFilter, resp.ids, resp.revision, resp.cache_hash);
		} catch (e) {
			console.warn('[review] could not fetch status ids', e);
		}
	} else {
		const seedParam = url.searchParams.get('seed');
		if (seedParam) {
			const s = Number.parseInt(seedParam, 10);
			if (Number.isFinite(s)) queue.setSeed(s);
		}
	}

	void ensureMirrorMapLoaded();

	const item = await queue.ensureLoaded(id);
	if (!item) throw error(404, 'utterance not found');

	void queue.topUp(id);

	return { item, seed: queue.seed(), statusFilter };
};
