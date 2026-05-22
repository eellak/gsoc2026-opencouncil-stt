import { error } from '@sveltejs/kit';
import * as queue from '$lib/client/group-queue.svelte';
import { ensureMirrorMapLoaded } from '$lib/client/audio-source';
import type { PageLoad } from './$types';

export const ssr = false;

export const load: PageLoad = async ({ params, url }) => {
	const id = params.utterance_id;
	const seedParam = url.searchParams.get('seed');
	if (seedParam) {
		const s = Number.parseInt(seedParam, 10);
		if (Number.isFinite(s)) queue.setSeed(s);
	}

	void ensureMirrorMapLoaded();

	const item = await queue.ensureLoaded(id);
	if (!item) throw error(404, 'utterance not found');

	void queue.topUp(id);

	return { item, seed: queue.seed() };
};
