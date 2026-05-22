import { error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { getCategoryCache } from '$lib/server/state/category-cache';
import { INGEST_CATEGORY_MAP } from '$lib/domain/categories';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params, url }) => {
	const cat = params.category;
	const meta = INGEST_CATEGORY_MAP.get(cat);
	if (!meta) throw error(404, `Άγνωστη κατηγορία: ${cat}`);

	const rawPage = Number.parseInt(url.searchParams.get('page') ?? '1', 10);
	const page = Number.isFinite(rawPage) && rawPage >= 1 ? rawPage : 1;
	const page_size = 50;

	const repo = await getRepo();
	const { items, total } = await getCategoryCache().getPage(repo, cat, page, page_size);

	return { meta, items, total, page, page_size };
};
