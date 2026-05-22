import { redirect } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { parseSeedParam, randomSeed, reviewHref } from '$lib/shared/urls';

/**
 * Landing page behaviour:
 *   - `/?seed=N` (or `/?seed=N&jump=1`) → redirect straight into the first
 *     item of that seed. Lets a share-link land the recipient on the same
 *     ordering immediately.
 *   - `/` (no seed) → render the seed input UI so the reviewer chooses
 *     (empty = a randomly generated seed).
 *
 * The seed propagates through every review URL via `reviewHref()`.
 */
export async function load({ url }) {
	const seedParam = parseSeedParam(url.searchParams.get('seed'));
	if (seedParam !== null) {
		const repo = await getRepo();
		const { groups } = repo.queue(seedParam, 0, 1);
		if (!groups.length) throw redirect(302, '/stats');
		throw redirect(302, reviewHref({ utterance_id: groups[0].utterance_id, seed: seedParam }));
	}
	// No seed → show the landing page. Pre-generate a random seed so the user
	// can submit-without-typing for a fresh exploration.
	return { suggestedSeed: randomSeed() };
}
