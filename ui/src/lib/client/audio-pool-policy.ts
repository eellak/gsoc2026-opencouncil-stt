/**
 * Pure retention + preload-tier policy for the audio pool.
 *
 * Why this exists: production DevTools snapshots on a slow connection showed
 * ~23 hidden `<audio preload="auto">` elements, almost all fully buffered
 * (buffered.end up to ~18000s — whole multi-hour meeting mp3s). Each utterance's
 * url IS the entire meeting file, and `preload="auto"` lets every element
 * download the whole thing. With ~24 of them alive (the old MAX_POOL fill-to-cap
 * eviction), they saturate a slow link and starve the element the user actually
 * lands on — the "audio loads at the moment of navigation, prefetch not working"
 * symptom. It's over-prefetching, not under.
 *
 * Policy:
 *   - keep ONLY current + the neighbours passed to setActive (no slack), so
 *     already-passed elements stop holding multi-hour buffers.
 *   - preload="auto" for current + the first `autoNeighbours` priority
 *     neighbours (the page passes neighbours forward-first, so neighbour[0] is
 *     the resolved next target). Everything else kept gets "metadata" — cheap,
 *     and it stops the background bandwidth contention.
 *
 * Keeping the decision pure makes it unit-testable without a DOM (vitest runs in
 * a node environment); the DOM side-effects live in audio-pool.svelte.ts.
 */

export interface PoolPolicy {
	/** ids to retain in the pool. */
	keep: Set<string>;
	/** ids present in the pool but no longer kept — remove these. */
	evict: string[];
	/** kept ids whose element should be preload="auto" (current + next target). */
	auto: Set<string>;
	/** kept ids whose element should be preload="metadata" (cheap warm). */
	metadata: Set<string>;
}

export function computePoolPolicy(
	currentId: string,
	neighbourIds: readonly string[],
	existingIds: readonly string[],
	autoNeighbours: number
): PoolPolicy {
	const keep = new Set<string>([currentId]);
	for (const id of neighbourIds) keep.add(id);

	// Current is always heavy-preloaded; never downgraded.
	const auto = new Set<string>([currentId]);
	let added = 0;
	for (const id of neighbourIds) {
		if (added >= autoNeighbours) break;
		if (auto.has(id)) continue; // skip a neighbour that equals current
		auto.add(id);
		added++;
	}

	const metadata = new Set<string>();
	for (const id of keep) if (!auto.has(id)) metadata.add(id);

	const evict: string[] = [];
	for (const id of existingIds) if (!keep.has(id)) evict.push(id);

	return { keep, evict, auto, metadata };
}
