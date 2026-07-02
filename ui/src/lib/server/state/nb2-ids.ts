/**
 * Fixed review-queue id sets (the `?queue=<name>` filters).
 *
 * Each is a bundled JSON import so it deploys with the server — no DB write, no
 * runtime fs path assumptions. Add a queue by importing its id list and adding it
 * to REGISTRY.
 *
 *   nb2      — auto-selected batch-2 (interestingness + LLM triage), ~13k
 *   nb2audio — city-balanced, per-item AUDIO-VERIFIED (Soniox), randomized order
 */
import nb2 from './nb2-ids.json';
import nb2audio from './nb2audio-ids.json';

const REGISTRY: Record<string, string[]> = {
	nb2: nb2 as string[],
	nb2audio: nb2audio as string[]
};

const _sets: Record<string, Set<string>> = {};

/** Memoised id set for a named queue, or null if the name is unknown. */
export function queueIdSet(name: string): Set<string> | null {
	if (!(name in REGISTRY)) return null;
	if (!_sets[name]) _sets[name] = new Set(REGISTRY[name]);
	return _sets[name];
}

export function isKnownQueue(name: string): boolean {
	return name in REGISTRY;
}
