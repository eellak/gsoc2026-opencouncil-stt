/**
 * Fixed review-queue id sets (the `?queue=<name>` filters).
 *
 * Each is a bundled JSON import so it deploys with the server — no DB write, no
 * runtime fs path assumptions. Add a queue by importing its id list and adding it
 * to REGISTRY.
 *
 *   nb2      — auto-selected batch-2 (interestingness + LLM triage), ~13k
 *   nb2audio — city-balanced, per-item AUDIO-VERIFIED (Soniox), randomized order
 *   gap1     — deletion/coverage gap candidates (2026-08-12 mining), ranked:
 *              restored-speech rows first, then rapid-turn/interjections, then
 *              name-dense; validation-split rows deliberately excluded
 *   gap2     — UNREVIEWED deletion-shaped rows (2026-08-13 corpus mining, top
 *              2000 of 12,949 by restored-words score; val cities excluded) —
 *              fresh material to judge, unlike gap1 which is mostly re-audit
 *   gap3     — gap2 candidates that PASSED Soniox audio verification (>=70% of
 *              the restored words audible in the clip ±3s); fixed-seed shuffle
 *              so the head of the queue is a random audit sample
 *   gap4     — wave-2 VERIFIED items in the small-but-clean cell (found_frac
 *              >=0.85 but 2-4 added words) — a calibration audit sample; the
 *              cell is NOT auto-accepted until this audit passes
 *   gap5     — final spot-check: 40 random items from the full auto-accepted
 *              set (both waves, both calibrated cells), fixed-seed sample
 */
import nb2 from './nb2-ids.json';
import nb2audio from './nb2audio-ids.json';
import gap1 from './gap1-ids.json';
import gap2 from './gap2-ids.json';
import gap3 from './gap3-ids.json';
import gap4 from './gap4-ids.json';
import gap5 from './gap5-ids.json';

const REGISTRY: Record<string, string[]> = {
	nb2: nb2 as string[],
	nb2audio: nb2audio as string[],
	gap1: gap1 as string[],
	gap2: gap2 as string[],
	gap3: gap3 as string[],
	gap4: gap4 as string[],
	gap5: gap5 as string[]
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
