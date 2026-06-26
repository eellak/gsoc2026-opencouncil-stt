/**
 * Start-time queue filters for the seeded review queue.
 *
 * Two independent, *narrowing* filters chosen on the landing page. They scope
 * which groups the seeded queue ever surfaces — and therefore which ones the
 * client ever fetches/prefetches — WITHOUT building a separate per-id queue.
 * The seeded paging path stays exactly as fast as before; filtered-out groups
 * are simply never returned by /api/review/queue.
 *
 *   1. `dropPunctOnly` — drop corrections whose *net* change (initial before →
 *      final after) is classified as ONLY punctuation/capitalization. Uses the
 *      deterministic `classify()` rules (zero false positives by design).
 *   2. `sources` — keep only groups whose edit-chain source profile is in the
 *      selected set. Profile is derived from the `edited_by` values across the
 *      chain: `task` (task edits only), `user` (user edits only), `both` (the
 *      chain has at least one task AND one user edit).
 *
 * Canonical-form rules (so caching + URLs stay stable):
 *   - `sources` is null when ALL three profiles are selected (= no constraint).
 *   - the serialized query string is empty when nothing is filtered, so the
 *     "no filter" path is byte-identical to today's behavior.
 *
 * This module is pure and shared: the server uses the predicates to scope the
 * queue scan; the client uses the (de)serializers to build URLs and to key its
 * seeded cache. Keep it dependency-light (only `classify` + the Group type).
 */

import { classify } from './auto-classify';
import type { Group } from '$lib/domain/groups';

export type SourceProfile = 'task' | 'both' | 'user';

/** All selectable source profiles, canonical order. */
export const ALL_SOURCE_PROFILES: readonly SourceProfile[] = ['task', 'both', 'user'];

export interface ReviewFilterSpec {
	/** Drop net-punctuation/capitalization-only corrections when true. */
	dropPunctOnly: boolean;
	/**
	 * Allowed source profiles, or null for "no source constraint" (all three).
	 * Always canonical: sorted by ALL_SOURCE_PROFILES order, deduped, and never
	 * the full set (the full set collapses to null).
	 */
	sources: SourceProfile[] | null;
}

export const EMPTY_FILTER: ReviewFilterSpec = { dropPunctOnly: false, sources: null };

/** True when the spec narrows anything (so the server must scan-filter). */
export function isFilterActive(spec: ReviewFilterSpec): boolean {
	return spec.dropPunctOnly || spec.sources !== null;
}

function canonicalizeSources(raw: Iterable<string>): SourceProfile[] | null {
	const valid = new Set<SourceProfile>(ALL_SOURCE_PROFILES);
	const picked = new Set<SourceProfile>();
	for (const s of raw) {
		const t = s.trim() as SourceProfile;
		if (valid.has(t)) picked.add(t);
	}
	// Empty selection is treated as "no constraint" (show all) rather than an
	// empty queue — the UI should prevent zero selection, but be forgiving.
	if (picked.size === 0 || picked.size === ALL_SOURCE_PROFILES.length) return null;
	return ALL_SOURCE_PROFILES.filter((p) => picked.has(p));
}

/** Parse a spec from URL search params (`punct=drop`, `src=task,both,user`). */
export function parseReviewFilter(params: URLSearchParams): ReviewFilterSpec {
	const dropPunctOnly = params.get('punct') === 'drop';
	const srcRaw = params.get('src');
	const sources = srcRaw ? canonicalizeSources(srcRaw.split(',')) : null;
	return { dropPunctOnly, sources };
}

/**
 * Canonical query string for a spec (no leading `?`/`&`). Empty when the spec
 * filters nothing — this is also the cache key, so two specs that mean the same
 * thing must serialize identically.
 */
export function serializeReviewFilter(spec: ReviewFilterSpec): string {
	const parts: string[] = [];
	if (spec.dropPunctOnly) parts.push('punct=drop');
	if (spec.sources) parts.push(`src=${spec.sources.join(',')}`);
	return parts.join('&');
}

/**
 * Write a spec's params onto an existing URLSearchParams (mutates it). Clears
 * any prior `punct`/`src` first, so an EMPTY_FILTER spec reliably *removes*
 * stale filters instead of leaving them behind. Sources are re-canonicalized so
 * the output is order-stable regardless of how the spec was constructed.
 */
export function applyReviewFilterParams(params: URLSearchParams, spec: ReviewFilterSpec): void {
	params.delete('punct');
	params.delete('src');
	if (spec.dropPunctOnly) params.set('punct', 'drop');
	const sources = spec.sources ? canonicalizeSources(spec.sources) : null;
	if (sources) params.set('src', sources.join(','));
}

/**
 * True when the group's NET change is classified as only punctuation /
 * capitalization. Judged on (initial_before_text → final_after_text): the net
 * correction, not each intermediate edit. A chain with a substantive
 * intermediate edit that is later reverted to a punctuation-only net change is
 * (correctly) treated as punctuation-only.
 */
export function isPunctuationOnly(g: Group): boolean {
	const ids = classify(g.initial_before_text, g.final_after_text);
	return ids.length === 1 && ids[0] === 'punctuation_capitalization';
}

/**
 * Source profile of a group from its edit chain, or `'unknown'` when no edit
 * carries a recognized `edited_by` (task/user). `'unknown'` never matches a
 * selected profile, so such groups are excluded whenever a source constraint is
 * active. Real data only ever has task/user, so this is a safety net.
 */
export function sourceProfile(g: Group): SourceProfile | 'unknown' {
	let hasTask = false;
	let hasUser = false;
	for (const e of g.edits) {
		if (e.edited_by === 'task') hasTask = true;
		else if (e.edited_by === 'user') hasUser = true;
	}
	if (hasTask && hasUser) return 'both';
	if (hasTask) return 'task';
	if (hasUser) return 'user';
	return 'unknown';
}

/** True when the group passes both filters (cheap pure predicate). */
export function matchesReviewFilter(g: Group, spec: ReviewFilterSpec): boolean {
	if (spec.dropPunctOnly && isPunctuationOnly(g)) return false;
	if (spec.sources) {
		const profile = sourceProfile(g);
		if (profile === 'unknown' || !spec.sources.includes(profile)) return false;
	}
	return true;
}
