/**
 * Centralised URL builders for the review UI.
 *
 * Reasons:
 *   - The seed must propagate across `/`, `/review`, `/category`, `/error-category`, `/edit`.
 *   - One place to encode utterance_id / category id parameters.
 *   - Avoids regressions like `/review/{edit_id}` being constructed by mistake (the
 *     review route expects utterance_id, not edit_id).
 */

import { applyReviewFilterParams, parseReviewFilter } from './review-filters';

export const UINT32_MAX = 4_294_967_295;
const INT_RE = /^\d+$/;

/** FNV-1a 32-bit hash — maps any string to a stable uint32. */
export function hashSeedString(s: string): number {
	const input = s.trim().toLowerCase();
	let h = 0x811c9dc5;
	for (let i = 0; i < input.length; i++) {
		h ^= input.charCodeAt(i);
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h;
}

export interface ReviewHrefArgs {
	utterance_id: string;
	seed: number;
	highlight?: string | null;
	/** Canonical review-filter query (e.g. "punct=drop&src=both,user"); preserved across nav. */
	filter?: string | null;
}

export function reviewHref({ utterance_id, seed, highlight, filter }: ReviewHrefArgs): string {
	const params = new URLSearchParams({ seed: String(seed) });
	if (highlight) params.set('highlight', highlight);
	// Whitelist: only the recognized review-filter keys are merged in, so a
	// malformed `filter` can never clobber seed/highlight or smuggle params.
	if (filter) applyReviewFilterParams(params, parseReviewFilter(new URLSearchParams(filter)));
	return `/review/${encodeURIComponent(utterance_id)}?${params.toString()}`;
}

/**
 * localStorage key for the "resume where I left off" pointer. Scoped by seed AND
 * the canonical filter signature so a position saved under one filtered queue
 * can't reopen inside a differently-filtered one. Empty sig → seed-only key
 * (byte-identical to the pre-filter format, so old pointers still resolve).
 */
export function resumeStorageKey(seed: number, filterSig: string): string {
	return filterSig ? `oc:resume:${seed}|${filterSig}` : `oc:resume:${seed}`;
}

export function editHref(edit_id: string, seed?: number): string {
	const base = `/edit/${encodeURIComponent(edit_id)}`;
	if (seed === undefined || seed === null) return base;
	return `${base}?seed=${seed}`;
}

export function errorCategoryHref(category: string, seed?: number): string {
	const base = `/error-category/${encodeURIComponent(category)}`;
	if (seed === undefined || seed === null) return base;
	return `${base}?seed=${seed}`;
}

export function categoryHref(category: string, seed?: number): string {
	const base = `/category/${encodeURIComponent(category)}`;
	if (seed === undefined || seed === null) return base;
	return `${base}?seed=${seed}`;
}

/**
 * Parse a seed param — accepts a uint32 digit string OR any non-empty string
 * (hashed via FNV-1a to a deterministic uint32). URLs always carry the numeric
 * form so share links are stable regardless of the input format.
 */
export function parseSeedParam(raw: string | null | undefined): number | null {
	if (raw == null || raw === '') return null;
	if (INT_RE.test(raw)) {
		const n = Number(raw);
		if (!Number.isFinite(n) || !Number.isInteger(n)) return null;
		if (n < 0 || n > UINT32_MAX) return null;
		return n;
	}
	return hashSeedString(raw);
}

/** Uniform random uint32. Uses crypto when available, falls back to Math.random. */
export function randomSeed(): number {
	const c: { getRandomValues?: (a: Uint32Array) => Uint32Array } | undefined =
		typeof globalThis !== 'undefined' && (globalThis as { crypto?: unknown }).crypto
			? ((globalThis as { crypto: { getRandomValues?: (a: Uint32Array) => Uint32Array } }).crypto)
			: undefined;
	if (c && typeof c.getRandomValues === 'function') {
		const buf = new Uint32Array(1);
		c.getRandomValues(buf);
		return buf[0]!;
	}
	return Math.floor(Math.random() * (UINT32_MAX + 1));
}
