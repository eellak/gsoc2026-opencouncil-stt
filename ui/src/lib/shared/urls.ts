/**
 * Centralised URL builders for the review UI.
 *
 * Reasons:
 *   - The seed must propagate across `/`, `/review`, `/category`, `/error-category`, `/edit`.
 *   - One place to encode utterance_id / category id parameters.
 *   - Avoids regressions like `/review/{edit_id}` being constructed by mistake (the
 *     review route expects utterance_id, not edit_id).
 */

export const UINT32_MAX = 4_294_967_295;
const INT_RE = /^\d+$/;

export interface ReviewHrefArgs {
	utterance_id: string;
	seed: number;
	highlight?: string | null;
}

export function reviewHref({ utterance_id, seed, highlight }: ReviewHrefArgs): string {
	const params = new URLSearchParams({ seed: String(seed) });
	if (highlight) params.set('highlight', highlight);
	return `/review/${encodeURIComponent(utterance_id)}?${params.toString()}`;
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
 * Strict integer parse for `?seed=N`. Accepts only `^\d+$` digits in [0, 2^32-1].
 * Rejects floats, NaN, negatives, scientific notation, overflow.
 */
export function parseSeedParam(raw: string | null | undefined): number | null {
	if (raw == null || raw === '') return null;
	if (!INT_RE.test(raw)) return null;
	const n = Number(raw);
	if (!Number.isFinite(n) || !Number.isInteger(n)) return null;
	if (n < 0 || n > UINT32_MAX) return null;
	return n;
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
