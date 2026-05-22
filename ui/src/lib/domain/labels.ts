/**
 * Canonical normalization for utterance review labels.
 *
 * Multi-category migration: the old shape stored a single `error_category: string | null`
 * per utterance. The new shape stores `error_categories: string[]`. We accept both on
 * read (lenient), but writes emit only the new shape.
 *
 * Single-value → array upgrade is lossy in spirit: historical records can carry at most
 * one category. Documented in docs/decisions/ui.md.
 */

import { TAXONOMY_MAP, normalizeTaxonomyId } from '../shared/taxonomy';
import type { GroupLabel } from './groups';
import { DEFAULT_LABEL } from './groups';

interface LegacyLabelShape {
	error_category?: unknown;
	error_categories?: unknown;
	[k: string]: unknown;
}

/**
 * Normalize an unknown record's category fields to `string[]`.
 * Precedence: `error_categories` (if array) wins over single-value `error_category`.
 * Dedupes, preserves first-seen order, drops non-string entries and empty strings.
 * Does NOT reject unknown taxonomy ids (lenient read — see `validateCategoriesForWrite`).
 */
export function normalizeLabelCategories(raw: unknown): string[] {
	if (!raw || typeof raw !== 'object') return [];
	const r = raw as LegacyLabelShape;
	const out: string[] = [];
	const seen = new Set<string>();

	const arr = r.error_categories;
	if (Array.isArray(arr)) {
		for (const item of arr) {
			if (typeof item !== 'string' || item === '') continue;
			if (seen.has(item)) continue;
			seen.add(item);
			out.push(item);
		}
		return out;
	}

	const single = r.error_category;
	if (typeof single === 'string' && single !== '') {
		return [single];
	}
	return [];
}

/**
 * Validate that every category id is in the active taxonomy (after legacy id mapping).
 * Used on write to keep the canonical store clean.
 */
export function validateCategoriesForWrite(cats: string[]): void {
	if (!Array.isArray(cats)) {
		throw new Error(`error_categories must be an array, got ${typeof cats}`);
	}
	for (const c of cats) {
		if (typeof c !== 'string' || c === '') {
			throw new Error(`error_categories entries must be non-empty strings, got ${JSON.stringify(c)}`);
		}
		const normalized = normalizeTaxonomyId(c);
		if (!normalized || !(normalized in TAXONOMY_MAP)) {
			throw new Error(`unknown taxonomy id: ${c}`);
		}
	}
}

/**
 * Take any label-shaped object (legacy or new) and return a canonical GroupLabel.
 * Missing fields fall back to DEFAULT_LABEL values.
 */
export function canonicalizeLabel(raw: unknown): GroupLabel {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
	const out: GroupLabel = {
		error_categories: normalizeLabelCategories(r),
		include_status: (r.include_status as GroupLabel['include_status']) ?? DEFAULT_LABEL.include_status,
		adjusted_start: (r.adjusted_start as number | null) ?? null,
		adjusted_end: (r.adjusted_end as number | null) ?? null,
		reviewer_notes: (r.reviewer_notes as string | null) ?? null,
		human_updated_at: (r.human_updated_at as string | null) ?? null
	};
	return out;
}
