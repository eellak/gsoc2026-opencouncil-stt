/**
 * Ingest-category filter (degenerate-bin exclusion).
 *
 * The CSV ingest classifier (`categorise()` in scripts/lib/csv-clean.ts) tags
 * every edit with an `ingest_category`. A handful of those categories are
 * *degenerate* for review and training: the latest edit carries no real
 * correction signal (an unchanged no-op, an all-empty `after`, or a
 * whitespace-only diff). This module decides, as policy, which categories to
 * drop and exposes a pure predicate the repos and the export use to scope them
 * out — WITHOUT deleting anything. It is a query-time view filter, the same
 * reversible posture as meeting-eligibility: `getGroup(id)` still resolves a
 * dropped utterance, and label state in `.state/` is never touched.
 *
 * Two layers share this one policy (see Codex review):
 *   1. review navigation — repos subtract degenerate ids from the eligible set;
 *   2. export — `/api/export` skips degenerate groups before writing a line.
 *
 * A group is judged by its LATEST edit's category, because the dataset uses
 * `final_after_text` (= the last edit). A group whose latest edit is degenerate
 * but whose history has clean edits is still dropped — the final state is what
 * ships. `getGroup()` keeps the full history for audit.
 */

import type { Group } from '$lib/domain/groups';

/** Every category `categorise()` can emit. Used to validate the env config. */
export const ALL_INGEST_CATEGORIES = [
	'clean',
	'noop_edit',
	'empty_after',
	'empty_before',
	'whitespace_only',
	'multiline_text',
	'embedded_reasoning',
	'reversed_timestamps'
] as const;

export type IngestCategory = (typeof ALL_INGEST_CATEGORIES)[number];

/**
 * Default degenerate set: clearly-useless bins only. `empty_before` (insertion
 * from nothing — may be a legit ASR miss), `multiline_text` (odd shape but the
 * text can be fine after normalisation), and `embedded_reasoning` (already
 * cleaned at ingest) are intentionally KEPT.
 */
export const DEFAULT_DEGENERATE_CATEGORIES: readonly IngestCategory[] = [
	'noop_edit',
	'empty_after',
	'whitespace_only'
];

/**
 * The configured degenerate set, from `DROP_INGEST_CATEGORIES` (comma list).
 * - unset            → the default set above;
 * - empty string     → drop NOTHING (explicit opt-out, filter disabled);
 * - any other value  → exactly that set.
 *
 * Fail-closed: an unknown category is a configuration error and throws, rather
 * than silently no-op'ing and shipping an unfiltered dataset.
 */
export function degenerateCategories(): Set<string> {
	const raw = process.env.DROP_INGEST_CATEGORIES;
	if (raw == null) return new Set(DEFAULT_DEGENERATE_CATEGORIES);
	if (raw.trim() === '') return new Set();
	const items = raw
		.split(',')
		.map((s) => s.trim())
		.filter(Boolean);
	const valid = new Set<string>(ALL_INGEST_CATEGORIES);
	for (const it of items) {
		if (!valid.has(it)) {
			throw new Error(
				`DROP_INGEST_CATEGORIES: unknown category "${it}". ` +
					`Valid categories: ${ALL_INGEST_CATEGORIES.join(', ')}.`
			);
		}
	}
	return new Set(items);
}

/** The ingest category of a group's latest edit, or null if it has no edits. */
export function latestEditCategory(g: Group): string | null {
	const last = g.edits[g.edits.length - 1];
	return last?.ingest_category ?? null;
}

/** True when the group's latest edit falls in the (non-empty) degenerate set. */
export function isDegenerate(g: Group, drop: Set<string>): boolean {
	if (drop.size === 0) return false;
	const cat = latestEditCategory(g);
	return cat != null && drop.has(cat);
}

/**
 * Narrow an already-eligible id list by removing ids whose latest edit is
 * degenerate. `categoryOf` returns the latest-edit category for an id (repos
 * supply it cheaply — SQL json_extract for sqlite, in-memory lookup for file).
 * A no-op when `drop` is empty.
 */
export function filterDegenerateIds(
	eligibleIds: readonly string[],
	categoryOf: (id: string) => string | null,
	drop: Set<string>
): { kept: string[]; dropped: string[] } {
	if (drop.size === 0) return { kept: [...eligibleIds], dropped: [] };
	const kept: string[] = [];
	const dropped: string[] = [];
	for (const id of eligibleIds) {
		const cat = categoryOf(id);
		if (cat != null && drop.has(cat)) dropped.push(id);
		else kept.push(id);
	}
	return { kept, dropped };
}
