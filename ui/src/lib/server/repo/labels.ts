import type { DbClient } from '../db';
import type { PatchLabelBody } from '$lib/domain/types';

// Whitelist of columns that patchLabel may write. Anything outside this set
// is rejected to keep the dynamic SET clause safe even if TS is bypassed.
const PATCHABLE_COLUMNS = new Set<keyof PatchLabelBody>([
	'error_category',
	'include_status',
	'adjusted_start',
	'adjusted_end',
	'reviewer_notes'
]);

export async function patchLabel(
	client: DbClient,
	edit_id: string,
	patch: PatchLabelBody
): Promise<{ updated: boolean }> {
	const existing = await client.execute({
		sql: `SELECT error_category, include_status, adjusted_start, adjusted_end, reviewer_notes
		      FROM review_labels WHERE edit_id = ?`,
		args: [edit_id]
	});
	if (!existing.rows.length) return { updated: false };

	const sets: string[] = [];
	const vals: (string | number | null)[] = [];

	for (const [key, val] of Object.entries(patch) as [keyof PatchLabelBody, unknown][]) {
		if (val === undefined) continue;
		if (!PATCHABLE_COLUMNS.has(key)) {
			throw new Error(`patchLabel: column '${String(key)}' is not patchable`);
		}
		sets.push(`${key} = ?`);
		vals.push(val === null ? null : (val as string | number));
	}

	if (!sets.length) return { updated: false };

	sets.push('human_updated_at = ?');
	vals.push(new Date().toISOString());
	vals.push(edit_id);

	await client.execute({
		sql: `UPDATE review_labels SET ${sets.join(', ')} WHERE edit_id = ?`,
		args: vals
	});
	return { updated: true };
}
