/**
 * GET /api/export — newline-delimited JSON export of included utterance groups.
 *
 * Streams directly off the repo's `iterGroups()` so a huge corpus never gets
 * materialised in memory before the first byte goes out. Each line is one
 * group in the canonical export shape (see encodeGroup).
 */

import { getRepo } from '$lib/server/repo';
import type { Group } from '$lib/domain/groups';

function encodeGroup(g: Group, cacheHash: string): string {
	return (
		JSON.stringify({
			utterance_id: g.utterance_id,
			meeting_id: g.meeting_id,
			city_id: g.city_id,
			meeting_name: g.meeting_name,
			meeting_date: g.meeting_date,
			audio_url: g.audio_url,
			audio_cdn_url: g.audio_cdn_url,
			start: g.label.adjusted_start ?? g.start,
			end: g.label.adjusted_end ?? g.end,
			initial_before_text: g.initial_before_text,
			final_after_text: g.final_after_text,
			edits: g.edits.map((e) => ({
				edit_id: e.edit_id,
				edit_timestamp: e.edit_timestamp,
				edited_by: e.edited_by,
				before_text: e.before_text,
				after_text: e.after_text,
				ingest_category: e.ingest_category,
				csv_row: e.csv_row
			})),
			chain_consistent: g.chain_consistent,
			error_categories: g.label.error_categories,
			include_status: g.label.include_status,
			reviewer_notes: g.label.reviewer_notes,
			human_updated_at: g.label.human_updated_at,
			cache_hash: cacheHash
		}) + '\n'
	);
}

export async function GET() {
	const repo = await getRepo();
	const cacheHash = repo.hash;
	// better-sqlite3's prepared-statement iterator is synchronous and tied to
	// the DB connection — we run the whole loop inside `start()` so the
	// cursor opens and closes within one tick, without yielding mid-iteration.
	const iter = repo.iterGroups();

	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const encoder = new TextEncoder();
			try {
				for (const g of iter) {
					if (g.label.include_status !== 'include') continue;
					controller.enqueue(encoder.encode(encodeGroup(g, cacheHash)));
				}
				controller.close();
			} catch (e) {
				controller.error(e);
			}
		}
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'application/jsonl',
			'Content-Disposition': 'attachment; filename="included-groups.jsonl"'
		}
	});
}
