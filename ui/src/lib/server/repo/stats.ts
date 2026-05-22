import type { DbClient } from '../db';
import type { IncludeStatus, StatsResponse } from '$lib/domain/types';

export async function getStats(client: DbClient): Promise<StatsResponse> {
	const [
		totalResult,
		byStatusResult,
		byCategoryResult,
		byIngestCategoryResult,
		byEditorResult,
		byDurationResult,
		byMeetingResult
	] = await Promise.all([
		client.execute({ sql: 'SELECT COUNT(*) as n FROM corrections', args: [] }),
		client.execute({ sql: `SELECT include_status, COUNT(*) as n FROM review_labels GROUP BY include_status`, args: [] }),
		client.execute({ sql: `SELECT error_category as category, COUNT(*) as count FROM review_labels GROUP BY error_category ORDER BY count DESC`, args: [] }),
		client.execute({
			// Postgres requires every non-aggregated column to appear in GROUP BY.
			sql: `SELECT c.ingest_category, d.label_el, d.reason_el, COALESCE(d.is_rejected, 0) as is_rejected, COUNT(*) as count
			      FROM corrections c LEFT JOIN category_descriptions d ON d.category = c.ingest_category
			      GROUP BY c.ingest_category, d.label_el, d.reason_el, d.is_rejected ORDER BY count DESC`,
			args: []
		}),
		client.execute({ sql: `SELECT edited_by, COUNT(*) as count FROM corrections GROUP BY edited_by ORDER BY count DESC NULLS LAST`, args: [] }),
		client.execute({
			// `duration_seconds` was dropped to save space — compute on the fly here.
			// 200-700ms typical on the full table; fine for an infrequent stats screen.
			sql: `SELECT
				CASE
					WHEN (utterance_end - utterance_start) < 2  THEN '0–2s'
					WHEN (utterance_end - utterance_start) < 5  THEN '2–5s'
					WHEN (utterance_end - utterance_start) < 10 THEN '5–10s'
					WHEN (utterance_end - utterance_start) < 30 THEN '10–30s'
					ELSE '30s+'
				END as bucket,
				COUNT(*) as count
				FROM corrections GROUP BY bucket`,
			args: []
		}),
		client.execute({
			sql: `SELECT m.meeting_name, m.meeting_date, COUNT(*) as count
			      FROM corrections c
			      JOIN meetings m ON m.meeting_id = c.meeting_id
			      GROUP BY m.meeting_name, m.meeting_date ORDER BY count DESC LIMIT 20`,
			args: []
		})
	]);

	const total = Number(totalResult.rows[0].n);

	const by_status: Record<IncludeStatus, number> = { unreviewed: 0, include: 0, exclude: 0, uncertain: 0 };
	for (const row of byStatusResult.rows) {
		by_status[row.include_status as IncludeStatus] = Number(row.n);
	}

	return {
		total,
		by_status,
		by_category: byCategoryResult.rows.map(r => ({ category: r.category as string | null, count: Number(r.count) })),
		by_ingest_category: byIngestCategoryResult.rows.map(r => ({
			ingest_category: r.ingest_category as string | null,
			label_el: r.label_el as string | null,
			reason_el: r.reason_el as string | null,
			is_rejected: Number(r.is_rejected),
			count: Number(r.count)
		})),
		by_editor: byEditorResult.rows.map(r => ({ edited_by: r.edited_by as string | null, count: Number(r.count) })),
		by_duration_bucket: byDurationResult.rows.map(r => ({ bucket: r.bucket as string, count: Number(r.count) })),
		by_meeting: byMeetingResult.rows.map(r => ({
			meeting_name: r.meeting_name as string | null,
			meeting_date: r.meeting_date as string | null,
			count: Number(r.count)
		}))
	};
}
