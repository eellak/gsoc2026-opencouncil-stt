import type { DbClient } from '../db';
import type { Correction, CorrectionWithLabel, ListOptions, NeighborRef } from '$lib/domain/types';

export async function insertCorrection(client: DbClient, c: Correction): Promise<void> {
	// Upsert the meeting first so the FK is satisfied. audio_cdn_url is
	// intentionally NOT overwritten — only apply-audio-cdn-map sets it.
	const stmts: Array<{ sql: string; args: readonly unknown[] }> = [];
	if (c.meeting_id) {
		stmts.push({
			sql: `INSERT INTO meetings (meeting_id, meeting_name, meeting_date, city_id, audio_url, youtube_url)
				VALUES (?, ?, ?, ?, ?, ?)
				ON CONFLICT (meeting_id) DO UPDATE SET
					meeting_name = EXCLUDED.meeting_name,
					meeting_date = EXCLUDED.meeting_date,
					city_id      = EXCLUDED.city_id,
					audio_url    = EXCLUDED.audio_url,
					youtube_url  = EXCLUDED.youtube_url`,
			args: [c.meeting_id, c.meeting_name, c.meeting_date, c.city_id, c.audio_url, c.youtube_url]
		});
	}
	stmts.push({
		sql: `INSERT INTO corrections
			(edit_id, utterance_id, meeting_id, edit_timestamp, edit_updated_at,
			 before_text, after_text, edited_by, utterance_start, utterance_end)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT (edit_id) DO UPDATE SET
				utterance_id = EXCLUDED.utterance_id,
				meeting_id   = EXCLUDED.meeting_id,
				edit_timestamp = EXCLUDED.edit_timestamp,
				edit_updated_at = EXCLUDED.edit_updated_at,
				before_text = EXCLUDED.before_text,
				after_text  = EXCLUDED.after_text,
				edited_by   = EXCLUDED.edited_by,
				utterance_start = EXCLUDED.utterance_start,
				utterance_end   = EXCLUDED.utterance_end`,
		args: [
			c.edit_id, c.utterance_id, c.meeting_id,
			c.edit_timestamp, c.edit_updated_at,
			c.before_text, c.after_text, c.edited_by,
			c.utterance_start, c.utterance_end
		]
	});
	stmts.push({
		sql: `INSERT INTO review_labels (edit_id, include_status) VALUES (?, 'unreviewed') ON CONFLICT (edit_id) DO NOTHING`,
		args: [c.edit_id]
	});
	await client.batch(stmts, 'write');
}

export async function getCorrection(client: DbClient, edit_id: string): Promise<CorrectionWithLabel | null> {
	const result = await client.execute({
		sql: `SELECT c.*,
		             m.meeting_name, m.meeting_date, m.city_id, m.audio_url, m.audio_cdn_url, m.youtube_url,
		             rl.error_category, rl.include_status, rl.adjusted_start, rl.adjusted_end,
		             rl.reviewer_notes, rl.human_updated_at
		      FROM corrections c
		      LEFT JOIN meetings m ON m.meeting_id = c.meeting_id
		      JOIN review_labels rl ON rl.edit_id = c.edit_id
		      WHERE c.edit_id = ?`,
		args: [edit_id]
	});
	if (!result.rows.length) return null;
	return rowToObj(result.rows[0]) as unknown as CorrectionWithLabel;
}

export async function listCorrections(
	client: DbClient,
	opts: ListOptions = {}
): Promise<{ items: CorrectionWithLabel[]; total: number }> {
	const { status = 'all', category = 'all', editor = 'all', meeting = 'all', page = 1, page_size = 50 } = opts;
	const offset = (page - 1) * page_size;

	const wheres: string[] = [];
	const params: (string | number | null)[] = [];

	if (status !== 'all') { wheres.push('rl.include_status = ?'); params.push(status); }
	if (category !== 'all') { wheres.push('rl.error_category = ?'); params.push(category); }
	if (editor !== 'all') { wheres.push('c.edited_by = ?'); params.push(editor); }
	if (meeting !== 'all') { wheres.push('m.meeting_name = ?'); params.push(meeting); }

	const where = wheres.length ? `WHERE ${wheres.join(' AND ')}` : '';
	const base = `FROM corrections c JOIN review_labels rl ON rl.edit_id = c.edit_id ${where}`;

	const [countResult, itemsResult] = await Promise.all([
		client.execute({ sql: `SELECT COUNT(*) as n ${base}`, args: params }),
		client.execute({
			sql: `SELECT c.*, rl.error_category, rl.include_status, rl.adjusted_start, rl.adjusted_end,
			             rl.reviewer_notes, rl.human_updated_at
			      ${base} ORDER BY c.edit_timestamp ASC LIMIT ? OFFSET ?`,
			args: [...params, page_size, offset]
		})
	]);

	const total = Number(countResult.rows[0].n);
	const items = itemsResult.rows.map(rowToObj) as unknown as CorrectionWithLabel[];
	return { items, total };
}

export async function getFirstUnreviewed(client: DbClient): Promise<string | null> {
	const result = await client.execute({
		sql: `SELECT c.edit_id FROM corrections c JOIN review_labels rl ON rl.edit_id = c.edit_id
		      WHERE rl.include_status = 'unreviewed' ORDER BY c.edit_timestamp ASC LIMIT 1`,
		args: []
	});
	return result.rows.length ? String(result.rows[0].edit_id) : null;
}

export async function getNeighbors(
	client: DbClient,
	edit_id: string,
	opts: Pick<ListOptions, 'status' | 'category' | 'editor'> = {}
): Promise<{ prev: NeighborRef | null; next: NeighborRef | null }> {
	const { status = 'all', category = 'all', editor = 'all' } = opts;
	const wheres: string[] = [];
	const params: (string | number | null)[] = [];

	if (status !== 'all') { wheres.push('rl.include_status = ?'); params.push(status); }
	if (category !== 'all') { wheres.push('rl.error_category = ?'); params.push(category); }
	if (editor !== 'all') { wheres.push('c.edited_by = ?'); params.push(editor); }

	const filter = wheres.length ? `AND ${wheres.join(' AND ')}` : '';

	const [prevResult, nextResult] = await Promise.all([
		client.execute({
			sql: `SELECT c.edit_id, m.audio_url, m.audio_cdn_url
			      FROM corrections c
			      LEFT JOIN meetings m ON m.meeting_id = c.meeting_id
			      JOIN review_labels rl ON rl.edit_id = c.edit_id
			      WHERE c.edit_timestamp < (SELECT edit_timestamp FROM corrections WHERE edit_id = ?) ${filter}
			      ORDER BY c.edit_timestamp DESC LIMIT 1`,
			args: [edit_id, ...params]
		}),
		client.execute({
			sql: `SELECT c.edit_id, m.audio_url, m.audio_cdn_url
			      FROM corrections c
			      LEFT JOIN meetings m ON m.meeting_id = c.meeting_id
			      JOIN review_labels rl ON rl.edit_id = c.edit_id
			      WHERE c.edit_timestamp > (SELECT edit_timestamp FROM corrections WHERE edit_id = ?) ${filter}
			      ORDER BY c.edit_timestamp ASC LIMIT 1`,
			args: [edit_id, ...params]
		})
	]);

	return {
		prev: prevResult.rows.length ? rowToObj(prevResult.rows[0]) as unknown as NeighborRef : null,
		next: nextResult.rows.length ? rowToObj(nextResult.rows[0]) as unknown as NeighborRef : null
	};
}

export async function getForwardCorrections(
	client: DbClient,
	edit_id: string,
	limit: number
): Promise<CorrectionWithLabel[]> {
	if (limit <= 0) return [];
	const result = await client.execute({
		sql: `SELECT c.*,
		             m.meeting_name, m.meeting_date, m.city_id, m.audio_url, m.audio_cdn_url, m.youtube_url,
		             rl.error_category, rl.include_status, rl.adjusted_start, rl.adjusted_end,
		             rl.reviewer_notes, rl.human_updated_at
		      FROM corrections c
		      LEFT JOIN meetings m ON m.meeting_id = c.meeting_id
		      JOIN review_labels rl ON rl.edit_id = c.edit_id
		      WHERE c.edit_timestamp > (SELECT edit_timestamp FROM corrections WHERE edit_id = ?)
		      ORDER BY c.edit_timestamp ASC LIMIT ?`,
		args: [edit_id, limit]
	});
	return result.rows.map(rowToObj) as unknown as CorrectionWithLabel[];
}

export async function listByIngestCategory(
	client: DbClient,
	ingest_category: string,
	page = 1,
	page_size = 50
): Promise<{ items: CorrectionWithLabel[]; total: number }> {
	const offset = (page - 1) * page_size;
	const [countResult, itemsResult] = await Promise.all([
		client.execute({
			sql: `SELECT COUNT(*) as n FROM corrections c JOIN review_labels rl ON rl.edit_id = c.edit_id
			      WHERE c.ingest_category = ?`,
			args: [ingest_category]
		}),
		client.execute({
			sql: `SELECT c.*, rl.error_category, rl.include_status, rl.adjusted_start, rl.adjusted_end,
			             rl.reviewer_notes, rl.human_updated_at
			      FROM corrections c JOIN review_labels rl ON rl.edit_id = c.edit_id
			      WHERE c.ingest_category = ?
			      ORDER BY c.edit_timestamp ASC LIMIT ? OFFSET ?`,
			args: [ingest_category, page_size, offset]
		})
	]);
	return {
		total: Number(countResult.rows[0].n),
		items: itemsResult.rows.map(rowToObj) as unknown as CorrectionWithLabel[]
	};
}

export async function getRandomFromDifferentMeeting(
	client: DbClient,
	currentMeetingName: string | null
): Promise<string | null> {
	const result = await client.execute({
		sql: `SELECT c.edit_id
		      FROM corrections c
		      JOIN meetings m ON m.meeting_id = c.meeting_id
		      JOIN review_labels rl ON rl.edit_id = c.edit_id
		      WHERE m.meeting_name IS NOT NULL
		        AND (?::text IS NULL OR m.meeting_name != ?::text)
		      ORDER BY RANDOM() LIMIT 1`,
		args: [currentMeetingName, currentMeetingName]
	});
	return result.rows.length ? String(result.rows[0].edit_id) : null;
}

function rowToObj(row: object): Record<string, unknown> {
	const obj: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(row)) {
		obj[k] = v === undefined ? null : v;
	}
	return obj;
}
