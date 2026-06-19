/**
 * Pure extraction: full OpenCouncil meeting JSON → flat transcript rows for the
 * local context index.
 *
 * Source shape (verified 2026-06-19 against
 * https://opencouncil.gr/api/cities/{city}/meetings/{meeting}):
 *   { meeting: { id, cityId }, transcript: [ segment ] }
 *   segment: { speakerTagId, utterances: [ { id, text, startTimestamp, endTimestamp } ] }
 *
 * `seq` is the ARRAY-ORDER position (segment order, then utterance order) over
 * the KEPT rows — NOT derived from timestamps, which upstream does not guarantee
 * to be unique or monotonic. seq is contiguous from 0 with no holes: a skipped
 * (malformed) utterance does not advance it.
 *
 * Malformed policy: an utterance with no usable `id` or no non-empty string
 * `text` is skipped and counted (never silently dropped, never poisons the whole
 * meeting). `start`/`end` that aren't finite numbers are stored as null.
 */

/**
 * Bumped only when the transcript table layout or extraction rules change. Kept
 * separate from CACHE_VERSION so transcript changes never invalidate the
 * CSV-derived label snapshots.
 */
export const TRANSCRIPT_SCHEMA_VERSION = 1;

/** One context neighbour, matching the upstream item shape exactly. */
export interface ContextNeighbour {
	id: string;
	text: string;
	start: number | null;
	end: number | null;
	speakerTagId: string | null;
}

/** Local equivalent of the upstream /context response (same field names). */
export interface LocalContextResult {
	meeting: { id: string; cityId: string };
	before: ContextNeighbour[];
	after: ContextNeighbour[];
}

export interface TranscriptRow {
	utterance_id: string;
	city_id: string;
	meeting_id: string;
	/** Array-order position within the meeting, contiguous from 0 over kept rows. */
	seq: number;
	text: string;
	start: number | null;
	end: number | null;
	/** speakerTagId — a grouping key only; never rendered as a name. */
	speaker_tag: string | null;
}

export interface ExtractResult {
	rows: TranscriptRow[];
	/** Utterances skipped for missing id / missing-or-empty text. */
	skipped: number;
}

interface RawUtterance {
	id?: unknown;
	text?: unknown;
	startTimestamp?: unknown;
	endTimestamp?: unknown;
}
interface RawSegment {
	speakerTagId?: unknown;
	utterances?: unknown;
}
interface RawMeetingJson {
	meeting?: { id?: unknown; cityId?: unknown };
	city?: { id?: unknown };
	transcript?: unknown;
}

function asString(v: unknown): string | null {
	return typeof v === 'string' && v.length > 0 ? v : null;
}
function asFiniteNumber(v: unknown): number | null {
	return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/**
 * Walk a meeting JSON into transcript rows. `city_id`/`meeting_id` come from the
 * top-level `meeting` object (falling back to `city.id`); both must be present
 * for the rows to be addressable — if either is missing the result is empty.
 */
export function extractTranscriptRows(meetingJson: unknown): ExtractResult {
	const json = (meetingJson ?? {}) as RawMeetingJson;
	const meetingId = asString(json.meeting?.id);
	const cityId = asString(json.meeting?.cityId) ?? asString(json.city?.id);
	const segments = Array.isArray(json.transcript) ? (json.transcript as RawSegment[]) : [];

	if (!meetingId || !cityId) return { rows: [], skipped: 0 };

	const rows: TranscriptRow[] = [];
	let skipped = 0;
	let seq = 0;

	for (const seg of segments) {
		const speakerTag = asString(seg?.speakerTagId);
		const utterances = Array.isArray(seg?.utterances) ? (seg.utterances as RawUtterance[]) : [];
		for (const u of utterances) {
			const id = asString(u?.id);
			const text = asString(u?.text);
			if (!id || text == null) {
				skipped++;
				continue;
			}
			rows.push({
				utterance_id: id,
				city_id: cityId,
				meeting_id: meetingId,
				seq,
				text,
				start: asFiniteNumber(u?.startTimestamp),
				end: asFiniteNumber(u?.endTimestamp),
				speaker_tag: speakerTag
			});
			seq++;
		}
	}

	return { rows, skipped };
}
