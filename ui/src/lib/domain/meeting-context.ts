/**
 * Shared shape returned by /api/meeting-context/[utterance_id].
 *
 * Lives under $lib/domain so both the client (review page) and the server
 * endpoint import the same type. The richer server-side type that includes
 * the meeting-fetch service helpers lives in $lib/server/state/meeting-context.
 */

export interface ContextUtterance {
	utterance_id: string;
	start: number;
	end: number;
	text: string;
	speaker_label: string | null;
	speaker_person_id: string | null;
	speaker_name: string | null;
	is_current: boolean;
	same_speaker_as_current: boolean;
}

export interface MeetingContext {
	city_id: string;
	meeting_id: string;
	current: ContextUtterance | null;
	prev: ContextUtterance[];
	next: ContextUtterance[];
	error: string | null;
	/**
	 * Why the fetch failed, when `error` is set.
	 * - 'private': upstream 401/403/404 — the meeting isn't publicly readable, so
	 *   the review UI auto-skips it.
	 * - 'transient': network/timeout/5xx — show the error and let the user retry.
	 */
	error_kind?: 'private' | 'transient' | null;
}
