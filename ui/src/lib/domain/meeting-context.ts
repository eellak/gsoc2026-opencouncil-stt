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
}
