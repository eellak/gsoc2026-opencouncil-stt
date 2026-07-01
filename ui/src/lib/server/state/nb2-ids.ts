/**
 * Fixed id set for the auto-selected "audio matches transcript" batch-2 edits.
 *
 * These 7,364 utterance_ids are the output of the next-batch faithfulness +
 * interestingness pipeline (data/next-batch/selected_edits.jsonl). Shipped as a
 * bundled JSON import so it deploys with the server — no DB write, no runtime
 * fs path assumptions. Consumed by the `?queue=nb2` review filter.
 */
import ids from './nb2-ids.json';

let _set: Set<string> | null = null;

/** Memoised set of the selected utterance_ids. */
export function nb2IdSet(): Set<string> {
	if (!_set) _set = new Set(ids as string[]);
	return _set;
}

export function nb2Count(): number {
	return (ids as string[]).length;
}
