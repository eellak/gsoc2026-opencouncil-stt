/**
 * Session-scoped state for auto-skipping unavailable utterances (Change 1).
 *
 * An utterance is "unavailable" when its OpenCouncil context fetch returns
 * 401/403/404 (classified as `error_kind: 'private'` in meeting-context). This
 * covers two cases that look identical to the reviewer and warrant the same
 * action — skip it:
 *  - a fully private / unpublished meeting (every utterance 404s), and
 *  - a public meeting with an individual utterance removed upstream (that one
 *    utterance 404s while its siblings return 200).
 *
 * Because the second case exists, we skip PER UTTERANCE on its own fetch — we do
 * NOT mark a whole meeting private and pre-skip siblings, which would silently
 * drop valid utterances. In the default seeded (shuffled) order a meeting's
 * utterances aren't adjacent anyway, so a per-meeting memo would buy almost
 * nothing.
 *
 * Skips advance in the last navigation direction (no wrap). A consecutive-skip
 * cap stops a run of failures from silently sweeping the queue: after
 * MAX_CONSECUTIVE_SKIPS the loop pauses and the page shows a banner.
 *
 * Nothing here is persisted and no label is written — this is navigation only.
 */

const MAX_CONSECUTIVE_SKIPS = 25;

let lastDirection: 'next' | 'prev' = 'next';
let consecutiveSkips = 0;

/** Reactive surface the page reads (banner + running count). */
export const autoSkip = $state<{ paused: boolean; skipped: number }>({
	paused: false,
	skipped: 0
});

/** Record which way the reviewer last moved, so skips continue that direction. */
export function rememberDirection(dir: 'next' | 'prev'): void {
	lastDirection = dir;
}

export function lastNavDirection(): 'next' | 'prev' {
	return lastDirection;
}

/**
 * A genuine available context load (ready/empty) landed: clear the skip streak
 * and lift any pause.
 */
export function noteSuccessfulLoad(): void {
	consecutiveSkips = 0;
	if (autoSkip.paused) autoSkip.paused = false;
}

/**
 * Ask permission to perform one auto-skip. Returns false once the cap is hit
 * (and flips `paused`), so the caller stops skipping and shows the banner.
 */
export function allowSkip(): boolean {
	if (autoSkip.paused) return false;
	consecutiveSkips += 1;
	autoSkip.skipped += 1;
	if (consecutiveSkips >= MAX_CONSECUTIVE_SKIPS) {
		autoSkip.paused = true;
		return false;
	}
	return true;
}

/** User dismissed the pause banner and wants to keep going. */
export function resumeAutoSkip(): void {
	autoSkip.paused = false;
	consecutiveSkips = 0;
}

/** The consecutive-skip cap (exposed for tests). */
export const MAX_CONSECUTIVE_SKIPS_FOR_TEST = MAX_CONSECUTIVE_SKIPS;

/** Test-only: clear all session state. */
export function _resetAutoSkip(): void {
	lastDirection = 'next';
	consecutiveSkips = 0;
	autoSkip.paused = false;
	autoSkip.skipped = 0;
}
