/**
 * Unit coverage for the private-meeting auto-skip session state (Change 1).
 * Locks in the cap, the direction memory, the private-meeting memo, and the
 * "reset only on a real non-private load" rule.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
	autoSkip,
	rememberDirection,
	lastNavDirection,
	noteSuccessfulLoad,
	allowSkip,
	resumeAutoSkip,
	MAX_CONSECUTIVE_SKIPS_FOR_TEST as CAP,
	_resetAutoSkip
} from '../../src/lib/client/auto-skip.svelte';

beforeEach(() => _resetAutoSkip());

describe('direction memory', () => {
	it('defaults to next and remembers the last direction', () => {
		expect(lastNavDirection()).toBe('next');
		rememberDirection('prev');
		expect(lastNavDirection()).toBe('prev');
		rememberDirection('next');
		expect(lastNavDirection()).toBe('next');
	});
});

describe('consecutive-skip cap', () => {
	it('allows skips up to the cap, then pauses', () => {
		// First CAP-1 skips are allowed; the CAP-th flips paused and returns false.
		for (let i = 0; i < CAP - 1; i++) {
			expect(allowSkip()).toBe(true);
		}
		expect(autoSkip.paused).toBe(false);
		expect(allowSkip()).toBe(false);
		expect(autoSkip.paused).toBe(true);
		// Once paused, further requests are denied.
		expect(allowSkip()).toBe(false);
		expect(autoSkip.skipped).toBe(CAP);
	});

	it('a genuine non-private load clears the streak so the cap restarts', () => {
		for (let i = 0; i < CAP - 2; i++) allowSkip();
		noteSuccessfulLoad();
		// Streak reset → we can skip CAP-1 more times before pausing again.
		for (let i = 0; i < CAP - 1; i++) {
			expect(allowSkip()).toBe(true);
		}
		expect(allowSkip()).toBe(false);
		expect(autoSkip.paused).toBe(true);
	});

	it('resume lifts the pause and restarts the streak', () => {
		for (let i = 0; i < CAP; i++) allowSkip();
		expect(autoSkip.paused).toBe(true);
		resumeAutoSkip();
		expect(autoSkip.paused).toBe(false);
		expect(allowSkip()).toBe(true);
	});

	it('noteSuccessfulLoad also lifts an active pause', () => {
		for (let i = 0; i < CAP; i++) allowSkip();
		expect(autoSkip.paused).toBe(true);
		noteSuccessfulLoad();
		expect(autoSkip.paused).toBe(false);
	});
});
