const AUTOPLAY_KEY = 'oc:review:autoplay';
const LOOP_KEY = 'oc:review:loop';
const LOOP_GAP_KEY = 'oc:review:loopGapMs';
const NUDGE_STEP_KEY = 'oc:review:nudgeStepMs';

const LOOP_GAP_MIN = 0;
const LOOP_GAP_MAX = 3000;
const LOOP_GAP_DEFAULT = 100;

// Manual segment fine-sync step ("delay"): how far each nudge moves a
// boundary. Stored/rounded to the nearest 50ms within [50, 1000].
const NUDGE_STEP_MIN = 50;
const NUDGE_STEP_MAX = 1000;
const NUDGE_STEP_DEFAULT = 200;
const NUDGE_STEP_GRID = 50;

function getBool(key: string, fallback: boolean): boolean {
	if (typeof localStorage === 'undefined') return fallback;
	const v = localStorage.getItem(key);
	return v === null ? fallback : v === 'true';
}

function getInt(key: string, fallback: number, min: number, max: number): number {
	if (typeof localStorage === 'undefined') return fallback;
	const v = localStorage.getItem(key);
	if (v === null) return fallback;
	const n = Number.parseInt(v, 10);
	if (!Number.isFinite(n)) return fallback;
	return Math.min(max, Math.max(min, n));
}

function snapNudgeStep(ms: number): number {
	const grid = Math.round(ms / NUDGE_STEP_GRID) * NUDGE_STEP_GRID;
	return Math.min(NUDGE_STEP_MAX, Math.max(NUDGE_STEP_MIN, grid));
}

function createPlaybackPrefs() {
	let autoplay = $state(getBool(AUTOPLAY_KEY, true));
	let loop = $state(getBool(LOOP_KEY, false));
	let loopGapMs = $state(getInt(LOOP_GAP_KEY, LOOP_GAP_DEFAULT, LOOP_GAP_MIN, LOOP_GAP_MAX));
	let nudgeStepMs = $state(snapNudgeStep(getInt(NUDGE_STEP_KEY, NUDGE_STEP_DEFAULT, NUDGE_STEP_MIN, NUDGE_STEP_MAX)));

	return {
		get autoplay() { return autoplay; },
		get loop() { return loop; },
		get loopGapMs() { return loopGapMs; },
		get nudgeStepMs() { return nudgeStepMs; },
		toggleAutoplay() {
			autoplay = !autoplay;
			if (typeof localStorage !== 'undefined') localStorage.setItem(AUTOPLAY_KEY, String(autoplay));
		},
		toggleLoop() {
			loop = !loop;
			if (typeof localStorage !== 'undefined') localStorage.setItem(LOOP_KEY, String(loop));
		},
		setLoopGapMs(ms: number) {
			const clamped = Math.min(LOOP_GAP_MAX, Math.max(LOOP_GAP_MIN, Math.round(ms)));
			loopGapMs = clamped;
			if (typeof localStorage !== 'undefined') localStorage.setItem(LOOP_GAP_KEY, String(clamped));
		},
		setNudgeStepMs(ms: number) {
			const snapped = snapNudgeStep(ms);
			nudgeStepMs = snapped;
			if (typeof localStorage !== 'undefined') localStorage.setItem(NUDGE_STEP_KEY, String(snapped));
		}
	};
}

export const playbackPrefs = createPlaybackPrefs();
