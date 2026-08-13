const AUTOPLAY_KEY = 'oc:review:autoplay';
const LOOP_KEY = 'oc:review:loop';
const LOOP_GAP_KEY = 'oc:review:loopGapMs';
const NUDGE_STEP_KEY = 'oc:review:nudgeStepMs';
const SPEED_KEY = 'oc:review:speed';

// Playback-rate ladder for quick skimming of long clips. `s` cycles through it.
export const SPEEDS = [1, 1.25, 1.5, 1.75, 2, 2.5, 3] as const;

const LOOP_GAP_MIN = 0;
const LOOP_GAP_MAX = 3000;
const LOOP_GAP_DEFAULT = 100;

// Manual segment fine-sync step ("delay"): how far each nudge moves a
// boundary. Stored/rounded to the nearest 10ms within [10, 1000].
const NUDGE_STEP_MIN = 10;
const NUDGE_STEP_MAX = 1000;
const NUDGE_STEP_DEFAULT = 200;
const NUDGE_STEP_GRID = 10;

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
	let speed = $state(readSpeed());

	function readSpeed(): number {
		if (typeof localStorage === 'undefined') return 1;
		const v = Number(localStorage.getItem(SPEED_KEY));
		return (SPEEDS as readonly number[]).includes(v) ? v : 1;
	}

	return {
		get autoplay() { return autoplay; },
		get loop() { return loop; },
		get loopGapMs() { return loopGapMs; },
		get nudgeStepMs() { return nudgeStepMs; },
		get speed() { return speed; },
		setSpeed(v: number) {
			if (!(SPEEDS as readonly number[]).includes(v)) return;
			speed = v;
			if (typeof localStorage !== 'undefined') localStorage.setItem(SPEED_KEY, String(v));
		},
		cycleSpeed(dir: 1 | -1 = 1) {
			const i = (SPEEDS as readonly number[]).indexOf(speed);
			const next = SPEEDS[(i + dir + SPEEDS.length) % SPEEDS.length];
			this.setSpeed(next);
		},
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
