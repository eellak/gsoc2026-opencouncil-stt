const AUTOPLAY_KEY = 'oc:review:autoplay';
const LOOP_KEY = 'oc:review:loop';
const LOOP_GAP_KEY = 'oc:review:loopGapMs';

const LOOP_GAP_MIN = 0;
const LOOP_GAP_MAX = 3000;
const LOOP_GAP_DEFAULT = 100;

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

function createPlaybackPrefs() {
	let autoplay = $state(getBool(AUTOPLAY_KEY, true));
	let loop = $state(getBool(LOOP_KEY, false));
	let loopGapMs = $state(getInt(LOOP_GAP_KEY, LOOP_GAP_DEFAULT, LOOP_GAP_MIN, LOOP_GAP_MAX));

	return {
		get autoplay() { return autoplay; },
		get loop() { return loop; },
		get loopGapMs() { return loopGapMs; },
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
		}
	};
}

export const playbackPrefs = createPlaybackPrefs();
