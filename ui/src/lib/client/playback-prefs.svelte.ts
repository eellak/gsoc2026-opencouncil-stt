const AUTOPLAY_KEY = 'oc:review:autoplay';
const LOOP_KEY = 'oc:review:loop';

function getBool(key: string, fallback: boolean): boolean {
	if (typeof localStorage === 'undefined') return fallback;
	const v = localStorage.getItem(key);
	return v === null ? fallback : v === 'true';
}

function createPlaybackPrefs() {
	let autoplay = $state(getBool(AUTOPLAY_KEY, true));
	let loop = $state(getBool(LOOP_KEY, false));

	return {
		get autoplay() { return autoplay; },
		get loop() { return loop; },
		toggleAutoplay() {
			autoplay = !autoplay;
			if (typeof localStorage !== 'undefined') localStorage.setItem(AUTOPLAY_KEY, String(autoplay));
		},
		toggleLoop() {
			loop = !loop;
			if (typeof localStorage !== 'undefined') localStorage.setItem(LOOP_KEY, String(loop));
		}
	};
}

export const playbackPrefs = createPlaybackPrefs();
