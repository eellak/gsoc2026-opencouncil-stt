const AUTO_ADVANCE_KEY = 'oc:review:autoAdvance';
const MOBILE_MODE_KEY = 'oc:review:mobileMode';
const TAP_NEXT_KEY = 'oc:review:tapAdvances';

function getBool(key: string, fallback: boolean): boolean {
	if (typeof localStorage === 'undefined') return fallback;
	const v = localStorage.getItem(key);
	return v === null ? fallback : v === 'true';
}

/** Coarse heuristic: touch-primary devices default to mobile mode. */
function detectMobile(): boolean {
	if (typeof window === 'undefined') return false;
	try {
		if (window.matchMedia('(pointer: coarse)').matches) return true;
	} catch { /* fine */ }
	return false;
}

function persist(key: string, value: boolean) {
	if (typeof localStorage !== 'undefined') localStorage.setItem(key, String(value));
}

function createReviewPrefs() {
	let autoAdvance = $state(getBool(AUTO_ADVANCE_KEY, false));
	let mobileMode = $state(getBool(MOBILE_MODE_KEY, detectMobile()));
	let tapAdvances = $state(getBool(TAP_NEXT_KEY, true));

	return {
		get autoAdvance() { return autoAdvance; },
		get mobileMode() { return mobileMode; },
		get tapAdvances() { return tapAdvances; },
		toggleAutoAdvance() {
			autoAdvance = !autoAdvance;
			persist(AUTO_ADVANCE_KEY, autoAdvance);
		},
		toggleMobileMode() {
			mobileMode = !mobileMode;
			persist(MOBILE_MODE_KEY, mobileMode);
		},
		toggleTapAdvances() {
			tapAdvances = !tapAdvances;
			persist(TAP_NEXT_KEY, tapAdvances);
		}
	};
}

export const reviewPrefs = createReviewPrefs();
