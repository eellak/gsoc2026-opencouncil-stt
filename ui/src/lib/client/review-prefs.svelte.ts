const AUTO_ADVANCE_KEY = 'oc:review:autoAdvance';
const MOBILE_MODE_KEY = 'oc:review:mobileMode';

function getBool(key: string, fallback: boolean): boolean {
	if (typeof localStorage === 'undefined') return fallback;
	const v = localStorage.getItem(key);
	return v === null ? fallback : v === 'true';
}

function persist(key: string, value: boolean) {
	if (typeof localStorage !== 'undefined') localStorage.setItem(key, String(value));
}

function createReviewPrefs() {
	let autoAdvance = $state(getBool(AUTO_ADVANCE_KEY, false));
	let mobileMode = $state(getBool(MOBILE_MODE_KEY, false));

	return {
		get autoAdvance() { return autoAdvance; },
		get mobileMode() { return mobileMode; },
		toggleAutoAdvance() {
			autoAdvance = !autoAdvance;
			persist(AUTO_ADVANCE_KEY, autoAdvance);
		},
		toggleMobileMode() {
			mobileMode = !mobileMode;
			persist(MOBILE_MODE_KEY, mobileMode);
		}
	};
}

export const reviewPrefs = createReviewPrefs();
