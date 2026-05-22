export interface AudioRegion {
	start: number;
	end: number;
}

const MIN_DURATION = 0.1;

function roundTime(value: number): number {
	return Math.round(value * 10) / 10;
}

export function clampRegion(start: number, end: number): AudioRegion {
	const clampedStart = Math.max(0, start);
	const clampedEnd = Math.max(clampedStart + MIN_DURATION, end);
	return { start: roundTime(clampedStart), end: roundTime(clampedEnd) };
}

export function nudgeRegionStart(region: AudioRegion, delta: number): AudioRegion {
	const maxStart = region.end - MIN_DURATION;
	return clampRegion(Math.min(maxStart, region.start + delta), region.end);
}

export function nudgeRegionEnd(region: AudioRegion, delta: number): AudioRegion {
	return clampRegion(region.start, Math.max(region.start + MIN_DURATION, region.end + delta));
}

export function expandRegion(region: AudioRegion, seconds: number): AudioRegion {
	return clampRegion(region.start - seconds, region.end + seconds);
}

export function contractRegion(region: AudioRegion, seconds: number): AudioRegion {
	const duration = region.end - region.start;
	const maxStep = Math.max(0, (duration - MIN_DURATION) / 2);
	const step = Math.min(seconds, maxStep);
	return clampRegion(region.start + step, region.end - step);
}

export function resetRegion(start: number, end: number): AudioRegion {
	return clampRegion(start, end);
}

export function shiftRegion(region: AudioRegion, delta: number): AudioRegion {
	return clampRegion(region.start + delta, region.end + delta);
}
