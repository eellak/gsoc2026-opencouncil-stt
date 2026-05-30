/**
 * Global mutex for full-corpus scans (stats aggregation, category index build).
 *
 * Each scan allocates large transient maps (~100 MB). On the 1 GB Oracle VM,
 * running two at once doubles the peak and OOMs the Node heap. The scans yield
 * to the event loop (so the server stays responsive), but that yielding is
 * exactly what lets a second scan interleave — so we serialise them here:
 * only one heavy scan runs at a time, the next waits its turn.
 */

let tail: Promise<unknown> = Promise.resolve();

export function runExclusiveScan<T>(fn: () => Promise<T>): Promise<T> {
	const run = tail.then(fn, fn);
	// Keep the chain alive regardless of individual failures.
	tail = run.then(
		() => undefined,
		() => undefined
	);
	return run;
}
