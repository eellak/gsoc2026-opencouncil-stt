import { describe, it, expect } from 'vitest';
import { computePoolPolicy } from '../../src/lib/client/audio-pool-policy';

describe('computePoolPolicy', () => {
	it('keeps ONLY current + neighbours and evicts everything else (no fill-to-max slack)', () => {
		const existing = ['stale1', 'stale2', 'cur', 'n0', 'n1', 'stale3'];
		const p = computePoolPolicy('cur', ['n0', 'n1'], existing, 1);
		expect([...p.keep].sort()).toEqual(['cur', 'n0', 'n1']);
		expect(p.evict.sort()).toEqual(['stale1', 'stale2', 'stale3']);
	});

	it('marks current + the first `autoNeighbours` priority neighbours as auto; rest metadata', () => {
		const p = computePoolPolicy('cur', ['n0', 'n1', 'n2'], ['cur', 'n0', 'n1', 'n2'], 1);
		// current + next target (n0) get the bandwidth-heavy preload=auto.
		expect([...p.auto].sort()).toEqual(['cur', 'n0']);
		// remaining kept neighbours are cheap metadata-only.
		expect([...p.metadata].sort()).toEqual(['n1', 'n2']);
	});

	it('always keeps current and always marks it auto, never metadata', () => {
		const p = computePoolPolicy('cur', [], ['cur', 'old'], 1);
		expect(p.keep.has('cur')).toBe(true);
		expect(p.auto.has('cur')).toBe(true);
		expect(p.metadata.has('cur')).toBe(false);
		expect(p.evict).toEqual(['old']);
	});

	it('protects the resolved next target (neighbour[0]) from eviction even amid many stale elements', () => {
		const existing = ['a', 'b', 'c', 'd', 'e', 'f', 'cur', 'nextTarget'];
		const p = computePoolPolicy('cur', ['nextTarget'], existing, 1);
		expect(p.keep.has('nextTarget')).toBe(true);
		expect(p.evict).not.toContain('nextTarget');
		expect(p.auto.has('nextTarget')).toBe(true);
	});

	it('does not double-count a neighbour that equals current', () => {
		const p = computePoolPolicy('cur', ['cur', 'n0'], ['cur', 'n0'], 1);
		expect([...p.keep].sort()).toEqual(['cur', 'n0']);
		expect([...p.auto].sort()).toEqual(['cur', 'n0']);
		expect(p.metadata.size).toBe(0);
	});

	it('autoNeighbours=2 promotes current + first two neighbours to auto', () => {
		const p = computePoolPolicy('cur', ['n0', 'n1', 'n2'], ['cur', 'n0', 'n1', 'n2'], 2);
		expect([...p.auto].sort()).toEqual(['cur', 'n0', 'n1']);
		expect([...p.metadata]).toEqual(['n2']);
	});
});
