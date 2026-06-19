import { describe, it, expect } from 'vitest';
import { cacheHashWithExclusions } from '../../src/lib/domain/groups';

describe('cacheHashWithExclusions', () => {
	it('returns the bare source hash when there are no exclusions', () => {
		expect(cacheHashWithExclusions('abc123', null)).toBe('abc123');
		expect(cacheHashWithExclusions('abc123', undefined)).toBe('abc123');
	});

	it('folds the exclusion digest in when present', () => {
		expect(cacheHashWithExclusions('abc123', 'deadbeef')).toBe('abc123+xdeadbeef');
	});

	it('different exclusion digests on the same CSV produce different cache_hash', () => {
		const a = cacheHashWithExclusions('csv', 'excl1');
		const b = cacheHashWithExclusions('csv', 'excl2');
		expect(a).not.toBe(b);
		// …and both differ from the unfiltered hash, so a filtered index can never
		// collide with the full index's dependent snapshots.
		expect(a).not.toBe('csv');
		expect(b).not.toBe('csv');
	});
});
