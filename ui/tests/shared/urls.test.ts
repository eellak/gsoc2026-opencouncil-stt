import { describe, it, expect } from 'vitest';
import { reviewHref, editHref, errorCategoryHref, parseSeedParam, randomSeed, hashSeedString } from '../../src/lib/shared/urls';

describe('reviewHref', () => {
	it('builds basic href', () => {
		expect(reviewHref({ utterance_id: 'u1', seed: 42 })).toBe('/review/u1?seed=42');
	});

	it('appends highlight when provided', () => {
		expect(reviewHref({ utterance_id: 'u1', seed: 42, highlight: 'e9' })).toBe(
			'/review/u1?seed=42&highlight=e9'
		);
	});

	it('encodes special characters in utterance_id', () => {
		expect(reviewHref({ utterance_id: 'u/1 a', seed: 1 })).toBe('/review/u%2F1%20a?seed=1');
	});

	it('encodes highlight value', () => {
		expect(reviewHref({ utterance_id: 'u1', seed: 1, highlight: 'a&b' })).toBe(
			'/review/u1?seed=1&highlight=a%26b'
		);
	});
});

describe('editHref', () => {
	it('builds /edit/{id}', () => {
		expect(editHref('e1')).toBe('/edit/e1');
	});
	it('preserves seed when provided', () => {
		expect(editHref('e1', 42)).toBe('/edit/e1?seed=42');
	});
});

describe('errorCategoryHref', () => {
	it('builds basic href', () => {
		expect(errorCategoryHref('homophone')).toBe('/error-category/homophone');
	});
	it('preserves seed', () => {
		expect(errorCategoryHref('homophone', 7)).toBe('/error-category/homophone?seed=7');
	});
	it('encodes the category id', () => {
		expect(errorCategoryHref('foo bar')).toBe('/error-category/foo%20bar');
	});
});

describe('parseSeedParam', () => {
	it('accepts a non-negative integer string', () => {
		expect(parseSeedParam('0')).toBe(0);
		expect(parseSeedParam('42')).toBe(42);
		expect(parseSeedParam('4294967295')).toBe(4294967295); // uint32 max
	});

	it('returns null for empty / null', () => {
		expect(parseSeedParam('')).toBeNull();
		expect(parseSeedParam(null)).toBeNull();
		expect(parseSeedParam(undefined)).toBeNull();
	});

	it('rejects overflow integer (> uint32 max)', () => {
		// digits-only but out of uint32 range → null
		expect(parseSeedParam('4294967296')).toBeNull();
	});

	it('hashes any non-empty non-integer string to a stable uint32', () => {
		const h = parseSeedParam('christos');
		expect(h).not.toBeNull();
		expect(Number.isInteger(h)).toBe(true);
		expect(h!).toBeGreaterThanOrEqual(0);
		expect(h!).toBeLessThanOrEqual(4294967295);
		// Stable: same input → same output
		expect(parseSeedParam('christos')).toBe(h);
		// Case-insensitive
		expect(parseSeedParam('CHRISTOS')).toBe(h);
		// Negative sign, dots, etc. become part of the string → hash
		expect(parseSeedParam('-1')).toBe(hashSeedString('-1'));
		expect(parseSeedParam('1.5')).toBe(hashSeedString('1.5'));
	});
});

describe('randomSeed', () => {
	it('returns a value in [0, 2^32-1]', () => {
		for (let i = 0; i < 50; i++) {
			const s = randomSeed();
			expect(Number.isInteger(s)).toBe(true);
			expect(s).toBeGreaterThanOrEqual(0);
			expect(s).toBeLessThanOrEqual(4294967295);
		}
	});
});
