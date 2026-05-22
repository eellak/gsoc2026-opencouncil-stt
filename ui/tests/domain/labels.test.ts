import { describe, it, expect } from 'vitest';
import {
	normalizeLabelCategories,
	canonicalizeLabel,
	validateCategoriesForWrite
} from '../../src/lib/domain/labels';

describe('normalizeLabelCategories', () => {
	it('returns [] for null / undefined / empty', () => {
		expect(normalizeLabelCategories(null)).toEqual([]);
		expect(normalizeLabelCategories(undefined)).toEqual([]);
		expect(normalizeLabelCategories({})).toEqual([]);
	});

	it('reads legacy single-value error_category', () => {
		expect(normalizeLabelCategories({ error_category: 'homophone' })).toEqual(['homophone']);
	});

	it('treats legacy error_category: null as empty', () => {
		expect(normalizeLabelCategories({ error_category: null })).toEqual([]);
	});

	it('reads new multi-value error_categories', () => {
		expect(normalizeLabelCategories({ error_categories: ['a', 'b'] })).toEqual(['a', 'b']);
	});

	it('prefers error_categories when both fields are present', () => {
		expect(
			normalizeLabelCategories({ error_categories: ['a'], error_category: 'b' })
		).toEqual(['a']);
	});

	it('dedupes while preserving first-seen order', () => {
		expect(
			normalizeLabelCategories({ error_categories: ['b', 'a', 'b', 'c', 'a'] })
		).toEqual(['b', 'a', 'c']);
	});

	it('drops non-string entries silently', () => {
		expect(
			normalizeLabelCategories({ error_categories: ['a', 42, null, 'b', undefined, ''] as unknown[] })
		).toEqual(['a', 'b']);
	});

	it('preserves unknown taxonomy ids on read (lenient)', () => {
		expect(normalizeLabelCategories({ error_categories: ['homophone', 'totally_bogus'] })).toEqual(
			['homophone', 'totally_bogus']
		);
	});
});

describe('validateCategoriesForWrite', () => {
	it('accepts known taxonomy ids', () => {
		expect(() => validateCategoriesForWrite(['homophone', 'accent_tonos'])).not.toThrow();
	});

	it('accepts empty array', () => {
		expect(() => validateCategoriesForWrite([])).not.toThrow();
	});

	it('rejects unknown taxonomy ids on write', () => {
		expect(() => validateCategoriesForWrite(['totally_bogus'])).toThrow(/unknown/i);
	});

	it('rejects non-array input', () => {
		expect(() => validateCategoriesForWrite('homophone' as unknown as string[])).toThrow();
	});
});

describe('canonicalizeLabel', () => {
	it('upgrades legacy single-field label to new shape', () => {
		const lbl = canonicalizeLabel({
			error_category: 'homophone',
			include_status: 'include'
		});
		expect(lbl.error_categories).toEqual(['homophone']);
		expect('error_category' in lbl).toBe(false);
		expect(lbl.include_status).toBe('include');
	});

	it('keeps already-canonical label intact', () => {
		const lbl = canonicalizeLabel({
			error_categories: ['a', 'b'],
			include_status: 'unreviewed'
		});
		expect(lbl.error_categories).toEqual(['a', 'b']);
	});
});
