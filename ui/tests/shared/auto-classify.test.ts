import { describe, it, expect } from 'vitest';
import { classify } from '../../src/lib/shared/auto-classify';

// Helper for readability — sorted set comparison.
function cats(a: string, b: string): string[] {
	return [...classify(a, b)].sort();
}

describe('classify — skip / empty cases', () => {
	it('returns [] when before === after', () => {
		expect(classify('δήμος', 'δήμος')).toEqual([]);
	});

	it('returns [] when only whitespace differs but text trims to equal', () => {
		expect(classify('  δήμος  ', 'δήμος')).toEqual([]);
	});

	it('returns [] when before is empty', () => {
		expect(classify('', 'γεια')).toEqual([]);
	});

	it('returns [] when after is empty', () => {
		expect(classify('γεια', '')).toEqual([]);
	});

	it('returns [] when text is multiline', () => {
		expect(classify('γεια\nσας', 'γεια σας')).toEqual([]);
	});

	it('NFC normalizes both sides — precomposed vs decomposed accents are equal', () => {
		// 'δήμος' NFC vs same string with decomposed accent
		const decomposed = 'δήμος'; // η + combining tonos
		expect(classify('δήμος', decomposed)).toEqual([]);
	});
});

describe('classify — accent_tonos (tonos only, not dialytika)', () => {
	it('fires for pure tonos addition', () => {
		expect(cats('δημος', 'δήμος')).toEqual(['accent_tonos']);
	});

	it('fires for pure tonos removal', () => {
		expect(cats('δήμος', 'δημος')).toEqual(['accent_tonos']);
	});

	it('does NOT fire for dialytika change (semantic distinction preserved)', () => {
		// μαιου -> μαΐου introduces dialytika+tonos — meaningful change, not pure accent
		expect(classify('μαιου', 'μαΐου')).not.toContain('accent_tonos');
	});

	it('does not misfire on identical text', () => {
		expect(classify('δήμος', 'δήμος')).toEqual([]);
	});
});

describe('classify — final_sigma', () => {
	it('fires for σ → ς at word end', () => {
		expect(cats('οπωσ', 'οπως')).toEqual(['final_sigma']);
	});

	it('fires combined with accent_tonos when both apply', () => {
		expect(cats('οπωσ', 'όπως')).toEqual(['accent_tonos', 'final_sigma']);
	});

	it('does not fire on pure uppercase→lowercase of word ending with Σ (case-folding alone)', () => {
		// 'ΟΠΩΣ'.toLowerCase() === 'οπως' in JS (Unicode context-sensitive case folding).
		// So 'ΟΠΩΣ' → 'οπως' is a pure case change — final_sigma should NOT fire,
		// only punctuation_capitalization.
		const result = cats('ΟΠΩΣ', 'οπως');
		expect(result).not.toContain('final_sigma');
		expect(result).toContain('punctuation_capitalization');
	});

	it('does not fire when sigma is unchanged', () => {
		expect(classify('δήμος', 'δημος')).not.toContain('final_sigma');
	});
});

describe('classify — punctuation_capitalization', () => {
	it('fires for pure punctuation addition', () => {
		expect(cats('ναι κύριε πρόεδρε', 'ναι, κύριε πρόεδρε.')).toEqual(['punctuation_capitalization']);
	});

	it('fires for pure capitalization change', () => {
		expect(cats('ναι', 'Ναι')).toEqual(['punctuation_capitalization']);
	});

	it('does not fire when content words change', () => {
		expect(classify('ναι κύριε', 'όχι κύριε')).not.toContain('punctuation_capitalization');
	});

	it('Greek question mark normalized', () => {
		// `?` vs `;` (Greek erotimatiko) — punctuation only
		expect(cats('τι;', 'τι?')).toEqual(['punctuation_capitalization']);
	});
});

describe('classify — word_boundary', () => {
	it('fires for word split', () => {
		expect(cats('τονδήμο', 'τον δήμο')).toEqual(['word_boundary']);
	});

	it('fires for word merge', () => {
		expect(cats('τον δήμο', 'τονδήμο')).toEqual(['word_boundary']);
	});

	it('does not fire when stripping spaces does not equal', () => {
		expect(classify('τον δήμο', 'στον δήμο')).not.toContain('word_boundary');
	});

	it('does not fire when only punctuation differs and content already explained', () => {
		// 'θα πάμε' -> 'θα, πάμε' — same after stripping spaces? no, comma differs.
		// stripAllSpaces('θα πάμε') === 'θαπάμε'
		// stripAllSpaces('θα, πάμε') === 'θα,πάμε' → not equal → word_boundary does NOT fire
		const result = classify('θα πάμε', 'θα, πάμε');
		expect(result).not.toContain('word_boundary');
		expect(result).toContain('punctuation_capitalization');
	});
});

describe('classify — homophone', () => {
	it('does NOT fire for η → ει diphthong (different token length — grammatical risk)', () => {
		// Diphthong changes can encode number/grammatical agreement. Tight rule
		// requires same-length tokens; leave diphthong fixes for human review.
		expect(classify('να γίνη', 'να γίνει')).not.toContain('homophone');
	});

	it('fires for single-letter ι ↔ υ swap (same length)', () => {
		expect(cats('Λιγιάς', 'Λυγιάς')).toEqual(['homophone']);
	});

	it('fires for single-letter η ↔ ι swap (same length)', () => {
		expect(cats('διευκρίνηση', 'διευκρίνιση')).toEqual(['homophone']);
	});

	it('fires for ω → ο', () => {
		expect(cats('καλώς', 'καλός')).toEqual(['homophone']);
	});

	it('does NOT fire when token count differs', () => {
		expect(classify('γίνη αυτό', 'γίνει')).not.toContain('homophone');
	});

	it('does NOT fire when changes go beyond vowel classes', () => {
		// 'έδωσε' → 'έδειξε' is a real word change, not homophone
		expect(classify('έδωσε', 'έδειξε')).not.toContain('homophone');
	});

	it('does NOT fire on completely unrelated words with similar vowel skeletons', () => {
		// 'πίνω' (drink) vs 'πένω' — not valid Greek, but mapping makes them equal
		// More realistic: 'τι' vs 'τη' — both map to 'τI' under homophone mapping but
		// they are distinct words. Want this to fire (it IS a η/ι homophone), but
		// 'πίνω' vs 'πινάω' should not.
		expect(classify('πίνω', 'πινάω')).not.toContain('homophone');
	});

	it('does NOT fire when only differing token changes length (grammatical risk)', () => {
		// 'γινη' → 'γίνει' is a length-changing vowel diphthong; rejected by
		// the strict same-length per-token guard.
		const result = cats('να γινη', 'να γίνει');
		expect(result).toEqual([]);
	});

	it('fires for plural-of-singular-looking single-letter swap that IS truly homophone', () => {
		// 'Λιγιάς' → 'Λυγιάς' — both 6 chars; ι↔υ pure spelling fix.
		expect(cats('Λιγιάς', 'Λυγιάς')).toEqual(['homophone']);
	});

	it('does NOT fire on noun_case-style plural endings of different length', () => {
		// 'άπλητη' (6) → 'άπλητοι' (7) — grammatical number change, not homophone.
		expect(classify('άπλητη', 'άπλητοι')).not.toContain('homophone');
	});

	it('does NOT fire on ρεζίλοι → ρεζίλι (lexical difference, length 7→6)', () => {
		expect(classify('ρεζίλοι', 'ρεζίλι')).not.toContain('homophone');
	});
});

describe('classify — acronym_abbreviation', () => {
	it('fires for spaced lowercase → uppercase concat', () => {
		expect(cats('δ ε υ α', 'ΔΕΥΑ')).toEqual(['acronym_abbreviation']);
	});

	it('fires reverse direction (acronym expanded to spaced letters)', () => {
		expect(cats('ΔΕΥΑ', 'δ ε υ α')).toEqual(['acronym_abbreviation']);
	});

	it('does NOT fire for mixed Latin/Greek', () => {
		expect(classify('d ε υ α', 'ΔΕΥΑ')).toEqual([]);
	});

	it('does NOT fire for 2-letter "acronyms" (too short, risk of false positives)', () => {
		// Require at least 3 letters
		expect(classify('δ ε', 'ΔΕ')).not.toContain('acronym_abbreviation');
	});

	it('fires when embedded in a larger sentence', () => {
		const result = cats('η δ ε υ α είναι', 'η ΔΕΥΑ είναι');
		expect(result).toContain('acronym_abbreviation');
	});
});

describe('classify — disfluency_cleanup', () => {
	it('fires for leading filler removal', () => {
		expect(cats('ε ε ναι θα πάμε', 'ναι θα πάμε')).toEqual(['disfluency_cleanup']);
	});

	it('fires for repeated-bigram cleanup (consecutive duplicate)', () => {
		expect(cats('ναι ναι θα πάμε', 'ναι θα πάμε')).toEqual(['disfluency_cleanup']);
	});

	it('does NOT fire when tokens are reordered', () => {
		expect(classify('ε ναι πάμε', 'πάμε ναι')).not.toContain('disfluency_cleanup');
	});

	it('does NOT fire when content words are removed', () => {
		expect(classify('ναι θα πάμε σπίτι', 'ναι θα πάμε')).not.toContain('disfluency_cleanup');
	});

	it('does NOT fire when a new token appears in after', () => {
		expect(classify('ε ναι', 'ναι σίγουρα')).not.toContain('disfluency_cleanup');
	});

	it('does NOT fire when non-consecutive duplicates are removed', () => {
		// 'ναι πάμε ναι' -> 'ναι πάμε' would require removing a non-consecutive duplicate
		expect(classify('ναι πάμε ναι', 'ναι πάμε')).not.toContain('disfluency_cleanup');
	});
});

describe('classify — multi-category combos', () => {
	it('accent_tonos + final_sigma', () => {
		const result = cats('οπωσ', 'όπως');
		expect(result).toEqual(['accent_tonos', 'final_sigma']);
	});

	it('returns empty when no rule fits — substantive word change', () => {
		expect(classify('είναι σωστό', 'συμφωνώ απόλυτα')).toEqual([]);
	});

	it('returns empty for real word substitution (substitution_phonetic case)', () => {
		expect(classify('έδωσε λεφτά', 'έδειξε λεφτά')).toEqual([]);
	});

	it('returns empty for number/date conversion (number_date — not auto)', () => {
		expect(classify('πέντε ευρώ', '5 ευρώ')).toEqual([]);
	});

	it('returns empty for proper-name capitalization alone (could be person_name — not auto)', () => {
		// Pure first-letter cap fires punctuation_capitalization. We do NOT distinguish
		// person_name from generic capitalization — assigning person_name needs NER.
		const result = cats('παπαδόπουλος', 'Παπαδόπουλος');
		expect(result).toEqual(['punctuation_capitalization']);
		expect(result).not.toContain('person_name');
	});
});
