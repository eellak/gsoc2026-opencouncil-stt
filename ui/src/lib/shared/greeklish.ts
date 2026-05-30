/**
 * Greek ↔ Latin transliteration helpers for fuzzy palette search.
 *
 * Goal: a reviewer typing "akronymio" should find Ακρωνύμιο. A reviewer
 * typing "punctuation" should still find the English label. We achieve that
 * by transliterating every entry's Greek label to a Latin form and including
 * the Latin form in the searchable haystack — no need to also transliterate
 * the query.
 *
 * Mapping is the common ISO-ish/colloquial Greeklish convention used in
 * casual writing (β→v, η→i, ω→o, χ→ch, ψ→ps, ξ→x, θ→th). Accents and
 * diaeresis are stripped on both sides via NFD; the result is folded to
 * lowercase ASCII so substring match Just Works.
 */

const SINGLE_MAP: Record<string, string> = {
	α: 'a', ά: 'a',
	β: 'v',
	γ: 'g',
	δ: 'd',
	ε: 'e', έ: 'e',
	ζ: 'z',
	η: 'i', ή: 'i',
	θ: 'th',
	ι: 'i', ί: 'i', ϊ: 'i', ΐ: 'i',
	κ: 'k',
	λ: 'l',
	μ: 'm',
	ν: 'n',
	ξ: 'x',
	ο: 'o', ό: 'o',
	π: 'p',
	ρ: 'r',
	σ: 's', ς: 's',
	τ: 't',
	υ: 'y', ύ: 'y', ϋ: 'y', ΰ: 'y',
	φ: 'f',
	χ: 'ch',
	ψ: 'ps',
	ω: 'o', ώ: 'o'
};

/** Transliterate a Greek string to its Greeklish/ASCII approximation. */
export function greekToLatin(input: string): string {
	const lower = input.toLowerCase();
	let out = '';
	for (const ch of lower) {
		out += SINGLE_MAP[ch] ?? ch;
	}
	// Strip any remaining combining marks (NFD decomposition leftovers).
	return out.normalize('NFD').replace(/[̀-ͯ]/g, '');
}

/** Lowercase + strip diacritics for accent-insensitive substring matching. */
export function foldAscii(input: string): string {
	return input
		.toLowerCase()
		.normalize('NFD')
		.replace(/[̀-ͯ]/g, '');
}
