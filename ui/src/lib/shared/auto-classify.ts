/**
 * Auto-classifier for correction edits.
 *
 * Given a (before, after) text pair, returns the set of TAXONOMY categories
 * that the diff *clearly* matches by purely mechanical rules. Returns an
 * empty array when the diff is ambiguous, requires NER/morphology, or no
 * rule fires cleanly. The bar is: zero false positives in exchange for
 * lower recall — anything uncertain stays unlabeled.
 *
 * Rules operate on NFC-normalized, trimmed inputs. Multiple categories may
 * fire on the same pair when their dimensions independently contribute
 * (e.g. accent_tonos + final_sigma on 'οπωσ' → 'όπως').
 */

import type { TaxonomyId } from './taxonomy';

// ── Greek letter classes ────────────────────────────────────────────────

// Greek letter inventory (excluding final sigma ς, which is handled via
// sigma normalization). Used for acronym detection.
const GREEK_LOWER = 'αβγδεζηθικλμνξοπρστυφχψω';
const GREEK_UPPER = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ';

// Combining tonos / oxia (U+0301). Stripped for accent_tonos comparisons.
// We deliberately do NOT strip combining dialytika (U+0308) so that
// μαιου vs μαΐου stay distinguishable (semantic spelling difference).
const COMBINING_TONOS_RE = /́/g;

// Disfluency fillers — strict allowlist, compared after accent strip + lower.
const FILLERS = new Set(['ε', 'εε', 'εεε', 'εμ', 'εμμ', 'αα', 'μμ', 'χμ']);

// ── String normalizers ──────────────────────────────────────────────────

function nfc(s: string): string {
	return s.normalize('NFC');
}

function stripAccentsOnly(s: string): string {
	// NFD decompose, drop tonos/oxia, recompose. Other combining marks survive.
	return s.normalize('NFD').replace(COMBINING_TONOS_RE, '').normalize('NFC');
}

function normalizeSigma(s: string): string {
	return s.replace(/ς/g, 'σ');
}

function stripNonSpacePunct(s: string): string {
	// Strip everything except letters, numbers, and whitespace. Greek letters
	// pass through \p{L}. Then collapse whitespace runs to a single space.
	return s.replace(/[^\p{L}\p{N}\s]+/gu, '').replace(/\s+/g, ' ').trim();
}

function stripAllSpaces(s: string): string {
	return s.replace(/\s+/g, '');
}

function tokens(s: string): string[] {
	const t = s.trim();
	if (!t) return [];
	return t.split(/\s+/);
}

function core(s: string): string {
	// Collapse case, accents, and sigma. If core(a) === core(b), the diff
	// between a and b lives entirely within {accent, sigma, case}.
	return stripAccentsOnly(normalizeSigma(s.toLowerCase()));
}

// ── Rule implementations ────────────────────────────────────────────────

function rulePunctuationCapitalization(a: string, b: string): boolean {
	// Same content (letters+digits+whitespace structure) ignoring case and
	// non-space punctuation. Whitespace structure preserved (so word_boundary
	// changes do NOT trigger this rule).
	const na = stripNonSpacePunct(a).toLowerCase();
	const nb = stripNonSpacePunct(b).toLowerCase();
	return na === nb;
}

function ruleAccentTonos(a: string, b: string): boolean {
	// Fires when diff lies wholly within {accent, sigma, case} AND tonos
	// pattern actually differs (so accent_tonos alone fires when it's the
	// only change, and co-fires with final_sigma when both contribute).
	if (core(a) !== core(b)) return false;
	const sa = normalizeSigma(a.toLowerCase());
	const sb = normalizeSigma(b.toLowerCase());
	return sa !== sb;
}

function ruleFinalSigma(a: string, b: string): boolean {
	// Fires when diff lies wholly within {accent, sigma, case} AND sigma
	// (ς↔σ) pattern actually differs.
	if (core(a) !== core(b)) return false;
	const sa = stripAccentsOnly(a.toLowerCase());
	const sb = stripAccentsOnly(b.toLowerCase());
	return sa !== sb;
}

function ruleWordBoundary(a: string, b: string): boolean {
	if (stripAllSpaces(a) !== stripAllSpaces(b)) return false;
	return a !== b;
}

// Per-token homophone classifier. A token differs by "homophone class" when
// its vowel-equivalence skeleton matches the other token's skeleton AND the
// tokens are NOT already equivalent under accent/sigma/case normalization
// alone (otherwise it's just a tonos/sigma fix, not a homophone substitution).
const SIMPLE_VOWEL_CLASS: Record<string, string> = {
	η: 'I', ι: 'I', υ: 'I',
	ω: 'O', ο: 'O',
	ε: 'E', α: 'A'
};

function homophoneSkeleton(token: string): string {
	const base = normalizeSigma(stripAccentsOnly(token).toLowerCase());
	let out = '';
	let i = 0;
	while (i < base.length) {
		const pair = base.slice(i, i + 2);
		if (pair === 'ει' || pair === 'οι') { out += 'I'; i += 2; continue; }
		if (pair === 'αι') { out += 'E'; i += 2; continue; }
		if (pair === 'ου') { out += 'U'; i += 2; continue; }
		const ch = base[i];
		out += SIMPLE_VOWEL_CLASS[ch] ?? ch;
		i++;
	}
	return out;
}

function tokenCoreEqual(x: string, y: string): boolean {
	// Two tokens are equivalent under accent/sigma/case only.
	return core(x) === core(y);
}

function ruleHomophone(a: string, b: string): boolean {
	// Strict: tokens 1:1, AND for each differing pair the lengths must be
	// equal (single-letter homophone swaps: η↔ι↔υ, ω↔ο). Different-length
	// pairs (e.g. 'γίνη'→'γίνει', 'άπλητη'→'άπλητοι') often encode
	// grammatical changes (number/case) that must not be silently labeled
	// as homophone. We accept the recall cost: ει/οι/αι/ου diphthong fixes
	// require human review.
	const ta = tokens(a);
	const tb = tokens(b);
	if (ta.length !== tb.length || ta.length === 0) return false;
	let anyHomophoneDiff = false;
	for (let i = 0; i < ta.length; i++) {
		if (ta[i] === tb[i]) continue;
		if (tokenCoreEqual(ta[i], tb[i])) continue; // accent/sigma/case only
		// Same length and skeleton-equivalent => single-letter vowel swap(s).
		const stripA = stripAccentsOnly(ta[i]).toLowerCase();
		const stripB = stripAccentsOnly(tb[i]).toLowerCase();
		if (stripA.length !== stripB.length) return false;
		if (homophoneSkeleton(ta[i]) !== homophoneSkeleton(tb[i])) return false;
		// Additionally require: every differing character position is in the
		// single-letter homophone equivalence set. This guards against
		// coincidental skeleton matches across non-vowel characters.
		for (let k = 0; k < stripA.length; k++) {
			if (stripA[k] === stripB[k]) continue;
			if (!isSimpleVowelSwap(stripA[k], stripB[k])) return false;
		}
		anyHomophoneDiff = true;
	}
	return anyHomophoneDiff;
}

function isSimpleVowelSwap(x: string, y: string): boolean {
	const I = new Set(['η', 'ι', 'υ']);
	const O = new Set(['ω', 'ο']);
	return (I.has(x) && I.has(y)) || (O.has(x) && O.has(y));
}

function ruleAcronymAbbreviation(a: string, b: string): boolean {
	return detectAcronym(a, b) || detectAcronym(b, a);
}

function detectAcronym(spaced: string, concat: string): boolean {
	// Token-based: find contiguous runs of single-Greek-letter tokens of
	// length ≥3, try each length & position, replace with uppercase concat,
	// compare against `concat` after whitespace normalization.
	const tks = tokens(spaced);
	if (tks.length === 0) return false;
	const isSingleGreek = (t: string) => t.length === 1 && GREEK_LETTER_RE.test(t);
	const targetTokens = tokens(concat);

	for (let i = 0; i < tks.length; i++) {
		if (!isSingleGreek(tks[i])) continue;
		// Extend run as far as possible.
		let j = i;
		while (j < tks.length && isSingleGreek(tks[j])) j++;
		const maxLen = j - i;
		if (maxLen < 3) continue;
		// Try each sub-run length from maxLen down to 3, anchored at start i,
		// and also try moving the start within the run.
		for (let start = i; start + 3 <= j; start++) {
			for (let len = j - start; len >= 3; len--) {
				const letters = tks.slice(start, start + len);
				const acronymUpper = letters.map((l) => l.toUpperCase()).join('');
				const candidateTokens = [
					...tks.slice(0, start),
					acronymUpper,
					...tks.slice(start + len)
				];
				if (candidateTokens.length !== targetTokens.length) continue;
				let allMatch = true;
				for (let k = 0; k < candidateTokens.length; k++) {
					if (nfc(candidateTokens[k]) !== nfc(targetTokens[k])) {
						allMatch = false;
						break;
					}
				}
				if (allMatch) return true;
			}
		}
		// Skip ahead to end of run to avoid re-scanning.
		i = j - 1;
	}
	return false;
}

const GREEK_LETTER_RE = new RegExp(`^[${GREEK_LOWER}${GREEK_UPPER}]$`);

function ruleDisfluencyCleanup(a: string, b: string): boolean {
	const ta = tokens(a);
	const tb = tokens(b);
	if (ta.length === 0 || tb.length === 0) return false;
	if (tb.length >= ta.length) return false;

	const norm = (t: string) => stripNonSpacePunct(stripAccentsOnly(t).toLowerCase());
	const na = ta.map(norm);
	const nb = tb.map(norm);

	let j = 0;
	const removed: { tok: string; prev: string | null }[] = [];
	for (let i = 0; i < na.length; i++) {
		if (j < nb.length && na[i] === nb[j]) {
			j++;
		} else {
			removed.push({ tok: na[i], prev: i > 0 ? na[i - 1] : null });
		}
	}
	if (j !== nb.length) return false;
	if (removed.length === 0) return false;

	for (const r of removed) {
		const isFiller = FILLERS.has(r.tok);
		const isConsecDup = r.prev !== null && r.prev === r.tok;
		if (!isFiller && !isConsecDup) return false;
	}
	return true;
}

// ── Entry point ─────────────────────────────────────────────────────────

export function classify(beforeRaw: string, afterRaw: string): TaxonomyId[] {
	if (typeof beforeRaw !== 'string' || typeof afterRaw !== 'string') return [];
	const a = nfc(beforeRaw).trim();
	const b = nfc(afterRaw).trim();
	if (!a || !b) return [];
	if (a === b) return [];
	if (/\n/.test(a) || /\n/.test(b)) return [];

	const out: TaxonomyId[] = [];

	if (ruleAccentTonos(a, b)) out.push('accent_tonos');
	if (ruleFinalSigma(a, b)) out.push('final_sigma');
	if (ruleWordBoundary(a, b)) out.push('word_boundary');
	if (ruleHomophone(a, b)) out.push('homophone');
	if (ruleAcronymAbbreviation(a, b)) out.push('acronym_abbreviation');
	if (ruleDisfluencyCleanup(a, b)) out.push('disfluency_cleanup');
	if (rulePunctuationCapitalization(a, b)) out.push('punctuation_capitalization');

	return out;
}

export function classifySorted(beforeRaw: string, afterRaw: string): TaxonomyId[] {
	return [...classify(beforeRaw, afterRaw)].sort();
}
