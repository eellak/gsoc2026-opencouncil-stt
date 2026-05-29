/**
 * Render the external-LLM categorization prompt from the active TAXONOMY.
 *
 * Composed once per batch (server-side). Items are appended as a compact JSON
 * array using short keys (`id`, `b`, `a`) so token spend per item is minimal.
 */

import { TAXONOMY } from '$lib/shared/taxonomy';

export interface BatchItem {
	id: string;
	b: string;
	a: string;
}

const RUBRIC = [
	'You are categorizing Greek ASR (automatic speech recognition) correction pairs.',
	'Each item has `b` (before / ASR raw) and `a` (after / human-corrected).',
	'',
	'Be analytical and PRODUCTIVE. A single pair often contains multiple errors at once',
	'(e.g. a phonetic mistake AND a capitalization fix AND a tonos fix). Pick up to 3',
	'categories that together explain the change. Use your linguistic understanding of',
	'Greek to classify hard / messy cases — ASR data is naturally noisy.',
	'',
	'Omit an item ONLY when:',
	'  • b and a are literally identical (zero change), OR',
	'  • the text is so garbled or truncated that no category honestly applies.',
	'In every other case, return at least one category.'
].join('\n');

const RULES = [
	'RULES & DISAMBIGUATION:',
	'- homophone is STRICT to η/ι/υ/ει/οι, ω/ο, αι/ε. Other vowel/consonant swaps → substitution_phonetic.',
	'- α↔ο, π↔φ, σ↔ψ etc → substitution_phonetic (NOT homophone).',
	'- ASR misheard one word as a phonetically similar but different word → substitution_phonetic.',
	'- Human truly paraphrased / changed meaning (not a mishear) → semantic_rewrite.',
	'- Punctuation / capitalization only → punctuation_capitalization.',
	'- Filler / repetition removal (ε, εε, ναι ναι, λοιπόν, εντάξει εντάξει) → disfluency_cleanup.',
	'- Tonos changes only → accent_tonos. Final σ vs ς only → final_sigma.',
	'- Numbers / dates / amounts spelled-out vs digit form → number_date.',
	'- Names (people, places, parties, organizations) → person_name / place_name / org_party_name.',
	'- Acronyms expanded or assembled (δ ε υ α → ΔΕΥΑ) → acronym_abbreviation.',
	'- Multiple errors at once → list all applicable categories (max 3).'
].join('\n');

const OUTPUT = [
	'OUTPUT FORMAT — return exactly this:',
	'',
	'BATCH_ID: <id>',
	'[',
	'  {',
	'    "id": "<id from input>",',
	'    "thought": "<one short sentence in Greek or English explaining what changed and why these categories>",',
	'    "c": ["<category_id_1>", "<category_id_2>"]',
	'  },',
	'  ...',
	']',
	'',
	'- The "thought" field forces you to analyze before deciding — keep it under ~20 words.',
	'- Use exact category ids from the list above. Max 3 categories per item.',
	'- Return at least one category whenever b != a unless the pair is truly garbled.',
	'- No markdown fences (no ```), no prose outside the JSON array, no JSONL.'
].join('\n');

function renderTaxonomy(): string {
	const lines: string[] = ['CATEGORIES (use exact id):'];
	for (const t of TAXONOMY) {
		const example = `${t.example_before} → ${t.example_after}`;
		lines.push(`- ${t.id} — ${t.en}  (e.g. ${example})`);
	}
	return lines.join('\n');
}

/**
 * Build the full prompt for a batch. The items array is included verbatim
 * inside the prompt as a fenced JSON block, so a single paste hands the
 * model both instructions and data.
 *
 * The prompt is bracketed by the `BATCH_ID:` marker so the ingest path can
 * detect operator paste-mismatches (one Gemini tab's reply accidentally
 * dropped into another batch's ingest box).
 */
export function renderPrompt(batch_id: string, items: BatchItem[]): string {
	const payload = JSON.stringify(items);
	const idHeader = `BATCH_ID: ${batch_id}`;
	const tellModel = [
		`This is batch ${batch_id}. Begin your reply with the exact line:`,
		idHeader,
		'Then on the next line return the JSON array described below.',
		'Do not omit the BATCH_ID line — the ingest tool uses it to verify the reply belongs to this batch.'
	].join('\n');
	return [
		idHeader,
		'',
		RUBRIC,
		'',
		tellModel,
		'',
		renderTaxonomy(),
		'',
		RULES,
		'',
		OUTPUT,
		'',
		'INPUT (UTF-8 JSON, do NOT modify id values):',
		payload
	].join('\n');
}
