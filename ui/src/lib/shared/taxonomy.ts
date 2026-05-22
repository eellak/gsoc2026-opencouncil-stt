// Error taxonomy for the correction review UI.
// Source of truth: docs/reference/ui-error-categories.md
//
// Each entry is an observable text-level error type. The optional shortcut binds
// a number key (1-9, 0) to the most common categories; entries without a shortcut
// are selectable only via the chip UI.

export type TaxonomyGroup = 'phonetic' | 'morphological' | 'named_entity' | 'formatting' | 'meta';

export interface TaxonomyEntry {
	id: string;
	en: string;
	el: string;
	group: TaxonomyGroup;
	example_before: string;
	example_after: string;
	shortcut?: string;
}

export const TAXONOMY: readonly TaxonomyEntry[] = [
	// ── Phonetic / Acoustic ──
	{ id: 'homophone', group: 'phonetic', en: 'Homophones (η/ι/υ/ει/οι, ω/ο, αι/ε)', el: 'Ομόηχα (η/ι/υ/ει/οι, ω/ο, αι/ε)', example_before: 'να γίνη', example_after: 'να γίνει', shortcut: '1' },
	{ id: 'accent_tonos', group: 'phonetic', en: 'Accent / diacritics', el: 'Τόνος / διακριτικά', example_before: 'δημος', example_after: 'δήμος', shortcut: '2' },
	{ id: 'final_sigma', group: 'phonetic', en: 'Final sigma (σ/ς)', el: 'Τελικό σίγμα (σ/ς)', example_before: 'οπωσ', example_after: 'όπως', shortcut: '3' },
	{ id: 'word_boundary', group: 'phonetic', en: 'Word boundary (merge/split)', el: 'Όρια λέξεων (κόλλημα / διαχωρισμός)', example_before: 'τονδήμο', example_after: 'τον δήμο', shortcut: '4' },
	{ id: 'substitution_phonetic', group: 'phonetic', en: 'Phonetic substitution', el: 'Φωνητική αντικατάσταση', example_before: 'έδωσε', example_after: 'έδειξε', shortcut: '5' },
	{ id: 'insertion', group: 'phonetic', en: 'Extra word/syllable inserted', el: 'Πλεονάζουσα λέξη/συλλαβή', example_before: 'θα να πάμε', example_after: 'θα πάμε' },
	{ id: 'deletion', group: 'phonetic', en: 'Spoken word missing', el: 'Παραλειφθείσα λέξη', example_before: 'πάμε σπίτι', example_after: 'πάμε στο σπίτι' },

	// ── Morphological / Grammatical ──
	{ id: 'verb_inflection', group: 'morphological', en: 'Verb inflection', el: 'Κλίση ρήματος', example_before: 'ψηφίζει', example_after: 'ψηφίζουμε', shortcut: '6' },
	{ id: 'noun_case', group: 'morphological', en: 'Noun/adjective case or number', el: 'Πτώση / αριθμός ουσιαστικού', example_before: 'του δήμου', example_after: 'τον δήμο' },
	{ id: 'article_pronoun', group: 'morphological', en: 'Article or pronoun', el: 'Άρθρο ή αντωνυμία', example_before: 'ο πρόεδρος', example_after: 'του προέδρου' },

	// ── Named entities / Domain ──
	{ id: 'person_name', group: 'named_entity', en: 'Person name', el: 'Όνομα προσώπου', example_before: 'κύριε παπαδόπουλε', example_after: 'κύριε Παπαδόπουλε', shortcut: '7' },
	{ id: 'place_name', group: 'named_entity', en: 'Place name', el: 'Τοπωνύμιο', example_before: 'στο πετράλωνα', example_after: 'στα Πετράλωνα' },
	{ id: 'org_party_name', group: 'named_entity', en: 'Organization / party name', el: 'Φορέας / παράταξη / οργανισμός', example_before: 'νέα δημοκρατία', example_after: 'Νέα Δημοκρατία' },
	{ id: 'acronym_abbreviation', group: 'named_entity', en: 'Acronym / abbreviation', el: 'Ακρωνύμιο / σύντμηση', example_before: 'δ ε υ α', example_after: 'ΔΕΥΑ', shortcut: '8' },
	{ id: 'legal_admin_term', group: 'named_entity', en: 'Legal / admin term', el: 'Νομικός / διοικητικός όρος', example_before: 'αρθρο 75 παρ 2', example_after: 'άρθρο 75 παρ. 2' },
	{ id: 'number_date', group: 'named_entity', en: 'Number, date, amount', el: 'Αριθμός, ημερομηνία, ποσό', example_before: 'πέντε εκατομμύρια ευρώ', example_after: '5.000.000 €' },

	// ── Formatting / Non-substantive ──
	{ id: 'punctuation_capitalization', group: 'formatting', en: 'Punctuation or capitalization', el: 'Στίξη ή κεφαλαία/πεζά', example_before: 'ναι κύριε πρόεδρε', example_after: 'Ναι, κύριε Πρόεδρε.', shortcut: '9' },
	{ id: 'disfluency_cleanup', group: 'formatting', en: 'Disfluency / filler cleanup', el: 'Καθαρισμός δισταγμών / επαναλήψεων', example_before: 'ε ε ναι ναι θα πάμε', example_after: 'ναι, θα πάμε', shortcut: '0' },

	// ── Meta / Quality ──
	{ id: 'semantic_rewrite', group: 'meta', en: 'Semantic rewrite (paraphrase)', el: 'Παράφραση: αλλάζει το νόημα', example_before: 'πιστεύω ότι είναι σωστό', example_after: 'συμφωνώ απόλυτα' },
	{ id: 'timestamp_misalignment', group: 'meta', en: 'Timestamp misalignment', el: 'Λάθος ευθυγράμμιση χρόνου', example_before: '(text unchanged)', example_after: '(adjust start/end)' },
	{ id: 'unusable', group: 'meta', en: 'Unusable / unclear', el: 'Άχρηστο / ασαφές', example_before: '[ασαφές]', example_after: '[ασαφές]' }
];

// Active taxonomy id union — keep in sync with TAXONOMY above.
export type TaxonomyId =
	| 'homophone' | 'accent_tonos' | 'final_sigma' | 'word_boundary'
	| 'substitution_phonetic' | 'insertion' | 'deletion'
	| 'verb_inflection' | 'noun_case' | 'article_pronoun'
	| 'person_name' | 'place_name' | 'org_party_name' | 'acronym_abbreviation'
	| 'legal_admin_term' | 'number_date'
	| 'punctuation_capitalization' | 'disfluency_cleanup'
	| 'semantic_rewrite' | 'timestamp_misalignment' | 'unusable';

export const TAXONOMY_MAP: Record<TaxonomyId, TaxonomyEntry> = Object.fromEntries(
	TAXONOMY.map((t) => [t.id, t])
) as Record<TaxonomyId, TaxonomyEntry>;

export const TAXONOMY_GROUP_ORDER: TaxonomyGroup[] = [
	'phonetic',
	'morphological',
	'named_entity',
	'formatting',
	'meta'
];

export const TAXONOMY_GROUP_LABELS: Record<TaxonomyGroup, { en: string; el: string }> = {
	phonetic: { en: 'Phonetic / Acoustic', el: 'Φωνητικά / Ακουστικά' },
	morphological: { en: 'Morphological / Grammatical', el: 'Μορφολογικά / Γραμματικά' },
	named_entity: { en: 'Named entities / Domain', el: 'Ονόματα / Όροι τομέα' },
	formatting: { en: 'Formatting / Non-substantive', el: 'Μορφοποίηση / Μη ουσιαστικά' },
	meta: { en: 'Meta / Quality', el: 'Μετα / Ποιότητα' }
};

// Older builds of the UI persisted a smaller 12-category set. Translate legacy
// ids on read so existing labels keep rendering. Keep this map alongside the
// active TAXONOMY; do not migrate the DB.
export const LEGACY_TAXONOMY_MAP: Record<string, TaxonomyId> = {
	asr_phonetic: 'substitution_phonetic',
	domain_term: 'legal_admin_term',
	named_entity: 'person_name',
	punctuation: 'punctuation_capitalization',
	capitalization: 'punctuation_capitalization',
	formatting_only: 'punctuation_capitalization',
	semantic_context: 'semantic_rewrite',
	speaker_or_attribution: 'person_name',
	timestamp_alignment: 'timestamp_misalignment',
	same_or_no_meaningful_change: 'unusable',
	malformed_or_unclear: 'unusable'
};

export function normalizeTaxonomyId(id: string | null | undefined): TaxonomyId | null {
	if (!id) return null;
	if (id in TAXONOMY_MAP) return id as TaxonomyId;
	return LEGACY_TAXONOMY_MAP[id] ?? null;
}

export type Lang = 'el' | 'en';

export function taxonomyLabel(id: string | null | undefined, lang: Lang): string {
	const normalized = normalizeTaxonomyId(id);
	if (!normalized) return '';
	return TAXONOMY_MAP[normalized]?.[lang] ?? normalized;
}
