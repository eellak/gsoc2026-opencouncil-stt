export interface IngestCategory {
	key: string;
	label_el: string;
	reason_el: string;
	is_rejected: boolean;
}

export const INGEST_CATEGORIES: IngestCategory[] = [
	{
		key: 'clean',
		label_el: 'Καθαρό',
		reason_el:
			'Η εγγραφή πέρασε όλους τους ελέγχους και είναι υποψήφια για το dataset εκπαίδευσης.',
		is_rejected: false
	},
	{
		key: 'noop_edit',
		label_el: 'Ίδιο πριν/μετά',
		reason_el:
			'Το πεδίο before_text ισούται ακριβώς με το after_text — δεν έγινε καμία πραγματική διόρθωση. Τέτοιες εγγραφές δεν προσφέρουν εκπαιδευτική αξία.',
		is_rejected: true
	},
	{
		key: 'whitespace_only',
		label_el: 'Μόνο κενά',
		reason_el:
			'Το before και το after διαφέρουν μόνο σε κενά, tabs ή αλλαγές γραμμής — η ουσία του κειμένου δεν άλλαξε. Δεν είναι χρήσιμο ως training pair.',
		is_rejected: true
	},
	{
		key: 'empty_before',
		label_el: 'Προσθήκη από το μηδέν',
		reason_el:
			'Κενό before_text — ο χρήστης πρόσθεσε κείμενο εκεί που το ASR δεν είχε γράψει τίποτα. Νόμιμη διόρθωση, δύσκολη ως training pair γιατί δεν υπάρχει αρχικό ASR output.',
		is_rejected: false
	},
	{
		key: 'empty_after',
		label_el: 'Διαγραφή',
		reason_el:
			'Κενό after_text — ο χρήστης διέγραψε τη μεταγραφή. Ενδεικτικό false-positive του ASR (μεταγράφηκε κάτι που δεν έπρεπε). Αξιόλογο για ανάλυση, όχι απαραίτητα για εκπαίδευση.',
		is_rejected: false
	},
	{
		key: 'embedded_reasoning',
		label_el: 'Ίχνη σκέψης μοντέλου',
		reason_el:
			'Το before_text περιέχει λεκτικά ίχνη σκέψης του γλωσσικού μοντέλου που δημιούργησε την αρχική μεταγραφή (π.χ. «Wait, let me reconsider…»). Αυτό δεν είναι πραγματικό ASR output αλλά διαρροή εσωτερικής λογικής.',
		is_rejected: true
	},
	{
		key: 'reversed_timestamps',
		label_el: 'Αντεστραμμένα timestamps',
		reason_el:
			'Το utterance_end είναι μικρότερο από το utterance_start — artifact ακρίβειας. Μικρές αποκλίσεις (<0.05s) διορθώθηκαν αυτόματα. Μεγαλύτερες χαρακτηρίστηκαν ως μη αξιόπιστες.',
		is_rejected: false
	},
	{
		key: 'multiline_text',
		label_el: 'Πολυγραμμικό κείμενο',
		reason_el:
			'Το κείμενο περιέχει νόμιμες αλλαγές γραμμής (π.χ. λίστες ή παραγράφους). Μόνο πληροφοριακή σήμανση — δεν απορρίπτεται αυτόματα.',
		is_rejected: false
	}
];

export const INGEST_CATEGORY_MAP = new Map(INGEST_CATEGORIES.map((c) => [c.key, c]));
