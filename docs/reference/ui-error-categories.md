# UI Error Categories

Proposed list of error categories for the exploration UI's classification dropdown. Each correction in the UI gets exactly one primary category.

These are **observable text-level error types**, not training-routing decisions. Routing (ASR fine-tuning vs LLM post-correction vs rules vs review) is decided **after** classification and lives in [error-taxonomy.md](error-taxonomy.md).

Each entry: English ID (for code) → Greek label (for UI) → short example (`before` → `after`).

## Phonetic / Acoustic

These are errors where the ASR misheard a sound. Most relevant for ASR fine-tuning.

1. **`homophone`** — Ομόηχα (η/ι/υ/ει/οι, ω/ο, αι/ε)
   `να γίνη` → `να γίνει`

2. **`accent_tonos`** — Τόνος / διακριτικά
   `δημος` → `δήμος`

3. **`final_sigma`** — Τελικό σίγμα (σ/ς)
   `οπωσ` → `όπως`

4. **`word_boundary`** — Όρια λέξεων (κόλλημα / διαχωρισμός)
   `τονδήμο` → `τον δήμο` · `απ' το` → `από το`

5. **`substitution_phonetic`** — Φωνητική αντικατάσταση λέξης (ηχητικά κοντινή λέξη, διαφορετικό νόημα)
   `έδωσε` → `έδειξε`

6. **`insertion`** — Πλεονάζουσα λέξη/συλλαβή που δεν ειπώθηκε
   `θα να πάμε` → `θα πάμε`

7. **`deletion`** — Παραλειφθείσα λέξη που ειπώθηκε
   `πάμε σπίτι` → `πάμε στο σπίτι`

## Morphological / Grammatical

Επιφανειακά κοντινά αλλά μορφολογικά διαφορετικά.

8. **`verb_inflection`** — Κλίση ρήματος (πρόσωπο, χρόνος, φωνή)
   `ψηφίζει` → `ψηφίζουμε`

9. **`noun_case`** — Πτώση / αριθμός ουσιαστικού ή επιθέτου
   `του δήμου` → `τον δήμο`

10. **`article_pronoun`** — Άρθρο ή αντωνυμία
    `ο πρόεδρος` → `του προέδρου`

## Named Entities / Domain

Συνήθως τα πιο κρίσιμα για fine-tuning σε δημοτικά πρακτικά.

11. **`person_name`** — Όνομα προσώπου (μέλος συμβουλίου, ομιλητής)
    `κύριε παπαδόπουλε` → `κύριε Παπαδόπουλε`

12. **`place_name`** — Τοπωνύμιο (δήμος, περιοχή, οδός)
    `στο πετράλωνα` → `στα Πετράλωνα`

13. **`org_party_name`** — Όνομα φορέα / παράταξης / οργανισμού
    `νέα δημοκρατία` → `Νέα Δημοκρατία`

14. **`acronym_abbreviation`** — Ακρωνύμιο ή σύντμηση
    `ε ε α` → `ΕΕΑ` · `δ ε υ α` → `ΔΕΥΑ`

15. **`legal_admin_term`** — Νομικός / διοικητικός όρος
    `αρθρο 75 παρ 2` → `άρθρο 75 παρ. 2`

16. **`number_date`** — Αριθμός, ημερομηνία, ποσό
    `δύο χιλιάδες είκοσι έξι` → `2026` · `πέντε εκατομμύρια ευρώ` → `5.000.000 €`

## Formatting / Non-substantive

Δεν αλλάζουν τι ειπώθηκε. Συνήθως **όχι** χρήσιμα για ASR fine-tuning.

17. **`punctuation_capitalization`** — Στίξη ή κεφαλαία/πεζά
    `ναι κύριε πρόεδρε` → `Ναι, κύριε Πρόεδρε.`

18. **`disfluency_cleanup`** — Καθαρισμός δισταγμών / επαναλήψεων
    `ε ε ναι ναι θα πάμε` → `ναι, θα πάμε`

## Meta / Quality

Όχι λάθος ASR αλλά πρόβλημα στο ίδιο το correction row.

19. **`semantic_rewrite`** — Παραφράζοντας: αλλάζει το νόημα, όχι το άκουσμα
    `πιστεύω ότι είναι σωστό` → `συμφωνώ απόλυτα`

20. **`timestamp_misalignment`** — Το text είναι σωστό αλλά τα start/end δεν δείχνουν στο σωστό audio span
    *(text δεν αλλάζει· σημείωση στο UI για χειροκίνητη διόρθωση timestamp)*

21. **`unusable`** — Junk / κενό / μη-ομιλία / αδύνατο να καταλάβει κανείς
    `[ασαφές]` → `[ασαφές]`

## Notes for the UI

- Ένα πεδίο "primary category" (single-select) + προαιρετικά "secondary" (multi-select) για cases που πέφτουν σε δύο (π.χ. και `homophone` και `person_name`).
- Default: ταξινόμηση κατά συχνότητα στο dataset μόλις έχουμε στατιστικά.
- Κάθε κατηγορία θα μαπάρει αργότερα σε **routing bucket** (training / post-correction / rules / drop) — αυτό δεν εμφανίζεται στο dropdown, ζει στο [error-taxonomy.md](error-taxonomy.md).
