# CPU census: σκάλα φιλτραρίσματος training data

Η σκάλα και τα diagnostics παγώθηκαν **πριν** τρέξει το census στο
[`2026-08-18-clean-data-filter-ladder.md`](../specs/2026-08-18-clean-data-filter-ladder.md).
Δεν χρησιμοποιήθηκαν GPU, API, validation/test/holdout δεδομένα. Το aggregate
αποτέλεσμα είναι στο
[`results_training_data_filter_census.json`](../../eval/results_training_data_filter_census.json).

## Τι μετρήθηκε

Το scope είναι τα σημερινά external packs `hparl2-v2`, `stoma-v2`, `cv-v2` και
`eurospeech-v2`, **χωριστά ανά source**. Το L0 είναι το σημερινό χαλαρό pack, το L1
απαιτεί τον ήδη αποθηκευμένο Soniox witness, το L2 απαιτεί `align >= 0.95` και
καθαρά edges, και το L3 απαιτεί και δεύτερο ASR witness με WER `<= 0.05`.

| level | rows | ώρες | ερμηνεία |
|---|---:|---:|---|
| L0 | 50.848 | 111,027 | σημερινή προσφορά |
| L1 | 50.848 | 111,027 | ίδιο με L0: κάθε row των v2 packs έχει ήδη Soniox witness |
| L2 | 30.044 | 60,069 | 59,1% των rows / 54,1% των ωρών επιβιώνει |
| L3 | 1.063 | 4,146 | μόνο EuroSpeech· 23.311 L2 passers των άλλων sources είναι unmeasurable, όχι failures |

Το aggregate δεν είναι recipe: HParl και EuroSpeech δεν επιτρέπεται να ενωθούν
σιωπηρά, επειδή ανήκουν στο ίδιο source domain και έχουν μη ελεγμένο overlap.

## Τι επιβιώνει ανά source

| source | L0 rows / h | L2 rows / h | L2 row retention |
|---|---:|---:|---:|
| HParl2 | 9.334 / 13,161 | 4.360 / 5,956 | 46,7% |
| STOMA | 14.407 / 24,187 | 10.332 / 16,823 | 71,7% |
| Common Voice | 12.971 / 15,108 | 8.619 / 9,723 | 66,5% |
| EuroSpeech | 14.136 / 58,571 | 6.733 / 27,567 | 47,6% |

Άρα το L2 έχει αρκετή χωρητικότητα για equal-hours screen. Δεν είναι όμως ουδέτερο
ως προς το είδος των δεδομένων που αφαιρεί.

## Representation guards

Τα diagnostics είναι proxies, όχι ground truth. Το `known_roster_name` κάνει exact
normalized match σε πλήρες όνομα από τα υπάρχοντα public rosters και χάνει
εξωτερικά ή κλιτά ονόματα. Το `capitalized_noninitial` είναι ευρύτερο αλλά θορυβώδες.

| diagnostic | L0 rows | L2 rows | row retention | L2 hour retention |
|---|---:|---:|---:|---:|
| γνωστό πλήρες roster name | 911 | 469 | 51,5% | 47,6% |
| κεφαλαίο μη αρχικό token | 19.146 | 9.366 | 48,9% | 46,7% |
| γρήγορη ομιλία (≥3 tokens/s) | 2.053 | 900 | 43,8% | 41,3% |
| hard example (`align < .85` ή clipped edge) | 11.959 | 0 | 0% | 0% |

Το συνολικό L2 row retention είναι 59,1%, άρα τόσο τα δύο name proxies όσο και η
γρήγορη ομιλία πέφτουν δυσανάλογα. Το hard-example retention είναι μηδέν **εκ
κατασκευής**: ο ίδιος κανόνας που ορίζει το L2 απομακρύνει αυτή την κατηγορία.
Ανά source, η διατήρηση γνωστών roster names είναι HParl 43,0%, STOMA 63,6%, Common
Voice 53,7%, EuroSpeech 43,7%. Η γρήγορη ομιλία είναι HParl 27,5%, STOMA 68,9%,
Common Voice 61,8%, EuroSpeech 34,0%.

## L3

Μόνο το EuroSpeech έχει ήδη δεύτερο witness: το corpus-provided Whisper-Turbo
`ds_wer`. Το L3 κρατά 1.063 rows / 4,146 h, δηλαδή 7,5% των EuroSpeech L0 rows.
Κρατά μόνο 18 από τα 403 known-roster-name rows και 9 από τα 300 fast rows.

Αυτό δεν είναι «εγγυημένα καθαρό». Είναι δείγμα συμφωνίας Soniox + Whisper-Turbo,
άρα επιλέγει ειδικά ό,τι είναι ήδη εύκολο για την ίδια οικογένεια μοντέλου που θα
εκπαιδεύσουμε. Δεν δικαιολογεί αγορά δεύτερων witnesses για τα άλλα sources.

## Απόφαση CPU preflight

Η ιδέα **περνά το capacity check αλλά αποτυγχάνει ως wholesale replacement**. Το
L2 αφήνει 60 ώρες, όμως καθαρίζει το dataset από όλα τα μετρημένα hard examples και
δυσανάλογα από names/fast speech — ακριβώς τις περιοχές που θέλουμε να βελτιώσουμε.
Δεν υπάρχει βάση να το ονομάσουμε καλύτερο training data και δεν προτείνεται GPU για
το ακατέργαστο «L0 εναντίον L2 παντού».

Αν θελήσουμε να συνεχίσουμε την υπόθεση, το επόμενο evidence-producing βήμα είναι
CPU/ανθρώπινο και τυφλό: audio audit retained/rejected rows ανά source, με
προκαθορισμένα name/fast/hard strata. Μετά μπορεί να οριστεί νέο treatment ως L2
clean core **συν προστατευμένο audited hard/name lane**. Αυτό είναι διαφορετικό arm
και πρέπει να παγώσει πριν δούμε training αποτέλεσμα.

Για οποιαδήποτε μελλοντική σύγκριση, το equal-hours σχέδιο είναι source-specific:
το L2 δίνει τις ώρες-στόχο του πίνακα και το L0 υποδειγματοληπτείται τυχαία στις
ίδιες ώρες με seeds 13/29/47. Ίδια optimizer updates, recipe και decode. Δεν κάνουμε
matching στα diagnostics, επειδή η αλλαγή της αναλογίας τους είναι μέρος της
παρέμβασης. Το L1 δεν χρειάζεται ξεχωριστό arm, αφού είναι byte-for-byte το ίδιο
population με το L0.

## Περιορισμός για το in-domain backbone

Το core OpenCouncil training parquet δεν μπήκε τεχνητά στη σκάλα: δεν έχει
ανεξάρτητο ASR witness για κάθε row. Υπάρχει witness μόνο σε επιλεγμένα audit/gap
subsets, οπότε L1 πάνω σε αυτά θα μετρούσε τον τρόπο επιλογής τους, όχι καθαρότητα.
Με τα σημερινά caches μπορούμε να εγγυηθούμε reproducible φίλτρα και μετρήσιμη
retention· δεν μπορούμε να εγγυηθούμε καλύτερες in-domain labels χωρίς νέο τυφλό
audio audit ή πλήρη independent-witness pass.
